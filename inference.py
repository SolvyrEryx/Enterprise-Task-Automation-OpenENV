#!/usr/bin/env python
"""
Inference Script — Enterprise Task Automation Environment
Meta AI OpenEnv Hackathon

Reads credentials from env vars:
  OPENAI_API_KEY  (required — standard key)
  HF_TOKEN        (fallback if OPENAI_API_KEY not set)
  MODEL_NAME      (optional, default gpt-3.5-turbo / gpt-4o-mini for hard)
  API_BASE_URL    (optional override)

Usage:
  python inference.py --task easy   --seed 42
  python inference.py --task medium --seed 42
  python inference.py --task hard   --seed 42
  python inference.py --task all    --seed 42
"""

import sys
import json
import argparse
import time
import os
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from src import EnterpriseEnv, Action, ActionType, EmailPriority, EmailCategory, TaskPriority
from src.graders import get_task_graders
from src.types import TaskStatus


# ──────────────────────────────────────────────────────────────────────────────
# Structured logging  (hackathon required: [START] [STEP] [END])
# ──────────────────────────────────────────────────────────────────────────────

class StructuredLogger:
    def __init__(self):
        self.start_time = None
        self.step_count = 0

    def start(self, config: Dict[str, Any]):
        self.start_time = time.time()
        self.step_count = 0
        print(f"[START] {json.dumps({'event':'START','timestamp':datetime.now().isoformat(),'config':config})}")

    def step(self, step_num: int, action: str, reward: float, obs_summary: Dict):
        self.step_count += 1
        print(f"[STEP] {json.dumps({'event':'STEP','step':step_num,'action':action,'reward':round(reward,4),'obs':obs_summary,'timestamp':datetime.now().isoformat()})}")

    def end(self, final_score: float, metadata: Dict):
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[END] {json.dumps({'event':'END','final_score':round(final_score,4),'total_steps':self.step_count,'elapsed_seconds':round(elapsed,2),'metadata':metadata,'timestamp':datetime.now().isoformat()})}")


# ──────────────────────────────────────────────────────────────────────────────
# State analysis helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_conflict_free_slots(obs, n: int = 20, step_minutes: int = 90) -> List[str]:
    """
    Returns up to n ISO timestamps that do not overlap any already-scheduled meeting.
    Each chosen slot is immediately reserved so successive calls don't collide.
    """
    windows: List[Tuple[datetime, datetime]] = [
        (m.scheduled_time, m.scheduled_time + timedelta(minutes=m.duration_minutes))
        for m in obs.meetings
        if m.status == "scheduled" and m.scheduled_time
    ]
    slots: List[str] = []
    base = obs.timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    for i in range(96):
        candidate = base + timedelta(minutes=i * step_minutes)
        cand_end  = candidate + timedelta(minutes=step_minutes)
        if any((candidate < we) and (ws < cand_end) for ws, we in windows):
            continue
        slots.append(candidate.strftime("%Y-%m-%dT%H:%M:00"))
        windows.append((candidate, cand_end))
        if len(slots) >= n:
            break
    return slots


def get_ready_tasks(obs) -> List:
    """PENDING tasks whose every dependency is COMPLETED, sorted by deadline."""
    done_ids = {t.task_id for t in obs.tasks if t.status == TaskStatus.COMPLETED}
    ready = [
        t for t in obs.tasks
        if t.status == TaskStatus.PENDING
        and all(dep in done_ids for dep in (t.dependencies or []))
    ]
    ready.sort(key=lambda t: t.deadline)
    return ready


def get_blocked_tasks(obs) -> List:
    done_ids = {t.task_id for t in obs.tasks if t.status == TaskStatus.COMPLETED}
    return [
        t for t in obs.tasks
        if t.status == TaskStatus.PENDING
        and any(dep not in done_ids for dep in (t.dependencies or []))
    ]


def get_unblocked_pending(obs) -> List:
    return [t for t in obs.tasks if t.status == TaskStatus.PENDING]


def current_score_breakdown(obs, metadata: Dict) -> Dict[str, float]:
    """Mirrors HardTask.grade() math for live score display in prompts."""
    email_score = min(obs.email_triage_rate / 0.90, 1.0)

    sched_r = obs.meeting_schedule_success_rate
    conf_p  = max(0.0, 1.0 - obs.meeting_conflicts * 0.15)
    meeting_score = (sched_r + conf_p) / 2.0

    total_t = max(len(obs.tasks), 1)
    ov_pen  = max(0.0, 1.0 - (obs.overdue_tasks_count / total_t) * 2.0)
    task_score = (obs.task_completion_rate + ov_pen) / 2.0

    esc_p = max(0.0, 1.0 - metadata.get("escalations", 0) * 0.20)
    dl_p  = max(0.0, 1.0 - metadata.get("deadline_misses", 0) * 0.15)
    health_score = (esc_p + dl_p) / 2.0

    return {
        "email":   round(email_score,   3),
        "meeting": round(meeting_score, 3),
        "task":    round(task_score,    3),
        "health":  round(health_score,  3),
        "total":   round((email_score + meeting_score + task_score + health_score) / 4.0, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders — one per difficulty
# ──────────────────────────────────────────────────────────────────────────────

def _easy_prompt(obs) -> str:
    unprocessed = sorted([e for e in obs.emails if not e.processed], key=lambda e: -e.urgency)
    email_lines = "\n".join(
        f"  {e.email_id}: urgency={e.urgency}  sender={e.sender}  \"{e.subject}\""
        for e in unprocessed[:12]
    )
    triaged = sum(1 for e in obs.emails if e.processed)
    total   = max(len(obs.emails), 1)
    need    = max(0, int(total * 0.80) - triaged + 1)

    return f"""You are an enterprise email triage agent.

TIME: {obs.timestamp.strftime("%Y-%m-%d %H:%M")}   STEP: {obs.step}
PROGRESS: {triaged}/{total} emails triaged ({obs.email_triage_rate:.0%}) — need {need} more to reach 80%

UNPROCESSED EMAILS (highest urgency first):
{email_lines if email_lines else "  (all triaged)"}

TRIAGE RULES:
  urgency 9-10 → category=urgent        priority=critical
  urgency 7-8  → category=urgent        priority=high
  urgency 5-6  → category=actionable    priority=medium
  urgency 3-4  → category=actionable    priority=low
  urgency 0-2  → category=informational priority=low

Triage the FIRST email listed.  If none remain, noop.
Respond with ONE JSON only:
{{"action":"triage_email","email_id":"email_X","category":"urgent","priority":"critical"}}
{{"action":"noop"}}"""


def _medium_prompt(obs, metadata: Dict) -> str:
    pending_meetings = sorted(
        [m for m in obs.meetings if m.status == "pending"],
        key=lambda m: -m.business_impact
    )
    # Pre-compute one conflict-free slot per pending meeting
    safe_slots = get_conflict_free_slots(obs, n=len(pending_meetings) + 5, step_minutes=90)
    slot_map = {m.meeting_id: safe_slots[i] for i, m in enumerate(pending_meetings) if i < len(safe_slots)}

    meeting_lines = "\n".join(
        f"  {m.meeting_id}: impact={m.business_impact:.2f}  dur={m.duration_minutes}min"
        f"  → SCHEDULE TO: {slot_map.get(m.meeting_id, 'NO_SLOT')}"
        for m in pending_meetings[:8]
    )

    needs_reprio = [
        t for t in get_unblocked_pending(obs)
        if (t.impact >= 0.7 or t.urgency >= 7)
        and t.priority.value not in ("critical", "high")
    ]
    reprio_lines = "\n".join(
        f"  {t.task_id}: impact={t.impact:.2f} urgency={t.urgency} current={t.priority.value}"
        for t in needs_reprio[:6]
    )

    triaged = sum(1 for e in obs.emails if e.processed)
    total_e = max(len(obs.emails), 1)
    sched   = sum(1 for m in obs.meetings if m.status == "scheduled")
    total_m = max(len(obs.meetings), 1)
    reprio_done = sum(
        1 for t in obs.tasks
        if (t.impact >= 0.7 or t.urgency >= 7) and t.priority.value in ("critical", "high")
    )
    high_impact_total = max(sum(1 for t in obs.tasks if t.impact >= 0.7 or t.urgency >= 7), 1)

    if pending_meetings:
        focus = "SCHEDULE MEETINGS NOW (use exact slot shown)"
    elif needs_reprio:
        focus = "REPRIORITISE TASKS"
    else:
        focus = "TRIAGE EMAILS"

    return f"""You are an enterprise workflow agent.  MEDIUM difficulty.

TIME: {obs.timestamp.strftime("%Y-%m-%d %H:%M")}   STEP: {obs.step}

SCORE:
  Meetings : {sched}/{total_m} scheduled ({obs.meeting_schedule_success_rate:.0%})  conflicts={obs.meeting_conflicts}  [target 70%+, 0 conflicts]
  Reprio   : {reprio_done}/{high_impact_total} high-impact tasks set to HIGH/CRITICAL  [target 60%+]
  Email    : {triaged}/{total_e} triaged ({obs.email_triage_rate:.0%})

FOCUS THIS STEP: {focus}

PENDING MEETINGS — use EXACT slot, no other time:
{meeting_lines if meeting_lines else "  (none pending)"}

TASKS TO REPRIORITISE (impact>=0.7 or urgency>=7, not yet HIGH/CRITICAL):
{reprio_lines if reprio_lines else "  (all done)"}

RULES:
  1. Pending meeting exists → schedule it to its listed slot RIGHT NOW.
  2. No meetings pending → reprioritise first task above to "high".
  3. All done → triage an email.
  4. NEVER use a time not in the slot list above.
  5. NEVER escalate.

Respond with ONE JSON only:
{{"action":"schedule_meeting","meeting_id":"meeting_X","scheduled_time":"SLOT_FROM_LIST"}}
{{"action":"reprioritize_task","task_id":"task_X","new_priority":"high"}}
{{"action":"triage_email","email_id":"email_X","category":"urgent","priority":"high"}}
{{"action":"noop"}}"""


def _hard_prompt(obs, metadata: Dict) -> str:
    now_str = obs.timestamp.strftime("%Y-%m-%d %H:%M")
    sc = current_score_breakdown(obs, metadata)

    unprocessed = sorted([e for e in obs.emails if not e.processed], key=lambda e: -e.urgency)
    email_lines = "\n".join(
        f"  {e.email_id}: urgency={e.urgency}  \"{e.subject}\""
        for e in unprocessed[:10]
    )

    pending_meetings = sorted(
        [m for m in obs.meetings if m.status == "pending"],
        key=lambda m: -m.business_impact
    )
    safe_slots = get_conflict_free_slots(obs, n=len(pending_meetings) + 8, step_minutes=90)
    slot_map = {m.meeting_id: safe_slots[i] for i, m in enumerate(pending_meetings) if i < len(safe_slots)}
    meeting_lines = "\n".join(
        f"  {m.meeting_id}: impact={m.business_impact:.2f}  dur={m.duration_minutes}min"
        f"  → USE SLOT: {slot_map.get(m.meeting_id, 'NONE')}"
        for m in pending_meetings[:10]
    )

    ready_tasks   = get_ready_tasks(obs)
    blocked_tasks = get_blocked_tasks(obs)

    ready_lines = "\n".join(
        f"  {t.task_id}: urgency={t.urgency} impact={t.impact:.2f}"
        f" deadline_in={max(0,int((t.deadline-obs.timestamp).total_seconds()/3600))}h"
        + (" ← URGENT" if (t.deadline - obs.timestamp).total_seconds() < 10800 else "")
        for t in ready_tasks[:10]
    )
    blocked_lines = "\n".join(
        f"  {t.task_id}: waiting on [{', '.join(t.dependencies)}]"
        for t in blocked_tasks[:6]
    )

    # Explicit next-action hint — removes ambiguity
    if sc["email"] < 1.0 and unprocessed:
        hint = f"TRIAGE {unprocessed[0].email_id}  (email score {sc['email']:.2f}, needs 1.0)"
    elif pending_meetings:
        m    = pending_meetings[0]
        slot = slot_map.get(m.meeting_id, "NONE")
        hint = f"SCHEDULE {m.meeting_id} → {slot}"
    elif ready_tasks:
        urgent = [t for t in ready_tasks if (t.deadline - obs.timestamp).total_seconds() < 10800]
        t = (urgent or ready_tasks)[0]
        hint = f"COMPLETE {t.task_id}  (deadline in {max(0,int((t.deadline-obs.timestamp).total_seconds()/3600))}h)"
    elif unprocessed:
        hint = f"TRIAGE {unprocessed[0].email_id}"
    else:
        hint = "noop — all work done"

    return f"""Enterprise AI — HARD mode.  All four scores need >= 0.90.

TIME: {now_str}   STEP: {obs.step}   REMAINING: {obs.time_until_end} min

SCORES (target each >= 0.90):
  Email   {sc['email']:.3f}  ({obs.email_triage_rate:.0%} triaged, 90% target)
  Meeting {sc['meeting']:.3f}  ({obs.meeting_schedule_success_rate:.0%} scheduled, {obs.meeting_conflicts} conflicts)
  Task    {sc['task']:.3f}  ({obs.task_completion_rate:.0%} done, {obs.overdue_tasks_count} overdue)
  Health  {sc['health']:.3f}  (escalations={metadata.get('escalations',0)}, deadline_misses={metadata.get('deadline_misses',0)})
  TOTAL   {sc['total']:.3f}

DO THIS NOW: {hint}

UNPROCESSED EMAILS ({len(unprocessed)}):
{email_lines if email_lines else "  (all triaged)"}
  Rules: urgency 9-10→urgent/critical | 7-8→urgent/high | 5-6→actionable/medium | <=4→informational/low

PENDING MEETINGS — use ONLY pre-assigned slot:
{meeting_lines if meeting_lines else "  (none)"}

READY TASKS (safe to complete):
{ready_lines if ready_lines else "  (none ready)"}

BLOCKED TASKS — do NOT touch:
{blocked_lines if blocked_lines else "  (none)"}

ABSOLUTE RULES:
  1. NEVER escalate — costs 20% health per use.
  2. NEVER schedule to a slot not listed above.
  3. NEVER complete a BLOCKED task.
  4. Email score < 1.0 AND emails exist → triage BEFORE completing tasks.

ONE JSON only, no explanation:
{{"action":"triage_email","email_id":"email_X","category":"urgent","priority":"critical"}}
{{"action":"schedule_meeting","meeting_id":"meeting_X","scheduled_time":"YYYY-MM-DDTHH:MM:00"}}
{{"action":"complete_task","task_id":"task_X"}}
{{"action":"reprioritize_task","task_id":"task_X","new_priority":"high"}}
{{"action":"noop"}}"""


def obs_to_prompt(obs, task_difficulty: str, metadata: Optional[Dict] = None) -> str:
    if metadata is None:
        metadata = {}
    if task_difficulty == "easy":
        return _easy_prompt(obs)
    elif task_difficulty == "medium":
        return _medium_prompt(obs, metadata)
    else:
        return _hard_prompt(obs, metadata)


# ──────────────────────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_llm_response(response_text: str, obs) -> Action:
    """Parse LLM JSON → Action. Smart heuristic fallback on any failure."""
    try:
        text = response_text.strip()
        # Strip markdown fences
        if "```" in text:
            for part in text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    text = part
                    break
        # Take first valid JSON line
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    json.loads(line)
                    text = line
                    break
                except Exception:
                    continue

        data = json.loads(text)
        action_str = data.get("action", "noop")

        if action_str == "triage_email":
            try:
                cat = EmailCategory(data.get("category", "actionable"))
            except ValueError:
                cat = EmailCategory.ACTIONABLE
            try:
                pri = EmailPriority(data.get("priority", "medium"))
            except ValueError:
                pri = EmailPriority.MEDIUM
                
            # FIX: Use .get() and validate to prevent KeyError
            email_id = data.get("email_id")
            if not email_id:
                raise ValueError("LLM forgot to provide email_id")

            return Action(action_type=ActionType.TRIAGE_EMAIL,
                          email_id=email_id, category=cat, priority=pri)

        elif action_str == "schedule_meeting":
            meeting_id = data.get("meeting_id")
            if not meeting_id:
                raise ValueError("LLM forgot to provide meeting_id")

            raw = data.get("scheduled_time", "")
            if raw.count(":") == 1:
                raw += ":00"
                
            # FIX: Catch ValueError if LLM hallucinates a non-ISO date string
            try:
                sched_time = datetime.fromisoformat(raw)
            except Exception:
                raise ValueError("LLM provided a malformed datetime string")

            return Action(action_type=ActionType.SCHEDULE_MEETING,
                          meeting_id=meeting_id,
                          scheduled_time=sched_time)

        elif action_str == "reprioritize_task":
            task_id = data.get("task_id")
            if not task_id:
                raise ValueError("LLM forgot to provide task_id")

            try:
                new_pri = TaskPriority(data.get("new_priority", "high"))
            except ValueError:
                new_pri = TaskPriority.HIGH

            return Action(action_type=ActionType.REPRIORITIZE_TASK,
                          task_id=task_id, new_priority=new_pri)

        elif action_str == "complete_task":
            task_id = data.get("task_id")
            if not task_id:
                raise ValueError("LLM forgot to provide task_id")
                
            return Action(action_type=ActionType.COMPLETE_TASK, task_id=task_id)

        else:
            return Action(action_type=ActionType.NOOP)

    except Exception:
        # Catch EVERYTHING from the JSON parsing phase (KeyError, ValueError, TypeError)
        pass

    # ── Heuristic fallback ────────────────────────────────────────────────────
    # FIX: Wrap the entire heuristic block in an ultimate safety net.
    # If the fallback fails for any bizarre reason, it will quietly default to NOOP.
    try:
        ready = get_ready_tasks(obs)
        if ready:
            return Action(action_type=ActionType.COMPLETE_TASK, task_id=ready[0].task_id)

        pending_meetings = [m for m in obs.meetings if m.status == "pending"]
        if pending_meetings:
            slots = get_conflict_free_slots(obs, n=5)
            if slots:
                return Action(action_type=ActionType.SCHEDULE_MEETING,
                              meeting_id=pending_meetings[0].meeting_id,
                              scheduled_time=datetime.fromisoformat(slots[0]))

        unprocessed = sorted([e for e in obs.emails if not e.processed], key=lambda e: -e.urgency)
        if unprocessed:
            e = unprocessed[0]
            pri = (EmailPriority.CRITICAL if e.urgency >= 9 else
                   EmailPriority.HIGH     if e.urgency >= 7 else
                   EmailPriority.MEDIUM   if e.urgency >= 5 else EmailPriority.LOW)
            cat = (EmailCategory.URGENT        if e.urgency >= 7 else
                   EmailCategory.ACTIONABLE    if e.urgency >= 4 else
                   EmailCategory.INFORMATIONAL)
            return Action(action_type=ActionType.TRIAGE_EMAIL,
                          email_id=e.email_id, category=cat, priority=pri)
                          
    except Exception:
        pass

    # The ultimate guarantee: if everything fails, do nothing.
    return Action(action_type=ActionType.NOOP)


# ──────────────────────────────────────────────────────────────────────────────
# Credentials  — single source of truth
# ──────────────────────────────────────────────────────────────────────────────

# Model pinned for hard tasks — never overridden by MODEL_NAME env var.
_HARD_MODEL   = "gpt-4o-mini"
_DEFAULT_MODEL = "gpt-3.5-turbo"


def get_openai_client(task_difficulty: str) -> Tuple[OpenAI, str]:
    """
    Credential resolution order:
      1. OPENAI_API_KEY  (preferred — standard OpenAI key)
      2. HF_TOKEN        (fallback for HF Space secrets)

    Model selection (strict priority — highest wins):
      Priority 1 — hard task: ALWAYS "gpt-4o-mini", regardless of MODEL_NAME.
      Priority 2 — easy/medium with MODEL_NAME set: use MODEL_NAME.
      Priority 3 — easy/medium without MODEL_NAME: fall back to "gpt-3.5-turbo".

    Root-cause note:
      os.getenv("MODEL_NAME", default) ignores `default` the moment MODEL_NAME
      exists in the environment, so the old hard-task branch was silently
      bypassed whenever MODEL_NAME was set via HF Secrets.  The fix evaluates
      task_difficulty FIRST, before consulting the environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None

    # ── Model resolution — task_difficulty is checked FIRST ──────────────────
    if task_difficulty == "hard":
        # Hard override: MODEL_NAME is intentionally ignored here.
        model = _HARD_MODEL
    else:
        # Easy / medium: respect MODEL_NAME if provided, else use cheap default.
        model = os.getenv("MODEL_NAME") or _DEFAULT_MODEL

    print(f"[MODEL SELECTED] task={task_difficulty!r} → model={model!r}"
          + (" (hard override — MODEL_NAME ignored)" if task_difficulty == "hard"
             else f" (MODEL_NAME={os.getenv('MODEL_NAME')!r})"),
          file=sys.stderr)

    return OpenAI(api_key=api_key, base_url=base_url), model


# ──────────────────────────────────────────────────────────────────────────────
# Step budgets and env sizes
# ──────────────────────────────────────────────────────────────────────────────

STEP_BUDGETS = {"easy": 60, "medium": 100, "hard": 130}

ENV_SIZES = {
    "easy":   dict(num_emails_per_day=8,  num_meetings_per_day=3, num_tasks_per_day=5),
    "medium": dict(num_emails_per_day=8,  num_meetings_per_day=4, num_tasks_per_day=6),
    "hard":   dict(num_emails_per_day=10, num_meetings_per_day=6, num_tasks_per_day=8),
}


# ──────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ──────────────────────────────────────────────────────────────────────────────

def run_llm_agent(
    task_difficulty: str = "medium",
    max_steps: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:

    if max_steps is None:
        max_steps = STEP_BUDGETS[task_difficulty]

    client, model_name = get_openai_client(task_difficulty)
    logger = StructuredLogger()
    random.seed(seed)

    env = EnterpriseEnv(**ENV_SIZES[task_difficulty], max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0
    step = 0

    logger.start({
        "task": task_difficulty, "model": model_name, "seed": seed,
        "max_steps": max_steps,
        "initial_emails": len(obs.emails),
        "initial_meetings": len(obs.meetings),
        "initial_tasks": len(obs.tasks),
    })

    while step < max_steps and not obs.done:
        step += 1
        prompt = obs_to_prompt(obs, task_difficulty, env.metadata)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise enterprise workflow AI. "
                            "Respond ONLY with a single valid JSON object — "
                            "no explanation, no markdown, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
                temperature=0.0,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] API error step {step}: {e}", file=sys.stderr)
            response_text = '{"action":"noop"}'

        action = parse_llm_response(response_text, obs)

        if verbose:
            print(f"[DBG step={step}] {response_text.strip()!r} → {action.action_type.value}",
                  file=sys.stderr)

        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward

        logger.step(step, action.action_type.value, reward, {
            "email_triage_rate":          round(obs.email_triage_rate, 3),
            "meeting_schedule_rate":      round(obs.meeting_schedule_success_rate, 3),
            "task_completion_rate":       round(obs.task_completion_rate, 3),
            "unprocessed_emails":         obs.unprocessed_emails_count,
            "meeting_conflicts":          obs.meeting_conflicts,
            "overdue_tasks":              obs.overdue_tasks_count,
        })

        if terminated or truncated:
            break

    graders   = get_task_graders()
    grader    = graders[task_difficulty]
    final_obs = env.state()
    score, explanation = grader.grade(final_obs, env.metadata)

    results = {
        "task":             grader.name,
        "difficulty":       task_difficulty,
        "model":            model_name,
        "seed":             seed,
        "score":            score,
        "explanation":      explanation,
        "cumulative_reward": cumulative_reward,
        "steps_completed":  step,
        "final_metrics": {
            "email_triage_rate":             final_obs.email_triage_rate,
            "task_completion_rate":          final_obs.task_completion_rate,
            "meeting_schedule_success_rate": final_obs.meeting_schedule_success_rate,
            "unprocessed_emails":            final_obs.unprocessed_emails_count,
            "overdue_tasks":                 final_obs.overdue_tasks_count,
            "meeting_conflicts":             final_obs.meeting_conflicts,
        },
        "metadata": env.metadata,
    }

    logger.end(score, results)
    return results


def run_all_tasks(seed: int = 42, verbose: bool = False) -> Dict[str, float]:
    scores = {}
    for difficulty in ("easy", "medium", "hard"):
        steps = STEP_BUDGETS[difficulty]
        print(f"\n{'='*60}\nRunning {difficulty.upper()} ({steps} steps, seed={seed})\n{'='*60}",
              file=sys.stderr)
        result = run_llm_agent(task_difficulty=difficulty, max_steps=steps,
                               seed=seed, verbose=verbose)
        scores[difficulty] = result["score"]
        print(f"  Score: {result['score']:.4f}  |  {result['explanation']}", file=sys.stderr)
    return scores


def evaluate_submission(task_difficulty: str = "medium",
                        max_steps: int = None, seed: int = 42) -> Dict[str, Any]:
    """Backwards-compatible wrapper used by app.py."""
    return run_llm_agent(task_difficulty=task_difficulty,
                         max_steps=max_steps, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Enterprise Task Automation — LLM Inference")
    parser.add_argument("--task",    choices=["easy","medium","hard","all"], default="medium")
    parser.add_argument("--steps",   type=int,  default=None,
                        help="Max steps (default: auto per difficulty)")
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.task == "all":
        scores = run_all_tasks(seed=args.seed, verbose=args.verbose)
        print("\n" + "="*60, file=sys.stderr)
        print("BASELINE SCORES  (seed=42, reproducible)", file=sys.stderr)
        print("="*60, file=sys.stderr)
        for diff, sc in scores.items():
            print(f"  {diff:8s}: {sc:.4f}/1.0", file=sys.stderr)
        sys.exit(0)

    results = run_llm_agent(
        task_difficulty=args.task,
        max_steps=args.steps,
        seed=args.seed,
        verbose=args.verbose,
    )
    print(f"\n{'='*60}\nTask:  {results['task']}\nScore: {results['score']:.4f}/1.0"
          f"\nExpl:  {results['explanation']}", file=sys.stderr)
    sys.exit(0 if results["score"] > 0.0 else 1)


if __name__ == "__main__":
    main()
