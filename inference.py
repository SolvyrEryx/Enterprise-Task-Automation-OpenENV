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
"""

import sys
import json
import argparse
import time
import os
import random
import math
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
        self._rewards = []          # Step 4: track every per-step reward

    def start(self, config: Dict[str, Any]):
        self.start_time = time.time()
        self.step_count = 0
        self._rewards = []
        # Step 2: env is always the fixed env name, not the task value
        print(
            f"[START] task={config.get('task','')} "
            f"env=enterprise-task-automation "
            f"model={config.get('model','')} "
            f"seed={config.get('seed','')} "
            f"max_steps={config.get('max_steps','')} "
            f"emails={config.get('initial_emails','')} "
            f"meetings={config.get('initial_meetings','')} "
            f"tasks={config.get('initial_tasks','')}",
            flush=True,
        )

    def step(self, step_num: int, action: str, reward: float, done_flag: bool):
        # Step 4: accumulate reward
        self._rewards.append(reward)
        self.step_count += 1
        # Step 3: strict minimal format — no extra fields
        print(
            f"[STEP] step={step_num} action={action} reward={reward:.2f} "
            f"done={str(done_flag).lower()} error=null",
            flush=True,
        )

    def end(self, final_score: float, steps: int):
        # Step 5 & 7: comma-separated rewards list, score at .3f precision
        score = max(0.01, min(float(final_score), 0.99))
        rewards_str = ",".join(f"{r:.2f}" for r in self._rewards) or "0.00"
        success = str(score >= 0.5).lower()
        print(
            f"[END] success={success} steps={steps} score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


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
        pass

    # ── Heuristic fallback ────────────────────────────────────────────────────
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

    return Action(action_type=ActionType.NOOP)


# ──────────────────────────────────────────────────────────────────────────────
# Credentials  — single source of truth
# ──────────────────────────────────────────────────────────────────────────────

# Model pinned for hard tasks — never overridden by MODEL_NAME env var.
_HARD_MODEL    = "gpt-4o-mini"
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
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None

    if task_difficulty == "hard":
        model = _HARD_MODEL
    else:
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

        # ── STRATEGY LAYER: focus on the weakest scoring dimension ──────────
        scores = current_score_breakdown(obs, env.metadata)
        sorted_dims = sorted(
            [(k, v) for k, v in scores.items() if k != "total"],
            key=lambda x: x[1]
        )
        priority_dimension  = sorted_dims[0][0]
        secondary_dimension = sorted_dims[1][0]

        if priority_dimension == "email":
            strategy = "FOCUS_EMAIL"
        elif priority_dimension == "meeting":
            strategy = "FOCUS_MEETINGS"
        elif priority_dimension == "task":
            strategy = "FOCUS_TASKS"
        else:
            strategy = "BALANCED"

        tradeoff_context = f"""
ADAPTIVE STRATEGY (MANDATORY):

Strategy Mode: {strategy} (Primary: {priority_dimension.upper()})
Secondary focus: {secondary_dimension.upper()}

You MUST prioritize improving the primary dimension.
Avoid actions that do not improve the primary dimension unless absolutely necessary.
If possible, also improve the secondary dimension.

Scores:
Email={scores['email']:.2f}, Meeting={scores['meeting']:.2f}, Task={scores['task']:.2f}, Health={scores['health']:.2f}

Reason:
{priority_dimension} is currently the lowest scoring dimension and is limiting overall system performance.

Decision Rule:
Prefer actions that improve {priority_dimension} more than other dimensions.
If two actions are similar, choose the one that also improves {secondary_dimension}.

Action Selection Priority:
1. Choose actions that improve the primary dimension.
2. If multiple actions improve it, choose the one that also improves the secondary dimension.
3. If no action improves the primary dimension, choose the safest action that avoids penalties.

Action Guidance:
* Email → use "triage_email"
* Meeting → use "schedule_meeting"
* Task → use "complete_task" or "reprioritize_task"

Goal:
Maximize TOTAL score by improving weakest dimensions first.

Penalty Awareness:
Avoid actions that introduce conflicts, overdue tasks, or missed deadlines, as they reduce overall score.

Consistency Rule:
Stick to the chosen strategy unless a clearly better opportunity appears.

Urgency Override:
If a task, meeting, or deadline is critically urgent, prioritize it immediately even if it is not the primary focus.

Tradeoff Hint:
Avoid actions that significantly harm other dimensions.
"""

        base_prompt = obs_to_prompt(obs, task_difficulty, env.metadata)
        prompt = tradeoff_context + "\n" + base_prompt

        # Controlled per-step seed for reproducibility
        random.seed(seed + step)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise enterprise workflow AI. "
                            "This system optimizes a multi-objective score across competing constraints. "
                            "Respond ONLY with a single valid JSON object — "
                            "no explanation, no markdown, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
                temperature=0.2,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] API error step {step}: {e}", file=sys.stderr)
            response_text = '{"action":"noop"}'

        action = parse_llm_response(response_text, obs)

        # Lightweight reasoning trace (stderr only — does not affect validator)
        print(
            f"[THINK] strategy={strategy} primary={priority_dimension} "
            f"secondary={secondary_dimension}",
            file=sys.stderr,
        )

        if verbose:
            print(
                f"[DBG step={step}] {response_text.strip()!r} → {action.action_type.value}",
                file=sys.stderr,
            )

        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward

        # Step 6: pass done_flag boolean instead of obs_summary dict
        done_flag = terminated or truncated
        logger.step(step, action.action_type.value, reward, done_flag)

        if done_flag:
            break

    graders   = get_task_graders()
    grader    = graders[task_difficulty]
    final_obs = env.state()
    raw_score, explanation = grader.grade(final_obs, env.metadata)

    epsilon = 1e-6

    try:
        score = float(raw_score)
    except Exception:
        score = 0.5

    if math.isnan(score) or math.isinf(score):
        score = 0.5

    if not (0 < score < 1):
        score = min(1 - epsilon, max(epsilon, score))

    score = float(score)

    results = {
        "task":              grader.name,
        "difficulty":        task_difficulty,
        "model":             model_name,
        "seed":              seed,
        "score":             score,
        "explanation":       explanation or "",
        "cumulative_reward": cumulative_reward,
        "steps_completed":   step,
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

    logger.end(score, step)
    return results


def evaluate_submission(task_difficulty: str = "medium",
                        max_steps: int = None, seed: int = 42) -> Dict[str, Any]:
    """Backwards-compatible wrapper used by app.py."""
    return run_llm_agent(task_difficulty=task_difficulty,
                         max_steps=max_steps, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Enterprise Task Automation — LLM Inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run_llm_agent(
        task_difficulty=args.task,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(result["score"])


if __name__ == "__main__":
    main()
