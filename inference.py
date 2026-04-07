#!/usr/bin/env python
"""
Inference Script for Enterprise Task Automation Environment
Meta AI OpenEnv Hackathon Submission — Optimised for Hard task > 0.9

Usage:
    python inference.py --task hard --steps 100 --seed 42
    python inference.py --task all --seed 42
"""

import sys
import json
import argparse
import time
import os
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from src import EnterpriseEnv, Action, ActionType, EmailPriority, EmailCategory, TaskPriority
from src.graders import get_task_graders
from src.types import TaskStatus


# ─── Structured logging ────────────────────────────────────────────────────────

class StructuredLogger:
    def __init__(self):
        self.start_time = None
        self.step_count = 0

    def start(self, config: Dict[str, Any]):
        self.start_time = time.time()
        self.step_count = 0
        print(f"[START] {json.dumps({'event': 'START', 'timestamp': datetime.now().isoformat(), 'config': config})}")

    def step(self, step_num: int, action: str, reward: float, obs_summary: Dict):
        self.step_count += 1
        print(f"[STEP] {json.dumps({'event': 'STEP', 'step': step_num, 'action': action, 'reward': round(reward, 4), 'obs': obs_summary, 'timestamp': datetime.now().isoformat()})}")

    def end(self, final_score: float, metadata: Dict):
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[END] {json.dumps({'event': 'END', 'final_score': round(final_score, 4), 'total_steps': self.step_count, 'elapsed_seconds': round(elapsed, 2), 'metadata': metadata, 'timestamp': datetime.now().isoformat()})}")


# ─── Hard-task helpers ─────────────────────────────────────────────────────────

def get_conflict_free_slots(obs, candidate_duration_minutes: int = 90) -> List[str]:
    """
    Return ISO-format time strings that do NOT overlap any already-scheduled meeting.
    Candidates are spaced 90 minutes apart to prevent back-to-back collisions.
    """
    scheduled_windows = [
        (m.scheduled_time, m.scheduled_time + timedelta(minutes=m.duration_minutes))
        for m in obs.meetings
        if m.status == "scheduled" and m.scheduled_time
    ]

    slots: List[str] = []
    base = obs.timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    for i in range(48):
        candidate = base + timedelta(minutes=i * candidate_duration_minutes)
        cand_end = candidate + timedelta(minutes=candidate_duration_minutes)

        conflict = any(
            (candidate < win_end) and (win_start < cand_end)
            for win_start, win_end in scheduled_windows
        )
        if not conflict:
            slots.append(candidate.strftime("%Y-%m-%dT%H:%M:00"))
            # Reserve this window so successive pending meetings get distinct slots
            scheduled_windows.append((candidate, cand_end))
        if len(slots) >= 16:
            break
    return slots


def get_ready_tasks(obs) -> List:
    """Return PENDING tasks whose every dependency is already completed."""
    completed_ids = {t.task_id for t in obs.tasks if t.status == TaskStatus.COMPLETED}
    ready = [
        t for t in obs.tasks
        if t.status == TaskStatus.PENDING
        and all(dep in completed_ids for dep in t.dependencies)
    ]
    ready.sort(key=lambda t: (t.deadline - obs.timestamp).total_seconds())
    return ready


def get_blocked_tasks(obs) -> List:
    completed_ids = {t.task_id for t in obs.tasks if t.status == TaskStatus.COMPLETED}
    return [
        t for t in obs.tasks
        if t.status == TaskStatus.PENDING
        and any(dep not in completed_ids for dep in t.dependencies)
    ]


# ─── Observation → prompt ──────────────────────────────────────────────────────

def obs_to_prompt(obs, task_difficulty: str) -> str:
    """Convert an Observation into a natural-language prompt for the LLM agent."""

    if task_difficulty == "hard":
        return _hard_prompt(obs)

    # ── Easy / Medium ──────────────────────────────────────────────────────────
    task_descriptions = {
        "easy":   "EMAIL TRIAGE: process at least 80% of emails and assign HIGH/CRITICAL to urgency >= 7.",
        "medium": "MEETING & TASK: schedule 70%+ meetings without conflicts AND reprioritize high-impact tasks (impact >= 0.7) to HIGH/CRITICAL.",
    }

    unprocessed = [e for e in obs.emails if not e.processed]
    pending_meetings = [m for m in obs.meetings if m.status == "pending"]
    pending_tasks = [t for t in obs.tasks if t.status.value == "pending"]

    email_lines = "\n".join(
        f"  - id={e.email_id} urgency={e.urgency}/10 sender={e.sender} subject=\"{e.subject}\""
        for e in unprocessed[:8]
    )
    meeting_lines = "\n".join(
        f"  - id={m.meeting_id} impact={m.business_impact:.2f} duration={m.duration_minutes}min title=\"{m.title}\""
        for m in pending_meetings[:6]
    )
    task_lines = "\n".join(
        f"  - id={t.task_id} priority={t.priority.value} urgency={t.urgency}/10 impact={t.impact:.2f} deadline_in={max(0, int((t.deadline - obs.timestamp).total_seconds()/3600))}h title=\"{t.title}\"" +
        (f" deps=[{','.join(t.dependencies)}]" if t.dependencies else "")
        for t in pending_tasks[:8]
    )
    now_str = obs.timestamp.strftime("%Y-%m-%d %H:%M")
    future_slots = [(obs.timestamp + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:00") for h in [1, 2, 3, 4, 5, 6]]
    urgent_tasks = [t for t in pending_tasks if (t.deadline - obs.timestamp).total_seconds() < 3600]
    urgent_warnings = ""
    if urgent_tasks:
        urgent_warnings = f"\nURGENT TASKS DUE SOON: {', '.join(t.task_id for t in urgent_tasks[:3])}"

    prompt = f"""You are an AI enterprise assistant.

CURRENT TIME: {now_str}
STEP: {obs.step} | TIME LEFT: {obs.time_until_end} min
TASK: {task_descriptions[task_difficulty]}

STATE: triage={obs.email_triage_rate:.0%} | schedule={obs.meeting_schedule_success_rate:.0%} | completion={obs.task_completion_rate:.0%} | conflicts={obs.meeting_conflicts} | overdue={obs.overdue_tasks_count}{urgent_warnings}

EMAILS: {email_lines if email_lines else "(none)"}
MEETINGS: {meeting_lines if meeting_lines else "(none)"}
TASKS: {task_lines if task_lines else "(none)"}
SLOTS: {json.dumps(future_slots)}

Actions (respond with ONE JSON only):
{{"action":"triage_email","email_id":"X","category":"urgent|actionable|informational|spam","priority":"critical|high|medium|low"}}
{{"action":"schedule_meeting","meeting_id":"X","scheduled_time":"YYYY-MM-DDTHH:MM:00"}}
{{"action":"reprioritize_task","task_id":"X","new_priority":"critical|high|medium|low"}}
{{"action":"complete_task","task_id":"X"}}
{{"action":"noop"}}

Rules: complete no-dep tasks first; schedule early; urgency>=8→critical, >=6→high; never noop if work remains."""

    return prompt


def _hard_prompt(obs) -> str:
    """
    Highly optimised prompt for Hard difficulty.
    - Pre-computes conflict-free meeting slots (no collisions possible).
    - Separates tasks into READY vs BLOCKED.
    - Explicit NEVER-ESCALATE instruction.
    - Decision tree makes the next action unambiguous.
    """
    now_str = obs.timestamp.strftime("%Y-%m-%d %H:%M")

    # Emails
    unprocessed = sorted([e for e in obs.emails if not e.processed], key=lambda x: -x.urgency)
    email_lines = "\n".join(
        f"  {e.email_id}: urgency={e.urgency}  \"{e.subject}\""
        for e in unprocessed[:10]
    )

    # Meetings + pre-assigned safe slots
    pending_meetings = [m for m in obs.meetings if m.status == "pending"]
    safe_slots = get_conflict_free_slots(obs, candidate_duration_minutes=90)
    slot_map = {m.meeting_id: safe_slots[i] for i, m in enumerate(pending_meetings) if i < len(safe_slots)}
    meeting_lines = "\n".join(
        f"  {m.meeting_id}: dur={m.duration_minutes}min impact={m.business_impact:.2f} → USE SLOT {slot_map.get(m.meeting_id, 'TBD')}"
        for m in pending_meetings[:8]
    )

    # Tasks
    ready_tasks = get_ready_tasks(obs)
    blocked_tasks = get_blocked_tasks(obs)
    ready_lines = "\n".join(
        f"  {t.task_id}: urgency={t.urgency} impact={t.impact:.2f} deadline_in={max(0,int((t.deadline-obs.timestamp).total_seconds()/3600))}h"
        for t in ready_tasks[:8]
    )
    blocked_lines = "\n".join(
        f"  {t.task_id}: blocked by [{', '.join(t.dependencies)}]"
        for t in blocked_tasks[:6]
    )

    # Per-dimension scoring
    email_comp  = min(obs.email_triage_rate / 0.90, 1.0)
    sched_rate  = obs.meeting_schedule_success_rate
    conf_pen    = max(0.0, 1.0 - obs.meeting_conflicts * 0.15)
    meeting_comp = (sched_rate + conf_pen) / 2.0
    total_tasks = max(len(obs.tasks), 1)
    ov_pen      = max(0.0, 1.0 - (obs.overdue_tasks_count / total_tasks) * 2.0)
    task_comp   = (obs.task_completion_rate + ov_pen) / 2.0

    urgent_ready = [t for t in ready_tasks if (t.deadline - obs.timestamp).total_seconds() < 7200]
    deadline_warn = ""
    if urgent_ready:
        deadline_warn = f"\nURGENT TASKS DUE SOON: {', '.join(t.task_id for t in urgent_ready)} - complete IMMEDIATELY"

    prompt = f"""Enterprise AI assistant — Hard mode.

TIME: {now_str}  STEP: {obs.step}  REMAINING: {obs.time_until_end} min

=== SCORE PROGRESS (each needs >= 0.90) =====================
  Email   (90% target): {obs.email_triage_rate:.0%} triaged -> {email_comp:.3f}/1.0
  Meeting (no conflicts):{obs.meeting_schedule_success_rate:.0%} scheduled, {obs.meeting_conflicts} conflicts -> {meeting_comp:.3f}/1.0
  Tasks   (no overdue): {obs.task_completion_rate:.0%} done, {obs.overdue_tasks_count} overdue -> {task_comp:.3f}/1.0
  Health  : 0 escalations required{deadline_warn}

ABSOLUTE RULES:
  1. NEVER use "escalate" — each one destroys 20% of your health score.
  2. NEVER schedule a meeting to any slot except the ones listed below.
  3. NEVER try to complete a BLOCKED task.

=== READY TASKS (complete these — all dependencies done) =====
{ready_lines if ready_lines else "  (none ready)"}

=== PENDING MEETINGS -> pre-assigned conflict-free slots =====
{meeting_lines if meeting_lines else "  (none)"}

=== UNPROCESSED EMAILS ================================
{email_lines if email_lines else "  (all triaged)"}
  Triage rules: urgency>=8 -> urgent/critical | 6-7 -> urgent/high | 4-5 -> actionable/medium | <=3 -> informational/low

=== BLOCKED TASKS (do NOT attempt) ====================
{blocked_lines if blocked_lines else "  (none)"}

=== DECISION (pick ONE) ===============================
• deadline_in < 4h on a READY task -> complete it NOW
• ready task exists -> complete earliest-deadline one
• pending meeting exists -> schedule it to its pre-assigned slot
• unprocessed email exists -> triage it
• else -> noop

Respond with ONE JSON, nothing else:
{{"action":"complete_task","task_id":"task_X"}}
{{"action":"schedule_meeting","meeting_id":"meeting_X","scheduled_time":"YYYY-MM-DDTHH:MM:00"}}
{{"action":"triage_email","email_id":"email_X","category":"urgent","priority":"critical"}}
{{"action":"reprioritize_task","task_id":"task_X","new_priority":"high"}}
{{"action":"noop"}}"""

    return prompt


# ─── Parse LLM response → Action ──────────────────────────────────────────────

def parse_llm_response(response_text: str, obs) -> Action:
    """Parse LLM JSON into Action; falls back to smart heuristic on failure."""
    try:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data = json.loads(text)
        action_str = data.get("action", "noop")

        if action_str == "triage_email":
            return Action(
                action_type=ActionType.TRIAGE_EMAIL,
                email_id=data["email_id"],
                category=EmailCategory(data.get("category", "actionable")),
                priority=EmailPriority(data.get("priority", "medium")),
            )
        elif action_str == "schedule_meeting":
            return Action(
                action_type=ActionType.SCHEDULE_MEETING,
                meeting_id=data["meeting_id"],
                scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            )
        elif action_str == "reprioritize_task":
            return Action(
                action_type=ActionType.REPRIORITIZE_TASK,
                task_id=data["task_id"],
                new_priority=TaskPriority(data.get("new_priority", "medium")),
            )
        elif action_str == "complete_task":
            return Action(action_type=ActionType.COMPLETE_TASK, task_id=data["task_id"])
        else:
            return Action(action_type=ActionType.NOOP)

    except Exception:
        # Smart fallback: tasks → meetings → emails
        ready = get_ready_tasks(obs)
        if ready:
            return Action(action_type=ActionType.COMPLETE_TASK, task_id=ready[0].task_id)

        pending_meetings = [m for m in obs.meetings if m.status == "pending"]
        if pending_meetings:
            slots = get_conflict_free_slots(obs)
            if slots:
                return Action(
                    action_type=ActionType.SCHEDULE_MEETING,
                    meeting_id=pending_meetings[0].meeting_id,
                    scheduled_time=datetime.fromisoformat(slots[0]),
                )

        unprocessed = [e for e in obs.emails if not e.processed]
        if unprocessed:
            email = max(unprocessed, key=lambda e: e.urgency)
            pri = EmailPriority.CRITICAL if email.urgency >= 8 else (
                EmailPriority.HIGH if email.urgency >= 6 else EmailPriority.MEDIUM
            )
            cat = EmailCategory.URGENT if email.urgency >= 6 else EmailCategory.ACTIONABLE
            return Action(action_type=ActionType.TRIAGE_EMAIL, email_id=email.email_id,
                          category=cat, priority=pri)

        return Action(action_type=ActionType.NOOP)


# ─── Main LLM agent loop ───────────────────────────────────────────────────────

def run_llm_agent(
    task_difficulty: str = "medium",
    max_steps: int = 50,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    logger = StructuredLogger()

    # Required hackathon environment variables (with defaults for testing)
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo" if task_difficulty != "hard" else "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        # For validation/testing purposes, allow missing HF_TOKEN but warn
        if os.getenv("VALIDATION_MODE"):
            hf_token = "dummy_token_for_validation"
        else:
            err = "HF_TOKEN environment variable not set (required for hackathon)"
            logger.end(0.0, {"error": err})
            print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(1)

    # Initialize OpenAI client with hackathon-required variables
    client = OpenAI(
        api_key=hf_token,  # HF_TOKEN used as API key
        base_url=api_base_url
    )

    logger.start({"task": task_difficulty, "model": model_name, "seed": seed, "max_steps": max_steps})
    random.seed(seed)

    if task_difficulty == "hard":
        env = EnterpriseEnv(num_emails_per_day=10, num_meetings_per_day=6,
                            num_tasks_per_day=8, max_steps=max_steps)
    else:
        env = EnterpriseEnv(num_emails_per_day=8, num_meetings_per_day=4,
                            num_tasks_per_day=6, max_steps=max_steps)

    obs, _ = env.reset(seed=seed)
    cumulative_reward = 0.0
    step = 0

    while step < max_steps and not obs.done:
        step += 1
        prompt = obs_to_prompt(obs, task_difficulty)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a precise enterprise AI. Respond ONLY with a single valid JSON action object — no explanation, no markdown."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            response_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[WARN] LLM error step {step}: {e}", file=sys.stderr)
            response_text = '{"action":"noop"}'

        action = parse_llm_response(response_text, obs)

        if verbose:
            print(f"[STEP {step}] {response_text.strip()} → {action.action_type.value}", file=sys.stderr)

        obs, reward, terminated, truncated, _ = env.step(action)
        cumulative_reward += reward

        logger.step(step, action.action_type.value, reward, {
            "email_triage_rate": round(obs.email_triage_rate, 3),
            "meeting_schedule_rate": round(obs.meeting_schedule_success_rate, 3),
            "task_completion_rate": round(obs.task_completion_rate, 3),
            "unprocessed_emails": obs.unprocessed_emails_count,
            "meeting_conflicts": obs.meeting_conflicts,
            "overdue_tasks": obs.overdue_tasks_count,
        })

        if terminated or truncated:
            break

    graders = get_task_graders()
    grader = graders[task_difficulty]
    final_obs = env.state()
    score, explanation = grader.grade(final_obs, env.metadata)

    results = {
        "task": grader.name,
        "difficulty": task_difficulty,
        "model": model_name,
        "seed": seed,
        "score": score,
        "explanation": explanation,
        "cumulative_reward": cumulative_reward,
        "steps_completed": step,
        "final_metrics": {
            "email_triage_rate": final_obs.email_triage_rate,
            "task_completion_rate": final_obs.task_completion_rate,
            "meeting_schedule_success_rate": final_obs.meeting_schedule_success_rate,
            "unprocessed_emails": final_obs.unprocessed_emails_count,
            "overdue_tasks": final_obs.overdue_tasks_count,
            "meeting_conflicts": final_obs.meeting_conflicts,
        },
        "metadata": env.metadata,
    }

    logger.end(score, results)
    return results


def run_all_tasks(seed: int = 42, verbose: bool = False) -> Dict[str, float]:
    configs = [("easy", 50), ("medium", 75), ("hard", 100)]
    scores = {}
    for difficulty, steps in configs:
        print(f"\n{'='*60}\nRunning {difficulty.upper()} ({steps} steps, seed={seed})\n{'='*60}", file=sys.stderr)
        result = run_llm_agent(task_difficulty=difficulty, max_steps=steps, seed=seed, verbose=verbose)
        scores[difficulty] = result["score"]
        print(f"Result — {difficulty}: {result['score']:.4f}/1.0\n  {result['explanation']}", file=sys.stderr)
    return scores


def evaluate_submission(task_difficulty: str = "medium", max_steps: int = 50, seed: int = 42) -> Dict[str, Any]:
    return run_llm_agent(task_difficulty=task_difficulty, max_steps=max_steps, seed=seed, verbose=False)


def main():
    parser = argparse.ArgumentParser(description="Enterprise Task Automation — LLM Inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="medium")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.task == "all":
        scores = run_all_tasks(seed=args.seed, verbose=args.verbose)
        print("\n" + "="*60 + "\nBASELINE SCORES (seed=42)\n" + "="*60, file=sys.stderr)
        for diff, score in scores.items():
            print(f"  {diff:8s}: {score:.4f}/1.0", file=sys.stderr)
        sys.exit(0)

    results = run_llm_agent(task_difficulty=args.task, max_steps=args.steps, seed=args.seed, verbose=args.verbose)
    print(f"\n{'='*60}\nINFERENCE RESULTS\n{'='*60}", file=sys.stderr)
    print(f"Task:  {results['task']}", file=sys.stderr)
    print(f"Score: {results['score']:.4f}/1.0", file=sys.stderr)
    print(f"Steps: {results['steps_completed']}/{args.steps}", file=sys.stderr)
    print(f"Expl:  {results['explanation']}", file=sys.stderr)
    sys.exit(0 if results["score"] > 0.0 else 1)


if __name__ == "__main__":
    main()
