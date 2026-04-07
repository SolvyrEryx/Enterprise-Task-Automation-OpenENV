"""Task definitions and graders for hackathon evaluation"""

from typing import Dict, Tuple
from .types import Observation, TaskStatus, EmailPriority, TaskPriority
from .environment import EnterpriseEnv


class TaskGrader:
    """Base class for task graders"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def grade(self, observation: Observation, metadata: Dict) -> Tuple[float, str]:
        raise NotImplementedError


class EasyTask(TaskGrader):
    """
    Easy Task: Email Triage Efficiency

    Goal: Process at least 80% of emails and correctly assign HIGH or CRITICAL
          priority to every email with urgency >= 7.

    Scoring:
        0.5 × triage_rate_score   (scaled so 80% = full 0.5)
        0.5 × urgency_accuracy    (fraction of urgent emails correctly prioritized)
    """

    def __init__(self):
        super().__init__(
            name="Email Triage Efficiency",
            description=(
                "Triage at least 80% of emails and correctly prioritize urgent ones "
                "(urgency >= 7 → HIGH or CRITICAL priority)."
            ),
        )

    def grade(self, observation: Observation, metadata: Dict) -> Tuple[float, str]:
        total_emails = len(observation.emails)
        if total_emails == 0:
            return 0.5, "No emails present — partial score awarded."

        # Component 1: triage rate (80% target → full marks)
        triaged = sum(1 for e in observation.emails if e.processed)
        triage_rate = triaged / total_emails
        triage_score = min(triage_rate / 0.80, 1.0)  # 80% = 1.0, scales linearly below

        # Component 2: urgent prioritization accuracy
        urgent_emails = [e for e in observation.emails if e.urgency >= 7]
        if urgent_emails:
            correctly_prioritized = sum(
                1 for e in urgent_emails
                if e.processed and e.priority in [EmailPriority.CRITICAL, EmailPriority.HIGH]
            )
            urgency_accuracy = correctly_prioritized / len(urgent_emails)
        else:
            urgency_accuracy = 1.0  # No urgent emails — full marks

        score = triage_score * 0.5 + urgency_accuracy * 0.5
        explanation = (
            f"Triage rate: {triage_rate:.1%} → score {triage_score*0.5:.3f}/0.5 | "
            f"Urgent accuracy: {urgency_accuracy:.1%} → score {urgency_accuracy*0.5:.3f}/0.5"
        )
        return round(min(score, 1.0), 4), explanation


class MediumTask(TaskGrader):
    """
    Medium Task: Meeting Scheduling & Task Prioritization

    Goal:
      - Schedule >= 70% of pending meetings without conflicts
      - Reprioritize >= 60% of high-impact tasks (impact >= 0.7 or urgency >= 7)
        to HIGH or CRITICAL

    Scoring:
        0.4 × scheduling_score    (70% meetings scheduled = 1.0)
        0.3 × conflict_score      (−0.1 per conflict, floored at 0)
        0.3 × prioritization_score
    """

    def __init__(self):
        super().__init__(
            name="Meeting Scheduling & Task Prioritization",
            description=(
                "Schedule at least 70% of meetings without conflicts AND "
                "correctly reprioritize high-impact/urgent tasks."
            ),
        )

    def grade(self, observation: Observation, metadata: Dict) -> Tuple[float, str]:
        details = []

        # Component 1: meeting scheduling (target 70%)
        total_meetings = len(observation.meetings)
        if total_meetings > 0:
            scheduled = sum(1 for m in observation.meetings if m.status == "scheduled")
            scheduling_rate = scheduled / total_meetings
            scheduling_score = min(scheduling_rate / 0.70, 1.0)
        else:
            scheduling_rate = 1.0
            scheduling_score = 1.0
        details.append(
            f"Meetings scheduled: {scheduling_rate:.1%} → {scheduling_score*0.4:.3f}/0.4"
        )

        # Component 2: conflict avoidance
        conflicts = observation.meeting_conflicts
        conflict_score = max(0.0, 1.0 - conflicts * 0.10)
        details.append(f"Conflicts: {conflicts} → {conflict_score*0.3:.3f}/0.3")

        # Component 3: task prioritization (target 60% of high-impact tasks)
        high_impact_tasks = [
            t for t in observation.tasks
            if t.impact >= 0.7 or t.urgency >= 7
        ]
        if high_impact_tasks:
            correctly_prioritized = sum(
                1 for t in high_impact_tasks
                if t.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]
            )
            prioritization_rate = correctly_prioritized / len(high_impact_tasks)
            prioritization_score = min(prioritization_rate / 0.60, 1.0)
        else:
            prioritization_rate = 1.0
            prioritization_score = 1.0
        details.append(
            f"Task prioritization: {prioritization_rate:.1%} → {prioritization_score*0.3:.3f}/0.3"
        )

        score = scheduling_score * 0.4 + conflict_score * 0.3 + prioritization_score * 0.3
        return round(min(score, 1.0), 4), " | ".join(details)


class HardTask(TaskGrader):
    """
    Hard Task: Comprehensive Workflow Optimization Under Pressure

    What makes this genuinely hard:
      - Requires excellence across ALL four dimensions simultaneously
      - Penalizes overdue tasks heavily (deadline pressure is real)
      - Escalations cost double compared to easy/medium
      - Blocked tasks (missed deadlines) directly penalize the task score
      - System health score considers both escalations AND missed deadlines
      - Meeting conflicts penalise at 15% each instead of 10%

    Scoring:
        0.25 × email_score        (triage rate, must reach 90% for full marks)
        0.25 × meeting_score      (scheduling rate + conflict avoidance at 15%/conflict)
        0.25 × task_score         (completion rate + overdue penalty combined)
        0.25 × health_score       (escalation penalty 20% each + deadline miss 15% each)
    """

    def __init__(self):
        super().__init__(
            name="Comprehensive Workflow Optimization Under Pressure",
            description=(
                "Simultaneously maximize email triage (90% target), meeting scheduling "
                "without conflicts, task completion before deadlines, and system health "
                "(minimal escalations and missed deadlines). All four dimensions must "
                "excel — weakness in any one pulls the total score below 0.75."
            ),
        )

    def grade(self, observation: Observation, metadata: Dict) -> Tuple[float, str]:
        details = []

        # 1. Email score — harder target (90%)
        email_triage_rate = observation.email_triage_rate
        email_score = min(email_triage_rate / 0.90, 1.0)
        details.append(f"Email (90% target): {email_triage_rate:.1%} → {email_score*0.25:.3f}/0.25")

        # 2. Meeting score — conflict penalty at 15%/conflict (stricter than medium)
        total_meetings = len(observation.meetings)
        if total_meetings > 0:
            scheduled_rate = observation.meeting_schedule_success_rate
            conflict_penalty = max(0.0, 1.0 - observation.meeting_conflicts * 0.15)
            meeting_score = (scheduled_rate + conflict_penalty) / 2.0
        else:
            meeting_score = 1.0
        details.append(f"Meetings: {meeting_score:.1%} → {meeting_score*0.25:.3f}/0.25")

        # 3. Task score — completion rate + heavy overdue penalty
        completion_rate = observation.task_completion_rate
        total_tasks = len(observation.tasks)
        if total_tasks > 0:
            # Each overdue task penalises by 15% of the overdue fraction
            overdue_fraction = observation.overdue_tasks_count / total_tasks
            overdue_penalty = max(0.0, 1.0 - overdue_fraction * 2.0)  # >50% overdue → 0
            task_score = (completion_rate + overdue_penalty) / 2.0
        else:
            task_score = 1.0
        details.append(f"Tasks: completion={completion_rate:.1%} overdue={observation.overdue_tasks_count} → {task_score*0.25:.3f}/0.25")

        # 4. System health — escalations (-20% each) + deadline misses (-15% each)
        escalations = metadata.get("escalations", 0)
        deadline_misses = metadata.get("deadline_misses", 0)
        escalation_penalty = max(0.0, 1.0 - escalations * 0.20)
        deadline_penalty = max(0.0, 1.0 - deadline_misses * 0.15)
        health_score = (escalation_penalty + deadline_penalty) / 2.0
        details.append(
            f"Health: escalations={escalations} deadline_misses={deadline_misses} → {health_score*0.25:.3f}/0.25"
        )

        score = email_score * 0.25 + meeting_score * 0.25 + task_score * 0.25 + health_score * 0.25
        return round(min(score, 1.0), 4), " | ".join(details)


# ─── Registry ──────────────────────────────────────────────────────────────────

def get_task_graders() -> Dict[str, TaskGrader]:
    """Return all available task graders keyed by difficulty."""
    return {
        "easy": EasyTask(),
        "medium": MediumTask(),
        "hard": HardTask(),
    }


def evaluate_agent_performance(
    env: "EnterpriseEnv",
    final_observation: Observation,
    task_difficulty: str = "medium",
) -> Dict:
    """
    Evaluate agent performance on a completed episode.

    Args:
        env: The environment instance (for metadata).
        final_observation: Observation at end of episode.
        task_difficulty: "easy", "medium", or "hard".

    Returns:
        Dict with score, explanation, metrics.
    """
    graders = get_task_graders()
    if task_difficulty not in graders:
        raise ValueError(f"Unknown difficulty: {task_difficulty}. Choose from {list(graders)}")

    grader = graders[task_difficulty]
    score, explanation = grader.grade(final_observation, env.metadata)

    return {
        "task_name": grader.name,
        "task_description": grader.description,
        "difficulty": task_difficulty,
        "score": score,
        "explanation": explanation,
        "metadata": env.metadata,
        "final_metrics": {
            "email_triage_rate": final_observation.email_triage_rate,
            "task_completion_rate": final_observation.task_completion_rate,
            "meeting_schedule_success_rate": final_observation.meeting_schedule_success_rate,
            "unprocessed_emails": final_observation.unprocessed_emails_count,
            "overdue_tasks": final_observation.overdue_tasks_count,
            "meeting_conflicts": final_observation.meeting_conflicts,
        },
    }
