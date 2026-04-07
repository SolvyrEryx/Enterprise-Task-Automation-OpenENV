"""
Hard Task: Comprehensive Workflow Optimization

Description:
  Master the complete enterprise workflow. Simultaneously optimize email management,
  meeting coordination, task execution, and system health while managing dependencies
  and deadline pressures.

Objective:
  Achieve balanced excellence across all workflow dimensions: process 85%+ emails,
  schedule 75%+ meetings without conflicts, complete 70%+ of high-priority tasks,
  and maintain system health (escalations/deadline misses).

Metrics:
  - Email Processing: % of emails processed accurately
  - Meeting Coordination: % meetings scheduled without conflicts
  - Task Execution: % of critical tasks completed on time
  - System Health: Escalation count, deadline misses
  - Overall Efficiency: Balanced performance across dimensions
  - Score Range: 0.0 - 1.0

Scoring Breakdown:
  - 25% from email triage metrics
  - 25% from meeting scheduling/coordination
  - 25% from task completion and prioritization
  - 25% from system health (escalations, deadline adherence)

Success Criteria:
  - Score >= 0.60: Acceptable (60% proficiency - very challenging!)
  - Score >= 0.75: Good (75% proficiency - excellent performance)
  - Score >= 0.88: Excellent (88% proficiency - expert level)

Constraints:
  - Maximum 100 steps
  - Up to 10 emails, 6 meetings, 8 tasks per day
  - Complex interdependencies and time pressures
  - Dense reward signals with partial progress

Agent Strategy Tips:
  1. Balance multiple objectives - don't ignore emails for meetings
  2. Identify task dependencies and schedule accordingly
  3. Proactively manage impending deadlines
  4. Reschedule meetings early to avoid cascading conflicts
  5. Complete critical-path tasks first
  6. Use escalation wisely (penalties apply for unnecessary ones)
  7. Monitor unread notifications for critical alerts
  8. Batch similar actions to improve efficiency
  9. Maintain buffer time to handle urgent items
  10. Track system health metrics throughout episode

Advanced Considerations:
  - Task dependencies can block progression
  - Meeting conflicts propagate if not handled early
  - Escalations trigger additional notifications
  - Deadline misses accumulate penalties
  - Cumulative stress increases item urgency
"""

TASK_CONFIG = {
    "name": "Comprehensive Workflow Optimization",
    "difficulty": "hard",
    "description": "Maximize all workflow dimensions simultaneously with balanced performance",
    "max_steps": 100,
    "environment_config": {
        "num_emails_per_day": 10,
        "num_meetings_per_day": 6,
        "num_tasks_per_day": 8,
    },
    "success_threshold": 0.60,
    "good_threshold": 0.75,
    "excellent_threshold": 0.88,
    "metrics": [
        "email_triage_rate",
        "meeting_scheduling_rate",
        "task_completion_rate",
        "meeting_conflict_count",
        "overdue_tasks_count",
        "escalation_count",
        "system_health_score",
    ],
    "hard_constraints": [
        "Must handle task dependencies",
        "Meeting time constraints strict",
        "Deadline enforcement mandatory",
        "Escalations flagged",
    ],
    "advanced_features": [
        "Task dependency chains",
        "Cascading meeting conflicts",
        "Escalation penalties",
        "Deadline miss accumulation",
        "Dynamic urgency adjustment",
    ],
}
