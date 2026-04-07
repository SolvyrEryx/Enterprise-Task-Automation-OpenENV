"""
Medium Task: Meeting Scheduling & Task Prioritization

Description:
  Elevate your workflow management skills. Coordinate multiple meetings without conflicts
  while intelligently reprioritizing tasks based on business impact and urgency.

Objective:
  Schedule 70%+ of meeting requests without conflicts while ensuring high-impact tasks
  receive appropriate priority levels.

Metrics:
  - Meeting Scheduling Rate: % of meetings successfully scheduled
  - Conflict Avoidance: Penalty for scheduling conflicts
  - Task Prioritization Accuracy: % high-impact tasks marked HIGH/CRITICAL
  - Score Range: 0.0 - 1.0

Scoring Breakdown:
  - 40% from meeting scheduling rate
  - 30% from conflict avoidance (-10% per conflict)
  - 30% from task prioritization accuracy

Success Criteria:
  - Score >= 0.65: Acceptable (65% proficiency)
  - Score >= 0.80: Good (80% proficiency)
  - Score >= 0.92: Excellent (92% proficiency)

Constraints:
  - Maximum 75 steps
  - Up to 5 emails, 5 meetings, 5 tasks per day
  - Medium difficulty reward signals

Agent Strategy Tips:
  1. Identify all pending meetings and their business impact
  2. Check for scheduling conflicts before confirming
  3. Prioritize high-impact meetings (business_impact >= 0.7)
  4. Scan tasks and identify high-impact ones (impact >= 0.7)
  5. Mark high-impact/urgent tasks with HIGH/CRITICAL priority
  6. Reschedule conflicting meetings to available time slots
  7. Balance email triage with meeting/task management
"""

TASK_CONFIG = {
    "name": "Meeting Scheduling & Task Prioritization",
    "difficulty": "medium",
    "description": "Schedule meetings efficiently without conflicts and prioritize high-impact tasks",
    "max_steps": 75,
    "environment_config": {
        "num_emails_per_day": 8,
        "num_meetings_per_day": 5,
        "num_tasks_per_day": 5,
    },
    "success_threshold": 0.65,
    "good_threshold": 0.80,
    "excellent_threshold": 0.92,
    "metrics": [
        "meeting_scheduling_rate",
        "meeting_conflict_count",
        "task_prioritization_accuracy",
        "email_triage_rate",
    ],
    "hard_constraints": [
        "Cannot schedule overlapping meetings",
        "Must respect time boundaries",
    ],
}
