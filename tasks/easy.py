"""
Easy Task: Email Triage Efficiency

Description:
  Master the fundamentals of email management. Efficiently categorize and prioritize
  a high volume of incoming emails while correctly identifying and flagging urgent messages.

Objective:
  Achieve at least 80% email triage rate while maintaining 80%+ priority accuracy
  on urgent emails.

Metrics:
  - Email Triage Rate: % of emails processed/categorized
  - Priority Accuracy: % of urgent emails correctly marked as HIGH/CRITICAL
  - Score Range: 0.0 - 1.0

Scoring Breakdown:
  - 50% of score from triage rate (process emails quickly)
  - 50% of score from priority accuracy (categorize correctly)

Success Criteria:
  - Score >= 0.7: Acceptable (70% proficiency)
  - Score >= 0.85: Good (85% proficiency) 
  - Score >= 0.95: Excellent (95% proficiency)

Constraints:
  - Maximum 50 steps
  - Up to 10 emails per day
  - Simple reward signals

Agent Strategy Tips:
  1. Scan all emails and identify urgent ones (urgency >= 7)
  2. Prioritize processing urgent emails first with HIGH/CRITICAL status
  3. Process non-urgent emails with MEDIUM/LOW status
  4. Batch similar emails to improve efficiency
"""

TASK_CONFIG = {
    "name": "Email Triage Efficiency",
    "difficulty": "easy",
    "description": "Efficiently triage incoming emails and correctly prioritize urgent messages",
    "max_steps": 50,
    "environment_config": {
        "num_emails_per_day": 10,
        "num_meetings_per_day": 2,
        "num_tasks_per_day": 2,
    },
    "success_threshold": 0.7,
    "good_threshold": 0.85,
    "excellent_threshold": 0.95,
    "metrics": [
        "email_triage_rate",
        "urgent_priority_accuracy",
    ],
}
