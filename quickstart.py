#!/usr/bin/env python
"""Quick start script for Enterprise Task Automation Environment"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import (
    EnterpriseEnv,
    Action,
    ActionType,
    EmailPriority,
    EmailCategory,
    TaskPriority,
)
from src.graders import evaluate_agent_performance, get_task_graders


def basic_example():
    """Basic usage example"""
    print("\n" + "=" * 70)
    print("QUICK START: Basic Usage")
    print("=" * 70)
    
    # Create environment
    env = EnterpriseEnv(
        num_emails_per_day=5,
        num_meetings_per_day=3,
        num_tasks_per_day=4,
        max_steps=30
    )
    
    # Reset to start an episode
    obs, info = env.reset(seed=42)
    print(f"\nInitial State:")
    print(f"  Emails in inbox: {len(obs.emails)}")
    print(f"  Meetings to schedule: {sum(1 for m in obs.meetings if m.status == 'pending')}")
    print(f"  Pending tasks: {sum(1 for t in obs.tasks if t.status.value == 'pending')}")
    
    # Take an action
    unprocessed = [e for e in obs.emails if not e.processed]
    if unprocessed:
        email = unprocessed[0]
        action = Action(
            action_type=ActionType.TRIAGE_EMAIL,
            email_id=email.email_id,
            category=EmailCategory.ACTIONABLE,
            priority=EmailPriority.HIGH,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nAction Taken: Triaged {email.email_id}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Cumulative Reward: {info['cumulative_reward']:.2f}")
    
    print("\nFor more examples, see EXAMPLES.md")


def task_showcase():
    """Showcase all task difficulties and graders"""
    print("\n" + "=" * 70)
    print("TASK SHOWCASE: All Difficulty Levels")
    print("=" * 70)
    
    graders = get_task_graders()
    
    for difficulty, grader in graders.items():
        print(f"\n[{difficulty.upper()}] {grader.name}")
        print(f"Description: {grader.description}")
        print("-" * 70)


def interactive_demo():
    """Interactive agent demonstration"""
    print("\n" + "=" * 70)
    print("INTERACTIVE DEMO: Simple Heuristic Agent")
    print("=" * 70)
    
    env = EnterpriseEnv(
        num_emails_per_day=8,
        num_meetings_per_day=4,
        num_tasks_per_day=6,
        max_steps=40
    )
    
    obs, info = env.reset(seed=123)
    cumulative_reward = 0.0
    step = 0
    
    print(f"\nStarting episode: {len(obs.emails)} emails, {len(obs.meetings)} meetings, {len(obs.tasks)} tasks")
    
    while not obs.done and step < 15:
        step += 1
        
        # Simple heuristic: prioritize urgent emails, then meetings, then tasks
        action = None
        
        if step % 5 == 1:
            urgent_emails = [e for e in obs.emails if not e.processed and e.urgency >= 7]
            if urgent_emails:
                email = urgent_emails[0]
                action = Action(
                    action_type=ActionType.TRIAGE_EMAIL,
                    email_id=email.email_id,
                    category=EmailCategory.URGENT,
                    priority=EmailPriority.CRITICAL,
                )
        
        if action is None:
            pending_meetings = [m for m in obs.meetings if m.status == "pending"]
            if pending_meetings and step % 5 == 2:
                from datetime import timedelta
                meeting = pending_meetings[0]
                action = Action(
                    action_type=ActionType.SCHEDULE_MEETING,
                    meeting_id=meeting.meeting_id,
                    scheduled_time=obs.timestamp + timedelta(hours=2),
                )
        
        if action is None:
            action = Action(action_type=ActionType.NOOP)
        
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        
        if reward > 0:
            print(f"Step {step}: {action.action_type.value:20s} | Reward: {reward:6.3f} | Cumulative: {cumulative_reward:6.2f}")
    
    # Evaluate performance
    result = evaluate_agent_performance(env, obs, task_difficulty="easy")
    print(f"\nEpisode Complete!")
    print(f"Final Score (Easy Task): {result['score']:.2f}/1.0")
    print(f"Email Triage Rate: {result['final_metrics']['email_triage_rate']:.1%}")
    print(f"Task Completion Rate: {result['final_metrics']['task_completion_rate']:.1%}")


def main():
    """Main demonstration runner"""
    print("\n" + "#" * 70)
    print("# ENTERPRISE TASK AUTOMATION ENVIRONMENT - QUICK START")
    print("#" * 70)
    
    basic_example()
    task_showcase()
    interactive_demo()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("""
1. Read the README.md for detailed documentation
2. Check DEVELOPMENT.md for deployment instructions
3. Review test_demo.py for more examples
4. Launch Gradio interface: python app.py
5. Build Docker image: docker build -t enterprise-env:latest .
""")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
