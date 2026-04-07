"""Test and demonstration script for Enterprise Automation Environment"""

import random
from datetime import datetime, timedelta
from src import EnterpriseEnv
from src.types import Action, ActionType, EmailPriority, EmailCategory, TaskPriority
from src.graders import evaluate_agent_performance, get_task_graders


def random_agent(env: EnterpriseEnv, max_steps: int = 100):
    """
    Random baseline agent that takes random valid actions
    
    Args:
        env: Environment instance
        max_steps: Maximum steps to run
        
    Returns:
        Tuple of (observations, rewards, metadata)
    """
    obs, info = env.reset(seed=42)
    
    cumulative_reward = 0.0
    step_count = 0
    
    while not obs.done and step_count < max_steps:
        # Get valid actions
        valid_actions = obs.valid_actions
        
        if not valid_actions:
            action = Action(action_type=ActionType.NOOP)
        else:
            # Sample random action
            action_type = random.choice(valid_actions)
            
            if action_type == ActionType.TRIAGE_EMAIL:
                unprocessed_emails = [e for e in obs.emails if not e.processed]
                if unprocessed_emails:
                    email = random.choice(unprocessed_emails)
                    action = Action(
                        action_type=ActionType.TRIAGE_EMAIL,
                        email_id=email.email_id,
                        category=random.choice(list(EmailCategory)),
                        priority=random.choice(list(EmailPriority)),
                    )
                else:
                    action = Action(action_type=ActionType.NOOP)
            
            elif action_type == ActionType.SCHEDULE_MEETING:
                pending_meetings = [m for m in obs.meetings if m.status == "pending"]
                if pending_meetings:
                    meeting = random.choice(pending_meetings)
                    time_slot = obs.timestamp + timedelta(hours=random.randint(1, 8))
                    action = Action(
                        action_type=ActionType.SCHEDULE_MEETING,
                        meeting_id=meeting.meeting_id,
                        scheduled_time=time_slot,
                    )
                else:
                    action = Action(action_type=ActionType.NOOP)
            
            elif action_type == ActionType.REPRIORITIZE_TASK:
                pending_tasks = [t for t in obs.tasks if t.status.value == "pending"]
                if pending_tasks:
                    task = random.choice(pending_tasks)
                    action = Action(
                        action_type=ActionType.REPRIORITIZE_TASK,
                        task_id=task.task_id,
                        new_priority=random.choice(list(TaskPriority)),
                    )
                else:
                    action = Action(action_type=ActionType.NOOP)
            
            elif action_type == ActionType.COMPLETE_TASK:
                pending_tasks = [t for t in obs.tasks if t.status.value == "pending"]
                if pending_tasks:
                    task = random.choice(pending_tasks)
                    action = Action(
                        action_type=ActionType.COMPLETE_TASK,
                        task_id=task.task_id,
                    )
                else:
                    action = Action(action_type=ActionType.NOOP)
            
            else:
                action = Action(action_type=ActionType.NOOP)
        
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action.action_type.value}, Reward={reward:.3f}, Cumulative={cumulative_reward:.2f}")
    
    return obs, cumulative_reward, env.metadata


def smart_agent(env: EnterpriseEnv, max_steps: int = 100):
    """
    Smarter agent that prioritizes urgent items
    """
    obs, info = env.reset(seed=42)
    
    cumulative_reward = 0.0
    step_count = 0
    
    while not obs.done and step_count < max_steps:
        # Priority: 1. Triage urgent emails, 2. Schedule meetings, 3. Complete high-impact tasks
        
        action = None
        
        # Try to triage urgent emails first
        urgent_emails = [
            e for e in obs.emails
            if not e.processed and e.urgency >= 7
        ]
        if urgent_emails:
            email = urgent_emails[0]
            action = Action(
                action_type=ActionType.TRIAGE_EMAIL,
                email_id=email.email_id,
                category=EmailCategory.URGENT,
                priority=EmailPriority.HIGH,
            )
        
        # If no urgent emails, schedule meetings
        if not action:
            pending_meetings = [m for m in obs.meetings if m.status == "pending"]
            if pending_meetings:
                # Schedule high-impact meetings first
                sorted_meetings = sorted(pending_meetings, key=lambda m: m.business_impact, reverse=True)
                meeting = sorted_meetings[0]
                time_slot = obs.timestamp + timedelta(hours=2)
                action = Action(
                    action_type=ActionType.SCHEDULE_MEETING,
                    meeting_id=meeting.meeting_id,
                    scheduled_time=time_slot,
                )
        
        # If no meetings, complete high-impact tasks
        if not action:
            high_impact_tasks = [
                t for t in obs.tasks
                if t.status.value == "pending" and (t.impact >= 0.7 or t.urgency >= 7)
            ]
            if high_impact_tasks:
                # Complete most urgent first
                task = sorted(high_impact_tasks, key=lambda t: t.urgency, reverse=True)[0]
                action = Action(
                    action_type=ActionType.COMPLETE_TASK,
                    task_id=task.task_id,
                )
        
        # Otherwise try to triage remaining emails
        if not action:
            unprocessed_emails = [e for e in obs.emails if not e.processed]
            if unprocessed_emails:
                email = unprocessed_emails[0]
                priority = EmailPriority.HIGH if email.urgency >= 5 else EmailPriority.MEDIUM
                action = Action(
                    action_type=ActionType.TRIAGE_EMAIL,
                    email_id=email.email_id,
                    category=EmailCategory.ACTIONABLE,
                    priority=priority,
                )
        
        if not action:
            action = Action(action_type=ActionType.NOOP)
        
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: Action={action.action_type.value}, Reward={reward:.3f}, Cumulative={cumulative_reward:.2f}")
    
    return obs, cumulative_reward, env.metadata


if __name__ == "__main__":
    print("=" * 80)
    print("ENTERPRISE TASK AUTOMATION ENVIRONMENT - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Print task descriptions
    print("Available Tasks:")
    graders = get_task_graders()
    for difficulty, grader in graders.items():
        print(f"  [{difficulty.upper()}] {grader.name}: {grader.description}")
    print()
    
    # Test with random agent on easy task
    print("-" * 80)
    print("TEST 1: Random Agent on Easy Task")
    print("-" * 80)
    env = EnterpriseEnv(num_emails_per_day=5, num_meetings_per_day=3, num_tasks_per_day=4, max_steps=30)
    final_obs, cum_reward, metadata = random_agent(env, max_steps=30)
    evaluation = evaluate_agent_performance(env, final_obs, task_difficulty="easy")
    print(f"\nTask: {evaluation['task_name']}")
    print(f"Score: {evaluation['score']:.2f}/1.0")
    print(f"Details: {evaluation['explanation']}")
    print()
    
    # Test with smart agent on medium task
    print("-" * 80)
    print("TEST 2: Smart Agent on Medium Task")
    print("-" * 80)
    env = EnterpriseEnv(num_emails_per_day=8, num_meetings_per_day=5, num_tasks_per_day=6, max_steps=50)
    final_obs, cum_reward, metadata = smart_agent(env, max_steps=50)
    evaluation = evaluate_agent_performance(env, final_obs, task_difficulty="medium")
    print(f"\nTask: {evaluation['task_name']}")
    print(f"Score: {evaluation['score']:.2f}/1.0")
    print(f"Details: {evaluation['explanation']}")
    print()
    
    # Test hard task
    print("-" * 80)
    print("TEST 3: Smart Agent on Hard Task")
    print("-" * 80)
    env = EnterpriseEnv(num_emails_per_day=10, num_meetings_per_day=6, num_tasks_per_day=8, max_steps=100)
    final_obs, cum_reward, metadata = smart_agent(env, max_steps=100)
    evaluation = evaluate_agent_performance(env, final_obs, task_difficulty="hard")
    print(f"\nTask: {evaluation['task_name']}")
    print(f"Score: {evaluation['score']:.2f}/1.0")
    print(f"Details: {evaluation['explanation']}")
    print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
