"""Core Environment Implementation - OpenEnv Compatible"""

import random
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

from .types import (
    Observation,
    Action,
    Reward,
    ActionType,
    Email,
    Meeting,
    Task,
    Notification,
    EmailCategory,
    EmailPriority,
    TaskPriority,
    TaskStatus,
)


# ─── Enterprise Content Templates ──────────────────────────────────────────────

EMAIL_TEMPLATES = [
    {
        "sender": "boss@company.com",
        "subjects": [
            "Q3 Strategy Discussion - Feedback Needed",
            "Urgent: Client escalation on Project X",
            "Budget Review for Team - Action Required",
            "Review: New Feature Rollout Plan",
        ],
        "bodies": [
            "Hi, I've attached the Q3 strategy document. Can you review and provide your thoughts by EOD? The client is expecting our recommendations.",
            "The client reported a critical issue in production. Our team needs to investigate immediately. Please prioritize this and update me every 2 hours.",
            "Finance needs our budget proposal for next quarter by tomorrow. Can you consolidate the numbers from your department and send to me?",
            "Please review the attached rollout plan for the new dashboard feature. We're targeting next Monday for launch. Any concerns?",
        ],
        "urgency_range": (7, 10),
    },
    {
        "sender": "client@partner.com",
        "subjects": [
            "Deliverable Due Tomorrow - Status Check",
            "Contract Amendment - Signature Required",
            "Integration API Issues",
            "Meeting Request: Quarterly Business Review",
        ],
        "bodies": [
            "We haven't received the deliverable yet. Can you confirm ETA? We need it before our board meeting tomorrow.",
            "Please review and sign the attached amendment to the contract. We need your signature by EOW.",
            "Our team is experiencing issues with your API endpoints. Could you help troubleshoot? Error logs attached.",
            "Would you be available for a quarterly business review call next Wednesday? We'd like to discuss our partnership roadmap.",
        ],
        "urgency_range": (6, 9),
    },
    {
        "sender": "team@company.com",
        "subjects": [
            "Code Review: Backend API Refactoring",
            "Team Standup Notes - Action Items",
            "Brainstorm: New Product Features",
            "Knowledge Share: Database Optimization",
        ],
        "bodies": [
            "I've submitted a PR for the backend API refactoring. Can you review the changes and provide feedback? The implementation uses the new async patterns we discussed.",
            "Here are the notes from today's standup. Please complete your assigned action items by Friday. The deployment is scheduled for next Monday.",
            "Let me know what ideas you have for our Q4 product roadmap. We're brainstorming on Tuesday at 2 PM.",
            "I found a great optimization for our database queries that reduced load times by 40%. Check out my blog post on this technique.",
        ],
        "urgency_range": (3, 6),
    },
    {
        "sender": "hr@company.com",
        "subjects": [
            "Benefits Enrollment Deadline Extended",
            "Performance Review Scheduled",
            "New Office Policy FAQ",
            "Team Event: Virtual Coffee Hour",
        ],
        "bodies": [
            "The benefits enrollment deadline has been extended to next Friday. Don't miss out on selecting your health plan options for 2024.",
            "Your performance review is scheduled for next Thursday at 2 PM with your manager. Please prepare your self-assessment.",
            "We've updated our remote work policy. Check the attached FAQ for details on the new hybrid schedule guidelines.",
            "Join us for our virtual coffee hour on Friday at 4 PM. This is a casual opportunity to connect with colleagues from across the company.",
        ],
        "urgency_range": (2, 5),
    },
    {
        "sender": "marketing@company.com",
        "subjects": [
            "Campaign Launch Feedback",
            "Event Registration: Industry Conference",
            "Market Research: Customer Insights",
            "Social Media Engagement Numbers",
        ],
        "bodies": [
            "We're launching the new campaign next week. Can you review the messaging and creative assets? We want your technical perspective on feasibility.",
            "The industry conference is next month and we need to confirm our booth setup. Can you help coordinate with the logistics team?",
            "We've gathered new customer insights from our recent survey. The data shows strong demand for feature X. Should we prioritize this?",
            "Our latest social media campaign reached 50K impressions. Here's the engagement breakdown by platform and demographic.",
        ],
        "urgency_range": (2, 7),
    },
]

MEETING_TEMPLATES = [
    {
        "titles": [
            "Daily Standup",
            "Sprint Planning",
            "Code Review Session",
            "Architecture Discussion",
        ],
        "description": "Regular team synchronization meetings",
        "duration_range": (15, 30),
        "impact_range": (0.3, 0.6),
        "required": True,
    },
    {
        "titles": [
            "Client Call: Project Update",
            "Partner Demo",
            "Vendor Negotiation",
            "Customer Success Review",
        ],
        "description": "External stakeholder meetings",
        "duration_range": (45, 90),
        "impact_range": (0.7, 0.95),
        "required": True,
    },
    {
        "titles": [
            "1:1 with Manager",
            "Mentoring Session",
            "Career Development Chat",
            "Feedback Discussion",
        ],
        "description": "One-on-one professional development",
        "duration_range": (30, 45),
        "impact_range": (0.4, 0.7),
        "required": False,
    },
    {
        "titles": [
            "All-Hands Meeting",
            "Company Town Hall",
            "Department Kickoff",
            "Quarterly Business Review",
        ],
        "description": "Large team or company-wide gatherings",
        "duration_range": (60, 120),
        "impact_range": (0.5, 0.8),
        "required": False,
    },
    {
        "titles": [
            "Brainstorm Session",
            "Design Workshop",
            "Product Strategy Review",
            "Roadmap Planning",
        ],
        "description": "Cross-functional planning and ideation",
        "duration_range": (60, 90),
        "impact_range": (0.6, 0.85),
        "required": False,
    },
]

TASK_TEMPLATES = [
    {
        "actions": ["Implement", "Develop", "Build"],
        "items": ["API endpoint", "database migration", "UI component", "microservice", "authentication",  "data pipeline"],
        "description": "Development tasks",
        "priority_range": (0.5, 1.0),
        "urgency_range": (4, 10),
        "hours_range": (4, 16),
    },
    {
        "actions": ["Fix", "Debug", "Troubleshoot"],
        "items": ["performance issue", "memory leak", "race condition", "data inconsistency", "crash in production"],
        "description": "Bug fixes",
        "priority_range": (0.7, 1.0),
        "urgency_range": (6, 10),
        "hours_range": (2, 8),
    },
    {
        "actions": ["Review", "Audit", "Analyze"],
        "items": ["code quality", "security compliance", "architecture", "test coverage", "documentation"],
        "description": "Review and analysis",
        "priority_range": (0.4, 0.8),
        "urgency_range": (3, 7),
        "hours_range": (2, 6),
    },
    {
        "actions": ["Document", "Write", "Create"],
        "items": ["API specification", "design document", "tutorial", "runbook", "architecture decision record"],
        "description": "Documentation tasks",
        "priority_range": (0.3, 0.7),
        "urgency_range": (2, 6),
        "hours_range": (2, 8),
    },
    {
        "actions": ["Optimize", "Refactor", "Improve"],
        "items": ["database queries", "caching strategy", "code structure", "build pipeline", "monitoring"],
        "description": "Performance and improvements",
        "priority_range": (0.4, 0.8),
        "urgency_range": (2, 5),
        "hours_range": (4, 12),
    },
    {
        "actions": ["Test", "QA", "Validate"],
        "items": ["feature launch", "security", "performance", "compatibility", "edge cases"],
        "description": "Testing and validation",
        "priority_range": (0.5, 0.9),
        "urgency_range": (4, 8),
        "hours_range": (3, 10),
    },
]


class EnterpriseEnv(Env):
    """
    OpenEnv-compatible Enterprise Task Automation Environment
    
    Simulates a realistic enterprise workflow with email triage, meeting scheduling,
    and task prioritization. The agent must efficiently manage these responsibilities
    to maximize cumulative reward.
    """
    
    def __init__(
        self,
        num_emails_per_day: int = 10,
        num_meetings_per_day: int = 5,
        num_tasks_per_day: int = 8,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Enterprise Environment
        
        Args:
            num_emails_per_day: Number of emails generated per simulated day
            num_meetings_per_day: Number of meeting scheduling requests per day
            num_tasks_per_day: Number of tasks per day
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.num_emails_per_day = num_emails_per_day
        self.num_meetings_per_day = num_meetings_per_day
        self.num_tasks_per_day = num_tasks_per_day
        self.max_steps = max_steps
        
        # Set seed for reproducibility
        if seed is not None:
            self.seed(seed)
        
        # Gymnasium spaces
        self.action_space = Discrete(len(ActionType))
        self.observation_space = Box(low=0, high=1, shape=(50,), dtype=np.float32)
        
        # Simulation state
        self.current_step = 0
        self.current_time: Optional[datetime] = None
        self.episode_start_time: Optional[datetime] = None
        
        # Environment state
        self.emails: Dict[str, Email] = {}
        self.meetings: Dict[str, Meeting] = {}
        self.tasks: Dict[str, Task] = {}
        self.notifications: Dict[str, Notification] = {}
        
        # Counters for ID generation
        self.email_counter = 0
        self.meeting_counter = 0
        self.task_counter = 0
        self.notification_counter = 0
        
        # Reward tracking
        self.cumulative_reward = 0.0
        self.last_reward = Reward(total=0.0, step_reward=0.0)
        
        # Metadata
        self.metadata = {
            "emails_triaged": 0,
            "meetings_scheduled": 0,
            "tasks_completed": 0,
            "escalations": 0,
            "deadline_misses": 0,
        }
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Args:
            seed: Optional random seed
            
        Returns:
            observation: Initial observation of the environment
            info: Metadata dictionary
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cumulative_reward = 0.0
        
        # Initialize simulation time (9 AM on a Monday)
        self.episode_start_time = datetime(2024, 1, 8, 9, 0, 0)
        self.current_time = self.episode_start_time
        
        # Clear previous state
        self.emails.clear()
        self.meetings.clear()
        self.tasks.clear()
        self.notifications.clear()
        self.email_counter = 0
        self.meeting_counter = 0
        self.task_counter = 0
        self.notification_counter = 0
        self.metadata = {
            "emails_triaged": 0,
            "meetings_scheduled": 0,
            "tasks_completed": 0,
            "escalations": 0,
            "deadline_misses": 0,
        }
        
        # Generate initial inbox
        self._generate_initial_inbox()
        
        observation = self._get_observation()
        info = {
            "step": self.current_step,
            "episode_start_time": self.episode_start_time.isoformat(),
        }
        
        return observation, info
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: The action to execute
            
        Returns:
            observation: New observation state
            reward: Reward for this step
            terminated: Whether episode ended (completed goal or failure)
            truncated: Whether episode was truncated (max_steps reached)
            info: Metadata dictionary
        """
        self.current_step += 1
        
        # Advance simulation time (5 minutes per step)
        self.current_time += timedelta(minutes=5)
        
        # Process action
        reward = self._process_action(action)
        self.cumulative_reward += reward.step_reward
        self.last_reward = reward
        
        # Check for deadline violations
        self._check_deadlines()
        
        # Potentially generate new items (25% chance per step)
        if random.random() < 0.25:
            self._generate_random_item()
        
        # Determine episode termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        done = terminated or truncated
        
        observation = self._get_observation()
        observation.done = done
        observation.truncated = truncated
        
        info = {
            "step": self.current_step,
            "current_time": self.current_time.isoformat(),
            "cumulative_reward": self.cumulative_reward,
            "metadata": self.metadata,
        }
        
        return observation, reward.step_reward, terminated, truncated, info
    
    def state(self) -> Observation:
        """
        Get current complete observation of environment state
        
        Returns:
            observation: Complete observation
        """
        return self._get_observation()
    
    def _get_observation(self) -> Observation:
        """Build current observation from environment state"""
        
        # Calculate metrics
        unprocessed_emails = sum(1 for e in self.emails.values() if not e.processed)
        urgent_emails = sum(
            1 for e in self.emails.values()
            if e.priority in [EmailPriority.CRITICAL, EmailPriority.HIGH] and not e.processed
        )
        
        scheduled_meetings = sum(
            1 for m in self.meetings.values() if m.status == "scheduled"
        )
        
        # Check for meeting conflicts
        meeting_conflicts = self._count_meeting_conflicts()
        
        overdue_tasks = sum(
            1 for t in self.tasks.values()
            if t.status != TaskStatus.COMPLETED and t.deadline < self.current_time
        )
        blocked_tasks = sum(
            1 for t in self.tasks.values() if t.status == TaskStatus.BLOCKED
        )
        total_pending_hours = sum(
            t.estimated_hours - t.actual_hours
            for t in self.tasks.values()
            if t.status != TaskStatus.COMPLETED
        )
        
        unread_notifications = sum(
            1 for n in self.notifications.values() if not n.read
        )
        
        # Calculate rates
        total_emails = len(self.emails) if len(self.emails) > 0 else 1
        email_triage_rate = sum(1 for e in self.emails.values() if e.processed) / total_emails
        
        total_tasks = len(self.tasks) if len(self.tasks) > 0 else 1
        task_completion_rate = sum(
            1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED
        ) / total_tasks
        
        total_meetings = len(self.meetings) if len(self.meetings) > 0 else 1
        meeting_schedule_success_rate = scheduled_meetings / total_meetings
        
        # Calculate time remaining in simulated day (9 AM - 6 PM = 9 hours = 540 minutes)
        end_of_day = self.episode_start_time.replace(hour=18, minute=0, second=0)
        time_until_end = max(0, int((end_of_day - self.current_time).total_seconds() / 60))
        
        # Determine valid actions
        valid_actions = self._get_valid_actions()
        
        return Observation(
            step=self.current_step,
            timestamp=self.current_time,
            time_until_end=time_until_end,
            emails=list(self.emails.values()),
            unprocessed_emails_count=unprocessed_emails,
            urgent_emails_count=urgent_emails,
            meetings=list(self.meetings.values()),
            scheduled_meetings_count=scheduled_meetings,
            meeting_conflicts=meeting_conflicts,
            tasks=list(self.tasks.values()),
            overdue_tasks_count=overdue_tasks,
            blocked_tasks_count=blocked_tasks,
            total_pending_hours=total_pending_hours,
            notifications=list(self.notifications.values()),
            unread_notifications_count=unread_notifications,
            email_triage_rate=email_triage_rate,
            task_completion_rate=task_completion_rate,
            meeting_schedule_success_rate=meeting_schedule_success_rate,
            valid_actions=valid_actions,
        )
    
    def _generate_initial_inbox(self):
        """Generate initial inbox with emails, meetings, and tasks"""
        # Generate emails
        for _ in range(self.num_emails_per_day):
            self._generate_email()
        
        # Generate meetings
        for _ in range(self.num_meetings_per_day):
            self._generate_meeting()
        
        # Generate tasks
        for _ in range(self.num_tasks_per_day):
            self._generate_task()
    
    def _generate_random_item(self):
        """Randomly generate a new item (email, meeting, or task)"""
        item_type = random.choice(["email", "meeting", "task"])
        if item_type == "email":
            self._generate_email()
        elif item_type == "meeting":
            self._generate_meeting()
        else:
            self._generate_task()
    
    def _generate_email(self):
        """Create a random email from realistic templates"""
        template = random.choice(EMAIL_TEMPLATES)
        
        email_id = f"email_{self.email_counter}"
        self.email_counter += 1
        
        sender = template["sender"]
        subject = random.choice(template["subjects"])
        body = random.choice(template["bodies"])
        urgency = random.randint(template["urgency_range"][0], template["urgency_range"][1])
        
        requires_response = random.random() < 0.6
        has_deadline = random.random() < 0.3
        
        deadline = None
        if has_deadline:
            deadline = self.current_time + timedelta(hours=random.randint(2, 48))
        
        email = Email(
            email_id=email_id,
            sender=sender,
            subject=subject,
            body=body,
            timestamp=self.current_time,
            urgency=urgency,
            topic=random.choice(["project", "budget", "schedule", "support", "urgent", "general"]),
            sentiment=random.choice(["positive", "neutral", "negative"]),
            deadline=deadline,
            requires_response=requires_response,
        )
        
        self.emails[email_id] = email
    
    def _generate_meeting(self):
        """Create a random meeting request from realistic templates"""
        template = random.choice(MEETING_TEMPLATES)
        
        meeting_id = f"meeting_{self.meeting_counter}"
        self.meeting_counter += 1
        
        title = random.choice(template["titles"])
        duration = random.randint(template["duration_range"][0], template["duration_range"][1])
        business_impact = random.uniform(template["impact_range"][0], template["impact_range"][1])
        num_participants = random.randint(2, 8)
        participants = [f"person_{i}@company.com" for i in range(num_participants)]
        
        meeting = Meeting(
            meeting_id=meeting_id,
            title=title,
            participants=participants,
            priority=random.randint(0, 10),
            duration_minutes=duration,
            required=template["required"],
            business_impact=business_impact,
            status="pending",
        )
        
        self.meetings[meeting_id] = meeting
    
    def _generate_task(self):
        """Create a random task from realistic templates"""
        template = random.choice(TASK_TEMPLATES)
        
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        action = random.choice(template["actions"])
        item = random.choice(template["items"])
        title = f"{action}: {item}"
        
        priority = random.choice(list(TaskPriority))
        urgency = random.randint(template["urgency_range"][0], template["urgency_range"][1])
        impact = random.uniform(template["priority_range"][0], template["priority_range"][1])
        hours = random.uniform(template["hours_range"][0], template["hours_range"][1])
        
        has_dependencies = random.random() < 0.2
        dependencies = []
        if has_dependencies and len(self.tasks) > 0:
            existing_ids = list(self.tasks.keys())
            dependencies = random.sample(existing_ids, k=min(2, len(existing_ids)))
        
        deadline = self.current_time + timedelta(hours=random.randint(4, 72))
        
        task = Task(
            task_id=task_id,
            title=title,
            description=f"{template['description']}: {title}",
            deadline=deadline,
            priority=priority,
            urgency=urgency,
            impact=impact,
            dependencies=dependencies,
            estimated_hours=hours,
            owner=f"person_{random.randint(0, 5)}@company.com",
        )
        
        self.tasks[task_id] = task
    
    def _process_action(self, action: Action) -> Reward:
        """
        Process an action and calculate reward
        
        Args:
            action: The action to process
            
        Returns:
            Reward object with detailed breakdown
        """
        reward = Reward(total=0.0, step_reward=0.0)
        
        if action.action_type == ActionType.TRIAGE_EMAIL:
            reward = self._handle_triage_email(action)
        elif action.action_type == ActionType.SCHEDULE_MEETING:
            reward = self._handle_schedule_meeting(action)
        elif action.action_type == ActionType.CREATE_TASK:
            reward = self._handle_create_task(action)
        elif action.action_type == ActionType.REPRIORITIZE_TASK:
            reward = self._handle_reprioritize_task(action)
        elif action.action_type == ActionType.COMPLETE_TASK:
            reward = self._handle_complete_task(action)
        elif action.action_type == ActionType.SEND_NOTIFICATION:
            reward = self._handle_send_notification(action)
        elif action.action_type == ActionType.ESCALATE:
            reward = self._handle_escalate(action)
        elif action.action_type == ActionType.NOOP:
            reward.step_reward = 0.0
            reward.details = "No action taken"
        
        reward.total = self.cumulative_reward + reward.step_reward
        return reward
    
    def _handle_triage_email(self, action: Action) -> Reward:
        """Handle email triage action"""
        reward = Reward(total=0.0, step_reward=0.0)
        
        if action.email_id not in self.emails:
            reward.step_reward = -0.5
            reward.details = "Email not found"
            return reward
        
        email = self.emails[action.email_id]
        
        if email.processed:
            reward.step_reward = -0.2
            reward.details = "Email already triaged"
            return reward
        
        email.processed = True
        email.category = action.category
        email.priority = action.priority
        self.metadata["emails_triaged"] += 1
        
        # Reward based on email properties
        base_reward = 0.5
        urgency_bonus = email.urgency / 10.0 * 0.3  # More urgent = more reward
        priority_bonus = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.0}.get(
            email.priority.value if email.priority else "low", 0.0
        )
        
        reward.email_processing_reward = base_reward + urgency_bonus + priority_bonus
        reward.step_reward = reward.email_processing_reward
        reward.details = f"Triaged email {action.email_id} as {action.category}/{action.priority}"
        
        return reward
    
    def _handle_schedule_meeting(self, action: Action) -> Reward:
        """Handle meeting scheduling action"""
        reward = Reward(total=0.0, step_reward=0.0)
        
        if action.meeting_id not in self.meetings:
            reward.step_reward = -0.3
            reward.details = "Meeting not found"
            return reward
        
        meeting = self.meetings[action.meeting_id]
        
        if meeting.status == "scheduled":
            reward.step_reward = -0.2
            reward.details = "Meeting already scheduled"
            return reward
        
        if action.scheduled_time is None:
            reward.step_reward = -0.3
            reward.details = "No time specified"
            return reward
        
        # Check for conflicts
        conflicts = self._check_time_conflict(action.scheduled_time, meeting.duration_minutes)
        
        if conflicts:
            reward.step_reward = -0.5
            reward.details = f"Scheduling conflict detected"
            return reward
        
        meeting.scheduled_time = action.scheduled_time
        meeting.status = "scheduled"
        self.metadata["meetings_scheduled"] += 1
        
        # Reward based on meeting impact
        base_reward = 0.4
        impact_bonus = meeting.business_impact * 0.3
        priority_bonus = (meeting.priority / 10.0) * 0.3
        
        reward.meeting_scheduling_reward = base_reward + impact_bonus + priority_bonus
        reward.step_reward = reward.meeting_scheduling_reward
        reward.details = f"Scheduled meeting {action.meeting_id} at {action.scheduled_time}"
        
        return reward
    
    def _handle_create_task(self, action: Action) -> Reward:
        """Handle task creation (extract from email) action"""
        reward = Reward(total=0.0, step_reward=0.0)
        reward.step_reward = 0.2
        reward.details = "Task created from correspondence"
        return reward
    
    def _handle_reprioritize_task(self, action: Action) -> Reward:
        """Handle task reprioritization action"""
        reward = Reward(total=0.0, step_reward=0.0)
        
        if action.task_id not in self.tasks:
            reward.step_reward = -0.3
            reward.details = "Task not found"
            return reward
        
        task = self.tasks[action.task_id]
        old_priority = task.priority
        task.priority = action.new_priority
        
        # Reward if deprioritizing lower-impact tasks or prioritizing urgent ones
        if action.new_priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            reward.step_reward = 0.3 * (task.impact + task.urgency / 10.0)
        else:
            reward.step_reward = 0.1
        
        reward.task_prioritization_reward = reward.step_reward
        reward.details = f"Reprioritized task {action.task_id} from {old_priority} to {action.new_priority}"
        return reward
    
    def _handle_complete_task(self, action: Action) -> Reward:
        """Handle task completion action"""
        reward = Reward(total=0.0, step_reward=0.0)
        
        if action.task_id not in self.tasks:
            reward.step_reward = -0.5
            reward.details = "Task not found"
            return reward
        
        task = self.tasks[action.task_id]
        
        if task.status == TaskStatus.COMPLETED:
            reward.step_reward = -0.2
            reward.details = "Task already completed"
            return reward
        
        # Check dependencies before completing
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if dep_task and dep_task.status != TaskStatus.COMPLETED:
                    reward.step_reward = -0.3
                    reward.details = f"Cannot complete {action.task_id}: dependency {dep_id} not yet done"
                    return reward
        
        # Check if deadline was met
        on_time = task.deadline >= self.current_time
        task.status = TaskStatus.COMPLETED
        self.metadata["tasks_completed"] += 1
        
        # Reward based on impact and timeliness
        base_reward = task.impact
        deadline_bonus = 0.5 if on_time else -0.3
        
        reward.step_reward = base_reward + deadline_bonus
        reward.deadline_adherence_reward = deadline_bonus
        reward.details = f"Completed task {action.task_id} {'on-time' if on_time else 'LATE'}"
        
        return reward
    
    def _handle_send_notification(self, action: Action) -> Reward:
        """Handle sending notification action"""
        reward = Reward(total=0.0, step_reward=0.0)
        
        if not action.target or not action.message:
            reward.step_reward = -0.1
            reward.details = "Invalid notification parameters"
            return reward
        
        notification_id = f"notif_{self.notification_counter}"
        self.notification_counter += 1
        
        notification = Notification(
            notification_id=notification_id,
            alert_type="info",
            title="Automated Notification",
            message=action.message,
            target=action.target,
        )
        
        self.notifications[notification_id] = notification
        reward.step_reward = 0.15
        reward.details = f"Sent notification to {action.target}"
        
        return reward
    
    def _handle_escalate(self, action: Action) -> Reward:
        """Handle escalation action - should be used sparingly"""
        reward = Reward(total=0.0, step_reward=0.0)
        self.metadata["escalations"] += 1
        
        # Escalations are penalized to encourage problem-solving first
        reward.escalation_penalty = -0.2
        reward.step_reward = reward.escalation_penalty
        reward.details = f"Escalated issue {action.issue_id}"
        
        return reward
    
    def _check_deadlines(self):
        """Check for missed deadlines and generate alerts"""
        for task in self.tasks.values():
            if task.status != TaskStatus.COMPLETED and task.deadline < self.current_time:
                if task.status != TaskStatus.BLOCKED:
                    task.status = TaskStatus.BLOCKED
                    self.metadata["deadline_misses"] += 1
                    
                    # Create escalation notification
                    notif_id = f"notif_{self.notification_counter}"
                    self.notification_counter += 1
                    self.notifications[notif_id] = Notification(
                        notification_id=notif_id,
                        alert_type="escalation",
                        severity="critical",
                        title=f"DEADLINE MISSED: {task.title}",
                        message=f"Task {task.task_id} missed deadline at {task.deadline}",
                        target=task.owner,
                        related_item=task.task_id,
                    )
        
        # Check email deadlines with responses required
        for email in self.emails.values():
            if email.requires_response and email.deadline and email.deadline < self.current_time and not email.processed:
                notif_id = f"notif_{self.notification_counter}"
                self.notification_counter += 1
                self.notifications[notif_id] = Notification(
                    notification_id=notif_id,
                    alert_type="reminder",
                    severity="high",
                    title=f"RESPONSE OVERDUE: {email.subject}",
                    message=f"Email from {email.sender} requires response",
                    target="user@company.com",
                    related_item=email.email_id,
                )
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # End at 6 PM or after max steps (already handled by truncated)
        end_of_day = self.episode_start_time.replace(hour=18, minute=0, second=0)
        return self.current_time >= end_of_day
    
    def _count_meeting_conflicts(self) -> int:
        """Count scheduling conflicts in meetings"""
        conflicts = 0
        scheduled = [m for m in self.meetings.values() if m.status == "scheduled"]
        
        for i, m1 in enumerate(scheduled):
            for m2 in scheduled[i + 1:]:
                if m1.scheduled_time and m2.scheduled_time:
                    m1_end = m1.scheduled_time + timedelta(minutes=m1.duration_minutes)
                    m2_end = m2.scheduled_time + timedelta(minutes=m2.duration_minutes)
                    
                    # Check overlap
                    if (m1.scheduled_time < m2_end) and (m2.scheduled_time < m1_end):
                        conflicts += 1
        
        return conflicts
    
    def _check_time_conflict(self, time_slot: datetime, duration: int) -> bool:
        """Check if a time slot conflicts with scheduled meetings"""
        proposed_end = time_slot + timedelta(minutes=duration)
        
        for meeting in self.meetings.values():
            if meeting.status == "scheduled" and meeting.scheduled_time:
                meeting_end = meeting.scheduled_time + timedelta(minutes=meeting.duration_minutes)
                
                # Check overlap
                if (time_slot < meeting_end) and (meeting.scheduled_time < proposed_end):
                    return True
        
        return False
    
    def _get_valid_actions(self) -> list:
        """Determine valid actions for current state"""
        valid = []
        
        if any(not e.processed for e in self.emails.values()):
            valid.append(ActionType.TRIAGE_EMAIL)
        
        if any(m.status == "pending" for m in self.meetings.values()):
            valid.append(ActionType.SCHEDULE_MEETING)
        
        if any(t.status == TaskStatus.PENDING for t in self.tasks.values()):
            valid.append(ActionType.REPRIORITIZE_TASK)
            valid.append(ActionType.COMPLETE_TASK)
        
        if len(self.emails) > 0:
            valid.append(ActionType.CREATE_TASK)
        
        if len(self.notifications) > 0:
            valid.append(ActionType.SEND_NOTIFICATION)
        
        valid.append(ActionType.ESCALATE)
        valid.append(ActionType.NOOP)
        
        return valid
    
    def render(self, mode: str = "human"):
        """Render current state (optional)"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass
