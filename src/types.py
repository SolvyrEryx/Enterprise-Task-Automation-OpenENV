"""Pydantic models for Enterprise Task Automation Environment"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class EmailCategory(str, Enum):
    """Email classification categories"""
    URGENT = "urgent"
    ACTIONABLE = "actionable"
    INFORMATIONAL = "informational"
    SPAM = "spam"


class EmailPriority(str, Enum):
    """Email priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Email(BaseModel):
    """Email representation"""
    email_id: str = Field(..., description="Unique email identifier")
    sender: str = Field(..., description="Email sender")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    timestamp: datetime = Field(..., description="Email arrival time")
    urgency: int = Field(default=0, ge=0, le=10, description="Urgency level (0-10)")
    topic: str = Field(default="general", description="Email topic/category")
    sentiment: str = Field(default="neutral", description="Sentiment: positive, neutral, negative")
    deadline: Optional[datetime] = Field(default=None, description="Task deadline if applicable")
    requires_response: bool = Field(default=False, description="Whether response is required")
    category: Optional[EmailCategory] = Field(default=None, description="Assigned category after triage")
    priority: Optional[EmailPriority] = Field(default=None, description="Assigned priority after triage")
    processed: bool = Field(default=False, description="Whether email has been processed")


class Meeting(BaseModel):
    """Meeting representation"""
    meeting_id: str = Field(..., description="Unique meeting identifier")
    title: str = Field(..., description="Meeting title")
    participants: List[str] = Field(default_factory=list, description="Participant list")
    priority: int = Field(default=0, ge=0, le=10, description="Meeting priority (0-10)")
    duration_minutes: int = Field(default=30, description="Duration in minutes")
    scheduled_time: Optional[datetime] = Field(default=None, description="Scheduled start time")
    required: bool = Field(default=False, description="Whether attendee is required")
    business_impact: float = Field(default=0.5, ge=0.0, le=1.0, description="Business impact score")
    status: str = Field(default="pending", description="pending, scheduled, completed, cancelled")


class Task(BaseModel):
    """Task representation"""
    task_id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Task title")
    description: str = Field(default="", description="Task description")
    deadline: datetime = Field(..., description="Task deadline")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority")
    urgency: int = Field(default=5, ge=0, le=10, description="Urgency level (0-10)")
    impact: float = Field(default=0.5, ge=0.0, le=1.0, description="Business impact (0.0-1.0)")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies (IDs)")
    estimated_hours: float = Field(default=1.0, description="Estimated effort in hours")
    actual_hours: float = Field(default=0.0, description="Actual effort in hours")
    owner: str = Field(default="", description="Task owner")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class Notification(BaseModel):
    """Notification/Alert representation"""
    notification_id: str = Field(..., description="Unique notification identifier")
    alert_type: str = Field(..., description="Type: reminder, escalation, warning, info")
    severity: str = Field(default="info", description="Severity: critical, high, medium, low, info")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    target: str = Field(..., description="Target recipient/user")
    timestamp: datetime = Field(default_factory=datetime.now, description="Notification creation time")
    read: bool = Field(default=False, description="Whether notification was read")
    related_item: Optional[str] = Field(default=None, description="Related email/task/meeting ID")


class ActionType(str, Enum):
    """Valid action types"""
    TRIAGE_EMAIL = "triage_email"
    SCHEDULE_MEETING = "schedule_meeting"
    RESCHEDULE_MEETING = "reschedule_meeting"
    CREATE_TASK = "create_task"
    REPRIORITIZE_TASK = "reprioritize_task"
    SEND_NOTIFICATION = "send_notification"
    ESCALATE = "escalate"
    COMPLETE_TASK = "complete_task"
    NOOP = "noop"


class Action(BaseModel):
    """Action representation - agent's decision at each step"""
    action_type: ActionType = Field(..., description="Type of action to perform")
    email_id: Optional[str] = Field(default=None, description="For triage_email action")
    category: Optional[EmailCategory] = Field(default=None, description="Category for triage")
    priority: Optional[EmailPriority] = Field(default=None, description="Priority for triage")
    meeting_id: Optional[str] = Field(default=None, description="For meeting-related actions")
    scheduled_time: Optional[datetime] = Field(default=None, description="For schedule_meeting action")
    task_id: Optional[str] = Field(default=None, description="For task-related actions")
    new_priority: Optional[TaskPriority] = Field(default=None, description="For reprioritize_task")
    target: Optional[str] = Field(default=None, description="For send_notification")
    message: Optional[str] = Field(default=None, description="Notification message")
    issue_id: Optional[str] = Field(default=None, description="For escalate action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional action metadata")


class Reward(BaseModel):
    """Reward signal from environment"""
    total: float = Field(..., description="Total cumulative reward")
    step_reward: float = Field(..., description="Reward for this step")
    
    # Component rewards for transparency
    email_processing_reward: float = Field(default=0.0, description="Reward for triaging email")
    meeting_scheduling_reward: float = Field(default=0.0, description="Reward for scheduling meetings")
    task_prioritization_reward: float = Field(default=0.0, description="Reward for prioritizing tasks")
    deadline_adherence_reward: float = Field(default=0.0, description="Reward for meeting deadlines")
    efficiency_reward: float = Field(default=0.0, description="Reward for efficient batching")
    escalation_penalty: float = Field(default=0.0, description="Penalty for unnecessary escalations")
    
    # Metadata
    details: str = Field(default="", description="Reward details/explanation")


class Observation(BaseModel):
    """Complete observation of environment state"""
    step: int = Field(..., description="Current step number")
    timestamp: datetime = Field(..., description="Current simulation time")
    time_until_end: int = Field(..., description="Minutes until end of simulated day")
    
    # Inbox state
    emails: List[Email] = Field(default_factory=list, description="Current emails in inbox")
    unprocessed_emails_count: int = Field(default=0, description="Count of unprocessed emails")
    urgent_emails_count: int = Field(default=0, description="Count of urgent/critical emails")
    
    # Calendar state
    meetings: List[Meeting] = Field(default_factory=list, description="Current/upcoming meetings")
    scheduled_meetings_count: int = Field(default=0, description="Count of scheduled meetings")
    meeting_conflicts: int = Field(default=0, description="Number of scheduling conflicts")
    
    # Task state
    tasks: List[Task] = Field(default_factory=list, description="Current tasks")
    overdue_tasks_count: int = Field(default=0, description="Count of overdue tasks")
    blocked_tasks_count: int = Field(default=0, description="Count of blocked tasks")
    total_pending_hours: float = Field(default=0.0, description="Total hours of pending work")
    
    # Notifications
    notifications: List[Notification] = Field(default_factory=list, description="Active notifications")
    unread_notifications_count: int = Field(default=0, description="Count of unread notifications")
    
    # Metrics
    email_triage_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of emails triaged")
    task_completion_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of tasks completed")
    meeting_schedule_success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate for scheduling")
    
    # Valid actions at this step
    valid_actions: List[ActionType] = Field(default_factory=list, description="Valid actions available")
    
    # Episode info
    done: bool = Field(default=False, description="Whether episode is complete")
    truncated: bool = Field(default=False, description="Whether episode was truncated")
