# Enterprise Task Automation Environment - API Documentation

## Overview

The Enterprise Task Automation Environment is an OpenEnv-compatible (Gymnasium-compatible) reinforcement learning environment for simulating and optimizing real-world enterprise workflows.

## Core Classes

### EnterpriseEnv

The main environment class that implements the standard RL environment interface.

```python
from src import EnterpriseEnv

env = EnterpriseEnv(
    num_emails_per_day: int = 10,       # Number of emails per simulated day
    num_meetings_per_day: int = 5,      # Number of meeting requests per day
    num_tasks_per_day: int = 8,         # Number of tasks per day
    max_steps: int = 100,                # Maximum steps per episode
    seed: Optional[int] = None           # Random seed for reproducibility
)
```

#### Methods

##### `reset(seed: Optional[int]) -> Tuple[Observation, Dict]`

Reset the environment to initial state and start a new episode.

```python
obs, info = env.reset(seed=42)
```

**Returns:**
- `obs` (Observation): Initial state observation
- `info` (Dict): Metadata including `step`, `episode_start_time`

##### `step(action: Action) -> Tuple[Observation, float, bool, bool, Dict]`

Execute one action and advance the environment by one step.

```python
obs, reward, terminated, truncated, info = env.step(action)
```

**Parameters:**
- `action` (Action): Action to execute

**Returns:**
- `obs` (Observation): New observation
- `reward` (float): Reward for this step
- `terminated` (bool): Whether episode ended (goal reached/failure)
- `truncated` (bool): Whether episode was truncated (max_steps reached)
- `info` (Dict): Metadata including step number, timestamp, cumulative reward

##### `state() -> Observation`

Get the current complete observation without taking an action.

```python
obs = env.state()
```

---

## Data Types

### Observation

Complete snapshot of the environment state at any point.

```python
@dataclass
class Observation:
    step: int                                    # Current step
    timestamp: datetime                          # Simulation time
    time_until_end: int                          # Minutes until end of simulated day
    
    emails: List[Email]                          # All emails in inbox
    unprocessed_emails_count: int                # Count of unprocessed
    urgent_emails_count: int                     # Count of urgent/critical
    
    meetings: List[Meeting]                      # All meetings
    scheduled_meetings_count: int                # Count scheduled
    meeting_conflicts: int                       # Number of conflicts
    
    tasks: List[Task]                            # All tasks
    overdue_tasks_count: int                     # Count overdue
    blocked_tasks_count: int                     # Count blocked
    total_pending_hours: float                   # Hours of pending work
    
    notifications: List[Notification]            # Active notifications
    unread_notifications_count: int              # Count unread
    
    email_triage_rate: float                     # Fraction triaged (0-1)
    task_completion_rate: float                  # Fraction completed (0-1)
    meeting_schedule_success_rate: float         # Success rate (0-1)
    
    valid_actions: List[ActionType]              # Valid actions now
    
    done: bool                                   # Episode complete
    truncated: bool                              # Episode truncated
```

### Action

Represents an agent's decision at each step.

```python
@dataclass
class Action:
    action_type: ActionType                  # Type of action
    email_id: Optional[str] = None          # For email actions
    category: Optional[EmailCategory] = None # Email category
    priority: Optional[EmailPriority] = None # Email priority
    meeting_id: Optional[str] = None        # For meeting actions
    scheduled_time: Optional[datetime] = None  # Meeting time
    task_id: Optional[str] = None           # For task actions
    new_priority: Optional[TaskPriority] = None  # New task priority
    target: Optional[str] = None            # For notifications
    message: Optional[str] = None           # Notification message
    issue_id: Optional[str] = None          # For escalations
    metadata: Dict[str, Any] = {}           # Additional data
```

#### ActionType Enum

```python
class ActionType(str, Enum):
    TRIAGE_EMAIL         # Classify and prioritize an email
    SCHEDULE_MEETING     # Schedule a meeting at a specific time
    RESCHEDULE_MEETING   # Move a meeting to a new time
    CREATE_TASK          # Extract task from email
    REPRIORITIZE_TASK    # Change task priority level
    COMPLETE_TASK        # Mark task as complete
    SEND_NOTIFICATION    # Send alert to recipient
    ESCALATE             # Escalate issue up chain
    NOOP                 # No operation (pass)
```

### Reward

Detailed reward signal with component breakdown.

```python
@dataclass
class Reward:
    total: float                        # Cumulative total
    step_reward: float                  # This step's reward
    
    # Component rewards (for analysis)
    email_processing_reward: float      # Email triage reward
    meeting_scheduling_reward: float    # Meeting reward
    task_prioritization_reward: float   # Task reward
    deadline_adherence_reward: float    # Deadline reward
    efficiency_reward: float            # Batching reward
    escalation_penalty: float           # Escalation penalty
    
    details: str                        # Explanation
```

### Email

```python
@dataclass
class Email:
    email_id: str                       # Unique ID
    sender: str                         # From address
    subject: str                        # Subject line
    body: str                           # Content
    timestamp: datetime                 # Arrival time
    urgency: int                        # 0-10 scale
    topic: str                          # Category
    sentiment: str                      # positive/neutral/negative
    deadline: Optional[datetime]        # Response deadline if any
    requires_response: bool             # Needs reply
    category: Optional[EmailCategory]   # Assigned after triage
    priority: Optional[EmailPriority]   # Assigned after triage
    processed: bool                     # Has been triaged
```

#### EmailCategory & EmailPriority Enums

```python
class EmailCategory(str, Enum):
    URGENT = "urgent"
    ACTIONABLE = "actionable"
    INFORMATIONAL = "informational"
    SPAM = "spam"

class EmailPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
```

### Meeting

```python
@dataclass
class Meeting:
    meeting_id: str                     # Unique ID
    title: str                          # Meeting title
    participants: List[str]             # Attendee list
    priority: int                       # 0-10 importance
    duration_minutes: int               # Duration
    scheduled_time: Optional[datetime]  # When scheduled
    required: bool                      # Mandatory attendance
    business_impact: float              # 0-1 business value
    status: str                         # pending/scheduled/completed
```

### Task

```python
@dataclass
class Task:
    task_id: str                        # Unique ID
    title: str                          # Task name
    description: str                    # Details
    deadline: datetime                  # Due date/time
    priority: TaskPriority              # Priority level
    urgency: int                        # 0-10 scale
    impact: float                       # 0-1 business impact
    status: TaskStatus                  # pending/in_progress/completed/blocked
    dependencies: List[str]             # Task IDs it depends on
    estimated_hours: float              # Estimated effort
    actual_hours: float                 # Time spent so far
    owner: str                          # Responsible person
    created_at: datetime                # Creation time
```

#### TaskPriority & TaskStatus Enums

```python
class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
```

### Notification

```python
@dataclass
class Notification:
    notification_id: str                # Unique ID
    alert_type: str                     # reminder/escalation/warning/info
    severity: str                       # critical/high/medium/low/info
    title: str                          # Title
    message: str                        # Content
    target: str                         # Recipient
    timestamp: datetime                 # When created
    read: bool                          # Whether read
    related_item: Optional[str]         # Related ID (email/task/meeting)
```

---

## Graders

### EasyTask

**Goal:** Triage 80%+ of emails with correct urgent prioritization

```python
from src import EasyTask

grader = EasyTask()
score, explanation = grader.grade(observation, metadata)
# Score: 0-1.0
# Scoring: 50% triage rate + 50% priority accuracy
```

### MediumTask

**Goal:** Schedule 70% of meetings, minimize conflicts, prioritize high-impact tasks

```python
from src import MediumTask

grader = MediumTask()
score, explanation = grader.grade(observation, metadata)
# Score: 0-1.0
# Scoring: 40% scheduling + 30% conflict avoidance + 30% prioritization
```

### HardTask

**Goal:** Maximize comprehensive workflow optimization across all dimensions

```python
from src import HardTask

grader = HardTask()
score, explanation = grader.grade(observation, metadata)
# Score: 0-1.0
# Scoring: 25% email + 25% meeting + 25% task + 25% system health
```

### Unified Evaluation

```python
from src import evaluate_agent_performance

result = evaluate_agent_performance(env, obs, task_difficulty="medium")

print(result['score'])                  # 0-1.0
print(result['explanation'])            # Detailed breakdown
print(result['task_name'])              # Task name
print(result['final_metrics'])          # {email_triage_rate, ...}
print(result['metadata'])               # {emails_triaged, escalations, ...}
```

---

## Usage Example

```python
from src import EnterpriseEnv, Action, ActionType, EmailPriority, EmailCategory
from src.graders import evaluate_agent_performance

# Create and reset environment
env = EnterpriseEnv(num_emails_per_day=10, max_steps=100)
obs, info = env.reset(seed=42)

# Run agent loop
cumulative_reward = 0.0
while not obs.done:
    # Get unprocessed emails
    unprocessed = [e for e in obs.emails if not e.processed]
    
    if unprocessed:
        # Triage first urgent email
        email = next((e for e in unprocessed if e.urgency >= 7), unprocessed[0])
        action = Action(
            action_type=ActionType.TRIAGE_EMAIL,
            email_id=email.email_id,
            category=EmailCategory.URGENT,
            priority=EmailPriority.HIGH,
        )
    else:
        action = Action(action_type=ActionType.NOOP)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward

# Evaluate final performance
result = evaluate_agent_performance(env, obs, task_difficulty="medium")
print(f"Final Score: {result['score']:.2f}/1.0")
```

---

## Performance Metrics

Available metrics in observation:

- `email_triage_rate`: Fraction of emails triaged
- `task_completion_rate`: Fraction of tasks completed
- `meeting_schedule_success_rate`: Fraction of meetings scheduled
- `unprocessed_emails_count`: Number of untriaged emails
- `overdue_tasks_count`: Number of missed deadlines
- `meeting_conflicts`: Number of scheduling conflicts
- `total_pending_hours`: Cumulative hours of pending work

---

## Integration with RL Frameworks

### Stable Baselines3

```python
from stable_baselines3 import PPO
from src import EnterpriseEnv

env = EnterpriseEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

### Ray RLlib

```python
from ray import tune
from ray.rllib.algorithms.ppo import PPO

config = PPO.get_default_config()
config.environment(env_class="src.EnterpriseEnv")
trainer = PPO(config)
```

---

## Troubleshooting

**Q: Episode ends immediately**
A: Check that `max_steps` is sufficient and `reset()` is called before `step()`

**Q: NaN in rewards**
A: Verify actions have required parameters (e.g., meeting has scheduled_time)

**Q: Slow execution**
A: Reduce `num_emails_per_day`, `num_meetings_per_day`, `num_tasks_per_day`

---

## Citation

If using this environment in research, cite as:

```bibtex
@software{enterprise_automation_env_2024,
  title={Enterprise Task Automation Environment},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/enterprise-env}
}
```
