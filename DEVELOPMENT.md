# Development & Deployment Guide

## Local Development

### Setup Environment

```bash
# Clone repository
cd enterprise_env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

### Run Tests & Demos

```bash
# Run comprehensive demo with all task difficulties
python test_demo.py

# Launch interactive Gradio interface
python app.py
# Visit http://localhost:7860
```

### Project Structure

```
enterprise_env/
├── src/
│   ├── __init__.py           # Package exports
│   ├── types.py              # Pydantic models
│   ├── environment.py        # Main environment class
│   └── graders.py            # Task graders (easy/medium/hard)
├── tests/
│   ├── test_environment.py   # Environment tests
│   └── test_graders.py       # Grader tests
├── tasks/
│   ├── easy.py               # Easy task spec
│   ├── medium.py             # Medium task spec
│   └── hard.py               # Hard task spec
├── app.py                    # Gradio web interface
├── test_demo.py              # Demo script
├── pyproject.toml            # Python packaging
├── requirements.txt          # Dependencies
├── Dockerfile                # Docker image
└── README.md                 # This file
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t enterprise-env:latest .
```

### Run Locally

```bash
# Run with default command (demo)
docker run -p 7860:7860 enterprise-env:latest

# Run interactive shell
docker run -it enterprise-env:latest /bin/bash

# Run Gradio app
docker run -p 7860:7860 enterprise-env:latest python app.py
```

### Push to Registry

```bash
# Tag image
docker tag enterprise-env:latest your-registry/enterprise-env:latest

# Push
docker push your-registry/enterprise-env:latest
```

## Hugging Face Spaces Deployment

### Method 1: Docker-based Space

1. Create a new Space on Hugging Face: https://huggingface.co/new-space
2. Select "Docker" as the SDK
3. Link this repository
4. The `README.md` and `Dockerfile` will be used automatically
5. Hugging Face will build and deploy the Docker image

### Method 2: GitHub Integration

1. Push code to GitHub
2. Create Hugging Face Space with GitHub integration
3. Select "Docker" SDK
4. Connect the GitHub repository

### Space Variables (optional)

Set these in Hugging Face Space settings:

```
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

## API Reference

### Environment Initialization

```python
env = EnterpriseEnv(
    num_emails_per_day: int = 10,
    num_meetings_per_day: int = 5,
    num_tasks_per_day: int = 8,
    max_steps: int = 100,
    seed: Optional[int] = None
)
```

### Core Methods

```python
# Reset environment
obs, info = env.reset(seed=42)

# Take action step
obs, reward, terminated, truncated, info = env.step(action)

# Get current state
obs = env.state()
```

### Expected Observation

```python
Observation(
    step: int,                          # Current step
    timestamp: datetime,                # Simulation time
    time_until_end: int,                # Minutes until end of day
    emails: List[Email],                # Current emails
    unprocessed_emails_count: int,      # Unprocessed count
    meetings: List[Meeting],            # Current meetings
    tasks: List[Task],                  # Current tasks
    notifications: List[Notification],  # Current notifications
    email_triage_rate: float,           # % triaged (0-1)
    task_completion_rate: float,        # % completed (0-1)
    meeting_schedule_success_rate: float,  # Success rate (0-1)
    valid_actions: List[ActionType],    # Valid actions now
)
```

### Action Types

```python
ActionType.TRIAGE_EMAIL          # Classify email
ActionType.SCHEDULE_MEETING      # Schedule a meeting
ActionType.RESCHEDULE_MEETING    # Move meeting
ActionType.CREATE_TASK           # Create task from email
ActionType.REPRIORITIZE_TASK     # Change task priority
ActionType.COMPLETE_TASK         # Mark task complete
ActionType.SEND_NOTIFICATION     # Send alert
ActionType.ESCALATE              # Escalate issue
ActionType.NOOP                  # Do nothing
```

### Reward Structure

```python
Reward(
    total: float,                      # Cumulative reward
    step_reward: float,                # Reward this step
    email_processing_reward: float,    # Email triage reward
    meeting_scheduling_reward: float,  # Meeting scheduling reward
    task_prioritization_reward: float, # Task priority reward
    deadline_adherence_reward: float,  # Meeting deadlines reward
    efficiency_reward: float,          # Batching/efficiency reward
    escalation_penalty: float,         # Escalation penalty
)
```

## Task Graders

### Easy Task Grader

```python
grader = EasyTask()
score, explanation = grader.grade(observation, metadata)
# Score: 0.0–1.0
# Scoring: Triage rate (50%) + Priority accuracy (50%)
```

### Medium Task Grader

```python
grader = MediumTask()
score, explanation = grader.grade(observation, metadata)
# Score: 0.0–1.0
# Scoring: Scheduling (40%) + Conflicts (30%) + Prioritization (30%)
```

### Hard Task Grader

```python
grader = HardTask()
score, explanation = grader.grade(observation, metadata)
# Score: 0.0–1.0
# Scoring: Email (25%) + Meeting (25%) + Task (25%) + Health (25%)
```

## Performance Benchmarks

Expected scores with different agents:

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random | 0.35 | 0.25 | 0.15 |
| Heuristic | 0.75 | 0.60 | 0.45 |
| RL (trained) | 0.92 | 0.85 | 0.72 |

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 7860
lsof -ti:7860 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :7860   # Windows
```

### Memory Issues with Large Environment

Reduce environment size:

```python
env = EnterpriseEnv(
    num_emails_per_day=3,
    num_meetings_per_day=2,
    num_tasks_per_day=2,
    max_steps=50
)
```

### Docker Build Fails

Ensure Python 3.11+ and all dependencies in `requirements.txt` are available:

```bash
docker build --no-cache -t enterprise-env:latest .
```

## Extending the Environment

### Add Custom Actions

1. Add action type to `ActionType` enum in `types.py`
2. Implement handler method in `EnterpriseEnv._process_action()`
3. Update `_get_valid_actions()` logic

### Add Custom Rewards

Modify `_process_action()` methods or `reward` object in `src/environment.py`

### Add Custom Tasks

1. Create new grader class inheriting from `TaskGrader`
2. Implement `grade()` method
3. Register in `get_task_graders()`

## Performance Optimization

### For Large-Scale Experiments

```python
# Use minimal environment for fast iteration
env = EnterpriseEnv(
    num_emails_per_day=3,
    num_meetings_per_day=2,
    num_tasks_per_day=2,
    max_steps=50
)
```

### Parallel Training

```python
import gym
import ray

ray.init()
env = ray.remote(EnterpriseEnv)
```

## Contributing

See CONTRIBUTING.md for guidelines.

## License

MIT License
