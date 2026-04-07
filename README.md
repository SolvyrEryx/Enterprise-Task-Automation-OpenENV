---
title: Enterprise Task Automation
emoji: chart_with_upwards_trend
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: latest
app_file: app.py
pinned: false
tags:
  - openenv
  - enterprise
  - workflow
  - reinforcement-learning
  - openai-gym
---

# Enterprise Task Automation Environment

**Meta AI OpenEnv Hackathon Submission**

An OpenEnv-compatible reinforcement learning environment for optimizing enterprise workflow management, including email triage, meeting scheduling, and task prioritization.

**Hackathon Status:** [CHECK] **SUBMISSION READY** - All 13 validation checks passed

## Features

- **OpenEnv API Compliance:** Standard `reset()`, `step()`, and `state()` interface
- **Pydantic Models:** Type-safe observations, actions, and rewards
- **3 Difficulty Levels:** Easy, Medium, Hard tasks with calibrated graders (0.0–1.0 scoring)
- **Advanced LLM Agent:** GPT-4o-mini powered baseline with strategic prompt engineering
- **Realistic Simulation:** Dynamic inbox generation, deadline tracking, conflict detection, task dependencies
- **Reward Shaping:** Detailed reward breakdown with partial progress signals
- **Docker Support:** Production-ready deployment to Hugging Face Spaces
- **Structured Logging:** `[START]`, `[STEP]`, `[END]` format for evaluation

## Quick Start

```python
from src import EnterpriseEnv
from src.types import Action, ActionType, EmailPriority, EmailCategory

# Create environment
env = EnterpriseEnv(
    num_emails_per_day=10,
    num_meetings_per_day=5,
    num_tasks_per_day=8,
    max_steps=100
)

# Reset to initial state
obs, info = env.reset()

# Take an action
action = Action(
    action_type=ActionType.TRIAGE_EMAIL,
    email_id="email_0",
    category=EmailCategory.URGENT,
    priority=EmailPriority.HIGH
)

obs, reward, terminated, truncated, info = env.step(action)
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- pip package manager
- Virtual environment (recommended)

### Quick Validation Check
```bash
# Verify all hackathon requirements are met
python validator.py
```

### Installation

```bash
# Clone repository
cd enterprise_env

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running the Environment

```bash
# Quick demo
python quickstart.py

# Run test suite
python test_demo.py

# Launch interactive web interface
python app.py
# Visit http://localhost:7860
```

# Run inference script (hackathon submission)
python inference.py --task medium --steps 50

# Run pre-submission validator
python validator.py
```


## Task Definitions

### Easy: Email Triage Efficiency
- **Goal:** Triage at least 80% of emails, correctly prioritizing urgent ones
- **Scoring:** 0.5 points (triage rate) + 0.5 points (priority accuracy)

### Medium: Meeting Scheduling & Task Prioritization
- **Goal:** Schedule 70% of meetings without conflicts, reprioritize 60% of tasks
- **Scoring:** 0.4 (meetings) + 0.3 (conflicts) + 0.3 (prioritization)

### Hard: Comprehensive Workflow Optimization
- **Goal:** Maximize all workflow dimensions simultaneously
- **Scoring:** 0.25 (emails) + 0.25 (meetings) + 0.25 (tasks) + 0.25 (system health)

## Reward Function

Rewards are designed to provide dense, informative signals:

```
- Email triage: 0.5–1.0 based on urgency and priority accuracy
- Meeting scheduling: 0.4–1.0 based on impact and conflict avoidance
- Task completion: 0.0–1.0 based on deadline adherence
- Escalations: -0.2 per unnecessary escalation
```

## Evaluation

```python
from src.graders import evaluate_agent_performance

final_obs = env.state()
result = evaluate_agent_performance(env, final_obs, task_difficulty="medium")
print(f"Score: {result['score']:.2f}/1.0")
print(f"Explanation: {result['explanation']}")
```

## Hackathon Submission

### [CHECK] Validation Status
All 13 OpenEnv validation checks pass:
- Environment registration and API compliance
- Action space and observation space validation
- Reward function and termination conditions
- Structured logging format verification
- Docker deployment compatibility

### [CHART] Baseline Scores

**Reproducible baseline using OpenAI GPT models, seed=42**

| Task | Difficulty | Score | Model | Description |
|------|-----------|-------|-------|-------------|
| Email Triage Efficiency | Easy | **1.000** | GPT-3.5-turbo | Perfect email triage with 100% accuracy |
| Meeting Scheduling & Task Prioritization | Medium | **0.933** | GPT-3.5-turbo | Excellent meeting scheduling + task management |
| Comprehensive Workflow Optimization | Hard | **0.950+** | GPT-4o-mini | Advanced multi-objective optimization |

### Reproduction Command
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-3.5-turbo"
export HF_TOKEN="your-hugging-face-api-key"
python inference.py --task all --seed 42
```

**Note:** Hard task uses GPT-4o-mini for superior reasoning on complex multi-objective optimization. All scores are reproducible and represent the baseline performance for hackathon evaluation.

### [ROCKET] Deployment Instructions
1. **Create Hugging Face Space:** Use Docker SDK, set to public
2. **Set Environment Variables:** Add `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` in Space settings
3. **Deploy:** Push code to trigger automatic build
4. **Test:** Verify baseline scores match documented values

### [LIST] Submission Checklist
- [x] OpenEnv API compliance (13/13 checks pass)
- [x] Baseline agent with reproducible scores
- [x] Docker deployment support
- [x] Comprehensive documentation
- [x] Type-safe implementation with Pydantic
- [x] Realistic enterprise simulation
- [x] Advanced prompt engineering for hard tasks

## AI Agent Enhancements

**Hackathon-optimized baseline agent with advanced prompt engineering:**

### Core Features
- **Multi-Model Strategy:** GPT-3.5-turbo (easy/medium) + GPT-4o-mini (hard) for optimal performance
- **Structured Logging:** `[START]`, `[STEP]`, `[END]` JSON format for evaluation
- **Smart Fallbacks:** Intelligent heuristic fallbacks when LLM calls fail
- **Environment Scaling:** Difficulty-appropriate environment sizes (8/4/6 → 10/6/8 for hard)

### Advanced Hard Mode Features
- **Dependency Graph Resolution:** Separates READY vs BLOCKED tasks with prerequisite checking
- **Conflict-Free Scheduling:** Pre-computed meeting slots eliminate scheduling conflicts
- **Strategic Decision Trees:** Priority hierarchy: deadlines → tasks → meetings → emails
- **Real-Time Score Tracking:** Per-dimension progress monitoring with target thresholds
- **Absolute Rules:** Explicit NEVER-ESCALATE and NEVER-BLOCKED-TASK rules

### Performance Optimizations
- **Temperature Control:** 0.0 temperature for deterministic, reproducible results
- **Token Optimization:** Efficient prompts with clear action specifications
- **Error Handling:** Robust JSON parsing with smart recovery mechanisms
- **Seed Consistency:** Deterministic randomization for reproducible baselines

## Design Principles

- **Enterprise Reality:** Simulates realistic workflows with dependencies, deadlines, and conflicts
- **Partial Progress:** Agents receive immediate feedback even without completing tasks
- **Scalability:** Easily configure environment complexity
- **Transparency:** Detailed reward breakdown and evaluation metrics

## Tech Stack

- **Framework:** Gymnasium (OpenAI Gym compatible)
- **Types:** Pydantic v2
- **API:** RESTful with Gradio UI
- **Deployment:** Docker + Hugging Face Spaces

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run demos
python test_demo.py

# Launch Gradio interface
python app.py
```

## License

MIT License - Open for hackathon participation

## Citation

```bibtex
@software{enterprise_env_2024,
  author = {Enterprise Automation Team},
  title = {Enterprise Task Automation Environment},
  year = {2026},
  url = https://github.com/SolvyrEryx/Enterprise-Task-Automation-OpenENV
}
```

## Support

For issues, questions, or contributions, please open an issue on GitHub or contact the hackathon organizers.
