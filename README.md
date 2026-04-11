# Enterprise Task Automation Environment

**Meta AI OpenEnv Hackathon Submission**

A strategy-driven, multi-objective decision system built on top of an OpenEnv-compatible enterprise simulation.

**Hackathon Status:** [CHECKED] **SUBMISSION READY** — All validation checks passed, strategy-driven agent integrated

---

## Features

- **OpenEnv API Compliance:** Standard `reset()`, `step()`, and `state()` interface
- **Pydantic Models:** Type-safe observations, actions, and rewards
- **3 Difficulty Levels:** Easy, Medium, Hard tasks with calibrated graders (0.0–1.0 scoring)
- **Strategy-Driven Multi-Objective Agent:** Adaptive decision system with dynamic prioritization
- **Realistic Simulation:** Dynamic inbox generation, deadline tracking, conflict detection, task dependencies
- **Reward Shaping:** Dense multi-component reward signals with penalties and progress feedback
- **Structured Logging:** Strict `[START]`, `[STEP]`, `[END]` format for OpenEnv evaluation
- **FastAPI + Gradio Interface:** Fully interactive + programmatic evaluation support
- **Docker Support:** Production-ready deployment to Hugging Face Spaces

---

## Core Innovation

This system is not a static prompt-based agent. It implements a **strategy-driven decision layer** that dynamically optimizes multiple competing objectives in real time.

At each step, the agent:
1. Evaluates performance across four dimensions: Email, Meeting, Task, and System Health
2. Identifies the weakest dimension (primary constraint)
3. Selects actions that maximize improvement on that dimension
4. Uses a secondary objective to maintain balance

This transforms the agent into a **multi-objective optimization policy**, inspired by reinforcement learning principles.

---

## Why This Is Not a Standard LLM Agent

Most baseline agents rely on:
- Static prompts
- Greedy decision making
- No explicit prioritization

This system introduces:

- **Adaptive Strategy Selection**
- **Multi-Objective Optimization**
- **Constraint-Aware Reasoning**
- **Policy-Level Control**

The LLM is used as an execution engine, while **decision intelligence is handled by the system itself**.

---

## Design Philosophy

- Workflows are interdependent
- Optimizing one metric can degrade another
- Tradeoffs must be managed explicitly

The system enforces:
- Dependency-aware execution
- Conflict-free scheduling
- Penalty-aware decisions
- Continuous performance tracking

Goal: **Sustained system optimization under constraints**

---

## Quick Start

```python
from src import EnterpriseEnv
from src.types import Action, ActionType, EmailPriority, EmailCategory

env = EnterpriseEnv(
    num_emails_per_day=10,
    num_meetings_per_day=5,
    num_tasks_per_day=8,
    max_steps=100
)

obs, info = env.reset()

action = Action(
    action_type=ActionType.TRIAGE_EMAIL,
    email_id="email_0",
    category=EmailCategory.URGENT,
    priority=EmailPriority.HIGH
)

obs, reward, terminated, truncated, info = env.step(action)
```

---

## Setup Instructions

```bash
python validator.py

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## Running

```bash
python app.py
python inference.py --task medium --seed 42
```

---

## Task Definitions

Easy → Email triage  
Medium → Meetings + tasks  
Hard → Multi-objective optimization  

---

## Evaluation

```python
from src.graders import evaluate_agent_performance

final_obs = env.state()
result = evaluate_agent_performance(env, final_obs, task_difficulty="medium")

print(result['score'])
```

---

## Baseline Scores

| Task | Score |
|------|------|
| Easy | 0.98+ |
| Medium | 0.98+|
| Hard | 0.90+ |

---

## Agent Architecture

- Multi-objective policy prioritization
- Adaptive strategy switching
- Constraint-aware execution
- Robust fallback system
- Deterministic logging

---

## Deployment

1. Create HF Space (Docker)
2. Add OPENAI_API_KEY
3. Deploy

---

## License

MIT License
