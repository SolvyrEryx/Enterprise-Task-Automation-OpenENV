"""
Gradio interface for Hugging Face Spaces deployment
Enterprise Task Automation — OpenEnv Hackathon
"""

import os
import gradio as gr
from datetime import timedelta

from inference import run_llm_agent, STEP_BUDGETS
from src import (
    EnterpriseEnv,
    Action,
    ActionType,
    EmailPriority,
    EmailCategory,
    TaskPriority,
)
from src.graders import evaluate_agent_performance

# ── Global environment instance (single user demo) ───────────────────────────
_env_instance: dict = {"env": None, "obs": None}


# ── Helper: ensure API key is available before calling inference ──────────────
def _check_api_key() -> str | None:
    """Returns None if a key is available, or an error string if not."""
    key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not key:
        return (
            "ERROR: No API key found.\n"
            "Set OPENAI_API_KEY in your HF Space secrets (Settings → Variables and Secrets)."
        )
    return None


# ── Tab callbacks ─────────────────────────────────────────────────────────────

def initialize_environment(num_emails, num_meetings, num_tasks, max_steps):
    env = EnterpriseEnv(
        num_emails_per_day=int(num_emails),
        num_meetings_per_day=int(num_meetings),
        num_tasks_per_day=int(num_tasks),
        max_steps=int(max_steps),
    )
    obs, _ = env.reset(seed=42)
    _env_instance["env"] = env
    _env_instance["obs"] = obs
    return (
        f"**Environment Initialized**\n"
        f"- Emails: {num_emails}  Meetings: {num_meetings}  Tasks: {num_tasks}\n"
        f"- Max Steps: {max_steps}\n\n"
        f"**Initial State:**\n"
        f"- Unprocessed Emails : {obs.unprocessed_emails_count}\n"
        f"- Pending Meetings   : {sum(1 for m in obs.meetings if m.status == 'pending')}\n"
        f"- Pending Tasks      : {sum(1 for t in obs.tasks if t.status.value == 'pending')}\n"
    )


def triage_email(email_id, category, priority):
    if not _env_instance["env"]:
        return "Error: Initialize the environment first (Setup tab)."
    action = Action(
        action_type=ActionType.TRIAGE_EMAIL,
        email_id=email_id.strip(),
        category=EmailCategory(category),
        priority=EmailPriority(priority),
    )
    obs, reward, terminated, truncated, info = _env_instance["env"].step(action)
    _env_instance["obs"] = obs
    return (
        f"Triaged: {email_id}\n"
        f"Reward : {reward:.3f}\n"
        f"Done   : {terminated or truncated}\n"
        f"Cumulative reward: {info['cumulative_reward']:.3f}"
    )


def schedule_meeting(meeting_id, hours_offset):
    if not _env_instance["env"]:
        return "Error: Initialize the environment first (Setup tab)."
    obs = _env_instance["obs"]
    time_slot = obs.timestamp + timedelta(hours=int(hours_offset))
    action = Action(
        action_type=ActionType.SCHEDULE_MEETING,
        meeting_id=meeting_id.strip(),
        scheduled_time=time_slot,
    )
    obs, reward, terminated, truncated, info = _env_instance["env"].step(action)
    _env_instance["obs"] = obs
    return (
        f"Scheduled: {meeting_id}\n"
        f"Time Slot: {time_slot.strftime('%Y-%m-%d %H:%M')}\n"
        f"Reward   : {reward:.3f}\n"
        f"Done     : {terminated or truncated}"
    )


def get_current_state():
    obs = _env_instance.get("obs")
    if obs is None:
        return "Environment not initialized."
    return (
        f"**Step {obs.step} — {obs.timestamp.strftime('%Y-%m-%d %H:%M')}**\n\n"
        f"**Emails:** {len(obs.emails)}\n"
        f"  Unprocessed : {obs.unprocessed_emails_count}\n"
        f"  Urgent      : {obs.urgent_emails_count}\n"
        f"  Triage Rate : {obs.email_triage_rate:.1%}\n\n"
        f"**Meetings:** {len(obs.meetings)}\n"
        f"  Scheduled   : {obs.scheduled_meetings_count}\n"
        f"  Conflicts   : {obs.meeting_conflicts}\n"
        f"  Success Rate: {obs.meeting_schedule_success_rate:.1%}\n\n"
        f"**Tasks:** {len(obs.tasks)}\n"
        f"  Overdue     : {obs.overdue_tasks_count}\n"
        f"  Blocked     : {obs.blocked_tasks_count}\n"
        f"  Completion  : {obs.task_completion_rate:.1%}\n\n"
        f"**Notifications:** {obs.unread_notifications_count} unread\n"
        f"**Time until end of day:** {obs.time_until_end} min\n"
    )


def evaluate_performance(difficulty):
    env = _env_instance.get("env")
    obs = _env_instance.get("obs")
    if not env or obs is None:
        return "Error: Initialize the environment first (Setup tab)."
    result = evaluate_agent_performance(env, obs, task_difficulty=difficulty)
    return (
        f"**Task:** {result['task_name']}\n"
        f"**Difficulty:** {result['difficulty']}\n"
        f"**Score:** {result['score']:.4f}/1.0\n\n"
        f"**Explanation:**\n{result['explanation']}\n\n"
        f"**Final Metrics:**\n"
        f"  Email Triage Rate   : {result['final_metrics']['email_triage_rate']:.1%}\n"
        f"  Task Completion Rate: {result['final_metrics']['task_completion_rate']:.1%}\n"
        f"  Meeting Success Rate: {result['final_metrics']['meeting_schedule_success_rate']:.1%}\n"
        f"  Unprocessed Emails  : {result['final_metrics']['unprocessed_emails']}\n"
        f"  Overdue Tasks       : {result['final_metrics']['overdue_tasks']}\n"
        f"  Meeting Conflicts   : {result['final_metrics']['meeting_conflicts']}\n"
    )


def run_baseline_inference(task, steps, seed):
    """
    Run the LLM inference agent and return a formatted result.
    Credentials resolved from OPENAI_API_KEY (preferred) or HF_TOKEN (fallback).
    """
    err = _check_api_key()
    if err:
        return err

    try:
        result = run_llm_agent(
            task_difficulty=task,
            max_steps=int(steps) if steps else None,
            seed=int(seed),
            verbose=False,
        )
        return (
            f"**Baseline Inference Results**\n\n"
            f"Task       : {result['task']}\n"
            f"Difficulty : {result['difficulty']}\n"
            f"Model      : {result['model']}\n"
            f"Seed       : {result['seed']}\n"
            f"**Score    : {result['score']:.4f}/1.0**\n"
            f"Steps      : {result['steps_completed']}\n\n"
            f"**Metrics:**\n"
            f"  Email Triage Rate   : {result['final_metrics']['email_triage_rate']:.1%}\n"
            f"  Task Completion Rate: {result['final_metrics']['task_completion_rate']:.1%}\n"
            f"  Meeting Success Rate: {result['final_metrics']['meeting_schedule_success_rate']:.1%}\n"
            f"  Unprocessed Emails  : {result['final_metrics']['unprocessed_emails']}\n"
            f"  Overdue Tasks       : {result['final_metrics']['overdue_tasks']}\n"
            f"  Meeting Conflicts   : {result['final_metrics']['meeting_conflicts']}\n\n"
            f"**Explanation:**\n{result['explanation']}"
        )
    except Exception as e:
        return f"Error running inference: {e}"


# ── Gradio layout ─────────────────────────────────────────────────────────────

def create_demo_interface():
    with gr.Blocks(title="Enterprise Task Automation — OpenEnv") as demo:
        gr.Markdown("# Enterprise Task Automation Environment")
        gr.Markdown(
            "OpenEnv-compatible RL environment for email triage, meeting scheduling, "
            "and task prioritization.  "
            "[OpenEnv Hackathon Submission]"
        )

        with gr.Tabs():

            # ── Setup ─────────────────────────────────────────────────────────
            with gr.TabItem("Setup"):
                num_emails   = gr.Slider(3,  20,  value=10, step=1,  label="Emails per day")
                num_meetings = gr.Slider(2,  10,  value=5,  step=1,  label="Meetings per day")
                num_tasks    = gr.Slider(3,  15,  value=8,  step=1,  label="Tasks per day")
                max_steps    = gr.Slider(20, 200, value=100, step=10, label="Max steps")
                init_btn     = gr.Button("Initialize Environment", variant="primary")
                init_output  = gr.Textbox(label="Status", lines=8)
                init_btn.click(
                    initialize_environment,
                    inputs=[num_emails, num_meetings, num_tasks, max_steps],
                    outputs=init_output,
                )

            # ── State ─────────────────────────────────────────────────────────
            with gr.TabItem("State"):
                state_btn    = gr.Button("Refresh State")
                state_output = gr.Textbox(label="Current State", lines=18)
                state_btn.click(get_current_state, outputs=state_output)

            # ── Manual Actions ────────────────────────────────────────────────
            with gr.TabItem("Actions"):
                gr.Markdown("### Email Triage")
                with gr.Row():
                    email_id_in  = gr.Textbox(label="Email ID",  placeholder="email_0")
                    email_cat    = gr.Dropdown(["urgent","actionable","informational","spam"],
                                               label="Category", value="urgent")
                    email_pri    = gr.Dropdown(["critical","high","medium","low"],
                                               label="Priority",  value="high")
                triage_btn    = gr.Button("Triage Email")
                triage_output = gr.Textbox(label="Result")
                triage_btn.click(triage_email,
                                 inputs=[email_id_in, email_cat, email_pri],
                                 outputs=triage_output)

                gr.Markdown("### Meeting Scheduling")
                with gr.Row():
                    meeting_id_in = gr.Textbox(label="Meeting ID", placeholder="meeting_0")
                    hours_off     = gr.Slider(1, 24, value=2, step=1, label="Hours from now")
                sched_btn    = gr.Button("Schedule Meeting")
                sched_output = gr.Textbox(label="Result")
                sched_btn.click(schedule_meeting,
                                inputs=[meeting_id_in, hours_off],
                                outputs=sched_output)

            # ── Grader Evaluation ─────────────────────────────────────────────
            with gr.TabItem("Evaluation"):
                difficulty_radio = gr.Radio(["easy","medium","hard"],
                                            value="medium", label="Task Difficulty")
                eval_btn    = gr.Button("Evaluate Performance", variant="primary")
                eval_output = gr.Textbox(label="Evaluation Result", lines=15)
                eval_btn.click(evaluate_performance,
                               inputs=difficulty_radio, outputs=eval_output)

            # ── Baseline LLM Inference ────────────────────────────────────────
            with gr.TabItem("Baseline AI"):
                gr.Markdown(
                    "### Run LLM Agent (OpenAI)\n"
                    "Requires `OPENAI_API_KEY` set in HF Space Secrets "
                    "(Settings → Variables and Secrets)."
                )
                with gr.Row():
                    bl_task  = gr.Radio(["easy","medium","hard"],
                                        value="medium", label="Task")
                    bl_steps = gr.Slider(25, 150, value=75, step=5, label="Max Steps")
                    bl_seed  = gr.Number(value=42, label="Seed", precision=0)
                run_btn    = gr.Button("Run Baseline Inference", variant="primary")
                run_output = gr.Textbox(label="Output", lines=20)
                run_btn.click(run_baseline_inference,
                              inputs=[bl_task, bl_steps, bl_seed],
                              outputs=run_output)

            # Documentation Tab
            with gr.TabItem("Documentation"):
                gr.Markdown("""
# Enterprise Task Automation Environment: Official Documentation

## Introduction
The Enterprise Task Automation Environment is a production-ready, OpenEnv-compatible reinforcement learning environment designed for optimizing enterprise workflow management. The application simulates realistic workplace scenarios, requiring the agent to balance email triage, meeting scheduling, and task prioritization.

## Features & Technical Details
The environment is built to be robust, scalable, and fully type-safe.

* **OpenEnv API Compliance:** The environment adheres strictly to standard reinforcement learning interfaces, providing `reset()`, `step()`, and `state()` methods.
* **Pydantic Data Models:** All observations, actions, and rewards are governed by Pydantic models to guarantee type safety and runtime validation.
* **Realistic Simulation:** The environment realistically simulates dynamic daily inbox generation, tracks deadlines, detects scheduling conflicts, and advances simulation time by 5 minutes per step.
* **Rich Reward Shaping:** Rewards are broken down into 7+ informative component signals to provide dense, partial progress feedback, including penalties for unnecessary task escalations.
* **Three Difficulty Levels:** * **Easy:** Focuses purely on email triage efficiency with a goal of processing 80%+ of emails accurately.
  * **Medium:** Incorporates meeting scheduling and task prioritization while avoiding scheduling conflicts.
  * **Hard:** Requires the agent to maximize all workflow dimensions simultaneously to maintain system health.

## Setup Instructions (Hugging Face Spaces)
The environment is fully containerized and configured for seamless deployment on Hugging Face Spaces. Evaluators can deploy the application using the following straightforward steps:

1. Create a new Space on Hugging Face and select the **Docker** SDK as your runtime environment.
2. Upload the entire project repository to the newly created Space.
3. The system will automatically utilize the provided `Dockerfile` to build the application and expose port 7860 for the interactive Gradio web interface.


## Usage Guidelines for Evaluators
Once deployed, evaluators can interact with the environment either programmatically or through the intuitive Gradio web interface.

### Using the Interactive Web Interface
Navigate to the web interface URL provided by Hugging Face Spaces, which contains several distinct tabs:

* **Setup Tab:** Use sliders to configure the number of emails, meetings, and tasks, as well as the maximum steps for the episode, then click "Initialize Environment".
* **State Tab:** Click "Refresh State" to monitor the real-time simulation, including pending tasks, meeting conflicts, unprocessed emails, and time remaining in the simulated day.
* **Actions Tab:** Manually execute specific actions, such as triaging an email with a chosen category/priority or scheduling a meeting with a designated time offset, to view the immediate reward and termination status.
* **Evaluation Tab:** Select a task difficulty (Easy, Medium, or Hard) and click "Evaluate Performance" to see the agent's score out of 1.0, accompanied by a detailed performance explanation.
* **Documentation Tab:** Review built-in documentation detailing the rules, workflows, and task goals.

### Programmatic Evaluation
Evaluators can also assess the environment programmatically using the provided command-line scripts:
* Run the pre-submission validator to ensure all configurations and Pydantic models are working properly by executing `python validator.py`.
* Run the mandatory inference script to test baseline agents with structured logs: `python inference.py --task medium --steps 50 --seed 42`.

## Troubleshooting Tips
* **"Error: Environment not initialized":** If you receive this error when attempting to take an action or evaluate performance in the web interface, you must return to the "Setup" tab and successfully initialize the environment first.
* **Failed Deployments:** If the Hugging Face Space fails to build or launch, ensure that your `Dockerfile` is intact and that you have selected the Docker runtime (not Gradio runtime) during Space creation.
* **Missing Baseline Scores:** If you attempt to reproduce baseline scores using the inference script and it fails, ensure that you have exported your actual OpenAI API key to the `OPENAI_API_KEY` environment variable.
                """)
        
        return demo


if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",   # required for HF Spaces
        server_port=7860,
        share=False,
        show_error=True,
    )
