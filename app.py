"""
Gradio interface for Hugging Face Spaces deployment

Provides interactive web interface for the Enterprise Task Automation Environment.
Compatible with HF Spaces health checks via Gradio's built-in API.
"""

import gradio as gr
import random
from datetime import datetime, timedelta
from typing import Dict, Any

from src import (
    EnterpriseEnv,
    Action,
    ActionType,
    EmailPriority,
    EmailCategory,
    TaskPriority,
)
from src.graders import evaluate_agent_performance

# Global environment instance
_env_instance = {"env": None, "obs": None}


def create_demo_interface():
    """Create Gradio interface for the environment"""
    
    # Use global environment instance
    env_instance = _env_instance
    
    def initialize_environment(num_emails, num_meetings, num_tasks, max_steps):
        """Initialize a new environment"""
        env = EnterpriseEnv(
            num_emails_per_day=num_emails,
            num_meetings_per_day=num_meetings,
            num_tasks_per_day=num_tasks,
            max_steps=max_steps,
        )
        obs, _ = env.reset(seed=42)
        env_instance["env"] = env
        env_instance["obs"] = obs
        
        summary = f"""
        **Environment Initialized**
        - Emails: {num_emails}
        - Meetings: {num_meetings}
        - Tasks: {num_tasks}
        - Max Steps: {max_steps}
        
        **Current State:**
        - Unprocessed Emails: {obs.unprocessed_emails_count}
        - Pending Meetings: {sum(1 for m in obs.meetings if m.status == 'pending')}
        - Pending Tasks: {sum(1 for t in obs.tasks if t.status.value == 'pending')}
        """
        return summary
    
    def triage_email(email_id, category, priority):
        """Triage an email"""
        if not env_instance["env"]:
            return "Error: Environment not initialized"
        
        env = env_instance["env"]
        action = Action(
            action_type=ActionType.TRIAGE_EMAIL,
            email_id=email_id,
            category=EmailCategory(category),
            priority=EmailPriority(priority),
        )
        
        obs, reward, terminated, truncated, info = env.step(action)
        env_instance["obs"] = obs
        
        return f"Action: Triaged {email_id}\nReward: {reward:.3f}\nTerminated: {terminated}"
    
    def schedule_meeting(meeting_id, hours_offset):
        """Schedule a meeting"""
        if not env_instance["env"]:
            return "Error: Environment not initialized"
        
        env = env_instance["env"]
        obs = env_instance["obs"]
        
        time_slot = obs.timestamp + timedelta(hours=hours_offset)
        action = Action(
            action_type=ActionType.SCHEDULE_MEETING,
            meeting_id=meeting_id,
            scheduled_time=time_slot,
        )
        
        obs, reward, terminated, truncated, info = env.step(action)
        env_instance["obs"] = obs
        
        return f"Action: Scheduled {meeting_id}\nReward: {reward:.3f}\nTerminated: {terminated}"
    
    def get_current_state():
        """Get current environment state"""
        if not env_instance["obs"]:
            return "Environment not initialized"
        
        obs = env_instance["obs"]
        
        state = f"""
        **Current State (Step {obs.step})**
        
        **Emails:** {len(obs.emails)}
        - Unprocessed: {obs.unprocessed_emails_count}
        - Urgent: {obs.urgent_emails_count}
        - Triage Rate: {obs.email_triage_rate:.1%}
        
        **Meetings:** {len(obs.meetings)}
        - Scheduled: {obs.scheduled_meetings_count}
        - Conflicts: {obs.meeting_conflicts}
        - Success Rate: {obs.meeting_schedule_success_rate:.1%}
        
        **Tasks:** {len(obs.tasks)}
        - Overdue: {obs.overdue_tasks_count}
        - Blocked: {obs.blocked_tasks_count}
        - Completion Rate: {obs.task_completion_rate:.1%}
        
        **Notifications:** {obs.unread_notifications_count} unread
        
        **Time Until End:** {obs.time_until_end} minutes
        """
        return state
    
    def evaluate_performance(difficulty):
        """Evaluate agent performance"""
        if not env_instance["env"]:
            return "Error: Environment not initialized"
        
        env = env_instance["env"]
        obs = env_instance["obs"]
        
        result = evaluate_agent_performance(env, obs, task_difficulty=difficulty)
        
        eval_summary = f"""
        **Task:** {result['task_name']}
        **Difficulty:** {result['difficulty']}
        **Score:** {result['score']:.2f}/1.0
        
        **Explanation:**
        {result['explanation']}
        
        **Final Metrics:**
        - Email Triage Rate: {result['final_metrics']['email_triage_rate']:.1%}
        - Task Completion Rate: {result['final_metrics']['task_completion_rate']:.1%}
        - Meeting Success Rate: {result['final_metrics']['meeting_schedule_success_rate']:.1%}
        - Unprocessed Emails: {result['final_metrics']['unprocessed_emails']}
        - Overdue Tasks: {result['final_metrics']['overdue_tasks']}
        - Meeting Conflicts: {result['final_metrics']['meeting_conflicts']}
        """
        return eval_summary
    
    # Build Gradio interface
    with gr.Blocks(title="Enterprise Task Automation") as demo:
        gr.Markdown("# Enterprise Task Automation Environment")
        gr.Markdown("OpenEnv-compatible environment for managing email, meetings, and tasks")
        
        with gr.Tabs():
            # Setup Tab
            with gr.TabItem("Setup"):
                with gr.Column():
                    num_emails = gr.Slider(3, 20, value=10, step=1, label="Number of Emails")
                    num_meetings = gr.Slider(2, 10, value=5, step=1, label="Number of Meetings")
                    num_tasks = gr.Slider(3, 15, value=8, step=1, label="Number of Tasks")
                    max_steps = gr.Slider(20, 200, value=100, step=10, label="Max Steps")
                    
                    init_btn = gr.Button("Initialize Environment", variant="primary")
                    init_output = gr.Textbox(label="Status", lines=8)
                    
                    init_btn.click(
                        initialize_environment,
                        inputs=[num_emails, num_meetings, num_tasks, max_steps],
                        outputs=init_output,
                    )
            
            # State Tab
            with gr.TabItem("State"):
                state_btn = gr.Button("Refresh State")
                state_output = gr.Textbox(label="Current State", lines=15)
                state_btn.click(get_current_state, outputs=state_output)
            
            # Actions Tab
            with gr.TabItem("Actions"):
                gr.Markdown("### Email Triage")
                with gr.Row():
                    email_id = gr.Textbox(label="Email ID", placeholder="email_0")
                    email_category = gr.Dropdown(
                        ["urgent", "actionable", "informational", "spam"],
                        label="Category",
                        value="urgent",
                    )
                    email_priority = gr.Dropdown(
                        ["critical", "high", "medium", "low"],
                        label="Priority",
                        value="high",
                    )
                
                triage_btn = gr.Button("Triage Email")
                triage_output = gr.Textbox(label="Result")
                triage_btn.click(
                    triage_email,
                    inputs=[email_id, email_category, email_priority],
                    outputs=triage_output,
                )
                
                gr.Markdown("### Meeting Scheduling")
                with gr.Row():
                    meeting_id = gr.Textbox(label="Meeting ID", placeholder="meeting_0")
                    hours_offset = gr.Slider(1, 24, value=2, step=1, label="Hours from Now")
                
                schedule_btn = gr.Button("Schedule Meeting")
                schedule_output = gr.Textbox(label="Result")
                schedule_btn.click(
                    schedule_meeting,
                    inputs=[meeting_id, hours_offset],
                    outputs=schedule_output,
                )
            
            # Evaluation Tab
            with gr.TabItem("Evaluation"):
                difficulty = gr.Radio(
                    ["easy", "medium", "hard"],
                    value="medium",
                    label="Task Difficulty",
                )
                eval_btn = gr.Button("Evaluate Performance", variant="primary")
                eval_output = gr.Textbox(label="Evaluation Result", lines=15)
                
                eval_btn.click(evaluate_performance, inputs=difficulty, outputs=eval_output)
            
            # Documentation Tab
            with gr.TabItem("Documentation"):
                gr.Markdown("""
                ## Enterprise Task Automation Environment
                
                OpenEnv-compatible environment for simulating enterprise workflow management.
                
                ### Key Components:
                - **Email Triage:** Classify and prioritize incoming emails
                - **Meeting Scheduling:** Schedule meetings while avoiding conflicts
                - **Task Management:** Prioritize and track task completion
                
                ### Tasks:
                
                **Easy:** Email Triage Efficiency
                - Triage at least 80% of emails
                - Correctly prioritize urgent ones
                
                **Medium:** Meeting Scheduling & Task Prioritization
                - Schedule 70% of meetings without conflicts
                - Reprioritize 60% of tasks
                
                **Hard:** Comprehensive Workflow Optimization
                - Maximize all workflow dimensions simultaneously
                - Balance email, meetings, tasks, and system health
                
                ### Workflow:
                1. Initialize environment with Setup tab
                2. Check current state in State tab
                3. Execute actions in Actions tab
                4. Evaluate performance in Evaluation tab
                """)
        
        return demo
    
    return demo


if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

