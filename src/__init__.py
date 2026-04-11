"""Enterprise Task Automation Environment - OpenEnv Compatible"""

__version__ = "0.1.0"

from .environment import EnterpriseEnv
from .types import (
    Observation,
    Action,
    Reward,
    Email,
    Meeting,
    Task,
    Notification,
    ActionType,
    EmailCategory,
    EmailPriority,
    TaskPriority,
    TaskStatus,
)
from .graders import (
    TaskGrader,
    EasyTask,
    MediumTask,
    HardTask,
    get_task_graders,
    evaluate_agent_performance,
    # OpenEnv proxy graders — referenced in openenv.yaml as src.graders:OpenEnv*
    OpenEnvEasyTask,
    OpenEnvMediumTask,
    OpenEnvHardTask,
)

__all__ = [
    "EnterpriseEnv",
    "Observation",
    "Action",
    "Reward",
    "Email",
    "Meeting",
    "Task",
    "Notification",
    "ActionType",
    "EmailCategory",
    "EmailPriority",
    "TaskPriority",
    "TaskStatus",
    "TaskGrader",
    "EasyTask",
    "MediumTask",
    "HardTask",
    "get_task_graders",
    "evaluate_agent_performance",
    "OpenEnvEasyTask",
    "OpenEnvMediumTask",
    "OpenEnvHardTask",
]
