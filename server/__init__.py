"""Drone Delivery RL Environment package."""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so sibling modules resolve
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import DroneDeliveryEnvironment
from models import (
    DroneAction, DroneObservation, FlightPhase,
    Position, Velocity, Obstacle, NearbyObstacle, ObstacleConfig,
)
from grader import (
    TASKS, TASK_CONFIGS, TaskDefinition, EpisodeResult,
    BaseGrader, Task1Grader, Task2Grader, Task3Grader, Task4Grader,
    grade_episode, grade_task, list_tasks, get_task_json, save_task_result,
)
from client import DroneEnv

__all__ = [
    "DroneDeliveryEnvironment",
    "DroneAction",
    "DroneObservation",
    "FlightPhase",
    "Position",
    "Velocity",
    "Obstacle",
    "NearbyObstacle",
    "ObstacleConfig",
    "DroneEnv",
    "TASKS",
    "TASK_CONFIGS",
    "TaskDefinition",
    "EpisodeResult",
    "BaseGrader",
    "Task1Grader",
    "Task2Grader",
    "Task3Grader",
    "Task4Grader",
    "grade_episode",
    "grade_task",
    "list_tasks",
    "get_task_json",
    "save_task_result",
]
