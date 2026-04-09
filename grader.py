"""
Drone Delivery RL – Grader & Task Definitions
================================================
Provides deterministic, reproducible tasks with difficulty progression
and a class-based grader hierarchy that scores episode outcomes in [0.0, 1.0].

Tasks
-----
  1. task_1_easy   – 2D ground-level direct flight, no obstacles, no liftoff
  2. task_2_medium – 3D full flight phases (takeoff → cruise → land), no obstacles
  3. task_3_hard   – 3D full flight phases with dense obstacles requiring avoidance
  4. task_4_expert – 3D obstacles + constant wind pushing the drone off course

Grading (BaseGrader → per-task subclass)
------------------------------------------
  Score is a weighted combination of:
    - delivery:   binary – did the package arrive?
    - progress:   fraction of distance covered
    - efficiency: steps used vs optimal
    - safety:     no collisions, no OOB
    - smoothness: low speed at landing

  All scores are [0.0, 1.0] and deterministic for a given seed+actions.
"""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .models import Position, ObstacleConfig
except ImportError:
    from models import Position, ObstacleConfig       # type: ignore[no-redef]


# ──────────────────────────────────────────────────────────────────────────────
#  Task Definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskDefinition:
    """Immutable definition of a grading task."""
    task_id: str
    name: str
    description: str
    difficulty: str                          # easy | medium | hard | expert
    difficulty_score: float                  # 0.0 (trivial) to 1.0 (brutal)
    seed: int                                # deterministic RNG seed
    world_size: float
    start: Tuple[float, float]               # (x, y)
    target: Tuple[float, float]              # (x, y)
    num_obstacles: int                       # for random generation
    custom_obstacles: List[ObstacleConfig]   # overrides random if non-empty
    delivery_radius: float                   # metres
    max_steps: int                           # episode budget
    movement_mode: str = "3d"                # "2d" or "3d"
    wind: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # constant wind accel


# ──────────────────────────────────────────────────────────────────────────────
#  TASK_CONFIGS – the canonical task registry (matches tester pattern)
# ──────────────────────────────────────────────────────────────────────────────

TASK_CONFIGS: Dict[str, dict] = {
    "task_1_easy": {
        "name": "Direct Flight",
        "description": (
            "Move directly towards the target on a 2D ground plane (X-Y only). "
            "No obstacles, no liftoff required. Tests basic 2D navigation."
        ),
        "difficulty": "easy",
        "difficulty_score": 0.2,
        "seed": 1001,
        "world_size": 100.0,
        "start": (10, 10),
        "target": (80, 80),
        "num_obstacles": 0,
        "custom_obstacles": [],
        "delivery_radius": 5.0,
        "max_steps": 500,
        "movement_mode": "2d",
        "wind": (0.0, 0.0, 0.0),
    },
    "task_2_medium": {
        "name": "Vertical Mission",
        "description": (
            "Lift off from the ground, cruise at altitude towards the target, "
            "descend and land. Full 3D movement. No obstacles. "
            "Tests takeoff, altitude hold, descent, precision landing."
        ),
        "difficulty": "medium",
        "difficulty_score": 0.5,
        "seed": 2002,
        "world_size": 200.0,
        "start": (10, 10),
        "target": (180, 180),
        "num_obstacles": 0,
        "custom_obstacles": [],
        "delivery_radius": 3.0,
        "max_steps": 1000,
        "movement_mode": "3d",
        "wind": (0.0, 0.0, 0.0),
    },
    "task_3_hard": {
        "name": "Obstacle Course",
        "description": (
            "Take off, navigate through dense obstacles (buildings, towers), "
            "and land at the target. Full 3D movement. "
            "Tests pathfinding, obstacle avoidance, altitude management."
        ),
        "difficulty": "hard",
        "difficulty_score": 0.75,
        "seed": 3003,
        "world_size": 200.0,
        "start": (10, 10),
        "target": (180, 180),
        "num_obstacles": 0,
        "custom_obstacles": [
            # Diagonal wall of buildings blocking direct path
            ObstacleConfig(x=60,  y=60,  height=20,  size_x=3, size_y=3, obstacle_type="building"),
            ObstacleConfig(x=75,  y=75,  height=18,  size_x=3, size_y=3, obstacle_type="building"),
            ObstacleConfig(x=90,  y=90,  height=22,  size_x=4, size_y=4, obstacle_type="tower"),
            ObstacleConfig(x=105, y=105, height=25,  size_x=4, size_y=4, obstacle_type="tower"),
            ObstacleConfig(x=120, y=120, height=19,  size_x=3, size_y=3, obstacle_type="building"),
            ObstacleConfig(x=135, y=135, height=21,  size_x=3, size_y=3, obstacle_type="building"),
            # Flanking obstacles to limit easy detours
            ObstacleConfig(x=70,  y=100, height=14, obstacle_type="antenna"),
            ObstacleConfig(x=100, y=70,  height=14, obstacle_type="antenna"),
            ObstacleConfig(x=110, y=130, height=12, obstacle_type="tree"),
            ObstacleConfig(x=130, y=110, height=12, obstacle_type="tree"),
            ObstacleConfig(x=50,  y=90,  height=8,  obstacle_type="tree"),
            ObstacleConfig(x=150, y=155, height=10, obstacle_type="antenna"),
        ],
        "delivery_radius": 2.0,
        "max_steps": 1500,
        "movement_mode": "3d",
        "wind": (0.0, 0.0, 0.0),
    },
    "task_4_expert": {
        "name": "Storm Run",
        "description": (
            "Navigate through dense obstacles with constant wind (2 m/s² eastward) "
            "pushing the drone off course. Full 3D movement. "
            "Tests wind compensation, optimal pathing under perturbation, precision."
        ),
        "difficulty": "expert",
        "difficulty_score": 1.0,
        "seed": 4004,
        "world_size": 200.0,
        "start": (10, 10),
        "target": (185, 185),
        "num_obstacles": 0,
        "custom_obstacles": [
            # Dense obstacle field
            ObstacleConfig(x=50,  y=50,  height=15, obstacle_type="building"),
            ObstacleConfig(x=65,  y=70,  height=20, size_x=3, size_y=3, obstacle_type="tower"),
            ObstacleConfig(x=80,  y=55,  height=12, obstacle_type="antenna"),
            ObstacleConfig(x=75,  y=90,  height=22, size_x=4, size_y=4, obstacle_type="tower"),
            ObstacleConfig(x=95,  y=80,  height=18, size_x=3, size_y=3, obstacle_type="building"),
            ObstacleConfig(x=100, y=100, height=25, size_x=5, size_y=5, obstacle_type="tower"),
            ObstacleConfig(x=115, y=90,  height=16, obstacle_type="building"),
            ObstacleConfig(x=90,  y=120, height=14, obstacle_type="antenna"),
            ObstacleConfig(x=130, y=120, height=21, size_x=3, size_y=3, obstacle_type="building"),
            ObstacleConfig(x=120, y=140, height=19, size_x=3, size_y=3, obstacle_type="tower"),
            ObstacleConfig(x=145, y=145, height=17, obstacle_type="building"),
            ObstacleConfig(x=155, y=130, height=11, obstacle_type="tree"),
            ObstacleConfig(x=140, y=160, height=15, obstacle_type="antenna"),
            ObstacleConfig(x=165, y=160, height=20, size_x=4, size_y=4, obstacle_type="building"),
            ObstacleConfig(x=170, y=175, height=10, obstacle_type="tree"),
        ],
        "delivery_radius": 1.5,
        "max_steps": 1500,
        "movement_mode": "3d",
        "wind": (2.0, 0.0, 0.0),
    },
}


def _build_task_definition(task_id: str, cfg: dict) -> TaskDefinition:
    """Build a TaskDefinition from a TASK_CONFIGS entry."""
    return TaskDefinition(
        task_id=task_id,
        name=cfg["name"],
        description=cfg["description"],
        difficulty=cfg["difficulty"],
        difficulty_score=cfg["difficulty_score"],
        seed=cfg["seed"],
        world_size=cfg["world_size"],
        start=tuple(cfg["start"]),
        target=tuple(cfg["target"]),
        num_obstacles=cfg["num_obstacles"],
        custom_obstacles=cfg.get("custom_obstacles", []),
        delivery_radius=cfg["delivery_radius"],
        max_steps=cfg["max_steps"],
        movement_mode=cfg.get("movement_mode", "3d"),
        wind=tuple(cfg.get("wind", (0.0, 0.0, 0.0))),
    )


# Build the TASKS dict from TASK_CONFIGS
TASKS: Dict[str, TaskDefinition] = {
    tid: _build_task_definition(tid, cfg)
    for tid, cfg in TASK_CONFIGS.items()
}


# ──────────────────────────────────────────────────────────────────────────────
#  Episode Result (filled by the environment after an episode)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Collected at episode end for grading."""
    delivered: bool = False
    collision: bool = False
    out_of_bounds: bool = False
    timed_out: bool = False
    final_dist: float = 0.0       # distance remaining to target
    initial_dist: float = 1.0     # distance at episode start
    steps_used: int = 0
    max_steps: int = 2000
    total_reward: float = 0.0
    landing_speed: float = 0.0    # speed at moment of landing
    delivery_radius: float = 2.0  # the task's delivery radius


# ──────────────────────────────────────────────────────────────────────────────
#  BaseGrader – abstract base class for all task graders
# ──────────────────────────────────────────────────────────────────────────────

# Some external evaluators require task scores to be strictly within (0, 1)
_STRICT_SCORE_EPS = 1e-4


def _strict_score(score: float) -> float:
    """Clamp score strictly within (0, 1)."""
    score = min(1.0 - _STRICT_SCORE_EPS, max(_STRICT_SCORE_EPS, score))
    return round(score, 4)


class BaseGrader(ABC):
    """
    Abstract base grader for drone delivery tasks.

    Subclasses override ``weights`` to customise scoring emphasis.
    The ``grade()`` method computes a weighted score from standard
    component scores and returns a result dict.

    Usage::

        grader = Task1Grader(result)
        score_dict = grader.grade()
    """

    def __init__(self, result: EpisodeResult, task_id: str = "unknown"):
        self.result = result
        self.task_id = task_id

    @property
    @abstractmethod
    def weights(self) -> Dict[str, float]:
        """Return grading weight dict: delivery, progress, efficiency, safety, smoothness."""
        ...

    def compute_components(self) -> Dict[str, float]:
        """Compute common component scores in [0, 1] (task-agnostic)."""
        result = self.result

        # ── Delivery (0 or 1) ─────────────────────
        delivery = 1.0 if result.delivered else 0.0

        # ── Progress (0 → didn't move, 1 → reached target vicinity) ───
        if result.initial_dist > 0:
            dist_covered = max(0.0, result.initial_dist - result.final_dist)
            progress = min(1.0, dist_covered / result.initial_dist)
        else:
            progress = 1.0

        # ── Efficiency (1 → used few steps, 0 → used all) ─────────────
        if result.delivered:
            # Optimal steps estimate: distance / (max_speed * dt)
            optimal = max(50.0, result.initial_dist / (16.67 * 0.1))
            ratio = optimal / max(1, result.steps_used)
            efficiency = min(1.0, ratio)
        else:
            efficiency = 0.0

        # ── Safety (1 → no incidents, 0 → collision or OOB) ───────────
        if result.collision:
            safety = 0.0
        elif result.out_of_bounds:
            safety = 0.2
        else:
            safety = 1.0

        # ── Smoothness (1 → gentle landing, 0 → fast crash-landing) ───
        if result.delivered:
            # Landing speed of 0 → perfect, ≥ 5 m/s → score 0
            smoothness = max(0.0, 1.0 - result.landing_speed / 5.0)
        else:
            smoothness = 0.0

        return {
            "delivery": round(delivery, 4),
            "progress": round(progress, 4),
            "efficiency": round(efficiency, 4),
            "safety": round(safety, 4),
            "smoothness": round(smoothness, 4),
        }

    def _weighted_score(self, components: Dict[str, float]) -> float:
        w = self.weights
        return (
            w["delivery"]   * components["delivery"]
            + w["progress"]   * components["progress"]
            + w["efficiency"] * components["efficiency"]
            + w["safety"]     * components["safety"]
            + w["smoothness"] * components["smoothness"]
        )

    def grade(self) -> Dict:
        """
        Grade the episode and return a result dict.

        Returns
        -------
        dict with keys: score, delivery, progress, efficiency, safety,
        smoothness, weights, task_id.
        """
        components = self.compute_components()
        raw_score = self._weighted_score(components)
        score = _strict_score(raw_score)
        return {
            "score": score,
            **components,
            "weights": self.weights,
            "task_id": self.task_id,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Task-specific grader subclasses
# ──────────────────────────────────────────────────────────────────────────────

class Task1Grader(BaseGrader):
    """
    Task 1 – Easy (2D direct flight).

    Emphasises progress and basic delivery, light smoothness requirement.
    """

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "delivery": 0.35,
            "progress": 0.35,
            "efficiency": 0.20,
            "safety": 0.08,
            "smoothness": 0.02,
        }


class Task2Grader(BaseGrader):
    """
    Task 2 – Medium (3D vertical mission).

    Emphasises landing smoothness and successful delivery (full phase control).
    """

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "delivery": 0.40,
            "progress": 0.20,
            "efficiency": 0.10,
            "safety": 0.10,
            "smoothness": 0.20,
        }


class Task3Grader(BaseGrader):
    """
    Task 3 – Hard (3D obstacle course).

    Emphasises safety (collision avoidance) while still rewarding progress.
    """

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "delivery": 0.30,
            "progress": 0.25,
            "efficiency": 0.10,
            "safety": 0.25,
            "smoothness": 0.10,
        }


class Task4Grader(BaseGrader):
    """
    Task 4 – Expert (3D storm run).

    Harsh on safety and requires meaningful progress under wind.
    """

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "delivery": 0.30,
            "progress": 0.30,
            "efficiency": 0.05,
            "safety": 0.30,
            "smoothness": 0.05,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Grader registry & dispatch
# ──────────────────────────────────────────────────────────────────────────────

TASK_GRADERS: Dict[str, type] = {
    "task_1_easy":   Task1Grader,
    "task_2_medium": Task2Grader,
    "task_3_hard":   Task3Grader,
    "task_4_expert": Task4Grader,
}


class DefaultGrader(BaseGrader):
    """Fallback grader with balanced weights (used when task_id is unknown)."""

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "delivery": 0.40,
            "progress": 0.25,
            "efficiency": 0.15,
            "safety": 0.10,
            "smoothness": 0.10,
        }


def grade_task(task_id: str, result: EpisodeResult) -> Dict:
    """
    Dispatch grading to the task-specific grader class.

    If task_id is unknown/empty, falls back to DefaultGrader
    which stays within (0, 1).
    """
    grader_cls = TASK_GRADERS.get(task_id, DefaultGrader)
    grader = grader_cls(result, task_id=task_id or "unknown")
    return grader.grade()


# Backwards-compatible name (older code imports grade_episode)
def grade_episode(result: EpisodeResult) -> Dict:
    return grade_task("unknown", result)


# ──────────────────────────────────────────────────────────────────────────────
#  Task listing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _action_schema(movement_mode: str) -> dict:
    """Return the action schema description for a task."""
    if movement_mode == "2d":
        return {
            "ax": "float — acceleration along X axis (m/s²), range [-5.0, 5.0]",
            "ay": "float — acceleration along Y axis (m/s²), range [-5.0, 5.0]",
            "az": "fixed at 0.0 — 2D mode, no vertical control",
        }
    return {
        "ax": "float — acceleration along X axis (m/s²), range [-5.0, 5.0]",
        "ay": "float — acceleration along Y axis (m/s²), range [-5.0, 5.0]",
        "az": "float — acceleration along Z axis (m/s²), range [-5.0, 5.0]",
    }


def _observation_space_summary(movement_mode: str) -> dict:
    """Return the observation space description for a task."""
    base = {
        "position": "dict — {x, y, z} current drone position in metres",
        "velocity": "dict — {vx, vy, vz} current drone velocity in m/s",
        "target_position": "dict — {x, y, z} delivery target position",
        "distance_to_target": "float — Euclidean distance to target (m)",
        "target_direction": "list — unit vector [3] pointing towards target",
        "nearby_obstacles": "list — obstacles within 25m sensor range",
        "done": "bool — whether the episode has ended",
        "reward": "float — per-step reward signal",
        "steps_remaining": "int — steps until episode timeout",
    }
    if movement_mode == "3d":
        base["flight_phase"] = "string — GROUND | LIFTING | CRUISING | DESCENDING | LANDED"
        base["cruise_altitude"] = "float — dynamic cruise altitude (m)"
        base["horizontal_distance_to_target"] = "float — XY-plane distance (m)"
        base["next_waypoint"] = "dict — next A* waypoint or null"
        base["path_length"] = "int — remaining waypoints on A* path"
    return base


def _success_criteria(task_id: str) -> List[str]:
    """Return human-readable success criteria for a task."""
    criteria = {
        "task_1_easy": [
            "Reach the delivery target within 5.0 metres (2D distance)",
            "Stay within the 100m world bounds",
            "Complete within 500 steps",
            "Achieve a score of at least 0.7",
        ],
        "task_2_medium": [
            "Complete all flight phases: GROUND → LIFTING → CRUISING → DESCENDING → LANDED",
            "Land within 3.0 metres of the target",
            "Maintain smooth landing (speed < 2 m/s at touchdown)",
            "Complete within 1000 steps",
        ],
        "task_3_hard": [
            "Navigate through 12 AABB obstacles without collision",
            "Land within 2.0 metres of the target",
            "Maintain crop_health above 0.5 for all zones",
            "Complete within 1500 steps",
        ],
        "task_4_expert": [
            "Navigate through 15 obstacles under constant 2 m/s² eastward wind",
            "Compensate for wind drift to land within 1.5 metres of target",
            "Avoid all collisions and out-of-bounds",
            "Complete within 1500 steps",
        ],
    }
    return criteria.get(task_id, ["Deliver the package successfully"])


def get_task_json(task_id: str) -> dict:
    """
    Return a full task description dict in the standardised JSON format.

    Matches the tester's expected schema with action_schema,
    observation_space, grader weights, and success_criteria.
    """
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task_id}'. Valid: {list(TASK_CONFIGS.keys())}")

    cfg = TASK_CONFIGS[task_id]
    task = TASKS[task_id]
    grader_cls = TASK_GRADERS.get(task_id, DefaultGrader)
    # Instantiate with a dummy result just to read weights
    dummy_weights = grader_cls(EpisodeResult(), task_id).weights

    target_scores = {
        "task_1_easy":   (0.7, 1.0),
        "task_2_medium": (0.4, 0.7),
        "task_3_hard":   (0.25, 0.5),
        "task_4_expert": (0.1, 0.35),
    }
    score_min, score_max = target_scores.get(task_id, (0.3, 1.0))

    return {
        "task_id": task_id,
        "name": cfg["name"],
        "description": cfg["description"],
        "difficulty": cfg["difficulty"],
        "max_steps": cfg["max_steps"],
        "world_size": cfg["world_size"],
        "start": list(cfg["start"]),
        "target": list(cfg["target"]),
        "num_obstacles": len(cfg.get("custom_obstacles", [])) or cfg["num_obstacles"],
        "delivery_radius": cfg["delivery_radius"],
        "movement_mode": cfg.get("movement_mode", "3d"),
        "wind": list(cfg.get("wind", (0.0, 0.0, 0.0))),
        "seed": cfg["seed"],
        "target_score_min": score_min,
        "target_score_max": score_max,
        "action_schema": _action_schema(cfg.get("movement_mode", "3d")),
        "observation_space": _observation_space_summary(cfg.get("movement_mode", "3d")),
        "grader": {
            **{f"{k}_weight": v for k, v in dummy_weights.items()},
            "description": (
                "Scores delivery success, distance progress, step efficiency, "
                "collision avoidance (safety), and landing smoothness"
            ),
        },
        "success_criteria": _success_criteria(task_id),
    }


def list_tasks() -> List[Dict]:
    """Return all available tasks as dicts (for API serialisation)."""
    return [get_task_json(tid) for tid in TASK_CONFIGS]


# ──────────────────────────────────────────────────────────────────────────────
#  JSON result file writer
# ──────────────────────────────────────────────────────────────────────────────

def save_task_result(
    task_id: str,
    episode_id: str,
    grade_result: Dict,
    episode_result: EpisodeResult,
    tasks_dir: str = "Tasks",
) -> str:
    """
    Save a completed episode's results to a JSON file in the Tasks/ directory.

    Parameters
    ----------
    task_id : str
        The task identifier (e.g. "task_1_easy").
    episode_id : str
        The episode UUID from the environment state.
    grade_result : dict
        The dict returned by grade_task().
    episode_result : EpisodeResult
        The raw episode result dataclass.
    tasks_dir : str
        Directory to write results into. Created if absent.

    Returns
    -------
    str — path to the written JSON file.
    """
    os.makedirs(tasks_dir, exist_ok=True)

    # Build the task metadata portion
    task_json = get_task_json(task_id) if task_id in TASK_CONFIGS else {"task_id": task_id}

    # Merge episode outcome
    output = {
        **task_json,
        "episode_id": episode_id,
        "episode_result": {
            "delivered": episode_result.delivered,
            "collision": episode_result.collision,
            "out_of_bounds": episode_result.out_of_bounds,
            "timed_out": episode_result.timed_out,
            "final_dist": round(episode_result.final_dist, 4),
            "initial_dist": round(episode_result.initial_dist, 4),
            "steps_used": episode_result.steps_used,
            "max_steps": episode_result.max_steps,
            "total_reward": round(episode_result.total_reward, 4),
            "landing_speed": round(episode_result.landing_speed, 4),
            "score": grade_result.get("score", 0.0),
            "components": {
                k: grade_result[k]
                for k in ("delivery", "progress", "efficiency", "safety", "smoothness")
                if k in grade_result
            },
        },
    }

    filename = f"{task_id}_{episode_id[:8]}.json"
    filepath = os.path.join(tasks_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    return filepath
