"""
Drone Delivery RL – Grader & Task Definitions
================================================
Provides deterministic, reproducible tasks with difficulty progression
and a grader that scores episode outcomes in [0.0, 1.0].

Tasks
-----
  1. clear_sky       (Easy)   – Short distance, no obstacles
  2. suburbs         (Medium) – Medium distance, few obstacles
  3. downtown        (Hard)   – Full distance, dense obstacles
  4. precision_drop  (Hard)   – Tight delivery radius, obstacles
  5. gauntlet        (Expert) – Maximum distance, 25 obstacles, tight everything

Grading
-------
  Score is a weighted combination of:
    - delivery: 0.4 weight (binary – did the package arrive?)
    - progress: 0.25 weight (fraction of distance covered)
    - efficiency: 0.15 weight (steps used vs optimal)
    - safety: 0.1 weight (no collisions, no OOB)
    - smoothness: 0.1 weight (low speed at landing)

  All scores are [0.0, 1.0] and deterministic for a given seed+actions.
"""

from __future__ import annotations

import math
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


#  Five canonical tasks with escalating challenge
TASKS: Dict[str, TaskDefinition] = {}

def _register(t: TaskDefinition):
    TASKS[t.task_id] = t


_register(TaskDefinition(
    task_id="clear_sky",
    name="Clear Sky",
    description="Short flight over flat terrain with no obstacles. "
                "Tests basic takeoff → cruise → land.",
    difficulty="easy",
    difficulty_score=0.2,
    seed=1001,
    world_size=100.0,
    start=(10, 10),
    target=(80, 80),
    num_obstacles=0,
    custom_obstacles=[],
    delivery_radius=3.0,
    max_steps=500,
))

_register(TaskDefinition(
    task_id="suburbs",
    name="Suburban Delivery",
    description="Medium distance with a handful of scattered obstacles. "
                "Tests navigation and basic avoidance.",
    difficulty="medium",
    difficulty_score=0.4,
    seed=2002,
    world_size=150.0,
    start=(10, 10),
    target=(130, 130),
    num_obstacles=6,
    custom_obstacles=[
        ObstacleConfig(x=50,  y=50,  height=8,  obstacle_type="tree"),
        ObstacleConfig(x=70,  y=80,  height=12, obstacle_type="building"),
        ObstacleConfig(x=90,  y=60,  height=10, obstacle_type="tower"),
        ObstacleConfig(x=100, y=100, height=15, obstacle_type="building"),
        ObstacleConfig(x=60,  y=110, height=6,  obstacle_type="antenna"),
        ObstacleConfig(x=110, y=80,  height=9,  obstacle_type="tree"),
    ],
    delivery_radius=2.5,
    max_steps=800,
))

_register(TaskDefinition(
    task_id="downtown",
    name="Downtown Corridor",
    description="Long flight through a dense urban grid. "
                "A corridor of tall buildings forces path planning.",
    difficulty="hard",
    difficulty_score=0.7,
    seed=3003,
    world_size=200.0,
    start=(10, 10),
    target=(180, 180),
    num_obstacles=0,
    custom_obstacles=[
        # A wall of buildings across the diagonal
        ObstacleConfig(x=60,  y=60,  height=20,  size_x=3, size_y=3, obstacle_type="building"),
        ObstacleConfig(x=70,  y=70,  height=18,  size_x=3, size_y=3, obstacle_type="building"),
        ObstacleConfig(x=80,  y=80,  height=22,  size_x=3, size_y=3, obstacle_type="tower"),
        ObstacleConfig(x=90,  y=90,  height=16,  size_x=3, size_y=3, obstacle_type="building"),
        ObstacleConfig(x=100, y=100, height=25,  size_x=4, size_y=4, obstacle_type="tower"),
        ObstacleConfig(x=110, y=110, height=19,  size_x=3, size_y=3, obstacle_type="building"),
        ObstacleConfig(x=120, y=120, height=21,  size_x=3, size_y=3, obstacle_type="building"),
        # Flanking obstacles to limit detour
        ObstacleConfig(x=75,  y=95,  height=14, obstacle_type="antenna"),
        ObstacleConfig(x=95,  y=75,  height=14, obstacle_type="antenna"),
        ObstacleConfig(x=105, y=125, height=12, obstacle_type="tree"),
        ObstacleConfig(x=125, y=105, height=12, obstacle_type="tree"),
        ObstacleConfig(x=140, y=140, height=10, obstacle_type="building"),
        ObstacleConfig(x=50,  y=100, height=8,  obstacle_type="tree"),
        ObstacleConfig(x=100, y=50,  height=8,  obstacle_type="tree"),
        ObstacleConfig(x=150, y=160, height=11, obstacle_type="antenna"),
    ],
    delivery_radius=2.0,
    max_steps=1500,
))

_register(TaskDefinition(
    task_id="precision_drop",
    name="Precision Drop",
    description="Medium distance but extremely tight landing radius (0.8m). "
                "Tests fine control and precision descent.",
    difficulty="hard",
    difficulty_score=0.8,
    seed=4004,
    world_size=150.0,
    start=(15, 15),
    target=(130, 130),
    num_obstacles=0,
    custom_obstacles=[
        ObstacleConfig(x=60,  y=70,  height=15, obstacle_type="building"),
        ObstacleConfig(x=80,  y=50,  height=12, obstacle_type="tower"),
        ObstacleConfig(x=100, y=90,  height=18, size_x=3, size_y=3, obstacle_type="building"),
        ObstacleConfig(x=70,  y=110, height=10, obstacle_type="antenna"),
        ObstacleConfig(x=110, y=120, height=8,  obstacle_type="tree"),
    ],
    delivery_radius=0.8,        # very tight! must brake precisely
    max_steps=1000,
))

_register(TaskDefinition(
    task_id="gauntlet",
    name="The Gauntlet",
    description="Maximum distance through 25 obstacles with a tiny delivery zone "
                "and limited step budget. Genuinely challenges frontier models.",
    difficulty="expert",
    difficulty_score=1.0,
    seed=5005,
    world_size=200.0,
    start=(5, 5),
    target=(190, 190),
    num_obstacles=0,
    custom_obstacles=[
        # Dense obstacle field
        ObstacleConfig(x=30,  y=30,  height=12, obstacle_type="building"),
        ObstacleConfig(x=40,  y=55,  height=18, size_x=3, size_y=3, obstacle_type="tower"),
        ObstacleConfig(x=55,  y=40,  height=15, obstacle_type="building"),
        ObstacleConfig(x=60,  y=70,  height=22, size_x=4, size_y=4, obstacle_type="tower"),
        ObstacleConfig(x=70,  y=55,  height=14, obstacle_type="antenna"),
        ObstacleConfig(x=80,  y=80,  height=25, size_x=5, size_y=5, obstacle_type="building"),
        ObstacleConfig(x=90,  y=65,  height=20, size_x=3, size_y=3, obstacle_type="tower"),
        ObstacleConfig(x=65,  y=95,  height=16, obstacle_type="building"),
        ObstacleConfig(x=100, y=100, height=28, size_x=5, size_y=5, obstacle_type="tower"),
        ObstacleConfig(x=110, y=85,  height=19, obstacle_type="building"),
        ObstacleConfig(x=85,  y=115, height=17, obstacle_type="antenna"),
        ObstacleConfig(x=120, y=110, height=21, size_x=3, size_y=3, obstacle_type="building"),
        ObstacleConfig(x=130, y=130, height=24, size_x=4, size_y=4, obstacle_type="tower"),
        ObstacleConfig(x=115, y=140, height=13, obstacle_type="tree"),
        ObstacleConfig(x=140, y=115, height=15, obstacle_type="antenna"),
        ObstacleConfig(x=145, y=145, height=20, size_x=3, size_y=3, obstacle_type="building"),
        ObstacleConfig(x=150, y=125, height=11, obstacle_type="tree"),
        ObstacleConfig(x=125, y=155, height=18, obstacle_type="building"),
        ObstacleConfig(x=160, y=150, height=16, obstacle_type="tower"),
        ObstacleConfig(x=155, y=170, height=14, obstacle_type="antenna"),
        ObstacleConfig(x=170, y=155, height=12, obstacle_type="tree"),
        ObstacleConfig(x=165, y=165, height=22, size_x=4, size_y=4, obstacle_type="building"),
        ObstacleConfig(x=175, y=175, height=10, obstacle_type="antenna"),
        ObstacleConfig(x=50,  y=120, height=9,  obstacle_type="tree"),
        ObstacleConfig(x=120, y=50,  height=9,  obstacle_type="tree"),
    ],
    delivery_radius=0.6,        # extremely tight
    max_steps=1200,              # tight budget for 260m+ distance
))


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
#  Grader
# ──────────────────────────────────────────────────────────────────────────────

# Scoring weights (must sum to 1.0)
W_DELIVERY   = 0.40   # did the package arrive?
W_PROGRESS   = 0.25   # how close did we get?
W_EFFICIENCY = 0.15   # step economy
W_SAFETY     = 0.10   # collision / OOB avoidance
W_SMOOTHNESS = 0.10   # landing quality


def grade_episode(result: EpisodeResult) -> Dict[str, float]:
    """
    Score an episode result.  Returns a dict with component scores
    plus the final ``score`` in [0.0, 1.0].

    Deterministic:  same EpisodeResult → same score.
    """
    # ── Delivery (0 or 1) ─────────────────────
    delivery = 1.0 if result.delivered else 0.0

    # ── Progress (0 → didn't move, 1 → reached target vicinity) ───
    if result.initial_dist > 0:
        dist_covered = max(0, result.initial_dist - result.final_dist)
        progress = min(1.0, dist_covered / result.initial_dist)
    else:
        progress = 1.0

    # ── Efficiency (1 → used few steps, 0 → used all) ─────────────
    if result.delivered:
        # Optimal steps estimate: distance / (max_speed * dt)
        optimal = max(50, result.initial_dist / (16.67 * 0.1))
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

    # ── Weighted score ────────────────────────────────────────────
    score = (
        W_DELIVERY   * delivery   +
        W_PROGRESS   * progress   +
        W_EFFICIENCY * efficiency +
        W_SAFETY     * safety     +
        W_SMOOTHNESS * smoothness
    )
    score = round(min(1.0, max(0.0, score)), 4)

    return {
        "score": score,
        "delivery": round(delivery, 4),
        "progress": round(progress, 4),
        "efficiency": round(efficiency, 4),
        "safety": round(safety, 4),
        "smoothness": round(smoothness, 4),
        "weights": {
            "delivery": W_DELIVERY,
            "progress": W_PROGRESS,
            "efficiency": W_EFFICIENCY,
            "safety": W_SAFETY,
            "smoothness": W_SMOOTHNESS,
        },
    }


def list_tasks() -> List[Dict]:
    """Return all available tasks as dicts (for API serialisation)."""
    return [
        {
            "task_id": t.task_id,
            "name": t.name,
            "description": t.description,
            "difficulty": t.difficulty,
            "difficulty_score": t.difficulty_score,
            "world_size": t.world_size,
            "start": list(t.start),
            "target": list(t.target),
            "num_obstacles": len(t.custom_obstacles) or t.num_obstacles,
            "delivery_radius": t.delivery_radius,
            "max_steps": t.max_steps,
        }
        for t in TASKS.values()
    ]
