# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Drone Delivery Environment.

Models the state space of a drone navigating a 3D grid to deliver packages
while avoiding obstacles.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  Action / Observation base shims
#  (used when openenv is not installed)
# ─────────────────────────────────────────────
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    class Action(BaseModel):        # type: ignore[no-redef]
        pass
    class Observation(BaseModel):   # type: ignore[no-redef]
        pass


# ─────────────────────────────────────────────
#  Primitive helpers
# ─────────────────────────────────────────────
class Position(BaseModel):
    """3-D position in the environment grid."""
    x: float = Field(..., description="X coordinate (East-West axis)")
    y: float = Field(..., description="Y coordinate (North-South axis)")
    z: float = Field(..., description="Z coordinate (Altitude)")


class Velocity(BaseModel):
    """3-D velocity vector of the drone."""
    vx: float = Field(default=0.0, description="Velocity along X axis (m/s)")
    vy: float = Field(default=0.0, description="Velocity along Y axis (m/s)")
    vz: float = Field(default=0.0, description="Velocity along Z axis (m/s)")


class Obstacle(BaseModel):
    """A static obstacle in the environment."""
    id: int = Field(..., description="Unique obstacle identifier")
    position: Position = Field(..., description="Center position of the obstacle")
    radius: float = Field(default=1.0, description="Collision radius (metres)")
    obstacle_type: str = Field(default="building", description="Type of obstacle")


class NearbyObstacle(BaseModel):
    """Obstacle within sensor range, relative to the drone."""
    id: int
    relative_x: float
    relative_y: float
    relative_z: float
    distance: float
    radius: float
    obstacle_type: str


# ─────────────────────────────────────────────
#  Action
# ─────────────────────────────────────────────
class DroneAction(Action):
    """
    Continuous thrust command sent to the drone.

    Each axis acceleration is clamped to [-max_accel, +max_accel] inside the
    environment so you can send raw network outputs without pre-clipping.
    """
    ax: float = Field(default=0.0, description="Acceleration along X axis (m/s²)")
    ay: float = Field(default=0.0, description="Acceleration along Y axis (m/s²)")
    az: float = Field(default=0.0, description="Acceleration along Z axis (m/s²)")


# ─────────────────────────────────────────────
#  Observation
# ─────────────────────────────────────────────
class DroneObservation(Observation):
    """
    Full observation returned after every step / reset.

    Includes drone kinematics, goal information, nearby hazards,
    mission status flags, and the current A* waypoint hint.
    """
    # ── Drone state ──────────────────────────
    position: Position = Field(..., description="Current drone position")
    velocity: Velocity = Field(..., description="Current drone velocity")

    # ── Goal info ────────────────────────────
    target_position: Position = Field(..., description="Package delivery target")
    distance_to_target: float = Field(..., description="Euclidean distance to target (m)")
    target_direction: Tuple[float, float, float] = Field(
        ..., description="Unit vector pointing towards the target"
    )

    # ── Hazards ──────────────────────────────
    nearby_obstacles: List[NearbyObstacle] = Field(
        default_factory=list,
        description="Obstacles within sensor range, sorted by distance",
    )
    min_obstacle_distance: float = Field(
        default=float("inf"),
        description="Distance to the nearest obstacle (m)",
    )

    # ── Mission status ───────────────────────
    package_delivered: bool = Field(default=False)
    collision_occurred: bool = Field(default=False)
    out_of_bounds: bool = Field(default=False)
    steps_remaining: int = Field(default=500)

    # ── Navigation hint ──────────────────────
    next_waypoint: Optional[Position] = Field(
        default=None,
        description="Next waypoint on the A* optimal path (None when no path found)",
    )
    path_length: int = Field(
        default=0,
        description="Remaining number of waypoints on the A* path",
    )

    # ── RL scalars ───────────────────────────
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    metadata: dict = Field(default_factory=dict)
