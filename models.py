# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Drone Delivery Environment.

Models the state space of a drone navigating a 3D grid to deliver packages
while avoiding obstacles.

Flight Phases
-------------
  GROUND      – Drone is on the ground, ready to take off
  LIFTING     – Drone is ascending to cruise altitude
  CRUISING    – Drone is flying at cruise altitude towards the target
  DESCENDING  – Drone is descending towards the delivery location
  LANDED      – Package delivered, episode done

Obstacle Model
--------------
  Obstacles are axis-aligned bounding boxes (AABB) with configurable
  dimensions (size_x, size_y, size_z).  Default footprint is 2×2 metres.
"""

from enum import Enum
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
#  Flight phase enum
# ─────────────────────────────────────────────
class FlightPhase(str, Enum):
    """Discrete flight phases for the package-delivery mission."""
    GROUND     = "GROUND"
    LIFTING    = "LIFTING"
    CRUISING   = "CRUISING"
    DESCENDING = "DESCENDING"
    LANDED     = "LANDED"


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
    """
    A static axis-aligned bounding-box obstacle in the environment.

    The obstacle is centred at ``position`` with half-extents
    ``size_x/2``, ``size_y/2``, ``size_z/2`` along each axis.
    Default footprint is 2×2 metres.
    """
    id: int = Field(..., description="Unique obstacle identifier")
    position: Position = Field(..., description="Center position of the obstacle")
    size_x: float = Field(default=2.0, description="Extent along X axis (metres)")
    size_y: float = Field(default=2.0, description="Extent along Y axis (metres)")
    size_z: float = Field(default=10.0, description="Extent along Z axis (metres)")
    obstacle_type: str = Field(default="building", description="Type of obstacle")


class NearbyObstacle(BaseModel):
    """Obstacle within sensor range, relative to the drone."""
    id: int
    relative_x: float
    relative_y: float
    relative_z: float
    distance: float
    size_x: float
    size_y: float
    size_z: float
    obstacle_type: str


class ObstacleConfig(BaseModel):
    """Input model for specifying a custom obstacle via the API."""
    x: float = Field(..., description="X position of obstacle centre")
    y: float = Field(..., description="Y position of obstacle centre")
    z: Optional[float] = Field(
        default=None,
        description="Z centre position (defaults to height/2, sitting on the ground)",
    )
    height: float = Field(default=10.0, ge=1.0, description="Obstacle height in metres")
    size_x: float = Field(default=2.0, ge=0.5, description="Footprint X extent (m)")
    size_y: float = Field(default=2.0, ge=0.5, description="Footprint Y extent (m)")
    obstacle_type: str = Field(default="building", description="Type: building, tower, tree, antenna")


# ─────────────────────────────────────────────
#  Action
# ─────────────────────────────────────────────
class DroneAction(Action):
    """
    Continuous thrust command sent to the drone.

    Each axis acceleration is clamped to [-max_accel, +max_accel] inside the
    environment so you can send raw network outputs without pre-clipping.

    The agent has full manual control: these accelerations are applied
    directly to the drone's velocity each timestep.
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
    mission status flags, flight phase, and the current A* waypoint hint.
    """
    # ── Drone state ──────────────────────────
    position: Position = Field(..., description="Current drone position")
    velocity: Velocity = Field(..., description="Current drone velocity")
    acceleration: Velocity = Field(
        default_factory=Velocity,
        description="Acceleration applied this step (m/s²)",
    )

    # ── Flight phase ─────────────────────────
    flight_phase: str = Field(
        default=FlightPhase.GROUND.value,
        description="Current flight phase: GROUND, LIFTING, CRUISING, DESCENDING, LANDED",
    )
    cruise_altitude: float = Field(
        default=15.0,
        description="Dynamic cruise altitude computed from obstacle heights (m)",
    )

    # ── Goal info ────────────────────────────
    target_position: Position = Field(..., description="Package delivery target")
    distance_to_target: float = Field(..., description="Euclidean distance to target (m)")
    horizontal_distance_to_target: float = Field(
        default=0.0,
        description="Horizontal (XY) distance to target (m)",
    )
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
    steps_remaining: int = Field(default=2000)

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
