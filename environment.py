# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Drone Delivery RL Environment
==============================
A 3-D continuous-state environment where a drone must navigate from a start
position to a delivery target while avoiding obstacles.

Physics model
-------------
  - Newtonian point-mass with drag.
  - Euler integration at configurable dt (default 0.1 s).
  - Acceleration commands are clamped to [-max_accel, +max_accel].
  - Speed is clamped to max_speed after integration.

Reward shaping
--------------
  +100   package delivered
  -100   collision
   -50   out-of-bounds
   +r    progress reward: Δ(distance_to_target) * progress_reward_scale
   -0.1  per-step living penalty (encourages efficiency)
   +10   if drone is on the A* path corridor (optional guidance bonus)

Pathfinding
-----------
  A* on a discretised 3-D grid (voxel_size configurable).
  Run once on reset; subsequent calls return cached waypoints.
  If the path is blocked the drone still receives shaping rewards.
"""

import math
import heapq
import random
from uuid import uuid4  
from typing import List, Optional, Tuple

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:                    # local / HF Spaces fallback
    from pydantic import BaseModel

    class State(BaseModel):            # type: ignore[no-redef]
        episode_id: str = ""
        step_count: int = 0

    class Environment:                 # type: ignore[no-redef]
        pass

try:
    from .models import (
        DroneAction, DroneObservation,
        Position, Velocity, Obstacle, NearbyObstacle,
    )
except ImportError:
    from models import (               # type: ignore[no-redef]
        DroneAction, DroneObservation,
        Position, Velocity, Obstacle, NearbyObstacle,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Tiny pure-Python A* on a voxelised grid
# ──────────────────────────────────────────────────────────────────────────────

def _astar(
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    obstacles: List[Obstacle],
    grid_size: int,
    voxel: float,
    drone_radius: float,
) -> List[Tuple[int, int, int]]:
    """
    A* pathfinding on a 3-D integer grid.

    Returns a list of voxel indices from start → goal (inclusive),
    or an empty list if no path is found.
    """
    def h(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def is_free(node):
        nx, ny, nz = node
        if not (0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size):
            return False
        # world-space centre of voxel
        wx = nx * voxel + voxel / 2
        wy = ny * voxel + voxel / 2
        wz = nz * voxel + voxel / 2
        for obs in obstacles:
            dx = wx - obs.position.x
            dy = wy - obs.position.y
            dz = wz - obs.position.z
            if math.sqrt(dx*dx + dy*dy + dz*dz) < obs.radius + drone_radius:
                return False
        return True

    # 26-connectivity
    neighbours = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    open_set: list = []
    heapq.heappush(open_set, (h(start, goal), 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    visited = set()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        for dx, dy, dz in neighbours:
            nb = (current[0]+dx, current[1]+dy, current[2]+dz)
            if nb in visited or not is_free(nb):
                continue
            step = math.sqrt(dx*dx + dy*dy + dz*dz)
            ng = g + step
            if ng < g_score.get(nb, float("inf")):
                g_score[nb] = ng
                came_from[nb] = current
                heapq.heappush(open_set, (ng + h(nb, goal), ng, nb))

    return []   # no path found


# ──────────────────────────────────────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────────────────────────────────────

class DroneDeliveryEnvironment(Environment):
    """
    Package-delivery drone RL environment.

    World
    -----
    A cubic space of side ``world_size`` metres (default 50 m).
    The drone starts near (5, 5, 5) with ± jitter and must reach
    a target near (45, 45, 10).  Random obstacles (buildings / towers)
    are scattered inside.

    State exposed via ``/state``
    ----------------------------
    episode_id, step_count, drone_position, drone_velocity,
    target_position, num_obstacles, current_reward_total, done.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Physics ──────────────────────────────
    MAX_SPEED:   float = 10.0    # m/s
    MAX_ACCEL:   float = 5.0     # m/s²
    DRAG:        float = 0.05    # proportional damping coefficient
    DT:          float = 0.1     # simulation timestep (s)

    # ── World ────────────────────────────────
    WORLD_SIZE:  float = 50.0    # metres per axis
    MIN_ALT:     float = 0.5     # minimum flight altitude (m)

    # ── Drone geometry ───────────────────────
    DRONE_RADIUS:    float = 0.4   # metres
    SENSOR_RANGE:    float = 10.0  # obstacle-detection radius (m)

    # ── Delivery ─────────────────────────────
    DELIVERY_RADIUS: float = 1.5   # metres – goal tolerance
    MAX_STEPS:       int   = 500

    # ── A* grid ──────────────────────────────
    VOXEL_SIZE: float = 2.0   # metres per voxel
    GRID_SIZE:  int   = 25    # voxels per axis  (50 / 2 = 25)

    # ── Reward shaping ───────────────────────
    DELIVERY_REWARD:      float =  100.0
    COLLISION_PENALTY:    float = -100.0
    OOB_PENALTY:          float =  -50.0
    PROGRESS_SCALE:       float =   1.0
    LIVING_PENALTY:       float =  -0.1
    PATH_BONUS:           float =   0.5   # bonus per step on A* path corridor

    def __init__(
        self,
        world_size: float = 50.0,
        num_obstacles: int = 8,
        seed: Optional[int] = None,
    ):
        self._world_size = world_size
        self._num_obstacles = num_obstacles
        self._rng = random.Random(seed)

        # will be populated on reset()
        self._pos = Position(x=0, y=0, z=0)
        self._vel = Velocity()
        self._target = Position(x=45, y=45, z=10)
        self._obstacles: List[Obstacle] = []
        self._path: List[Tuple[int,int,int]] = []
        self._waypoint_idx: int = 0
        self._done = False
        self._collision = False
        self._oob = False
        self._delivered = False
        self._prev_dist = 0.0
        self._total_reward = 0.0
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def reset(self) -> DroneObservation:
        """Randomise start / target / obstacles and plan an A* path."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._collision = False
        self._oob = False
        self._delivered = False
        self._total_reward = 0.0

        ws = self._world_size
        jitter = lambda base, spread: base + self._rng.uniform(-spread, spread)

        self._pos = Position(
            x=jitter(5.0, 2.0),
            y=jitter(5.0, 2.0),
            z=jitter(5.0, 1.5),
        )
        self._vel = Velocity()

        self._target = Position(
            x=jitter(ws - 5.0, 3.0),
            y=jitter(ws - 5.0, 3.0),
            z=jitter(10.0, 3.0),
        )

        self._obstacles = self._generate_obstacles()
        self._prev_dist = self._dist_to_target()
        self._plan_path()

        return self._make_observation(reward=0.0)

    def step(self, action: DroneAction) -> DroneObservation:  # type: ignore[override]
        """Apply thrust, integrate physics, compute reward, return observation."""
        if self._done:
            return self._make_observation(reward=0.0)

        self._state.step_count += 1

        # ── physics ──────────────────────────
        ax = self._clamp(action.ax, -self.MAX_ACCEL, self.MAX_ACCEL)
        ay = self._clamp(action.ay, -self.MAX_ACCEL, self.MAX_ACCEL)
        az = self._clamp(action.az, -self.MAX_ACCEL, self.MAX_ACCEL)

        self._vel.vx = (self._vel.vx + ax * self.DT) * (1 - self.DRAG)
        self._vel.vy = (self._vel.vy + ay * self.DT) * (1 - self.DRAG)
        self._vel.vz = (self._vel.vz + az * self.DT) * (1 - self.DRAG)
        self._vel = self._clamp_velocity(self._vel)

        self._pos.x += self._vel.vx * self.DT
        self._pos.y += self._vel.vy * self.DT
        self._pos.z += self._vel.vz * self.DT

        # ── termination checks ───────────────
        reward = self.LIVING_PENALTY
        self._oob = self._check_oob()
        self._collision = self._check_collision()
        self._delivered = self._dist_to_target() <= self.DELIVERY_RADIUS

        if self._oob:
            reward += self.OOB_PENALTY
            self._done = True
        elif self._collision:
            reward += self.COLLISION_PENALTY
            self._done = True
        elif self._delivered:
            reward += self.DELIVERY_REWARD
            self._done = True
        else:
            # progress shaping
            curr_dist = self._dist_to_target()
            progress = self._prev_dist - curr_dist
            reward += progress * self.PROGRESS_SCALE
            self._prev_dist = curr_dist

            # path-following bonus
            if self._on_path_corridor():
                reward += self.PATH_BONUS

        # max-step timeout
        if self._state.step_count >= self.MAX_STEPS:
            self._done = True

        self._total_reward += reward
        return self._make_observation(reward=reward)

    @property
    def state(self) -> State:
        return self._state

    # ──────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────

    def _generate_obstacles(self) -> List[Obstacle]:
        ws = self._world_size
        margin = 8.0
        obstacles = []
        for i in range(self._num_obstacles):
            # keep clear of start and goal
            for _ in range(100):
                x = self._rng.uniform(margin, ws - margin)
                y = self._rng.uniform(margin, ws - margin)
                z = self._rng.uniform(2.0, 20.0)
                r = self._rng.uniform(1.5, 4.0)
                pos = Position(x=x, y=y, z=z)
                if (self._euclidean(pos, self._pos) > r + 5.0 and
                        self._euclidean(pos, self._target) > r + 5.0):
                    obstacles.append(Obstacle(
                        id=i,
                        position=pos,
                        radius=r,
                        obstacle_type=self._rng.choice(
                            ["building", "tower", "tree", "antenna"]
                        ),
                    ))
                    break
        return obstacles

    def _plan_path(self):
        """Run A* and cache voxel waypoints."""
        def to_voxel(p: Position) -> Tuple[int,int,int]:
            v = self.VOXEL_SIZE
            return (
                max(0, min(self.GRID_SIZE-1, int(p.x / v))),
                max(0, min(self.GRID_SIZE-1, int(p.y / v))),
                max(0, min(self.GRID_SIZE-1, int(p.z / v))),
            )

        start_v = to_voxel(self._pos)
        goal_v  = to_voxel(self._target)

        self._path = _astar(
            start_v, goal_v,
            self._obstacles,
            self.GRID_SIZE,
            self.VOXEL_SIZE,
            self.DRONE_RADIUS,
        )
        self._waypoint_idx = 0

    def _current_waypoint(self) -> Optional[Position]:
        """Return the world-space centre of the current A* waypoint."""
        # Advance waypoint index when drone is close enough
        while (self._waypoint_idx < len(self._path) - 1):
            vx, vy, vz = self._path[self._waypoint_idx]
            v = self.VOXEL_SIZE
            wp = Position(x=vx*v + v/2, y=vy*v + v/2, z=vz*v + v/2)
            if self._euclidean(self._pos, wp) < v * 0.75:
                self._waypoint_idx += 1
            else:
                break

        if self._waypoint_idx >= len(self._path):
            return None
        vx, vy, vz = self._path[self._waypoint_idx]
        v = self.VOXEL_SIZE
        return Position(x=vx*v + v/2, y=vy*v + v/2, z=vz*v + v/2)

    def _on_path_corridor(self) -> bool:
        """True if the drone is within VOXEL_SIZE of any path waypoint."""
        v = self.VOXEL_SIZE
        for (vx, vy, vz) in self._path:
            wp = Position(x=vx*v + v/2, y=vy*v + v/2, z=vz*v + v/2)
            if self._euclidean(self._pos, wp) < v:
                return True
        return False

    def _nearby_obstacles(self) -> List[NearbyObstacle]:
        nearby = []
        for obs in self._obstacles:
            dx = obs.position.x - self._pos.x
            dy = obs.position.y - self._pos.y
            dz = obs.position.z - self._pos.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= self.SENSOR_RANGE:
                nearby.append(NearbyObstacle(
                    id=obs.id,
                    relative_x=dx,
                    relative_y=dy,
                    relative_z=dz,
                    distance=dist,
                    radius=obs.radius,
                    obstacle_type=obs.obstacle_type,
                ))
        nearby.sort(key=lambda o: o.distance)
        return nearby

    def _check_collision(self) -> bool:
        for obs in self._obstacles:
            if self._euclidean(self._pos, obs.position) < obs.radius + self.DRONE_RADIUS:
                return True
        if self._pos.z < self.MIN_ALT:
            return True
        return False

    def _check_oob(self) -> bool:
        ws = self._world_size
        return not (0 <= self._pos.x <= ws and
                    0 <= self._pos.y <= ws and
                    0 <= self._pos.z <= ws)

    def _dist_to_target(self) -> float:
        return self._euclidean(self._pos, self._target)

    def _target_direction(self) -> Tuple[float, float, float]:
        dx = self._target.x - self._pos.x
        dy = self._target.y - self._pos.y
        dz = self._target.z - self._pos.z
        mag = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-8
        return (dx/mag, dy/mag, dz/mag)

    def _make_observation(self, reward: float) -> DroneObservation:
        nearby = self._nearby_obstacles()
        min_dist = nearby[0].distance if nearby else float("inf")
        wp = self._current_waypoint()
        remaining_path = max(0, len(self._path) - self._waypoint_idx)

        return DroneObservation(
            position=Position(**self._pos.model_dump()),
            velocity=Velocity(**self._vel.model_dump()),
            target_position=Position(**self._target.model_dump()),
            distance_to_target=self._dist_to_target(),
            target_direction=self._target_direction(),
            nearby_obstacles=nearby,
            min_obstacle_distance=min_dist,
            package_delivered=self._delivered,
            collision_occurred=self._collision,
            out_of_bounds=self._oob,
            steps_remaining=self.MAX_STEPS - self._state.step_count,
            next_waypoint=wp,
            path_length=remaining_path,
            done=self._done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "total_reward": self._total_reward,
                "num_obstacles": len(self._obstacles),
                "speed": math.sqrt(
                    self._vel.vx**2 + self._vel.vy**2 + self._vel.vz**2
                ),
            },
        )

    # ──────────────────────────────────────────
    #  Utilities
    # ──────────────────────────────────────────

    @staticmethod
    def _euclidean(a: Position, b: Position) -> float:
        return math.sqrt(
            (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2
        )

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _clamp_velocity(self, vel: Velocity) -> Velocity:
        speed = math.sqrt(vel.vx**2 + vel.vy**2 + vel.vz**2)
        if speed > self.MAX_SPEED:
            factor = self.MAX_SPEED / speed
            return Velocity(
                vx=vel.vx * factor,
                vy=vel.vy * factor,
                vz=vel.vz * factor,
            )
        return vel
