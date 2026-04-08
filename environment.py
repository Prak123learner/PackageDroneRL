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

Flight Phases
-------------
  GROUND      → LIFTING → CRUISING → DESCENDING → LANDED
  The drone starts on the ground, lifts to a dynamic cruise altitude
  (computed from the tallest obstacle + safety margin), cruises towards
  the target, then descends and lands.

Physics model
-------------
  - Newtonian point-mass with drag.
  - Euler integration at configurable dt (default 0.1 s).
  - Acceleration commands are clamped to [-max_accel, +max_accel].
  - Speed is clamped to max_speed after integration.
  - Agent has full control over (ax, ay, az) thrust commands.

Obstacle Model
--------------
  Obstacles are axis-aligned bounding boxes (AABB) with a 2×2 m footprint
  and random heights.  Collision is detected as AABB-vs-sphere.

Reward shaping
--------------
  +100   package delivered (LANDED)
  -100   collision
   -50   out-of-bounds
   +r    progress reward: Δ(distance_to_target) * progress_reward_scale
   -0.1  per-step living penalty (encourages efficiency)
   +0.5  if drone is on the A* path corridor (optional guidance bonus)

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
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .models import (
        DroneAction, DroneObservation, FlightPhase,
        Position, Velocity, Obstacle, NearbyObstacle, ObstacleConfig,
    )
    from .grader import EpisodeResult, TaskDefinition, TASKS, grade_task
except ImportError:
    from models import (               # type: ignore[no-redef]
        DroneAction, DroneObservation, FlightPhase,
        Position, Velocity, Obstacle, NearbyObstacle, ObstacleConfig,
    )
    from grader import EpisodeResult, TaskDefinition, TASKS, grade_task  # type: ignore[no-redef]


# ──────────────────────────────────────────────────────────────────────────────
#  AABB helpers
# ──────────────────────────────────────────────────────────────────────────────

def _aabb_point_distance(obs: Obstacle, px: float, py: float, pz: float) -> float:
    """
    Distance from point (px, py, pz) to the surface of an AABB obstacle.
    Returns 0 if the point is inside the box.
    """
    # Half-extents
    hx = obs.size_x / 2.0
    hy = obs.size_y / 2.0
    hz = obs.size_z / 2.0
    # Clamp point to box extents
    cx = max(obs.position.x - hx, min(px, obs.position.x + hx))
    cy = max(obs.position.y - hy, min(py, obs.position.y + hy))
    cz = max(obs.position.z - hz, min(pz, obs.position.z + hz))
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _aabb_contains_point(obs: Obstacle, px: float, py: float, pz: float) -> bool:
    """True if point is inside the AABB."""
    hx = obs.size_x / 2.0
    hy = obs.size_y / 2.0
    hz = obs.size_z / 2.0
    return (abs(px - obs.position.x) <= hx and
            abs(py - obs.position.y) <= hy and
            abs(pz - obs.position.z) <= hz)


def _aabb_overlaps_voxel(
    obs: Obstacle,
    voxel_x: float, voxel_y: float, voxel_z: float,
    voxel_size: float, drone_radius: float,
) -> bool:
    """
    True if the obstacle AABB (inflated by drone_radius) overlaps
    the given voxel centre.
    """
    hx = obs.size_x / 2.0 + drone_radius
    hy = obs.size_y / 2.0 + drone_radius
    hz = obs.size_z / 2.0 + drone_radius
    return (abs(voxel_x - obs.position.x) < hx + voxel_size / 2.0 and
            abs(voxel_y - obs.position.y) < hy + voxel_size / 2.0 and
            abs(voxel_z - obs.position.z) < hz + voxel_size / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Tiny pure-Python A* on a voxelised grid (updated for AABB obstacles)
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
    A* pathfinding on a 3-D integer grid with AABB obstacles.

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
            if _aabb_overlaps_voxel(obs, wx, wy, wz, voxel, drone_radius):
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
    Package-delivery drone RL environment with flight phases.

    World
    -----
    A cubic space of side ``world_size`` metres (default 200 m).
    The drone starts near (10, 10, 0) with ± jitter and must reach
    a target near (180, 180, 0).  Random box obstacles (buildings / towers)
    are scattered inside.

    Flight Phases
    -------------
    GROUND → LIFTING → CRUISING → DESCENDING → LANDED

    The drone starts on the ground and must lift off, fly at cruise altitude
    (dynamically computed from tallest obstacle + margin), navigate to
    the target, then descend and land.

    Agent has full manual control over (ax, ay, az) throughout.

    State exposed via ``/state``
    ----------------------------
    episode_id, step_count, drone_position, drone_velocity,
    target_position, flight_phase, cruise_altitude, num_obstacles,
    current_reward_total, done.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Physics ──────────────────────────────
    MAX_SPEED:   float = 16.67   # m/s  (≈ 60 km/h)
    MAX_ACCEL:   float = 5.0     # m/s²
    DRAG:        float = 0.05    # proportional damping coefficient
    DT:          float = 0.1     # simulation timestep (s)

    # ── World ────────────────────────────────
    WORLD_SIZE:  float = 200.0   # metres per axis
    MIN_ALT:     float = 0.0     # ground level (m)

    # ── Drone geometry ───────────────────────
    DRONE_RADIUS:    float = 0.4   # metres
    SENSOR_RANGE:    float = 25.0  # obstacle-detection radius (m)

    # ── Delivery ─────────────────────────────
    DELIVERY_RADIUS: float = 2.0   # metres – goal tolerance
    MAX_STEPS:       int   = 2000

    # ── A* grid ──────────────────────────────
    VOXEL_SIZE: float = 4.0    # metres per voxel
    GRID_SIZE:  int   = 50     # voxels per axis  (200 / 4 = 50)

    # ── Flight phase thresholds ──────────────
    CRUISE_ALT_MARGIN: float = 5.0     # metres above tallest obstacle
    MIN_CRUISE_ALT:    float = 10.0    # minimum cruise altitude
    LIFT_THRESHOLD:    float = 1.0     # z above this → LIFTING phase
    HORIZONTAL_CLOSE:  float = 5.0     # horizontal distance to trigger DESCENDING
    GROUND_DRAG:       float = 0.8     # horizontal damping while on ground

    # ── Reward shaping ───────────────────────
    DELIVERY_REWARD:      float =  100.0
    COLLISION_PENALTY:    float = -10000.0
    OOB_PENALTY:          float =  -50.0
    PROGRESS_SCALE:       float =   1.0
    HEADING_BONUS:        float =   1.0   # bonus for velocity aligned toward target
    ALT_PENALTY:          float =  -0.2   # penalty for wrong altitude per phase
    NEAR_MISS_BONUS:      float =   0.2   # bonus for flying near obstacles without colliding
    LANDING_BONUS:        float =   5.0   # bonus for smooth, slow landing
    LIVING_PENALTY:       float =  -0.1
    PATH_BONUS:           float =   0.5   # bonus per step on A* path corridor

    def __init__(
        self,
        world_size: float = 200.0,
        num_obstacles: int = 15,
        seed: Optional[int] = None,
    ):
        self._world_size = world_size
        self._num_obstacles = num_obstacles
        self._rng = random.Random(seed)

        # will be populated on reset()
        self._pos = Position(x=0, y=0, z=0)
        self._vel = Velocity()
        self._accel = Velocity()  # last applied acceleration
        self._target = Position(x=180, y=180, z=0)
        self._obstacles: List[Obstacle] = []
        self._path: List[Tuple[int, int, int]] = []
        self._waypoint_idx: int = 0
        self._done = False
        self._collision = False
        self._oob = False
        self._delivered = False
        self._prev_dist = 0.0
        self._initial_dist = 0.0
        self._total_reward = 0.0
        self._delivery_radius = self.DELIVERY_RADIUS  # can be overridden per task
        self._max_steps = self.MAX_STEPS  # instance-level budget (overridden by tasks)
        self._flight_phase = FlightPhase.GROUND
        self._cruise_altitude = self.MIN_CRUISE_ALT
        self._wind = Velocity()  # constant wind acceleration (set by task)
        self._task_id: Optional[str] = None
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ──────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────

    def reset(
        self,
        *,
        start_pos: Optional[Position] = None,
        target_pos: Optional[Position] = None,
        custom_obstacles: Optional[List[ObstacleConfig]] = None,
        task_id: Optional[str] = None,
    ) -> DroneObservation:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        start_pos : Position, optional
            Custom start position.  Defaults to random ~(10, 10, 0).
        target_pos : Position, optional
            Custom delivery target. Defaults to random ~(180, 180, 0).
        custom_obstacles : list[ObstacleConfig], optional
            User-defined obstacles.  When provided, random generation is
            skipped and these obstacles are used instead.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._collision = False
        self._oob = False
        self._delivered = False
        self._task_id = task_id
        self._total_reward = 0.0
        self._flight_phase = FlightPhase.GROUND

        ws = self._world_size
        jitter = lambda base, spread: base + self._rng.uniform(-spread, spread)

        # ── Start position ──
        if start_pos is not None:
            self._pos = Position(x=start_pos.x, y=start_pos.y, z=0.0)
        else:
            self._pos = Position(
                x=jitter(10.0, 3.0),
                y=jitter(10.0, 3.0),
                z=0.0,
            )
        self._vel = Velocity()
        self._accel = Velocity()

        # ── Target position ──
        if target_pos is not None:
            self._target = Position(x=target_pos.x, y=target_pos.y, z=0.0)
        else:
            self._target = Position(
                x=jitter(ws - 20.0, 5.0),
                y=jitter(ws - 20.0, 5.0),
                z=0.0,
            )

        # ── Obstacles ──
        if custom_obstacles is not None:
            self._obstacles = self._build_custom_obstacles(custom_obstacles)
        else:
            self._obstacles = self._generate_obstacles()

        self._cruise_altitude = self._compute_cruise_altitude()
        self._prev_dist = self._dist_to_target()
        self._initial_dist = self._prev_dist   # save for grader normalisation
        self._plan_path()

        return self._make_observation(reward=0.0)

    def step(self, action: DroneAction) -> DroneObservation:  # type: ignore[override]
        """Apply the agent's thrust commands, integrate physics, update flight phase, compute reward."""
        if self._done:
            return self._make_observation(reward=0.0)

        self._state.step_count += 1

        # ── clamp agent acceleration ─────────
        ax = self._clamp(action.ax, -self.MAX_ACCEL, self.MAX_ACCEL)
        ay = self._clamp(action.ay, -self.MAX_ACCEL, self.MAX_ACCEL)
        az = self._clamp(action.az, -self.MAX_ACCEL, self.MAX_ACCEL)

        # ── flight-phase specific physics ────
        if self._flight_phase == FlightPhase.GROUND:
            # On the ground: dampen horizontal movement, only z matters
            ax *= (1.0 - self.GROUND_DRAG)
            ay *= (1.0 - self.GROUND_DRAG)
            # Prevent going below ground
            if self._pos.z <= 0.0 and az < 0:
                az = 0.0

        elif self._flight_phase == FlightPhase.LIFTING:
            # Full control, but prevent going below ground
            if self._pos.z <= 0.0 and az < 0:
                az = 0.0

        elif self._flight_phase == FlightPhase.CRUISING:
            # Full agent control — no restrictions
            pass

        elif self._flight_phase == FlightPhase.DESCENDING:
            # Full agent control during descent
            pass

        # Store applied acceleration for observation
        self._accel = Velocity(vx=ax, vy=ay, vz=az)

        # ── apply wind acceleration ──────────
        ax += self._wind.vx
        ay += self._wind.vy
        az += self._wind.vz

        # ── integrate velocity ───────────────
        self._vel.vx = (self._vel.vx + ax * self.DT) * (1 - self.DRAG)
        self._vel.vy = (self._vel.vy + ay * self.DT) * (1 - self.DRAG)
        self._vel.vz = (self._vel.vz + az * self.DT) * (1 - self.DRAG)
        self._vel = self._clamp_velocity(self._vel)

        # ── integrate position ───────────────
        self._pos.x += self._vel.vx * self.DT
        self._pos.y += self._vel.vy * self.DT
        self._pos.z += self._vel.vz * self.DT

        # Clamp to ground (can't go underground)
        if self._pos.z < 0.0:
            self._pos.z = 0.0
            self._vel.vz = 0.0

        # ── update flight phase ──────────────
        self._update_flight_phase()

        # ── termination checks ───────────────
        reward = self.LIVING_PENALTY
        self._oob = self._check_oob()
        self._collision = self._check_collision()
        self._delivered = (
            self._flight_phase == FlightPhase.LANDED
            or (self._dist_to_target() <= self._delivery_radius and self._pos.z <= 1.0)
        )

        if self._oob:
            reward += self.OOB_PENALTY
            self._done = True
        elif self._collision:
            reward += self.COLLISION_PENALTY
            self._done = True
        elif self._delivered:
            self._flight_phase = FlightPhase.LANDED
            reward += self.DELIVERY_REWARD
            # Smooth landing bonus: reward low speed at delivery
            speed = math.sqrt(self._vel.vx**2 + self._vel.vy**2 + self._vel.vz**2)
            if speed < 2.0:
                reward += self.LANDING_BONUS * (1.0 - speed / 2.0)
            self._done = True
        else:
            # ── progress shaping (Euclidean distance to target) ──
            curr_dist = self._dist_to_target()
            progress = self._prev_dist - curr_dist
            reward += progress * self.PROGRESS_SCALE
            self._prev_dist = curr_dist

            # ── heading alignment bonus ──
            # Reward when velocity vector points toward the target
            speed = math.sqrt(self._vel.vx**2 + self._vel.vy**2 + self._vel.vz**2)
            if speed > 0.5:
                td = self._target_direction()
                dot = (
                    self._vel.vx * td[0] +
                    self._vel.vy * td[1] +
                    self._vel.vz * td[2]
                ) / speed
                # dot in [-1, 1]; only reward positive alignment
                if dot > 0:
                    reward += self.HEADING_BONUS * dot

            # ── altitude management ──
            if self._flight_phase == FlightPhase.CRUISING:
                alt_error = abs(self._pos.z - self._cruise_altitude)
                if alt_error > 3.0:  # more than 3m off cruise altitude
                    reward += self.ALT_PENALTY  # encourages altitude hold
            elif self._flight_phase == FlightPhase.GROUND:
                if self._pos.z < 0.5 and az > 0:  # trying to lift -> small encouragement
                    reward += 0.05

            # ── path-following bonus ──
            if self._on_path_corridor():
                reward += self.PATH_BONUS

            # ── near-miss navigation bonus ──
            if self._flight_phase in (FlightPhase.CRUISING, FlightPhase.DESCENDING):
                for obs in self._obstacles:
                    dist = _aabb_point_distance(obs, self._pos.x, self._pos.y, self._pos.z)
                    if self.DRONE_RADIUS < dist < self.DRONE_RADIUS + 3.0:
                        reward += self.NEAR_MISS_BONUS * (1.0 - dist / (self.DRONE_RADIUS + 3.0))
                        break

        # max-step timeout
        if self._state.step_count >= self._max_steps:
            self._done = True

        self._total_reward += reward
        return self._make_observation(reward=reward)

    def grade(self) -> Dict:
        """
        Grade the completed episode.

        Returns a dict with ``score`` in [0.0, 1.0] and component scores.
        Must be called after the episode is done (``self._done == True``).
        """
        speed = math.sqrt(
            self._vel.vx ** 2 + self._vel.vy ** 2 + self._vel.vz ** 2
        )
        result = EpisodeResult(
            delivered=self._delivered,
            collision=self._collision,
            out_of_bounds=self._oob,
            timed_out=(self._state.step_count >= self._max_steps and not self._delivered),
            final_dist=self._dist_to_target(),
            initial_dist=self._initial_dist,
            steps_used=self._state.step_count,
            max_steps=self._max_steps,
            total_reward=self._total_reward,
            landing_speed=speed if self._delivered else 0.0,
            delivery_radius=self._delivery_radius,
        )
        return grade_task(self._task_id or "", result)

    def reset_from_task(self, task: TaskDefinition) -> DroneObservation:
        """
        Reset the environment using a TaskDefinition.

        Overrides world_size, delivery_radius, max_steps, positions,
        obstacles, and wind from the task spec — ensures reproducibility.
        """
        self._world_size = task.world_size
        self._delivery_radius = task.delivery_radius
        self._max_steps = task.max_steps
        self._rng = random.Random(task.seed)

        # Set wind from task definition
        if hasattr(task, 'wind') and task.wind:
            self._wind = Velocity(vx=task.wind[0], vy=task.wind[1], vz=task.wind[2])
        else:
            self._wind = Velocity()

        return self.reset(
            start_pos=Position(x=task.start[0], y=task.start[1], z=0),
            target_pos=Position(x=task.target[0], y=task.target[1], z=0),
            custom_obstacles=task.custom_obstacles if task.custom_obstacles else [],
            task_id=task.task_id,
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def flight_phase(self) -> FlightPhase:
        return self._flight_phase

    @property
    def cruise_altitude(self) -> float:
        return self._cruise_altitude

    # ──────────────────────────────────────────
    #  Flight phase management
    # ──────────────────────────────────────────

    def _update_flight_phase(self):
        """Transition between flight phases based on current state."""
        phase = self._flight_phase

        if phase == FlightPhase.GROUND:
            if self._pos.z > self.LIFT_THRESHOLD:
                self._flight_phase = FlightPhase.LIFTING

        elif phase == FlightPhase.LIFTING:
            if self._pos.z >= self._cruise_altitude:
                self._flight_phase = FlightPhase.CRUISING
            elif self._pos.z <= 0.0:
                # Fell back to ground
                self._flight_phase = FlightPhase.GROUND

        elif phase == FlightPhase.CRUISING:
            hdist = self._horizontal_dist_to_target()
            if hdist <= self.HORIZONTAL_CLOSE:
                self._flight_phase = FlightPhase.DESCENDING

        elif phase == FlightPhase.DESCENDING:
            if self._dist_to_target() <= self._delivery_radius and self._pos.z <= 1.0:
                self._flight_phase = FlightPhase.LANDED
            elif self._horizontal_dist_to_target() > self.HORIZONTAL_CLOSE * 2:
                # Drifted too far away, go back to cruising
                self._flight_phase = FlightPhase.CRUISING

    def _compute_cruise_altitude(self) -> float:
        """Compute cruise altitude from tallest obstacle + safety margin."""
        if not self._obstacles:
            return self.MIN_CRUISE_ALT

        tallest = 0.0
        for obs in self._obstacles:
            top = obs.position.z + obs.size_z / 2.0
            if top > tallest:
                tallest = top

        cruise = tallest + self.CRUISE_ALT_MARGIN
        return max(cruise, self.MIN_CRUISE_ALT)

    # ──────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────

    def _generate_obstacles(self) -> List[Obstacle]:
        """Generate random AABB obstacles with 2×2 m footprint."""
        ws = self._world_size
        margin = 20.0  # keep clear of world edges
        obstacles = []
        for i in range(self._num_obstacles):
            # keep clear of start and goal
            for _ in range(100):
                x = self._rng.uniform(margin, ws - margin)
                y = self._rng.uniform(margin, ws - margin)
                height = self._rng.uniform(5.0, 20.0)
                z = height / 2.0  # centre of box sits at half-height
                pos = Position(x=x, y=y, z=z)
                # Ensure obstacle isn't too close to start or target
                if (self._euclidean(pos, self._pos) > 10.0 and
                        self._euclidean(pos, self._target) > 10.0):
                    obstacles.append(Obstacle(
                        id=i,
                        position=pos,
                        size_x=2.0,
                        size_y=2.0,
                        size_z=height,
                        obstacle_type=self._rng.choice(
                            ["building", "tower", "tree", "antenna"]
                        ),
                    ))
                    break
        return obstacles

    def _build_custom_obstacles(self, configs: List[ObstacleConfig]) -> List[Obstacle]:
        """Convert user-supplied ObstacleConfig list into Obstacle objects."""
        obstacles = []
        for i, cfg in enumerate(configs):
            z = cfg.z if cfg.z is not None else cfg.height / 2.0
            obstacles.append(Obstacle(
                id=i,
                position=Position(x=cfg.x, y=cfg.y, z=z),
                size_x=cfg.size_x,
                size_y=cfg.size_y,
                size_z=cfg.height,
                obstacle_type=cfg.obstacle_type,
            ))
        return obstacles

    def _plan_path(self):
        """
        Run A* and cache voxel waypoints.

        Plans at the median obstacle height so the path must route *around*
        tall obstacles rather than trivially flying above everything.
        If the low-altitude plan fails, falls back to cruise altitude.
        """
        def to_voxel(p: Position) -> Tuple[int, int, int]:
            v = self.VOXEL_SIZE
            return (
                max(0, min(self.GRID_SIZE-1, int(p.x / v))),
                max(0, min(self.GRID_SIZE-1, int(p.y / v))),
                max(0, min(self.GRID_SIZE-1, int(p.z / v))),
            )

        # Compute a planning altitude at the median obstacle height
        # so that taller obstacles force the path around them
        if self._obstacles:
            heights = sorted(
                obs.position.z + obs.size_z / 2.0 for obs in self._obstacles
            )
            planning_alt = heights[len(heights) // 2]   # median top
        else:
            planning_alt = self.MIN_CRUISE_ALT

        # Ensure minimum safe planning altitude
        planning_alt = max(planning_alt, self.VOXEL_SIZE)

        plan_start = Position(
            x=self._pos.x, y=self._pos.y, z=planning_alt
        )
        plan_target = Position(
            x=self._target.x, y=self._target.y, z=planning_alt
        )

        start_v = to_voxel(plan_start)
        goal_v  = to_voxel(plan_target)

        path = _astar(
            start_v, goal_v,
            self._obstacles,
            self.GRID_SIZE,
            self.VOXEL_SIZE,
            self.DRONE_RADIUS,
        )

        # Fallback: if no path at low altitude, plan at cruise altitude
        if not path:
            cruise_start = Position(
                x=self._pos.x, y=self._pos.y, z=self._cruise_altitude
            )
            cruise_target = Position(
                x=self._target.x, y=self._target.y, z=self._cruise_altitude
            )
            path = _astar(
                to_voxel(cruise_start), to_voxel(cruise_target),
                self._obstacles,
                self.GRID_SIZE,
                self.VOXEL_SIZE,
                self.DRONE_RADIUS,
            )

        self._path = path
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
            dist = _aabb_point_distance(obs, self._pos.x, self._pos.y, self._pos.z)
            if dist <= self.SENSOR_RANGE:
                nearby.append(NearbyObstacle(
                    id=obs.id,
                    relative_x=dx,
                    relative_y=dy,
                    relative_z=dz,
                    distance=dist,
                    size_x=obs.size_x,
                    size_y=obs.size_y,
                    size_z=obs.size_z,
                    obstacle_type=obs.obstacle_type,
                ))
        nearby.sort(key=lambda o: o.distance)
        return nearby

    def _check_collision(self) -> bool:
        """Check AABB collision against all obstacles."""
        for obs in self._obstacles:
            dist = _aabb_point_distance(
                obs, self._pos.x, self._pos.y, self._pos.z
            )
            if dist < self.DRONE_RADIUS:
                return True
        # Ground collision only matters if drone is somehow below ground
        # (handled by clamping, so this rarely triggers)
        if self._pos.z < -0.1:
            return True
        return False

    def _check_oob(self) -> bool:
        ws = self._world_size
        return not (0 <= self._pos.x <= ws and
                    0 <= self._pos.y <= ws and
                    0 <= self._pos.z <= ws)

    def _dist_to_target(self) -> float:
        return self._euclidean(self._pos, self._target)

    def _horizontal_dist_to_target(self) -> float:
        """XY-plane distance to target (ignores altitude)."""
        dx = self._target.x - self._pos.x
        dy = self._target.y - self._pos.y
        return math.sqrt(dx * dx + dy * dy)

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
            acceleration=Velocity(**self._accel.model_dump()),
            flight_phase=self._flight_phase.value,
            cruise_altitude=self._cruise_altitude,
            target_position=Position(**self._target.model_dump()),
            distance_to_target=self._dist_to_target(),
            horizontal_distance_to_target=self._horizontal_dist_to_target(),
            target_direction=self._target_direction(),
            nearby_obstacles=nearby,
            min_obstacle_distance=min_dist,
            package_delivered=self._delivered,
            collision_occurred=self._collision,
            out_of_bounds=self._oob,
            steps_remaining=self._max_steps - self._state.step_count,
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
                "flight_phase": self._flight_phase.value,
                "cruise_altitude": self._cruise_altitude,
                "wind": {
                    "wx": self._wind.vx,
                    "wy": self._wind.vy,
                    "wz": self._wind.vz,
                },
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
