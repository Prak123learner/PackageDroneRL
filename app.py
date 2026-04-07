# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Drone Delivery RL Environment
==========================================================

Endpoints
---------
POST /reset                  – start a new episode
POST /step                   – advance the simulation by one step
GET  /state                  – query current episode state
GET  /health                 – liveness probe
GET  /info                   – environment constants / hyper-parameters
GET  /render                 – human-readable ASCII snapshot of the world

HuggingFace Spaces
------------------
This file is the entry point. Set

    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

in your Dockerfile (HF Spaces exposes port 7860 by default).

Multi-session
-------------
Each client session can create its own environment instance by passing
``?session_id=<uuid>`` to any endpoint.  Without a session_id, all
requests share a single default environment.
"""

import math
import os
from typing import Any, Dict, Optional
from uuid import uuid4

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from environment import DroneDeliveryEnvironment
from models import DroneAction, DroneObservation, Position

# ──────────────────────────────────────────────────────────────────────────────
#  App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drone Delivery RL Environment",
    description=(
        "A 3-D physics-based reinforcement-learning environment where a drone "
        "navigates from a start location to a delivery target while avoiding "
        "obstacles.  Features flight phases (GROUND → LIFTING → CRUISING → "
        "DESCENDING → LANDED), AABB box obstacles, and a 200m world.  "
        "Compatible with the openenv protocol."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent / "frontend"

# ──────────────────────────────────────────────────────────────────────────────
#  Session registry
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_SESSION = "default"
_sessions: Dict[str, DroneDeliveryEnvironment] = {}


def _get_env(session_id: str) -> DroneDeliveryEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = DroneDeliveryEnvironment(
            world_size=float(os.getenv("WORLD_SIZE", "200")),
            num_obstacles=int(os.getenv("NUM_OBSTACLES", "15")),
            seed=None,
        )
    return _sessions[session_id]


# ──────────────────────────────────────────────────────────────────────────────
#  Request / response schemas
# ──────────────────────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0


class ResetRequest(BaseModel):
    world_size: Optional[float] = None
    num_obstacles: Optional[int] = None
    seed: Optional[int] = None


def _obs_to_dict(obs: DroneObservation) -> Dict[str, Any]:
    """Serialise observation to a plain dict (handles non-JSON-native types)."""
    d = obs.model_dump()
    # Convert tuple → list for JSON compatibility
    if isinstance(d.get("target_direction"), tuple):
        d["target_direction"] = list(d["target_direction"])
    # Replace inf with a large sentinel
    if d.get("min_obstacle_distance") == float("inf"):
        d["min_obstacle_distance"] = 9999.0
    return d


# ──────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    """Liveness probe – returns 200 when the server is up."""
    return {"status": "ok", "sessions": len(_sessions)}


@app.get("/info", tags=["Meta"])
async def info():
    """Return environment constants and hyper-parameters."""
    env = DroneDeliveryEnvironment   # read class-level constants
    return {
        "world_size": env.WORLD_SIZE,
        "max_speed_ms": env.MAX_SPEED,
        "max_accel_ms2": env.MAX_ACCEL,
        "drag": env.DRAG,
        "timestep_s": env.DT,
        "drone_radius_m": env.DRONE_RADIUS,
        "sensor_range_m": env.SENSOR_RANGE,
        "delivery_radius_m": env.DELIVERY_RADIUS,
        "max_steps": env.MAX_STEPS,
        "voxel_size_m": env.VOXEL_SIZE,
        "flight_phases": ["GROUND", "LIFTING", "CRUISING", "DESCENDING", "LANDED"],
        "cruise_alt_margin_m": env.CRUISE_ALT_MARGIN,
        "min_cruise_alt_m": env.MIN_CRUISE_ALT,
        "horizontal_close_m": env.HORIZONTAL_CLOSE,
        "obstacle_model": "AABB (axis-aligned bounding box, 2x2 footprint)",
        "observation_space": {
            "position": "3-D float (x, y, z)",
            "velocity": "3-D float (vx, vy, vz)",
            "acceleration": "3-D float (vx, vy, vz) — last applied accel",
            "flight_phase": "str: GROUND | LIFTING | CRUISING | DESCENDING | LANDED",
            "cruise_altitude": "float (m) — dynamic, computed from obstacles",
            "target_position": "3-D float",
            "distance_to_target": "float",
            "horizontal_distance_to_target": "float",
            "target_direction": "unit-vector [3]",
            "nearby_obstacles": "list[NearbyObstacle] (sensor range)",
            "min_obstacle_distance": "float",
            "next_waypoint": "Position | null",
            "path_length": "int",
            "done": "bool",
            "reward": "float",
        },
        "action_space": {
            "ax": f"float in [-{env.MAX_ACCEL}, {env.MAX_ACCEL}]",
            "ay": f"float in [-{env.MAX_ACCEL}, {env.MAX_ACCEL}]",
            "az": f"float in [-{env.MAX_ACCEL}, {env.MAX_ACCEL}]",
            "note": "Full manual control — agent directly controls drone acceleration",
        },
        "reward_structure": {
            "delivery": env.DELIVERY_REWARD,
            "collision": env.COLLISION_PENALTY,
            "out_of_bounds": env.OOB_PENALTY,
            "living_penalty_per_step": env.LIVING_PENALTY,
            "progress_scale": env.PROGRESS_SCALE,
            "path_bonus_per_step": env.PATH_BONUS,
        },
    }


@app.post("/reset", tags=["Environment"])
async def reset(
    body: ResetRequest = ResetRequest(),
    session_id: str = Query(default=_DEFAULT_SESSION, description="Session UUID"),
):
    """
    Reset the environment and start a new episode.

    Optionally override ``world_size``, ``num_obstacles``, and ``seed``
    to create a custom scenario.  If a new ``seed`` is provided the
    obstacle layout is deterministic.
    """
    if body.world_size or body.num_obstacles or body.seed is not None:
        # Create / replace session with custom settings
        _sessions[session_id] = DroneDeliveryEnvironment(
            world_size=body.world_size or DroneDeliveryEnvironment.WORLD_SIZE,
            num_obstacles=body.num_obstacles or 15,
            seed=body.seed,
        )

    env = _get_env(session_id)
    obs = env.reset()
    return JSONResponse(_obs_to_dict(obs))


@app.post("/step", tags=["Environment"])
async def step(
    body: StepRequest,
    session_id: str = Query(default=_DEFAULT_SESSION, description="Session UUID"),
):
    """
    Advance the simulation by one time-step.

    Send acceleration commands ``(ax, ay, az)`` in m/s².
    Values are internally clamped to ``[-MAX_ACCEL, MAX_ACCEL]``.
    The agent has full manual control over the drone's acceleration.

    Returns a full ``DroneObservation`` including:
    * current position & velocity & acceleration
    * flight phase
    * distance / direction to target
    * nearby obstacles (within sensor range)
    * next A* waypoint
    * ``done`` flag and per-step ``reward``
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    env = _sessions[session_id]
    action = DroneAction(ax=body.ax, ay=body.ay, az=body.az)
    obs = env.step(action)
    return JSONResponse(_obs_to_dict(obs))


@app.get("/state", tags=["Environment"])
async def get_state(
    session_id: str = Query(default=_DEFAULT_SESSION, description="Session UUID"),
):
    """
    Return the current lightweight episode state.

    Unlike ``/step``, this does **not** advance the simulation.
    Useful for dashboards, logging, and state synchronisation.
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    env = _sessions[session_id]
    s = env.state
    inner = env  # type: DroneDeliveryEnvironment

    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "drone_position": inner._pos.model_dump(),
        "drone_velocity": inner._vel.model_dump(),
        "drone_acceleration": inner._accel.model_dump(),
        "target_position": inner._target.model_dump(),
        "distance_to_target": round(inner._dist_to_target(), 4),
        "horizontal_distance_to_target": round(inner._horizontal_dist_to_target(), 4),
        "speed": round(
            math.sqrt(inner._vel.vx**2 + inner._vel.vy**2 + inner._vel.vz**2), 4
        ),
        "flight_phase": inner.flight_phase.value,
        "cruise_altitude": round(inner.cruise_altitude, 2),
        "num_obstacles": len(inner._obstacles),
        "path_length": len(inner._path),
        "total_reward": round(inner._total_reward, 4),
        "done": inner._done,
        "delivered": inner._delivered,
        "collision": inner._collision,
        "out_of_bounds": inner._oob,
    }


@app.get("/render", tags=["Environment"])
async def render(
    session_id: str = Query(default=_DEFAULT_SESSION),
    axis: str = Query(default="xy", description="Projection plane: 'xy' or 'xz'"),
    size: int = Query(default=40, ge=10, le=80, description="Grid characters"),
):
    """
    ASCII top-down (XY) or side (XZ) projection of the environment.

    Legend:
      D = drone, T = target, # = obstacle, * = A*-path, . = free space
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Session not found.")

    env = _sessions[session_id]
    ws = env._world_size
    grid = [["." for _ in range(size)] for _ in range(size)]

    def to_cell(a, b):
        ci = int(a / ws * (size - 1))
        cj = int(b / ws * (size - 1))
        return max(0, min(size-1, ci)), max(0, min(size-1, cj))

    if axis == "xz":
        get = lambda p: (p.x, p.z)
    else:
        get = lambda p: (p.x, p.y)

    # Obstacles (AABB — mark all cells covered by the box footprint)
    for obs in env._obstacles:
        if axis == "xz":
            # show X × Z extents
            x_min, x_max = obs.position.x - obs.size_x/2, obs.position.x + obs.size_x/2
            z_min, z_max = obs.position.z - obs.size_z/2, obs.position.z + obs.size_z/2
            ci_min, _ = to_cell(x_min, 0)
            ci_max, _ = to_cell(x_max, 0)
            _, cj_min = to_cell(0, z_min)
            _, cj_max = to_cell(0, z_max)
        else:
            # show X × Y extents
            x_min, x_max = obs.position.x - obs.size_x/2, obs.position.x + obs.size_x/2
            y_min, y_max = obs.position.y - obs.size_y/2, obs.position.y + obs.size_y/2
            ci_min, _ = to_cell(x_min, 0)
            ci_max, _ = to_cell(x_max, 0)
            _, cj_min = to_cell(0, y_min)
            _, cj_max = to_cell(0, y_max)

        for ci in range(ci_min, ci_max + 1):
            for cj in range(cj_min, cj_max + 1):
                if 0 <= ci < size and 0 <= cj < size:
                    grid[cj][ci] = "#"

    # Waypoints
    v = env.VOXEL_SIZE
    for (vx, vy, vz) in env._path:
        wp = Position(x=vx*v+v/2, y=vy*v+v/2, z=vz*v+v/2)
        ci, cj = to_cell(*get(wp))
        if grid[cj][ci] == ".":
            grid[cj][ci] = "*"

    # Target
    ci, cj = to_cell(*get(env._target))
    grid[cj][ci] = "T"

    # Drone
    ci, cj = to_cell(*get(env._pos))
    grid[cj][ci] = "D"

    lines = ["".join(row) for row in reversed(grid)]
    legend = "D=drone  T=target  #=obstacle  *=A*-path  .=free"
    return {"render": "\n".join(lines), "legend": legend, "axis": axis}


@app.get("/sessions", tags=["Meta"])
async def list_sessions():
    """List all active session IDs."""
    return {"sessions": list(_sessions.keys()), "count": len(_sessions)}


@app.delete("/sessions/{session_id}", tags=["Meta"])
async def delete_session(session_id: str):
    """Remove a session from memory."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": session_id}
    raise HTTPException(status_code=404, detail="Session not found.")


@app.get("/obstacles", tags=["Environment"])
async def get_obstacles(
    session_id: str = Query(default=_DEFAULT_SESSION),
):
    """Return full obstacle list with absolute positions and AABB dimensions."""
    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Session not found.")
    env = _sessions[session_id]
    return {
        "obstacles": [
            {
                "id": obs.id,
                "position": obs.position.model_dump(),
                "size_x": obs.size_x,
                "size_y": obs.size_y,
                "size_z": obs.size_z,
                "obstacle_type": obs.obstacle_type,
            }
            for obs in env._obstacles
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Frontend serving
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point (local dev)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
