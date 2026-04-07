# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Drone Delivery Environment – WebSocket client (openenv-based)."""

from typing import Dict, Optional, Tuple

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        DroneAction, DroneObservation, FlightPhase,
        Position, Velocity, NearbyObstacle,
    )
except ImportError:
    from models import (               # type: ignore[no-redef]
        DroneAction, DroneObservation, FlightPhase,
        Position, Velocity, NearbyObstacle,
    )


def _parse_position(d: Dict) -> Position:
    return Position(x=d.get("x", 0.0), y=d.get("y", 0.0), z=d.get("z", 0.0))


def _parse_velocity(d: Dict) -> Velocity:
    return Velocity(vx=d.get("vx", 0.0), vy=d.get("vy", 0.0), vz=d.get("vz", 0.0))


def _parse_observation(payload: Dict) -> DroneObservation:
    obs_data = payload.get("observation", payload)   # tolerate flat payloads

    nearby_raw = obs_data.get("nearby_obstacles", [])
    nearby = [
        NearbyObstacle(
            id=o.get("id", 0),
            relative_x=o.get("relative_x", 0.0),
            relative_y=o.get("relative_y", 0.0),
            relative_z=o.get("relative_z", 0.0),
            distance=o.get("distance", 0.0),
            size_x=o.get("size_x", 2.0),
            size_y=o.get("size_y", 2.0),
            size_z=o.get("size_z", 10.0),
            obstacle_type=o.get("obstacle_type", "unknown"),
        )
        for o in nearby_raw
    ]

    wp_raw = obs_data.get("next_waypoint")
    next_waypoint: Optional[Position] = _parse_position(wp_raw) if wp_raw else None

    td_raw = obs_data.get("target_direction", [0.0, 0.0, 0.0])
    if isinstance(td_raw, (list, tuple)) and len(td_raw) == 3:
        target_direction: Tuple[float, float, float] = tuple(td_raw)  # type: ignore
    else:
        target_direction = (0.0, 0.0, 0.0)

    return DroneObservation(
        position=_parse_position(obs_data.get("position", {})),
        velocity=_parse_velocity(obs_data.get("velocity", {})),
        acceleration=_parse_velocity(obs_data.get("acceleration", {})),
        flight_phase=obs_data.get("flight_phase", FlightPhase.GROUND.value),
        cruise_altitude=obs_data.get("cruise_altitude", 15.0),
        target_position=_parse_position(obs_data.get("target_position", {})),
        distance_to_target=obs_data.get("distance_to_target", 0.0),
        horizontal_distance_to_target=obs_data.get("horizontal_distance_to_target", 0.0),
        target_direction=target_direction,
        nearby_obstacles=nearby,
        min_obstacle_distance=obs_data.get("min_obstacle_distance", float("inf")),
        package_delivered=obs_data.get("package_delivered", False),
        collision_occurred=obs_data.get("collision_occurred", False),
        out_of_bounds=obs_data.get("out_of_bounds", False),
        steps_remaining=obs_data.get("steps_remaining", 2000),
        next_waypoint=next_waypoint,
        path_length=obs_data.get("path_length", 0),
        done=payload.get("done", obs_data.get("done", False)),
        reward=payload.get("reward", obs_data.get("reward", 0.0)),
        metadata=obs_data.get("metadata", {}),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  openenv-based client
# ──────────────────────────────────────────────────────────────────────────────

class DroneEnv(EnvClient[DroneAction, DroneObservation, State]):
    """
    WebSocket client for the Drone Delivery environment.

    Example::

        with DroneEnv(base_url="http://localhost:8000") as client:
            result = client.reset()
            obs = result.observation
            print(obs.position, obs.distance_to_target)

            result = client.step(DroneAction(ax=1.0, ay=0.5, az=0.0))
            print(result.observation.package_delivered)

    Docker example::

        client = DroneEnv.from_docker_image("drone-delivery-env:latest")
        try:
            result = client.reset()
            result = client.step(DroneAction(ax=0.5, ay=0.5, az=0.2))
        finally:
            client.close()
    """

    def _step_payload(self, action: DroneAction) -> Dict:
        return {"ax": action.ax, "ay": action.ay, "az": action.az}

    def _parse_result(self, payload: Dict) -> "StepResult[DroneObservation]":
        observation = _parse_observation(payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> "State":
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
