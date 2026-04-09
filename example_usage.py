"""
example_usage.py
================
Demonstrates how to interact with the Drone Delivery environment
both directly (in-process) and via HTTP client.

Tasks:
  task_1_easy   — 2D ground-level direct flight (XY only)
  task_2_medium — 3D full flight phases, no obstacles
  task_3_hard   — 3D dense obstacles
  task_4_expert — 3D obstacles + wind

The drone follows a flight-phase-aware policy:
  GROUND     → thrust upward to lift off
  LIFTING    → continue ascending until cruise altitude, start horizontal
  CRUISING   → follow A* waypoints / target direction, avoid obstacles
  DESCENDING → descend towards the target landing position
  LANDED     → done

Run in-process:
    python example_usage.py

Run against a live server (set SERVER_URL env-var):
    SERVER_URL=http://localhost:8000 python example_usage.py --remote
"""

import argparse
import json
import math
import os
import sys
import uuid

import requests

from environment import DroneDeliveryEnvironment
from models import DroneAction, FlightPhase
from grader import TASK_CONFIGS




# ──────────────────────────────────────────────────────────────────────────────
#  In-process usage
# ──────────────────────────────────────────────────────────────────────────────

def run_local_episode(num_steps: int = 1000, seed: int = 42, task_id: str = "task_2_medium"):
    """Run one episode directly against the Python environment object."""

    print("=" * 70)
    print(f"LOCAL in-process episode  (task={task_id})")
    print("=" * 70)

    env = DroneDeliveryEnvironment(task_id=task_id)
    obs = env.reset()

    print(f"Start        : ({obs.position.x:.1f}, {obs.position.y:.1f}, {obs.position.z:.1f})")
    print(f"Target       : ({obs.target_position.x:.1f}, {obs.target_position.y:.1f}, {obs.target_position.z:.1f})")
    print(f"Distance     : {obs.distance_to_target:.2f} m")
    print(f"Cruise Alt   : {obs.cruise_altitude:.1f} m")
    print(f"Flight Phase : {obs.flight_phase}")
    print(f"Movement     : {obs.metadata.get('movement_mode', '3d').upper()}")
    print(f"A* path      : {obs.path_length} waypoints")
    print(f"Obstacles    : {obs.metadata.get('num_obstacles', 0)}")
    print()

    total_reward = 0.0
    is_2d = obs.metadata.get('movement_mode', '3d') == '2d'

    for step_idx in range(num_steps):
        phase = obs.flight_phase

        if is_2d:
            # ── 2D MODE: steer towards target on XY plane ──
            td = obs.target_direction
            ax = td[0] * 4.0
            ay = td[1] * 4.0
            az = 0.0

        elif phase == FlightPhase.GROUND.value:
            # ── GROUND: thrust upward to lift off ──
            ax, ay, az = 0.0, 0.0, 5.0  # full upward thrust

        elif phase == FlightPhase.LIFTING.value:
            # ── LIFTING: ascend + start moving towards target ──
            # Gentle horizontal acceleration towards target while climbing
            td = obs.target_direction
            az = 4.0  # strong upward
            ax = td[0] * 1.0
            ay = td[1] * 1.0

        elif phase == FlightPhase.CRUISING.value:
            # ── CRUISING: navigate towards target avoiding obstacles ──
            td = obs.target_direction
            scale = 4.0

            # Use A* waypoint if available
            if obs.next_waypoint:
                wp = obs.next_waypoint
                dx = wp.x - obs.position.x
                dy = wp.y - obs.position.y
                dz = wp.z - obs.position.z
                mag = math.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
                td = (dx/mag, dy/mag, dz/mag)

            ax = td[0] * scale
            ay = td[1] * scale
            az = td[2] * scale

            # Obstacle avoidance: repulsive force from nearest obstacle
            if obs.nearby_obstacles:
                nearest = obs.nearby_obstacles[0]
                if nearest.distance < 8.0:
                    rep_x = -nearest.relative_x / (nearest.distance + 1e-6)
                    rep_y = -nearest.relative_y / (nearest.distance + 1e-6)
                    rep_z = -nearest.relative_z / (nearest.distance + 1e-6)
                    strength = max(0, (8.0 - nearest.distance) / 8.0) * 5.0
                    ax += rep_x * strength
                    ay += rep_y * strength
                    az += rep_z * strength

            # Maintain cruise altitude
            alt_error = obs.cruise_altitude - obs.position.z
            az += alt_error * 1.5  # proportional altitude hold

        elif phase == FlightPhase.DESCENDING.value:
            # ── DESCENDING: move towards target and descend ──
            dx = obs.target_position.x - obs.position.x
            dy = obs.target_position.y - obs.position.y
            dz = obs.target_position.z - obs.position.z
            mag = math.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
            ax = (dx / mag) * 2.0
            ay = (dy / mag) * 2.0
            # Controlled descent
            az = -2.0 if obs.position.z > 1.0 else -0.5

            # Brake horizontally when very close
            if obs.horizontal_distance_to_target < 2.0:
                ax = -obs.velocity.vx * 3.0
                ay = -obs.velocity.vy * 3.0
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        action = DroneAction(ax=ax, ay=ay, az=az)
        obs = env.step(action)
        total_reward += obs.reward

        if step_idx % 50 == 0 or obs.done:
            print(
                f"Step {step_idx+1:>4} | "
                f"phase={obs.flight_phase:<11} | "
                f"pos=({obs.position.x:.1f},{obs.position.y:.1f},{obs.position.z:.1f}) | "
                f"dist={obs.distance_to_target:.1f}m | "
                f"hdist={obs.horizontal_distance_to_target:.1f}m | "
                f"alt={obs.position.z:.1f}m | "
                f"near_obs={len(obs.nearby_obstacles)}"
            )

        if obs.done:
            print()
            if obs.package_delivered:
                print(f"✅  Package delivered in {step_idx+1} steps!")
            elif obs.collision_occurred:
                print("💥  Collision!")
            elif obs.out_of_bounds:
                print("🚫  Out of bounds!")
            else:
                print("⏱️  Timeout.")
            break

    print(f"\nTotal reward: {total_reward:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
#  Remote HTTP usage
# ──────────────────────────────────────────────────────────────────────────────

def run_remote_episode(base_url: str, num_steps: int = 1000):
    """Run one episode against a live server using plain requests."""


    session_id = str(uuid.uuid4())[:8]
    print("=" * 70)
    print(f"REMOTE episode  →  {base_url}  (session={session_id})")
    print("=" * 70)

    params = {"session_id": session_id}

    # Reset with a task
    reset_body = {"task_id": "task_2_medium"}
    r = requests.post(f"{base_url}/reset", params=params, json=reset_body, timeout=10)
    obs = r.json()
    print(f"Start        : {obs['position']}")
    print(f"Target       : {obs['target_position']}")
    print(f"Distance     : {obs['distance_to_target']:.2f} m")
    print(f"Cruise Alt   : {obs['cruise_altitude']:.1f} m")
    print(f"Flight Phase : {obs['flight_phase']}")
    print()

    total_reward = 0.0
    for step_idx in range(num_steps):
        phase = obs["flight_phase"]

        if phase == "GROUND":
            ax, ay, az = 0.0, 0.0, 5.0

        elif phase == "LIFTING":
            td = obs["target_direction"]
            ax, ay, az = td[0] * 1.0, td[1] * 1.0, 4.0

        elif phase == "CRUISING":
            td = obs["target_direction"]
            scale = 4.0

            if obs.get("next_waypoint"):
                wp = obs["next_waypoint"]
                pos = obs["position"]
                dx = wp["x"] - pos["x"]
                dy = wp["y"] - pos["y"]
                dz = wp["z"] - pos["z"]
                mag = math.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
                td = [dx/mag, dy/mag, dz/mag]

            ax = td[0] * scale
            ay = td[1] * scale
            az = td[2] * scale

            # Altitude hold
            alt_error = obs["cruise_altitude"] - obs["position"]["z"]
            az += alt_error * 1.5

        elif phase == "DESCENDING":
            pos = obs["position"]
            tgt = obs["target_position"]
            dx = tgt["x"] - pos["x"]
            dy = tgt["y"] - pos["y"]
            mag = math.sqrt(dx**2 + dy**2) + 1e-8
            ax = (dx / mag) * 2.0
            ay = (dy / mag) * 2.0
            az = -2.0 if pos["z"] > 1.0 else -0.5

            if obs.get("horizontal_distance_to_target", 999) < 2.0:
                vel = obs["velocity"]
                ax = -vel["vx"] * 3.0
                ay = -vel["vy"] * 3.0
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        payload = {"ax": ax, "ay": ay, "az": az}
        r = requests.post(f"{base_url}/step", params=params, json=payload, timeout=10)
        obs = r.json()
        total_reward += obs.get("reward", 0.0)

        if step_idx % 50 == 0 or obs.get("done"):
            pos = obs["position"]
            print(
                f"Step {step_idx+1:>4} | "
                f"phase={obs['flight_phase']:<11} | "
                f"pos=({pos['x']:.1f},{pos['y']:.1f},{pos['z']:.1f}) | "
                f"dist={obs['distance_to_target']:.1f}m | "
                f"hdist={obs.get('horizontal_distance_to_target', 0):.1f}m"
            )

        if obs.get("done"):
            print()
            if obs.get("package_delivered"):
                print(f"✅  Delivered in {step_idx+1} steps!")
            elif obs.get("collision_occurred"):
                print("💥  Collision!")
            else:
                print("⏱️  Timeout / OOB.")
            break

    # Final state
    r = requests.get(f"{base_url}/state", params=params, timeout=10)
    print("\nFinal state:", json.dumps(r.json(), indent=2))
    print(f"\nTotal reward: {total_reward:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--url", default=os.getenv("SERVER_URL", "http://localhost:8000"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="task_2_medium",
                        choices=list(TASK_CONFIGS.keys()),
                        help="Task to run")
    args = parser.parse_args()

    if args.remote:
        run_remote_episode(args.url, args.steps)
    else:
        run_local_episode(args.steps, args.seed, task_id=args.task)
