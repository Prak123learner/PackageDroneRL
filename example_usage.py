"""
example_usage.py
================
Demonstrates how to interact with the Drone Delivery environment
both directly (in-process) and via HTTP client.

Run in-process:
    python example_usage.py

Run against a live server (set SERVER_URL env-var):
    SERVER_URL=http://localhost:8000 python example_usage.py --remote
"""

import argparse
import math
import sys
import os

# ──────────────────────────────────────────────────────────────────────────────
#  In-process usage
# ──────────────────────────────────────────────────────────────────────────────

def run_local_episode(num_steps: int = 200, seed: int = 42):
    """Run one episode directly against the Python environment object."""
    from drone_env.environment import DroneDeliveryEnvironment
    from drone_env.models import DroneAction

    print("=" * 60)
    print("LOCAL in-process episode")
    print("=" * 60)

    env = DroneDeliveryEnvironment(num_obstacles=6, seed=seed)
    obs = env.reset()

    print(f"Start   : {obs.position}")
    print(f"Target  : {obs.target_position}")
    print(f"Distance: {obs.distance_to_target:.2f} m")
    print(f"A* path : {obs.path_length} waypoints")
    print()

    total_reward = 0.0
    for step_idx in range(num_steps):
        # Greedy policy: thrust along target_direction
        td = obs.target_direction
        scale = 3.0

        # Obstacle avoidance: steer away from nearest obstacle
        if obs.nearby_obstacles:
            nearest = obs.nearby_obstacles[0]
            if nearest.distance < 5.0:
                rep_x = -nearest.relative_x / (nearest.distance + 1e-6)
                rep_y = -nearest.relative_y / (nearest.distance + 1e-6)
                rep_z = -nearest.relative_z / (nearest.distance + 1e-6)
                strength = max(0, (5.0 - nearest.distance) / 5.0) * 4.0
                td = (
                    td[0] + rep_x * strength,
                    td[1] + rep_y * strength,
                    td[2] + rep_z * strength,
                )
                # re-normalise
                mag = math.sqrt(td[0]**2 + td[1]**2 + td[2]**2) + 1e-8
                td = (td[0]/mag, td[1]/mag, td[2]/mag)

        # Use A* waypoint if available
        if obs.next_waypoint:
            wp = obs.next_waypoint
            dx = wp.x - obs.position.x
            dy = wp.y - obs.position.y
            dz = wp.z - obs.position.z
            mag = math.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
            td = (dx/mag, dy/mag, dz/mag)

        action = DroneAction(
            ax=td[0] * scale,
            ay=td[1] * scale,
            az=td[2] * scale,
        )
        obs = env.step(action)
        total_reward += obs.reward

        if step_idx % 25 == 0 or obs.done:
            print(
                f"Step {step_idx+1:>3} | "
                f"pos=({obs.position.x:.1f},{obs.position.y:.1f},{obs.position.z:.1f}) | "
                f"dist={obs.distance_to_target:.2f} m | "
                f"reward={obs.reward:.2f} | "
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

def run_remote_episode(base_url: str, num_steps: int = 200):
    """Run one episode against a live server using plain requests."""
    import requests, json, uuid

    session_id = str(uuid.uuid4())[:8]
    print("=" * 60)
    print(f"REMOTE episode  →  {base_url}  (session={session_id})")
    print("=" * 60)

    params = {"session_id": session_id}

    # Reset
    r = requests.post(f"{base_url}/reset", params=params, timeout=10)
    obs = r.json()
    print(f"Start   : {obs['position']}")
    print(f"Target  : {obs['target_position']}")
    print(f"Distance: {obs['distance_to_target']:.2f} m")

    total_reward = 0.0
    for step_idx in range(num_steps):
        td = obs["target_direction"]
        scale = 3.0

        # Waypoint override
        if obs.get("next_waypoint"):
            wp = obs["next_waypoint"]
            pos = obs["position"]
            dx = wp["x"] - pos["x"]
            dy = wp["y"] - pos["y"]
            dz = wp["z"] - pos["z"]
            mag = math.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
            td = [dx/mag, dy/mag, dz/mag]

        payload = {"ax": td[0]*scale, "ay": td[1]*scale, "az": td[2]*scale}
        r = requests.post(f"{base_url}/step", params=params, json=payload, timeout=10)
        obs = r.json()
        total_reward += obs.get("reward", 0.0)

        if step_idx % 25 == 0 or obs.get("done"):
            pos = obs["position"]
            print(
                f"Step {step_idx+1:>3} | "
                f"pos=({pos['x']:.1f},{pos['y']:.1f},{pos['z']:.1f}) | "
                f"dist={obs['distance_to_target']:.2f} m | "
                f"reward={obs.get('reward',0):.2f}"
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
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.remote:
        run_remote_episode(args.url, args.steps)
    else:
        run_local_episode(args.steps, args.seed)
