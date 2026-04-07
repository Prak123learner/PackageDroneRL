"""
Inference Script – Drone Delivery RL Environment
===================================================
An LLM agent controls a 3D delivery drone through flight phases
(GROUND → LIFTING → CRUISING → DESCENDING → LANDED) by choosing
acceleration commands (ax, ay, az) each step.

MANDATORY ENV VARS
------------------
    API_BASE_URL       LLM API endpoint.
    MODEL_NAME         Model identifier.
    HF_TOKEN           HuggingFace / API key.
    IMAGE_NAME         Docker image for the environment.

STDOUT FORMAT
-------------
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from client import DroneEnv
from models import DroneAction, DroneObservation

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("DRONE_TASK", "package-delivery")
BENCHMARK = os.getenv("DRONE_BENCHMARK", "drone_delivery_rl")
MAX_STEPS = int(os.getenv("DRONE_MAX_STEPS", "500"))
TEMPERATURE = 0.3           # lower = more deterministic flight decisions
MAX_TOKENS = 200

# Scoring: delivery = +100, progress ≈ +240, path bonus ≈ +150, living ≈ -50
# A successful delivery yields roughly +440. We normalize to [0, 1].
MAX_TOTAL_REWARD = 450.0
SUCCESS_SCORE_THRESHOLD = 0.3   # score ≥ 0.3 → success (drone made significant progress)

# ──────────────────────────────────────────────────────────────────────────────
#  System prompt – tells the LLM how to fly the drone
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI pilot controlling a delivery drone in a 3D physics simulation.

OBJECTIVE: Fly from the start position to the delivery target, land, and deliver the package.

FLIGHT PHASES (automatic transitions based on altitude/position):
  GROUND     → You start here. Thrust upward (az > 0) to lift off.
  LIFTING    → Keep climbing until you reach cruise altitude.
  CRUISING   → Fly horizontally toward the target. Follow A* waypoints if available.
  DESCENDING → You are near the target horizontally. Descend toward ground (az < 0).
  LANDED     → Package delivered! Episode ends with +100 reward.

PHYSICS:
  - You control acceleration: ax (east-west), ay (north-south), az (up-down)
  - Each value must be between -5.0 and +5.0 m/s²
  - Max speed: 16.67 m/s (60 km/h)
  - Drag slows you down naturally
  - Timestep: 0.1 seconds per step

REWARDS:
  +100  delivery (land within 2m of target)
  -100  collision with obstacle
  -50   out of bounds
  +Δdist progress toward target
  +0.5  per step on A* path corridor
  -0.1  per step (efficiency penalty)

OUTPUT FORMAT: Reply with ONLY a JSON object, nothing else:
{"ax": <float>, "ay": <float>, "az": <float>}

STRATEGY HINTS:
  - GROUND: set az=5.0 to lift off quickly
  - LIFTING: keep az=3.0-4.0, gently steer toward target with small ax/ay
  - CRUISING: use target_direction or waypoint to set ax/ay (scale 3-4), hold altitude with az
  - DESCENDING: reduce az to -2.0, keep ax/ay aimed at target
  - If obstacle is nearby (< 8m), steer away from it
""")


# ──────────────────────────────────────────────────────────────────────────────
#  Logging helpers (mandatory stdout format)
# ──────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Observation → LLM prompt
# ──────────────────────────────────────────────────────────────────────────────

def format_observation(obs: DroneObservation, step: int, last_reward: float) -> str:
    """Format the drone observation into a concise prompt for the LLM."""
    pos = obs.position
    vel = obs.velocity
    td = obs.target_direction
    tp = obs.target_position

    lines = [
        f"Step: {step} / {obs.steps_remaining + step}",
        f"Flight Phase: {obs.flight_phase}",
        f"Position: x={pos.x:.1f}, y={pos.y:.1f}, z={pos.z:.1f}",
        f"Velocity: vx={vel.vx:.1f}, vy={vel.vy:.1f}, vz={vel.vz:.1f}",
        f"Speed: {obs.metadata.get('speed', 0):.1f} m/s",
        f"Target: x={tp.x:.1f}, y={tp.y:.1f}, z={tp.z:.1f}",
        f"Target Direction (unit vec): [{td[0]:.3f}, {td[1]:.3f}, {td[2]:.3f}]",
        f"Distance to Target: {obs.distance_to_target:.1f} m",
        f"Horizontal Distance: {obs.horizontal_distance_to_target:.1f} m",
        f"Cruise Altitude: {obs.cruise_altitude:.1f} m",
        f"Last Reward: {last_reward:.2f}",
    ]

    # Waypoint hint
    if obs.next_waypoint:
        wp = obs.next_waypoint
        lines.append(f"Next Waypoint: x={wp.x:.1f}, y={wp.y:.1f}, z={wp.z:.1f} ({obs.path_length} remaining)")

    # Nearby obstacles (top 3)
    if obs.nearby_obstacles:
        obs_strs = []
        for o in obs.nearby_obstacles[:3]:
            obs_strs.append(
                f"  - {o.obstacle_type} at distance {o.distance:.1f}m "
                f"(rel: x={o.relative_x:.1f}, y={o.relative_y:.1f}, z={o.relative_z:.1f}, "
                f"size: {o.size_x:.0f}×{o.size_y:.0f}×{o.size_z:.0f})"
            )
        lines.append(f"Nearby Obstacles ({len(obs.nearby_obstacles)} total):")
        lines.extend(obs_strs)
    else:
        lines.append("Nearby Obstacles: none in sensor range")

    lines.append("")
    lines.append("Decide your acceleration. Reply with ONLY: {\"ax\": <float>, \"ay\": <float>, \"az\": <float>}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  LLM → action parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_action(text: str) -> Tuple[float, float, float]:
    """
    Parse LLM response into (ax, ay, az).
    Tries JSON first, then regex fallback for robustness.
    """
    # Try JSON parse
    try:
        # Extract JSON object from response (handles markdown code blocks too)
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            data = json.loads(json_match.group())
            return (
                float(data.get("ax", 0.0)),
                float(data.get("ay", 0.0)),
                float(data.get("az", 0.0)),
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex fallback: look for three numbers
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if len(nums) >= 3:
        return float(nums[0]), float(nums[1]), float(nums[2])

    # Default: gentle upward thrust (safe fallback)
    return (0.0, 0.0, 2.0)


def get_llm_action(
    client: OpenAI,
    obs: DroneObservation,
    step: int,
    last_reward: float,
    history: List[Dict],
) -> Tuple[float, float, float, str]:
    """
    Ask the LLM for acceleration commands given the current observation.
    Returns (ax, ay, az, raw_response_text).
    """
    user_prompt = format_observation(obs, step, last_reward)

    # Build message history (keep last 4 exchanges for context)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-4:]:
        messages.append({"role": "user", "content": h["prompt"]})
        messages.append({"role": "assistant", "content": h["response"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        ax, ay, az = parse_action(text)
        return ax, ay, az, text

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback: use simple heuristic based on phase
        phase = obs.flight_phase
        if phase == "GROUND":
            return 0.0, 0.0, 5.0, '{"ax":0,"ay":0,"az":5}'
        elif phase == "LIFTING":
            td = obs.target_direction
            return td[0] * 1.0, td[1] * 1.0, 4.0, "fallback-lifting"
        elif phase == "CRUISING":
            td = obs.target_direction
            return td[0] * 3.5, td[1] * 3.5, 0.0, "fallback-cruising"
        elif phase == "DESCENDING":
            td = obs.target_direction
            return td[0] * 1.5, td[1] * 1.5, -2.0, "fallback-descending"
        return 0.0, 0.0, 0.0, "fallback-default"


# ──────────────────────────────────────────────────────────────────────────────
#  Main episode loop
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await DroneEnv.from_docker_image(IMAGE_NAME)

    history: List[Dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Ask LLM for acceleration
            ax, ay, az, raw_text = get_llm_action(client, obs, step, last_reward, history)

            # Execute action
            action = DroneAction(ax=ax, ay=ay, az=az)
            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # Format action string for logging
            action_str = f"ax={ax:.1f},ay={ay:.1f},az={az:.1f}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            # Track conversation history
            prompt_text = format_observation(obs, step, last_reward)
            history.append({"prompt": prompt_text, "response": raw_text})

            if done:
                # Log final outcome
                if obs.package_delivered:
                    print(f"[DEBUG] 📦 Package delivered in {step} steps!", flush=True)
                elif obs.collision_occurred:
                    print(f"[DEBUG] 💥 Collision at step {step}", flush=True)
                elif obs.out_of_bounds:
                    print(f"[DEBUG] ⚠ Out of bounds at step {step}", flush=True)
                else:
                    print(f"[DEBUG] ⏱ Timeout at step {step}", flush=True)
                break

        # Compute normalized score in [0, 1]
        total = sum(rewards)
        score = total / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())