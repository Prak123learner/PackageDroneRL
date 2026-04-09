"""
Inference Script – Drone Delivery RL Environment
===================================================
An LLM agent controls a delivery drone through flight phases
(GROUND → LIFTING → CRUISING → DESCENDING → LANDED) by choosing
acceleration commands (ax, ay, az) each step.

TASKS (4 difficulty tiers)
--------------------------
    1. task_1_easy   – 2D ground-level direct movement, no obstacles
    2. task_2_medium – 3D full flight phases, takeoff → cruise → land
    3. task_3_hard   – 3D dense obstacles, requires pathfinding + avoidance
    4. task_4_expert – 3D obstacles + constant wind (2 m/s² eastward)

STANDALONE: This script only requires `requests` and `openai` — no
openenv-core or other project files needed.

MANDATORY ENV VARS
------------------
    HF_TOKEN           HuggingFace / API key.

OPTIONAL ENV VARS
-----------------
    ENV_URL            URL of the running environment server.
                       Defaults to http://localhost:7860 (HF Spaces port).
    API_BASE_URL       LLM API endpoint (default: HF router).
    MODEL_NAME         Model identifier (default: Qwen/Qwen2.5-72B-Instruct).
    DRONE_TASK_ID      Task to run: task_1_easy | task_2_medium |
                       task_3_hard | task_4_expert  (default: none / LLM chooses).

STDOUT FORMAT
-------------
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import textwrap
import time
from typing import Dict, List, Optional, Tuple

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

ENV_URL = os.getenv("ENV_URL", "https://prototype05-droneenv2.hf.space")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("DRONE_TASK", "package-delivery")
BENCHMARK = os.getenv("DRONE_BENCHMARK", "drone_delivery_rl")
MAX_STEPS = int(os.getenv("DRONE_MAX_STEPS", "500"))
TEMPERATURE = 0.3           # lower = more deterministic flight decisions
MAX_TOKENS = 200
HTTP_TIMEOUT = 30           # seconds per HTTP request to the environment

# ── Task configs (mirrors environment tasks) ───────────────────────────────────
TASKS = ["task_1_easy", "task_2_medium", "task_3_hard", "task_4_expert"]

TASK_DESCRIPTIONS = {
    "task_1_easy": (
        "Easy: 2D ground-level direct flight (XY plane only), no obstacles. "
        "Reach target area and deliver with minimal steps."
    ),
    "task_2_medium": (
        "Medium: 3D full flight phases (takeoff → cruise → descend → land), no obstacles. "
        "Maintain altitude control and land smoothly."
    ),
    "task_3_hard": (
        "Hard: 3D dense AABB obstacles. "
        "Avoid collisions while navigating to target and landing within delivery radius."
    ),
    "task_4_expert": (
        "Expert: 3D dense obstacles + constant eastward wind acceleration. "
        "Compensate for drift while avoiding collisions and landing precisely."
    ),
}

# Scoring: delivery = +100, progress ~ +240, path bonus ~ +150, living ~ -50
# A successful delivery yields roughly +440. We normalize to [0, 1].
MAX_TOTAL_REWARD = 450.0
SUCCESS_SCORE_THRESHOLD = 0.3   # score >= 0.3 -> success (drone made significant progress)

# ──────────────────────────────────────────────────────────────────────────────
#  System prompt – tells the LLM how to fly the drone
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an AI pilot controlling a delivery drone in a physics simulation.

OBJECTIVE: Fly from the start position to the delivery target, land, and deliver the package.

TASKS (escalating difficulty):
  1. EASY   (task_1_easy)   - 2D only: Move directly to the target on the XY ground plane. az=0 always.
  2. MEDIUM (task_2_medium) - 3D: Take off, cruise at altitude, descend and land. No obstacles.
  3. HARD   (task_3_hard)   - 3D: Full flight through dense obstacles. Requires pathfinding.
  4. EXPERT (task_4_expert) - 3D: Obstacles + constant wind (2 m/s^2) pushing you off course.

FLIGHT PHASES (3D mode, automatic transitions based on altitude/position):
  GROUND     - You start here. Thrust upward (az > 0) to lift off.
  LIFTING    - Keep climbing until you reach cruise altitude.
  CRUISING   - Fly horizontally toward the target. Follow A* waypoints if available.
  DESCENDING - You are near the target horizontally. Descend toward ground (az < 0).
  LANDED     - Package delivered! Episode ends with +100 reward.

PHYSICS:
  - You control acceleration: ax (east-west), ay (north-south), az (up-down)
  - Each value must be between -5.0 and +5.0 m/s^2
  - Max speed: 16.67 m/s (60 km/h)
  - Drag slows you down naturally
  - Timestep: 0.1 seconds per step
  - WIND: Some tasks have constant wind that pushes the drone. Compensate by
    steering against the wind direction. Wind info is in the observation.

REWARDS:
  +100  delivery (land within delivery radius of target)
  -100  collision with obstacle
  -50   out of bounds
  +dist progress toward target
  +0.5  per step on A* path corridor
  -0.1  per step (efficiency penalty)

OUTPUT FORMAT: Reply with ONLY a JSON object, nothing else:
{"ax": <float>, "ay": <float>, "az": <float>}

STRATEGY HINTS:
  - EASY (2D): Just steer toward the target with ax/ay, keep az=0
  - GROUND: set az=5.0 to lift off quickly
  - LIFTING: keep az=3.0-4.0, gently steer toward target with small ax/ay
  - CRUISING: use target_direction or waypoint to set ax/ay (scale 3-4), hold altitude with az
  - DESCENDING: reduce az to -2.0, keep ax/ay aimed at target
  - If obstacle is nearby (<8m), steer away from it
  - WIND: If wind is blowing east (+wx), add negative ax to compensate
""")


# ──────────────────────────────────────────────────────────────────────────────
#  Simple HTTP client for the Drone Environment REST API
# ──────────────────────────────────────────────────────────────────────────────

class DroneEnvClient:
    """
    Lightweight HTTP client for the Drone Delivery REST API.

    Connects directly to the FastAPI server endpoints (/health, /reset, /step,
    /grade). No openenv-core dependency required.
    """

    def __init__(self, base_url: str, session_id: str = "inference"):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def health(self) -> Dict:
        """Check if the environment server is alive."""
        resp = self._session.get(
            f"{self.base_url}/health",
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: Optional[str] = None) -> Dict:
        """Reset the environment and start a new episode."""
        body: Dict = {}
        if task_id:
            body["task_id"] = task_id
        resp = self._session.post(
            f"{self.base_url}/reset",
            params={"session_id": self.session_id},
            json=body,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, ax: float, ay: float, az: float) -> Dict:
        """Advance simulation by one timestep with the given accelerations."""
        resp = self._session.post(
            f"{self.base_url}/step",
            params={"session_id": self.session_id},
            json={"ax": ax, "ay": ay, "az": az},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> Dict:
        """Grade the completed episode (call after done=true)."""
        resp = self._session.get(
            f"{self.base_url}/grade",
            params={"session_id": self.session_id},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP session."""
        self._session.close()


# ──────────────────────────────────────────────────────────────────────────────
#  Wait for environment to become ready
# ──────────────────────────────────────────────────────────────────────────────

def wait_for_env(client: DroneEnvClient, max_wait: int = 120) -> bool:
    """
    Poll /health until the environment is ready.
    Returns True if healthy, False if timed out.
    """
    print(f"[DEBUG] Waiting for environment at {client.base_url} ...", flush=True)
    start = time.time()
    delay = 2.0
    while time.time() - start < max_wait:
        try:
            result = client.health()
            print(f"[DEBUG] Environment ready: {result}", flush=True)
            return True
        except Exception:
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)
    print(f"[ERROR] Environment not ready after {max_wait}s", flush=True)
    return False


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
#  Observation -> LLM prompt
# ──────────────────────────────────────────────────────────────────────────────

def format_observation(obs: Dict, step: int, last_reward: float) -> str:
    """Format the drone observation dict into a concise prompt for the LLM."""
    pos = obs.get("position", {})
    vel = obs.get("velocity", {})
    td = obs.get("target_direction", [0.0, 0.0, 0.0])
    tp = obs.get("target_position", {})
    meta = obs.get("metadata", {})
    movement_mode = meta.get("movement_mode", "3d")

    lines = [
        f"Step: {step} / {obs.get('steps_remaining', 0) + step}",
        f"Movement Mode: {movement_mode.upper()}",
        f"Flight Phase: {obs.get('flight_phase', 'GROUND')}",
        f"Position: x={pos.get('x', 0):.1f}, y={pos.get('y', 0):.1f}, z={pos.get('z', 0):.1f}",
        f"Velocity: vx={vel.get('vx', 0):.1f}, vy={vel.get('vy', 0):.1f}, vz={vel.get('vz', 0):.1f}",
        f"Speed: {meta.get('speed', 0):.1f} m/s",
        f"Target: x={tp.get('x', 0):.1f}, y={tp.get('y', 0):.1f}, z={tp.get('z', 0):.1f}",
        f"Target Direction (unit vec): [{td[0]:.3f}, {td[1]:.3f}, {td[2]:.3f}]",
        f"Distance to Target: {obs.get('distance_to_target', 0):.1f} m",
    ]

    if movement_mode == "3d":
        lines.append(f"Horizontal Distance: {obs.get('horizontal_distance_to_target', 0):.1f} m")
        lines.append(f"Cruise Altitude: {obs.get('cruise_altitude', 15):.1f} m")

    lines.append(f"Last Reward: {last_reward:.2f}")

    # Waypoint hint (3D only)
    wp = obs.get("next_waypoint")
    if wp:
        lines.append(
            f"Next Waypoint: x={wp.get('x', 0):.1f}, y={wp.get('y', 0):.1f}, "
            f"z={wp.get('z', 0):.1f} ({obs.get('path_length', 0)} remaining)"
        )

    # Nearby obstacles (top 3)
    nearby = obs.get("nearby_obstacles", [])
    if nearby:
        obs_strs = []
        for o in nearby[:3]:
            obs_strs.append(
                f"  - {o.get('obstacle_type', 'unknown')} at distance {o.get('distance', 0):.1f}m "
                f"(rel: x={o.get('relative_x', 0):.1f}, y={o.get('relative_y', 0):.1f}, "
                f"z={o.get('relative_z', 0):.1f}, "
                f"size: {o.get('size_x', 2):.0f}x{o.get('size_y', 2):.0f}x{o.get('size_z', 10):.0f})"
            )
        lines.append(f"Nearby Obstacles ({len(nearby)} total):")
        lines.extend(obs_strs)
    else:
        lines.append("Nearby Obstacles: none in sensor range")

    lines.append("")

    # Wind info
    wind = meta.get("wind", {})
    wx = wind.get("wx", 0)
    wy = wind.get("wy", 0)
    wz = wind.get("wz", 0)
    if wx != 0 or wy != 0 or wz != 0:
        lines.append(f"Wind: wx={wx:.1f}, wy={wy:.1f}, wz={wz:.1f} m/s^2")

    if movement_mode == "2d":
        lines.append('Decide your acceleration (2D only). Reply with ONLY: {"ax": <float>, "ay": <float>, "az": 0}')
    else:
        lines.append('Decide your acceleration. Reply with ONLY: {"ax": <float>, "ay": <float>, "az": <float>}')
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  LLM -> action parsing
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


def _parse_task_id(text: str) -> str:
    """
    Parse an LLM response into a valid task id.
    Expected JSON: {"task_id": "<one of TASKS>"}.
    """
    try:
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            data = json.loads(json_match.group())
            tid = str(data.get("task_id", "")).strip()
            if tid in TASKS:
                return tid
    except Exception:
        pass

    # Fallback: look for a known task id anywhere in the text
    for tid in TASKS:
        if tid in text:
            return tid
    return TASKS[0]


def choose_task_id(client: OpenAI) -> str:
    """
    Ask the LLM to choose a task id from TASKS.
    This removes dependency on environment variables for task selection.
    """
    desc_lines = "\n".join(f"- {tid}: {TASK_DESCRIPTIONS.get(tid, '')}" for tid in TASKS)
    prompt = textwrap.dedent(f"""\
    Choose ONE task_id for this run from the list below.

    Available tasks:
    {desc_lines}

    Reply with ONLY a JSON object:
    {{"task_id": "<one of: {', '.join(TASKS)} >"}}
    """)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a strict JSON-only selector."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=80,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    return _parse_task_id(text)


def get_llm_action(
    client: OpenAI,
    obs: Dict,
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
        phase = obs.get("flight_phase", "GROUND")
        td = obs.get("target_direction", [0.0, 0.0, 0.0])
        movement_mode = obs.get("metadata", {}).get("movement_mode", "3d")

        if movement_mode == "2d":
            # 2D mode: just steer toward target
            return td[0] * 3.5, td[1] * 3.5, 0.0, "fallback-2d"

        if phase == "GROUND":
            return 0.0, 0.0, 5.0, '{"ax":0,"ay":0,"az":5}'
        elif phase == "LIFTING":
            return td[0] * 1.0, td[1] * 1.0, 4.0, "fallback-lifting"
        elif phase == "CRUISING":
            return td[0] * 3.5, td[1] * 3.5, 0.0, "fallback-cruising"
        elif phase == "DESCENDING":
            return td[0] * 1.5, td[1] * 1.5, -2.0, "fallback-descending"
        return 0.0, 0.0, 0.0, "fallback-default"


# ──────────────────────────────────────────────────────────────────────────────
#  Main episode loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # -- Validate configuration --
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable is required.", flush=True)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DroneEnvClient(base_url=ENV_URL)

    # Wait for environment to be ready (important for cold-start on HF Spaces)
    if not wait_for_env(env):
        print("[ERROR] Environment not reachable. Exiting.", flush=True)
        sys.exit(1)

    def run_episode(task_id: str) -> None:
        history: List[Dict] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

        try:
            if task_id:
                print(f"[DEBUG] Using task_id: {task_id}", flush=True)
            obs = env.reset(task_id=task_id or None)
            last_reward = 0.0
            done = obs.get("done", False)

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                ax, ay, az, raw_text = get_llm_action(llm_client, obs, step, last_reward, history)
                obs = env.step(ax, ay, az)

                reward = obs.get("reward", 0.0)
                done = obs.get("done", False)
                error = None

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                action_str = f"ax={ax:.1f},ay={ay:.1f},az={az:.1f}"
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                prompt_text = format_observation(obs, step, last_reward)
                history.append({"prompt": prompt_text, "response": raw_text})

                if done:
                    if obs.get("package_delivered", False):
                        print(f"[DEBUG] Package delivered in {step} steps!", flush=True)
                    elif obs.get("collision_occurred", False):
                        print(f"[DEBUG] Collision at step {step}", flush=True)
                    elif obs.get("out_of_bounds", False):
                        print(f"[DEBUG] Out of bounds at step {step}", flush=True)
                    else:
                        print(f"[DEBUG] Timeout at step {step}", flush=True)
                    break

            # Local fallback score (kept for robustness if /grade fails)
            total = sum(rewards)
            score = total / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

            if done:
                try:
                    grade_result = env.grade()
                    server_score = grade_result.get("score", score)
                    print(f"[DEBUG] Server grade: {grade_result}", flush=True)
                    score = float(server_score)
                    success = score >= SUCCESS_SCORE_THRESHOLD
                except Exception as e:
                    print(f"[DEBUG] Could not fetch grade: {e}", flush=True)

        except requests.exceptions.ConnectionError as e:
            print(f"[ERROR] Lost connection to environment: {e}", flush=True)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}", flush=True)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    try:
        chosen_task_id = choose_task_id(llm_client)
        run_episode(chosen_task_id)
    finally:
        env.close()


if __name__ == "__main__":
    main()
