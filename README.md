---
title: Drone Delivery RL Environment
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🚁 Drone Delivery RL Environment

A **3-D physics-based reinforcement learning environment** where a drone must
navigate from a start location to a delivery target while avoiding obstacles.
Served as a REST API via **FastAPI**, compatible with the **openenv** protocol.

## Quickstart

### Reset (start a new episode)
```bash
curl -X POST https://<your-space>.hf.space/reset
```

### Step (send thrust commands)
```bash
curl -X POST https://<your-space>.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"ax": 1.0, "ay": 0.5, "az": 0.2}'
```

### Get state (no simulation advance)
```bash
curl https://<your-space>.hf.space/state
```

### Run a grading task
```bash
curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "downtown"}'
```

### Grade the episode
```bash
curl https://<your-space>.hf.space/grade
```

### ASCII render
```bash
curl "https://<your-space>.hf.space/render?axis=xy&size=25"
```

---

## Multi-session support

Add `?session_id=<uuid>` to any request to isolate your episode from other users:

```bash
SESSION="my-unique-id-123"
curl -X POST "https://<your-space>.hf.space/reset?session_id=$SESSION"
curl -X POST "https://<your-space>.hf.space/step?session_id=$SESSION" \
  -H "Content-Type: application/json" -d '{"ax":1,"ay":0,"az":0}'
```

---

## Python client (with openenv)

```python
from client import DroneEnv
from models import DroneAction

with DroneEnv(base_url="https://<your-space>.hf.space") as client:
    result = client.reset()
    obs = result.observation
    print(obs.position, obs.distance_to_target)

    for _ in range(100):
        action = DroneAction(
            ax=obs.target_direction[0] * 3.0,
            ay=obs.target_direction[1] * 3.0,
            az=obs.target_direction[2] * 3.0,
        )
        result = client.step(action)
        obs = result.observation
        if result.done:
            print("Delivered!" if obs.package_delivered else "Failed.")
            break
```

---

## Grading Tasks

Five pre-defined tasks with escalating difficulty:

| Task ID | Difficulty | World | Obstacles | Delivery Radius | Max Steps |
|---|---|---|---|---|---|
| `clear_sky` | Easy | 100m | 0 | 3.0m | 500 |
| `suburbs` | Medium | 150m | 6 | 2.5m | 800 |
| `downtown` | Hard | 200m | 15 | 2.0m | 1500 |
| `precision_drop` | Hard | 150m | 5 | 0.8m | 1000 |
| `gauntlet` | Expert | 200m | 25 | 0.6m | 1200 |

Scores are normalized to **[0.0, 1.0]** using weighted components:
delivery (40%), progress (25%), efficiency (15%), safety (10%), smoothness (10%).

---

## Environment details

| Property | Value |
|---|---|
| World size | 200 × 200 × 200 m |
| Max speed | 16.67 m/s (60 km/h) |
| Max acceleration | ±5 m/s² |
| Drone collision radius | 0.4 m |
| Sensor range | 25 m |
| Default delivery radius | 2.0 m |
| Default max steps | 2000 |
| Default obstacles | 15 random buildings/towers |

### Observation space

| Field | Type | Description |
|---|---|---|
| `position` | `{x,y,z}` | Current drone position (m) |
| `velocity` | `{vx,vy,vz}` | Current drone velocity (m/s) |
| `acceleration` | `{vx,vy,vz}` | Applied acceleration (m/s²) |
| `flight_phase` | `string` | GROUND, LIFTING, CRUISING, DESCENDING, or LANDED |
| `cruise_altitude` | `float` | Computed safe cruising altitude (m) |
| `target_position` | `{x,y,z}` | Delivery target |
| `distance_to_target` | `float` | Euclidean distance to goal (m) |
| `horizontal_distance_to_target` | `float` | XY distance to target (m) |
| `target_direction` | `[3]` | Unit vector towards target |
| `nearby_obstacles` | `list` | Obstacles within 25 m sensor range |
| `min_obstacle_distance` | `float` | Distance to nearest obstacle (m) |
| `next_waypoint` | `{x,y,z}` | Next A* waypoint (navigation hint) |
| `path_length` | `int` | Remaining waypoints in A* path |
| `steps_remaining` | `int` | Steps until episode timeout |
| `package_delivered` | `bool` | Whether delivery succeeded |
| `collision_occurred` | `bool` | Whether drone hit an obstacle |
| `out_of_bounds` | `bool` | Whether drone left world bounds |
| `done` | `bool` | Episode ended |
| `reward` | `float` | Per-step reward |

### Action space

```json
{ "ax": float, "ay": float, "az": float }
```
Accelerations in m/s², clamped to ±5 m/s² internally.

### Reward structure

| Event | Reward |
|---|---|
| Package delivered | +100 |
| Smooth landing (speed < 2 m/s) | +5 × (1 - speed/2) |
| Collision | −100 |
| Out of bounds | −50 |
| Progress toward goal | +Δdist × 1.0 |
| Heading alignment | +0.3 × dot(vel, target_dir) |
| On A* path corridor | +0.5 / step |
| Near-miss navigation | +0.3 × factor |
| Altitude error > 3m | −0.2 / step |
| Living penalty | −0.1 / step |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/info` | Environment constants |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Advance simulation |
| `GET` | `/state` | Episode state snapshot |
| `GET` | `/render` | ASCII world projection |
| `GET` | `/obstacles` | Full obstacle list |
| `GET` | `/sessions` | List active sessions |
| `DELETE` | `/sessions/{id}` | Remove a session |
| `GET` | `/tasks` | List available grading tasks |
| `GET` | `/grade` | Grade completed episode (returns 0.0–1.0) |
| `GET` | `/docs` | Interactive Swagger UI |
