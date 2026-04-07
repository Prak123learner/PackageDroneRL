---
title: Drone Delivery RL Environment
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: docker
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
from drone_env.client import DroneEnv
from drone_env.models import DroneAction

with DroneEnv(base_url="https://<your-space>.hf.space") as client:
    result = client.reset()
    print(result.position, result.distance_to_target)

    for _ in range(100):
        # naive greedy policy: thrust towards target
        obs = result
        action = DroneAction(
            ax=obs.target_direction[0] * 3.0,
            ay=obs.target_direction[1] * 3.0,
            az=obs.target_direction[2] * 3.0,
        )
        result = client.step(action)
        if result.done:
            print("Delivered!" if result.package_delivered else "Failed.")
            break
```

---

## Environment details

| Property | Value |
|---|---|
| World size | 50 × 50 × 50 m |
| Max speed | 10 m/s |
| Max acceleration | 5 m/s² |
| Drone collision radius | 0.4 m |
| Sensor range | 10 m |
| Delivery tolerance | 1.5 m |
| Max steps per episode | 500 |
| Default obstacles | 8 random buildings/towers |

### Observation space

| Field | Type | Description |
|---|---|---|
| `position` | `{x,y,z}` | Current drone position (m) |
| `velocity` | `{vx,vy,vz}` | Current drone velocity (m/s) |
| `target_position` | `{x,y,z}` | Delivery target |
| `distance_to_target` | `float` | Euclidean distance to goal |
| `target_direction` | `[3]` | Unit vector towards target |
| `nearby_obstacles` | `list` | Obstacles within 10 m sensor range |
| `next_waypoint` | `{x,y,z}` | Next A* waypoint (navigation hint) |
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
| Collision | −100 |
| Out of bounds | −50 |
| Progress toward goal | +Δdist × 1.0 |
| On A* path corridor | +0.5 / step |
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
| `GET` | `/sessions` | List active sessions |
| `DELETE` | `/sessions/{id}` | Remove a session |
| `GET` | `/docs` | Interactive Swagger UI |
