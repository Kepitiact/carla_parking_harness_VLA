# CARLA Closed-Loop Harness

Runs the fine-tuned OpenDriveVLA maneuver model live in CARLA:
**perceive → build prompt → model predicts 6 waypoints → controller tracks → step sim → repeat**,
until the car parks or fails. Measures real outcomes (success / collision / final-pose
error / # gear changes / timeout), not offline L2.

Additive: imports and reuses the existing ParkingScenes scenario-spawn, slot, 6-cam
capture, and expert-planner code. The collection loop (`scripts/generate_episodes.py`)
is **not** modified.

## Why two processes (Option C)

`carla` has no Python-3.10 wheel for 0.9.14 (the server version that produced the
training data), so a single-process build would force a server upgrade to 0.9.15 —
changing the renderer the model was trained against. Instead:

| Process | venv | Python | Owns |
|---|---|---|---|
| **CARLA client** | `parking_data_gen/venv` | 3.8 | scenario spawn, 6-cam capture, controllers, viewer, metrics, loop |
| **Model server** | `openvla_nuscenes/.venv` | 3.10 | UniAD + LLM on the GPU |

They talk over a localhost socket (`protocol.py`). At ~2 Hz sync-mode planner cadence the
IPC cost (6 JPEGs + small JSON) is negligible next to UniAD+LLM inference.

## Layout

```
harness/
  config.py            # ALL paths/hosts/tunables in one place (CLI/env overridable)
  protocol.py          # length-framed JSON+blob wire format; request/response builders
  model_server/        # Py3.10 — run with openvla_nuscenes/.venv
    server.py          #   socket server
    live_uniad.py      #   build UniAD input from 6 live imgs (no NuScenes DB); carry BEV state
    live_prompt.py     #   reuse generate_user_message to assemble the exact prompt
  client/              # Py3.8 — run with parking_data_gen/venv
    controller_adapter.py  # 6 waypoints -> existing MPC -> carla.VehicleControl (+gear)
    slot_pick_ui.py        # top-down slot bboxes + click; + auto-pick/headless
    live_viewer.py         # 6 cams + verbatim prompt + pred/executed/GT overlay + pose error
    metrics.py             # success/collision/final-pose/gear-changes/ticks
  run_episode.py       # orchestrator
  tests/               # byte-match prompt test, coordinate sanity check
  runs/                # per-episode replays + metrics (gitignored)
```

Each piece is independently runnable (see each module's `__main__`). The loop only
orchestrates them.

## Coordinate frames

CARLA is left-handed (`y_odr = -y_carla`). All bridge poses are in the nuScenes/OpenDRIVE
global frame; the server converts the slot to ego-local (x=right, y=forward) every step,
matching the model's waypoint output and the controller. See `../ARCHITECTURE.md`.
