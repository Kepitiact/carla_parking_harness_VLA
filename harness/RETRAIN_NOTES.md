# Retrain TODOs — training-data fixes deferred during the harness build

> **RESOLVED 2026-06-25.** Both bugs below were fixed at the source: `build_infos_pkl.py` now
> carries the measured `ego_*` fields, `generate_cached_nuscenes_info.build_entry` reads them
> (real velocity/yaw-rate/steering, no future-derivation), and the model was retrained. The
> harness now feeds the MEASURED ego-state unconditionally — the `next_step`/`COLD_START` /
> `steer=0` workarounds described below have been **deleted** (see live_prompt.py, run_episode.py).
> Kept as a historical record of the bugs. (The retrained model also emits a per-waypoint heading
> output; the controller does not consume it yet — position-only tracking for now.)

These are **training-data / cache-generation bugs** found while building the closed-loop
harness. The current checkpoint depends on them, so the harness works around each one. The
proper fix is to change the cache generator, regenerate the cache + conversations, and
**retrain**. After retraining, remove the corresponding harness workaround (noted below).

All file/line references are in the **model repo** (`~/projects/openvla_nuscenes`).

---

## 1. Ego-state is derived from the FUTURE trajectory (information leak) — HIGH PRIORITY

`scripts/generate_cached_nuscenes_info.py` → `build_entry()` computes the ego-state
velocity AND yaw-rate from the **future** trajectory:

```python
dx_right = fut_traj[1,0] - fut_traj[0,0]      # next-step displacement
dy_fwd   = fut_traj[1,1] - fut_traj[0,1]
vx_true  = dy_fwd  / dt                        # -> gt_ego_lcf_feat[0]
vy_true  = dx_right / dt                       # -> gt_ego_lcf_feat[1]
yaw_rate = arctan2(vy_true, vx_true) / dt      # -> gt_ego_lcf_feat[4]  (the "v_yaw" token)
```

So every training prompt's `Ego states ... v_yaw` carries a hint of where the car is about
to go. **The model learned to depend on it**: feeding the honest *measured* ego velocity
(0 at a dead stop) makes the model emit its degenerate all-zero trajectory. Verified by
swapping ONLY the `v_yaw` token with perception held fixed: `0.04 → correct`, `0.00 → zeros`.

**Proper fix:** derive the ego-state from the **measured/current** motion instead of the
future — i.e. use the recorded `vx_world`/`vy_world` (rotated into the ego frame) and the
IMU yaw-rate already present in `poses.json` / the infos `can_bus`. Then regenerate the
cache + conversations and retrain.

**Harness workaround (remove after retrain):** `harness/model_server/live_prompt.py`
reconstructs the ego-state from a `next_step` displacement (the model's previous prediction's
first waypoint, or the GT path's first step at frame 0) to mimic the future-derivation. After
retraining on measured ego-state, switch `live_prompt` back to measured velocity and drop the
`next_step` plumbing (protocol/server).

## 2. Steering is hardcoded to 0.0 — MEDIUM PRIORITY  (your earlier note)

`scripts/generate_cached_nuscenes_info.py` ~L149–162: `gt_ego_lcf_feat[8]` is hardcoded to
`0.0`. VERIFIED 0.0 across all 49,620 training frames, even though `poses.json` records a
real `steer_normalized` in [-1, 1]. So the model trained on `Steering: (0.00)` always.

**Proper fix:** populate `gt_ego_lcf_feat[8]` with the real steering (from the infos/poses
`can_bus`/`steer_normalized`) instead of `0.0`; regenerate cache + retrain.

**Harness workaround (remove after retrain):** `live_prompt.build_data_dict` plumbs
`ego["steer"]` but the client sends `0.0` to match the current model. After retraining, send
the real measured steering.

---

_Once both are retrained, the harness ego-state should equal the measured CARLA state and
these workarounds can be deleted._
