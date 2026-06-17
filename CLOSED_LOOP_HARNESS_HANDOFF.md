# CARLA Closed-Loop Harness — Build Handoff

> Self-contained brief for a fresh coding session. Goal: run the fine-tuned
> OpenDriveVLA parking model **live in CARLA** (perceive → predict → control →
> step → repeat) and measure whether the car actually parks.

---

## 0. TL;DR
Build a loop where, every step, CARLA gives 6 camera images + ego state, the
**model** predicts a 6-waypoint trajectory, an existing **controller** turns that
into steering/throttle, the sim advances, repeat. Measure success / collision /
final-pose error. Reuse as much of `ParkingScenes` as possible.

---

## 1. Context — what already exists

**The model (in `~/projects/openvla_nuscenes`, Python 3.10 `.venv`):**
- OpenDriveVLA-0.5B (LLaVA = Qwen2-0.5B LLM + UniAD BEV vision tower), fine-tuned
  with LoRA (rank 64, attention + mm_projectors) on 49,620 CARLA parking frames.
- Merged, ready-to-run checkpoint:
  `~/projects/openvla_nuscenes/checkpoints/OpenDriveVLA-0.5B-carla/merged`
- It **works**: produces correct reverse-parking trajectories (validated visually).
  Residual error is at the far horizon; precision is being worked separately.
- **Model interface:**
  - Inputs: 6 cameras (`CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK,
    CAM_BACK_LEFT, CAM_BACK_RIGHT`), ego state (velocity, last-2s history),
    mission goal (`reverse`/`keep forward`/`turn left|right`), and camera
    intrinsics+extrinsics.
  - Output: **6 waypoints, ego-local frame, x=right, y=forward, 0.5 s spacing
    (3 s horizon)**, emitted as a text string `[(x1,y1),...,(x6,y6)]`.
  - Offline inference entry: `OpenDriveVLA/drivevla/inference_drivevla.py`
    (reads from a NuScenes DB — see §5, the live path is the hard part).

**The CARLA side (this repo, `~/projects/parking_data_gen`, Python 3.8.20):**
- CARLA **0.9.14** client (3.7/3.8 wheels only).
- `ParkingScenes/` has the working sim integration used for data collection:
  client, vehicle spawn, 6-camera capture, scenario generation, expert planner.

**An offline full inference run is currently going on the shared 8 GB GPU (~20 h).**
Model-in-the-loop testing needs the GPU free; non-GPU code can be written anytime.

---

## 2. What we're building — the loop
```
CARLA frame ─► 6 cam images + ego state ─► MODEL ─► 6 waypoints (ego-local)
     ▲                                                      │
     │                                                      ▼
   step sim ◄── steering/throttle ◄── CONTROLLER (Stanley/MPC) ◄── waypoints
```
Run over generated parking scenarios; log success / collision / final-pose / comfort.

---

## 3. DECISION 1 (do this first): Python / CARLA version
The model is Python 3.10; CARLA 0.9.14 is Python 3.8 → can't `import` both in one
process. Three options:

- **A — Unify on 3.10 (try first):** upgrade CARLA to **0.9.15** (added 3.10
  wheels), `pip install carla==0.9.15` into the model's `.venv`, run client+model
  in ONE process. Quick test: does `carla` import in the `.venv`? Can the CARLA
  **server** be upgraded to 0.9.15? Does `ParkingScenes` still work on 0.9.15?
- **B — Unify on 3.8:** DON'T. Rebuilding the model's torch/mmcv/mmdet3d/UniAD
  stack on 3.8 risks breaking a working setup.
- **C — Two-process bridge (reliable fallback):** CARLA client (3.8) ↔ model
  server (3.10) over a local socket / ZeroMQ. Zero version risk, more glue code.

**Plan: spend ~30 min testing A; if painful, use C. Never B.**

---

## 4. Reuse map — existing code (don't rebuild)
In `ParkingScenes/tool/`:
- `MotionPlanning/Control/Stanley.py`, `MotionPlanning/Control/MPC.py` —
  **trajectory-following controllers** (path → steering/throttle). **Reuse one**,
  feeding it the *model's* waypoints instead of the planner's path.
- `AutomatedValetParking/path_plan/rs_curve.py` — Reeds-Shepp paths (the expert;
  used because hybrid A* was buggy). Not needed for model eval, but the expert
  path generation is here if needed.
- `auto_park_1.py`, `world.py`, `sensors.py`, `data_generator.py`,
  `parking_position.py`, `bev_render.py` — the existing collection loop, 6-camera
  setup, scenario/slot generation. The harness = this loop with the model's
  prediction swapped in for the expert trajectory.

The 6-camera rig here MUST match the calibration the model was trained on (it
does — same data-gen rig). Keep the same camera mounting/intrinsics.

---

## 5. The hard part — live model inference
The offline pipeline (`inference_drivevla.py` + `LLaVANuScenesDataset`) reads
samples from a **NuScenes DB + cached pkl**. Closed-loop has **no DB** — you must
construct the model input from **live CARLA data** each step:
1. 6 live camera images (numpy/PIL).
2. Ego state: recent pose history (last ~2 s), velocity — from CARLA.
3. Mission goal: the target command (e.g. `reverse`) for the scenario.
4. Camera intrinsics/extrinsics (from the rig).

The model server must replicate the dataset's preprocessing (the mmdet3d image
pipeline + `build_llava_conversation` prompt assembly) from these live inputs,
then run UniAD → LLM → parse the `[(x,y),...]` output.

**Files to study in `~/projects/openvla_nuscenes/OpenDriveVLA`:**
- `drivevla/data_utils/nuscenes_llava_dataset.py` (`_get_uniad_test_data` — how a
  sample becomes model input)
- `drivevla/data_utils/build_llava_conversation.py` (prompt assembly, ego-state
  encoding, mission-goal text)
- `projects/mmdet3d_plugin/datasets/pipelines/loading.py` (image loading/transform)
- `drivevla/inference_drivevla.py` (`inference_data` — the per-sample forward)

Building this "live inference" function is the main model-side task. Consider an
MVP that constructs the minimal `input_dict` the pipeline needs and calls the
model directly.

---

## 6. Step-by-step plan
1. **Version decision** (§3): test Option A; else set up the Option C bridge skeleton.
2. **Model server / live inference** (§5): a function/endpoint that takes 6 images
   + ego history + mission goal + calibration → returns 6 ego-local waypoints.
   Validate it against a known frame (compare to an offline prediction for the
   same inputs).
3. **Controller adapter**: convert the 6 ego-local waypoints (x=right, y=forward)
   into the path/pose format `Stanley.py`/`MPC.py` expects; produce
   `carla.VehicleControl(steer, throttle, brake)`.
4. **The loop**: spawn a scenario (reuse scenario gen) → each tick: capture cams +
   ego state → model → controller → apply control → step → record.
5. **Metrics**: success (reached slot within tolerance), collision (CARLA sensor),
   final-pose error (cm), comfort (jerk). Save a per-run log + a replay
   (camera + chosen-path overlay) for visual debugging.
6. **Run** a handful of scenarios; iterate on the controller and coordinate frames.

---

## 7. Key paths
- Model checkpoint: `~/projects/openvla_nuscenes/checkpoints/OpenDriveVLA-0.5B-carla/merged`
- Model repo (run inference from here): `~/projects/openvla_nuscenes/OpenDriveVLA`
  (its `drivevla/*.py` entry points self-bootstrap `llava`/`projects`/`mmdet3d`
  paths + an nvcc shim).
- Model `.venv`: `~/projects/openvla_nuscenes/.venv` (Python 3.10)
- CARLA repo: `~/projects/parking_data_gen/ParkingScenes`
- CARLA venv: `~/projects/parking_data_gen/venv` (Python 3.8)

---

## 8. Gotchas / constraints
- **Coordinate frames**: model output is ego-local **x=right, y=forward, metres,
  0.5 s steps**. CARLA uses its own (left-handed) frame. Get this transform right —
  it's the #1 source of silent bugs.
- **GPU**: single 8 GB shared *desktop* GPU. Model-in-loop needs ~3.7 GB and the
  display already takes ~2.2 GB. Don't run alongside other GPU jobs.
- **Distribution shift**: the model was trained open-loop; live, its small errors
  compound into states it never saw. Expect drift; that's the whole point of
  measuring closed-loop. (DAgger to fix it = a *later* phase, not now.)
- **Control rate**: the model predicts at ~2 Hz (0.5 s steps); the controller can
  run faster (interpolate between waypoints) and re-query the model each ~0.5 s.

---

## 9. Initial prompt to paste into the new session
(see the README/handoff; paste the block your planning session gave you)
