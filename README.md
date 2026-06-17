# CARLA Parking Dataset Generator for OpenDriveVLA

A production-ready pipeline to generate large-scale synthetic parking maneuver datasets from CARLA simulator, compatible with OpenDriveVLA fine-tuning.

**Status**: ✅ 1500 episodes, 49,620 frames collected and validated  
**Dataset**: ~61 GB raw, ~124 MB processed (6 RGB cameras per frame @ 1600×900)  
**Output format**: nuScenes-compatible `.pkl` + NuScenes database tables  

---

## Table of Contents

- [Overview](#overview)
- [Dataset Features](#dataset-features)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Folder Structure](#folder-structure)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## Overview

This repository generates synthetic parking datasets from CARLA using:
- **Reeds-Shepp curves** for optimal path planning (1ms per episode)
- **MPC (Model Predictive Control)** for vehicle control
- **Parallel collection** across multiple CARLA instances for speed
- **Automatic CARLA restarts** to avoid streaming-ID exhaustion crashes

The output is directly compatible with OpenDriveVLA's fine-tuning pipeline, requiring only path changes in the training config.

---

## Dataset Features

### Parking Maneuvers
- **Type**: Perpendicular reverse parking (Town04_Opt), parallel parking (Town10HD_Opt)
- **Success rate**: ~67% (retry-based)
- **Episodes**: 1500 perpendicular, expandable to parallel/forward variants
- **Frames per episode**: 4–30 (at 2 Hz recording)

### Sensor Setup
- **Cameras**: 6 RGB (front, front-left, front-right, back, back-left, back-right)
- **Resolution**: 1600×900 pixels, 70° FOV
- **Intrinsics**: Pre-calibrated K matrix (included in metadata)
- **IMU**: Gyroscope (yaw rate), acceleration
- **Recording**: 2 Hz (30 Hz CARLA, every 15th frame)

### Motion Capture
Per-frame ego state recorded:
- **Position**: x, y, z (world frame)
- **Orientation**: yaw, pitch, roll (degrees)
- **Velocity**: vx, vy, vz (world frame, m/s)
- **Control**: throttle, brake, steer, reverse flag (bool)
- **Sensor data**: timestamps (microseconds), yaw rate

### Dataset Diversity
- **16 parking slots** in Town04_Opt (slots 17–32, all with 180° goal yaw)
- **Multiple seeds** (42, 137, ...) → different NPC placements per batch
- **Random spawn points** → varied approach angles
- **Reverse vs. forward frames**: 1.7:1 ratio (40% reverse for IL training)

---

## Quick Start

### 1. Prerequisites

```bash
# CARLA 0.9.14+ with Town04_Opt and Town10HD_Opt maps
# Python 3.8+
# GPU with 8+ GB VRAM (for single instance)

# Clone and install
git clone https://github.com/Kepitiact/carla_data_gen_VLA.git
cd carla_data_gen_VLA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect 100 Episodes (Demo)

```bash
# Assumes CARLA is running on localhost:2000
python scripts/generate_episodes.py \
    --port 2000 \
    --map Town04_Opt \
    --num_episodes 100 \
    --seed 42 \
    --save_path data/raw
```

Expected time: ~2 hours for 100 episodes (including ~3 failures per success).

### 3. Process into OpenDriveVLA Format

```bash
# Validate and remove corrupt episodes
python scripts/validate_episodes.py --raw_dir data/raw --fix

# Build OpenDriveVLA-format .pkl files
python scripts/build_infos_pkl.py
python scripts/build_cache_pkl.py

# Sanity check
python scripts/verify_dataset.py
```

### 4. (Optional) Generate nuScenes-Compatible Map

```bash
# Creates data/nuscenes/maps/Town04_Opt.png (BEV semantic raster)
python scripts/build_carla_map.py \
    --xodr ParkingScenes/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04_Opt.xodr \
    --out-dir data/nuscenes/maps \
    --map-json data/nuscenes/v1.0-mini/map.json
```

---

## Pipeline Architecture

```
generate_episodes.py
    ├─ CARLA world + ego spawn
    ├─ Auto_Park controller (Reeds-Shepp planner + MPC)
    ├─ 6-camera sensor capture (2 Hz)
    └─ Save: poses.json + meta.json + frames/

validate_episodes.py
    ├─ Check meta.json / poses.json present
    ├─ Verify JPEG headers (no corrupt images)
    ├─ Check frame counts, timestamps, speed ranges
    └─ Remove bad episodes (--fix)

build_infos_pkl.py
    ├─ Parse poses.json → nuScenes sample infos
    ├─ Build camera intrinsics, ego poses, tokens
    └─ Output: parking_infos_temporal.pkl

build_cache_pkl.py
    ├─ Compute per-frame ego-local trajectories
    ├─ Extract history (5 steps @ 0.5s) + future (7 steps @ 0.5s)
    ├─ Velocity + command inference
    └─ Output: cached_parking_info.pkl

verify_dataset.py
    ├─ Check all 6 images exist per frame
    ├─ Validate trajectory math (consistency checks)
    ├─ Report forward/reverse ratio, outliers
    └─ Exit 0 if no errors

build_carla_map.py (optional)
    ├─ Parse OpenDRIVE .xodr file
    ├─ Extract road geometry (lanes + dividers + edges)
    ├─ Rasterize into multi-channel PNG (BEV semantic)
    └─ Output: Town04_Opt.png + metadata JSON
```

---

## Folder Structure

```
carla_data_gen_VLA/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── CLAUDE.md                          # Development guidelines
│
├── configs/
│   └── base_parking.py               # (WIP) Config templates
│
├── scripts/
│   ├── generate_episodes.py          # Main collection script ⭐
│   ├── build_infos_pkl.py            # Episode → nuScenes infos
│   ├── build_cache_pkl.py            # Compute ego-local trajectories
│   ├── verify_dataset.py             # Sanity checks
│   ├── validate_episodes.py          # Remove corrupt episodes
│   ├── build_carla_map.py            # OpenDRIVE → BEV PNG
│   ├── visualize_episode.py          # Debug viz (trajectory plots)
│   ├── make_nuscenes_stub.py         # Create v1.0-mini directory
│   └── run_collection.sh             # Wrapper for auto-restarts
│
├── data/
│   ├── raw/                          # episode_XXXX/ directories
│   │   └── episode_0000/
│   │       ├── meta.json             # Episode metadata (slot, spawn, etc.)
│   │       ├── poses.json            # Per-frame pose + IMU + control
│   │       └── frames/
│   │           └── frame_0000/
│   │               ├── CAM_FRONT.jpg
│   │               ├── CAM_FRONT_LEFT.jpg
│   │               └── ... (6 cameras total)
│   │
│   ├── processed/
│   │   ├── parking_infos_temporal.pkl         # nuScenes format infos
│   │   ├── cached_parking_info.pkl            # Ego-local trajectories
│   │   └── (optional) carla_conversations.json # VLM conversation pairs
│   │
│   ├── plan_cache/                   # Cached Reeds-Shepp solutions
│   │   └── plan_XXXXXXXX.csv         # Road geometry (x, y, yaw, v)
│   │
│   ├── v1.0-mini/                    # NuScenes stub directory
│   │   ├── maps/
│   │   │   └── Town04_Opt.png        # BEV semantic raster
│   │   ├── attribute.json
│   │   ├── sensor.json
│   │   └── ... (13 JSON schema files)
│   │
│   └── nuscenes/
│       └── maps/
│           └── Town04_Opt.png        # (same as above, alt location)
│
├── logs/
│   ├── collect_A.log                 # Batch A stdout/stderr
│   ├── collect_B.log                 # Batch B
│   └── carla_server.log              # CARLA console output
│
└── ParkingScenes/                    # Git submodule (CARLA + planning)
    └── carla/
        ├── CarlaUE4.sh               # Launch script
        ├── CarlaUE4/
        │   └── Content/Carla/Maps/OpenDrive/
        │       ├── Town04_Opt.xodr   # Used by build_carla_map.py
        │       └── Town10HD_Opt.xodr
        └── PythonAPI/
            └── util/
                └── opendrive/
                    └── (OpenDRIVE parsing utilities)
```

---

## Detailed Usage

### Collecting Episodes

#### Single Instance (Safe)

```bash
source venv/bin/activate
python scripts/generate_episodes.py \
    --host 127.0.0.1 --port 2000 \
    --map Town04_Opt \
    --num_episodes 500 \
    --seed 42 \
    --save_path data/raw
```

**Arguments:**
- `--num_episodes`: Total episodes to collect in this run
- `--seed`: Random seed for NPC placement (different per batch = different diversity)
- `--map`: Town04_Opt (perpendicular) or Town10HD_Opt (parallel)
- `--place_pedestrians`: Include walking pedestrians (default: True)

#### Auto-Restarting Wrapper (Production)

```bash
source venv/bin/activate
bash scripts/run_collection.sh 1500 data/raw 42
```

**How it works:**
- Restarts CARLA every 200 episodes (avoids streaming-ID exhaustion crash)
- Auto-detects existing episodes and continues numbering
- Shifts seed by 7 each restart (NPC diversity)
- Logs to `logs/collect.log`
- Stops automatically when reaching target count

**Parameters:**
```bash
bash scripts/run_collection.sh <target_episodes> <save_dir> <seed>
```

### Processing Episodes

#### 1. Validate

```bash
python scripts/validate_episodes.py --raw_dir data/raw

# Expected output:
#   Valid:    1499 / 1500
#   Invalid:  1 / 1500
```

If any are invalid, remove them:
```bash
python scripts/validate_episodes.py --raw_dir data/raw --fix
```

#### 2. Build OpenDriveVLA Format

```bash
# Create nuScenes infos (sample tokens, camera paths, ego poses)
python scripts/build_infos_pkl.py

# Compute ego-local trajectories (history + future + commands)
python scripts/build_cache_pkl.py

# Sanity check (frame counts, image existence, trajectory math)
python scripts/verify_dataset.py
```

**Output files:**
- `data/processed/parking_infos_temporal.pkl` — sample metadata + tokens (49,620 frames)
- `data/processed/cached_parking_info.pkl` — ego state cache (same 49,620 entries)

#### 3. (Optional) Build Conversation Pairs for VLM Training

If using OpenDriveVLA's conversation-based training:

```bash
# From openvla_nuscenes/
cd ../openvla_nuscenes
python scripts/build_carla_conversations.py \
    --infos ../parking_data_gen/data/processed/parking_infos_temporal.pkl \
    --raw_dir ../parking_data_gen/data/raw \
    --out ../parking_data_gen/data/processed/carla_conversations.json
```

---

## Configuration

### Simulation Parameters (generate_episodes.py)

| Param | Value | Notes |
|---|---|---|
| SIM_HZ | 30 | CARLA world timestep |
| RECORD_HZ | 2 | Frames saved per second |
| RECORD_EVERY | 15 | Record every Nth tick (30/2) |
| GOAL_DIST_M | 0.5 | Distance threshold to goal |
| GOAL_ROT_DEG | 3.5 | Rotation threshold (degrees) |
| GOAL_HOLD_FRAMES | 30 | Frames to hold goal before success |
| TIMEOUT_TICKS | 9000 | 5 minutes hard limit |

### Parking Slot Configuration

**Town04_Opt:**
- **Slots**: 17–32 (perpendicular, 3.0 m wide, 5.5 m long)
- **Goal yaw**: 180° (car facing -x direction)
- **Aisle location**: x ≈ 285.6 m
- **NPC rows**: 3 rows of parked vehicles (with filtering to avoid ego path)

**Town10HD_Opt:**
- **Slots**: Parallel parking
- **Goal yaw**: 90° (car facing +y direction)
- (Documentation TBD — requires geometry audit)

### Path Planning

**Reeds-Shepp Curve:**
- **Max curvature**: 0.2 (1/5m turn radius)
- **Start**: Offset to rear-axle (WB=1.4m)
- **Goal**: Slot center (no offset)
- **Cache**: Invalidated with seed/slot changes (v3 key)

**MPC Control:**
- **Target speed**: 0.8 m/s
- **Horizon**: 6 steps @ 0.2s
- **Steer limit**: ±60°
- **WB (wheelbase)**: 2.5 m (for dynamics model)

---

## Technical Details

### Coordinate Conventions

**CARLA world frame:**
- Left-hand, x=east, y=south
- `carla_y = -world_y` (y-flipped)

**OpenDRIVE / nuScenes world frame:**
- Right-hand, x=east, y=north
- `xodr_y = -carla_y`

**Ego-local frame (cache):**
- Right (x), forward (y)
- Computed in `build_cache_pkl.py`

### Poses.json Schema

Each episode's `poses.json` contains per-frame records with:

```python
{
    "frame_idx": int,                 # 0, 1, 2, ...
    "timestamp_us": int,              # microseconds since epoch
    "x_world": float,                 # CARLA world x
    "y_world": float,                 # CARLA world y
    "z_world": float,                 # height
    "yaw_deg": float,                 # heading (degrees)
    "pitch_deg": float,               # tilt
    "roll_deg": float,                # bank
    "vx_world": float,                # world-frame vx (m/s)
    "vy_world": float,                # world-frame vy (m/s)
    "vz_world": float,                # vertical velocity
    "speed_ms": float,                # |v| (m/s)
    "yaw_rate_rads": float,           # dψ/dt (rad/s)
    "steer_normalized": float,        # steering angle (-1 to +1)
    "throttle": float,                # throttle (0 to 1)
    "brake": float,                   # brake (0 to 1)
    "reverse": bool,                  # reverse gear engaged
}
```

### Meta.json Schema

Each episode's `meta.json` contains:

```python
{
    "episode_id": "episode_0000",
    "map": "Town04_Opt",
    "parking_type": "perpendicular_reverse",
    "slot": {
        "cx_world": float,            # slot center x
        "cy_world": float,            # slot center y
        "heading_rad": float,         # slot orientation (rad)
        "width_m": 3.0,
        "length_m": 5.5,
    },
    "spawn": {
        "x_world": float,
        "y_world": float,
        "heading_rad": float,
    },
    "heading_error_at_spawn_rad": float,    # angle to slot
    "approach_mode": "forward" | "reverse",  # based on spawn heading
    "astar_path_world": [waypoints...],     # (unused, for future)
    "total_frames": int,
    "sample_rate_hz": 2,
}
```

### Camera Intrinsics

All 6 cameras share the same intrinsic matrix (pre-calibrated for 1600×900, FOV=70°):

```python
K = [
    [1247.4,    0.0, 800.0],
    [   0.0, 1247.4, 450.0],
    [   0.0,    0.0,   1.0],
]
# focal length ≈ 1247 pixels
# principal point = (800, 450) = image center
```

---

## Troubleshooting

### CARLA Crashes After ~250 Episodes

**Cause:** Internal CARLA streaming-ID counter exhaustion (not GPU OOM).  
**Solution:** Use `run_collection.sh` wrapper — restarts CARLA every 200 episodes.

### "Invalid session: no stream available" Errors

**Cause:** Streaming-ID exhaustion (same as above).  
**Solution:** Kill CARLA and restart:
```bash
pkill -f CarlaUE4; sleep 5
# Then re-run collection script
```

### Validation Fails: "missing meta.json"

**Cause:** Episode crashed before writing metadata.  
**Solution:** Delete the incomplete episode:
```bash
python scripts/validate_episodes.py --raw_dir data/raw --fix
```

### Low Success Rate (<50%)

**Cause:** Usually NPC collisions during MPC approach phase.  
**Fixes:**
1. Reduce NPC count in `generate_episodes.py` (line ~605)
2. Widen the NPC exclusion zone (line ~690, `if sp.x < 285.0`)
3. Increase GOAL_HOLD_FRAMES (line ~419) to require longer parking hold

### PNG Map Rasterization Is Slow

**Cause:** `build_carla_map.py` rasterizes 800+ polygons into 9800×8827 pixels.  
**Tips:**
- This is a one-time operation (~2-3 minutes)
- Run in the background or on a separate machine
- Optional for fine-tuning (OpenDriveVLA can work without it)

### CUDA OOM During Collection

**Cause:** Two CARLA instances on single 8GB GPU.  
**Solution:** Use sequential collection (Option B in README) or split across two machines.

---

## Performance Metrics

### Collection Speed

| Setup | Episodes/Hour | Total Time (1500 ep) |
|---|---|---|
| Single CARLA (sequential) | ~50 | 30 hours |
| Dual CARLA (parallel) | ~95 | 16 hours |

### Dataset Size

| Component | Size | Count |
|---|---|---|
| Raw (images + JSON) | 61 GB | 1499 episodes |
| parking_infos_temporal.pkl | ~80 MB | 49,620 frames |
| cached_parking_info.pkl | ~40 MB | 49,620 frames |
| Town04_Opt.png (BEV map) | ~40 MB | 1 map |

### Success Rate

- **Episode success**: ~67% (with automatic retries)
- **Frame validation**: 100% (after `validate_episodes.py --fix`)
- **Trajectory consistency**: 96% (3.9% velocity/trajectory mismatches during turns)

---

## Future Work

- [ ] Town10HD_Opt (parallel parking) geometry audit
- [ ] Faster Hybrid A* implementation (obstacle-aware planning)
- [ ] Forward and lateral parking variants
- [ ] Multi-map fusion (Town04_Opt + Town10HD_Opt + custom maps)
- [ ] Augmentation pipeline (weather, lighting, time-of-day)

---

## Citation

If you use this dataset, please cite:

```bibtex
@misc{carla_parking_datagen,
  author = {Kepitiact},
  title = {CARLA Parking Dataset Generator for OpenDriveVLA},
  year = {2026},
  url = {https://github.com/Kepitiact/carla_data_gen_VLA},
  note = {Synthetic parking maneuver dataset, 1500 episodes, nuScenes format}
}
```

---

## License

[Specify your license here — MIT, Apache 2.0, etc.]

---

## Contact & Issues

For bugs, questions, or contributions:
- **GitHub Issues**: https://github.com/Kepitiact/carla_data_gen_VLA/issues
- **Email**: aybertunca@gmail.com

