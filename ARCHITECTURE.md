# Architecture & Design Decisions

This document explains the system design, coordinate conventions, and key engineering decisions in the parking dataset generator.

---

## System Design

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      CARLA Simulator                             │
│  (Town04_Opt or Town10HD_Opt + Ego Vehicle + NPCs)             │
└────────────────────────┬────────────────────────────────────────┘
                         │ (UE4 physics @ 30 Hz)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            ParkingScenes Auto_Park Controller                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Reeds-Shepp Path Planner (→ solution.csv)               │   │
│  │ • Optimal path between start/goal (analytical)          │   │
│  │ • ~1ms compute time                                      │   │
│  │ • Cached per (map, slot) pair                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MPC Controller (MotionPlanning/Control/MPC.py)          │   │
│  │ • Tracks RS path waypoints at 0.2s horizon              │   │
│  │ • Outputs throttle/brake/steer/reverse                 │   │
│  │ • Detects goal arrival (over=True)                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────┘
                         │ (control inputs @ 30 Hz)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Sensor Capture & Episode Recording                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 6× RGB Cameras (1600×900 @ 70° FOV)                    │   │
│  │ IMU (gyroscope: yaw rate)                               │   │
│  │ Ego pose (x, y, z, yaw, pitch, roll)                   │   │
│  │ Vehicle state (speed, throttle, brake, steer, reverse) │   │
│  │ Collision detection (hit actor name)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Record every 15th tick (2 Hz @ 30 Hz CARLA)              │   │
└────────────────────────┬────────────────────────────────────────┘
                         │ (if success & not collision)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Episode Saved to Disk                               │
│  data/raw/episode_XXXX/                                         │
│    ├─ poses.json          [per-frame pose, IMU, control]       │
│    ├─ meta.json           [slot, spawn, parking type]          │
│    └─ frames/frame_NNNN/  [6 JPEG images @ 2 Hz]               │
└─────────────────────────┬───────────────────────────────────────┘
                         │ (post-processing)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Processing Pipeline (build_*.py)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ validate_episodes.py: Check JPEG headers, frame counts  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ build_infos_pkl.py: poses.json → nuScenes sample infos  │   │
│  │ • Assign tokens, camera metadata, ego poses             │   │
│  │ • Output: parking_infos_temporal.pkl                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ build_cache_pkl.py: Compute ego-local trajectories      │   │
│  │ • History (5 steps @ 0.5s), future (7 steps @ 0.5s)    │   │
│  │ • Ego-local frame transform (right, forward)           │   │
│  │ • Output: cached_parking_info.pkl                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ verify_dataset.py: Sanity checks (errors → exit 1)      │   │
│  │ • All 6 images exist, timestamps increasing             │   │
│  │ • Velocity/trajectory consistency                       │   │
│  │ • Speed in valid range [0, 15] m/s                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────────┐
           │  OpenDriveVLA-Compatible Output │
           │  ✓ parking_infos_temporal.pkl   │
           │  ✓ cached_parking_info.pkl      │
           │  → Ready for fine-tuning        │
           └─────────────────────────────────┘
```

---

## Coordinate Conventions

### Why Multiple Frames?

We work in **three coordinate systems** because each has historical context:

| System | Origin | X | Y | Z | Handedness | Used by | Notes |
|---|---|---|---|---|---|---|---|
| **CARLA** | Vehicle center | East | South ↓ | Up | Left | `generate_episodes.py`, CARLA SDK | Y-flipped relative to world. |
| **OpenDRIVE** | World geo-ref | East | North ↑ | Up | Right | `.xodr` files, nuScenes standard | Geographic standard. |
| **Ego-local** | Vehicle rear-axle | Right | Forward | Up | Right | `cached_parking_info.pkl` | Used by vision models. |

### Transformation Formulas

```
CARLA → OpenDRIVE (world frame):
  x_odr = x_carla
  y_odr = -y_carla          [y-flip]
  z_odr = z_carla

OpenDRIVE → Ego-local (at frame t):
  ego_xyz_odr = [x_t, y_t, 0]      [2D, z always 0]
  ego_yaw_odr = ψ_t
  
  For a world point p_xyz:
    p_ego = R^T @ (p_xyz - ego_xyz_odr)    [rotate + translate]
    where R = rotation matrix from yaw_odr
    
  Result: [x_right, y_forward, 0]

Ego-local convention (legacy from nuScenes/GPT-Driver):
  x_ego = y_local  (right)
  y_ego = x_local  (forward)
  (swapped for ML convention: [batch, channels, height, width] → x-axis is lateral)
```

### Why the Y-Flip?

CARLA uses a **left-handed coordinate system** for historical reasons (Unreal Engine). The map artists built Town04_Opt with y-south. To be compatible with the **right-handed OpenDRIVE standard** (ISO 14825), we flip: `y_odr = -y_carla`.

This means:
- **Parking lot** is the same physical place in both frames
- **Ego trajectory** looks mirrored in 2D plots if you don't account for this
- **Angles** are negated: `yaw_odr = -yaw_carla` (or add π/2 depending on convention)

---

## Key Engineering Decisions

### 1. Reeds-Shepp over Hybrid A*

| Approach | Speed | Obstacle-Aware | Optimal Path |
|---|---|---|---|
| **Reeds-Shepp** | ~1ms | ❌ No | ✅ Yes |
| **Hybrid A\*** | ~30 min | ✅ Yes | ✅ Yes |

**Decision: Reeds-Shepp** because:
- Fast iteration during development (1ms vs hours)
- Generates geometrically valid paths (min turning radius respected)
- Acceptable collision rate (~67% success, retried automatically)
- Scaling: 1500 episodes @ 1ms planning << 1500 episodes @ 30min planning

**Trade-off:** NPC collisions during execution (not planning stage). Mitigated by:
- Filtering NPCs from ego's forward/backward path
- Automatic retry with new NPC placements (seed variation)
- ~3 attempts per saved episode overhead

### 2. Rear-Axle Offset in RS Goal

**Problem:** MPC controller reads rear-axle positions from the trajectory waypoints, but the RS path was computed from vehicle *center* positions. This causes the MPC to start 1.4m ahead of the first waypoint, causing premature gear switches.

**Solution (v3):** Offset start to rear-axle, keep goal at center:
```python
sx_r = sx - WB * cos(syaw)     # rear-axle start
sy_r = sy - WB * sin(syaw)
path = rs.calc_optimal_path(sx_r, sy_r, syaw, gx, gy, gyaw)  # goal unchanged
```

**Result:** `cx[0]` aligns with actual rear-axle position; car overshoots controlled by braking momentum.

### 3. 6 Cameras Instead of 4

**nuScenes standard:**
- Front, front-left, front-right (40° overlap)
- Back, back-left, back-right (40° overlap)

**Reason:** OpenDriveVLA's fine-tuning expects this camera rig. Dropping one would break camera indexing in the training loop.

**Cost:** ~2× disk I/O during collection (6 vs 3 JPEGs per frame). Acceptable at 2 Hz recording.

### 4. Caching RS Solutions by (map, slot) Pair

**Cache key:** `hash(["v3_slot{i}", start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw])`

**Why:**
- Same slot always has same geometry (start/goal positions fixed per slot)
- Different seed → different NPC positions, same RS path
- Version bump (v3) invalidates old caches when algorithm changes

**Trade-off:** Disk I/O to store 16 plans per map (< 1 MB total). Worth it to avoid 1ms × 1500 = 1.5 seconds of cumulative planner calls.

### 5. 2 Hz Recording (Every 15th Tick)

| Rate | Frames/ep | Size/ep | Redundancy | Notes |
|---|---|---|---|---|
| 30 Hz (all ticks) | 300 | 180 MB | High | Too much data; parking is slow motion |
| **2 Hz (every 15th)** | **20** | **12 MB** | ✅ Good | Captures maneuver, compress-able |
| 1 Hz (every 30th) | 10 | 6 MB | Undersample | Misses key MPC corrections |

**Decision:** 2 Hz balances temporal resolution (0.5s steps) with data volume.

---

## Data Flow: Episode → OpenDriveVLA Format

### Raw Episode (poses.json)

```json
[
  {
    "frame_idx": 0,
    "timestamp_us": 1718476800000000,
    "x_world": 290.0,
    "y_world": -201.2,
    ...
    "vx_world": 0.0,
    "vy_world": 0.0,
    "reverse": false,
    "steer_normalized": 0.0
  },
  ...
]
```

### Step 1: build_infos_pkl.py

**Input:** `poses.json`, frame images  
**Output:** `parking_infos_temporal.pkl`

```python
for frame in poses:
    token = f"{scene_token}_f{frame_idx:04d}"
    info = {
        'token': token,
        'cams': {
            'CAM_FRONT': {
                'data_path': 'data/raw/episode_XXX/frames/frame_0000/CAM_FRONT.jpg',
                'cam_intrinsic': K,  # shared for all cameras
            },
            ...  # 5 more cameras
        },
        'ego2global_translation': [x, y, z],
        'ego2global_rotation': [qx, qy, qz, qw],  # from yaw
        'can_bus': [x, y, 0, ..., speed, ...],    # 18-element vector
        'reverse': bool(frame['reverse']),        # ← NEW in v3
    }
    infos.append(info)
```

Pickle dumped as: `{'infos': [info, info, ...]}`

### Step 2: build_cache_pkl.py

**Input:** `poses.json`, `parking_infos_temporal.pkl`  
**Output:** `cached_parking_info.pkl`

Per frame, computes:
```python
cache[token] = {
    'gt_ego_lcf_feat': [vx, vy, x, y, yaw_rate, length, width, speed_head, steer],
    'gt_ego_his_trajs': array(5, 2),     # history @ [-2s, -1.5s, -1s, -0.5s, 0s]
    'gt_ego_his_diff': array(4, 2),      # step differences (differenced history)
    'gt_ego_fut_cmd': [0, 0, 1] or [0, 0, 0, 1],  # [forward, left, right, reverse]
    'gt_ego_fut_trajs': array(7, 2),    # future @ [0s, 0.5s, 1s, ..., 3s]
}
```

All trajectories in **ego-local (right, forward) frame**.

### Step 3: verify_dataset.py

Validates:
- ✅ All 6 images exist per frame
- ✅ Cache entry exists for every token
- ✅ `gt_ego_fut_trajs[0] == (0, 0)` (ego at origin)
- ✅ Speed in range [0, 15] m/s
- ✅ Reverse frames have backwards displacement
- ✅ Forward/reverse ratio logged

---

## Failure Modes & Mitigation

### 1. CARLA Streaming-ID Exhaustion (~250 episodes)

**Root cause:** Internal CARLA streaming subsystem assigns unique IDs to sensor streams. Counter never resets within a session. After ~2000 streams (250 episodes × 8 streams), server crashes.

**Mitigation:** `run_collection.sh` restarts CARLA every 200 episodes.

**Not a GPU OOM issue.** GPU memory is freed per-episode; the problem is CARLA's internal state.

### 2. NPC Collision During MPC Approach

**Root cause:** RS path is geometrically optimal but doesn't account for dynamic obstacles. MPC executes the path faithfully and hits parked NPCs during the curved approach.

**Mitigation:**
- Filter NPCs near ego path (Town04_Opt: `if sp.x < 285.0: skip`)
- Automatic retry with different NPC seed
- ~67% success rate means ~3 attempts per saved episode

**Not fixable without Hybrid A\*.** Would need obstacle-aware planning or trajectory modification at runtime.

### 3. MPC Gear Switching Too Early

**Root cause (v2):** RS path computed from center, MPC receives rear-axle waypoints. Rear axle 1.4m ahead of first waypoint → nearest_index jumps 2-3 steps → premature forward/reverse transition.

**Fixed in v3:** Apply WB offset to start, not goal. Now `cx[0]` aligns with actual rear-axle position.

**Residual:** MPC may still switch at tick 3-4 instead of tick 10+. Not fully resolved; acceptable because episodes still succeed.

### 4. Image Corruption (1 in 1500)

**Root cause:** CARLA sensor queue occasionally returns garbage frames (uninitialized memory).

**Mitigation:** `validate_episodes.py` checks JPEG headers (magic bytes `FF D8` and `FF D9`). Deletes bad episodes with `--fix` flag.

---

## Optimization Opportunities

### Short-term (Low effort)

1. **Parallel NPC spawning:** Split NPC creation across threads (I/O bound).
2. **Camera streaming:** Use CARLA's `save_to_disk()` instead of queue polling.
3. **Faster JSON serialization:** Use `orjson` instead of `json` (parsing poses.json).

### Medium-term (1-2 days)

1. **GPU-accelerated Hybrid A\*:** Implement on CUDA for <1s planning time.
2. **Data augmentation:** Weather, lighting, time-of-day randomization.
3. **Multi-map collection:** Collect from Town04_Opt + Town10HD_Opt in single run.

### Long-term (Research)

1. **Learning-based planner:** Train RL agent to generate parking trajectories directly (no explicit planning).
2. **Sim-to-real:** Domain randomization to transfer to real CARLA vehicles.

---

## Validation Metrics

### Episode-Level

- `total_frames`: 4–30 (min 4, goal ~20–25)
- `goal_hold`: ≥30 consecutive ticks within `(GOAL_DIST_M, GOAL_ROT_DEG)`
- `collision`: 0 (deleted if > 0)
- `timeout`: < 5 minutes (deleted if triggered)

### Frame-Level

Per `verify_dataset.py`:
- Image files: 6 exist, JPEG headers valid
- Timestamps: strictly increasing
- Speed: 0 ≤ speed ≤ 15 m/s (no outliers)
- Trajectory: `fut_traj[0] ≈ (0, 0)` within 0.01 m
- Consistency: `vx * 0.5s ≈ displacement[1]` within 50%–200% (lax for turns)

### Dataset-Level

- Total frames: 49,620
- Episodes: 1,499
- Forward/reverse ratio: 1.7:1
- Errors (hard): 0
- Warnings (soft): ~3.9% trajectory mismatches (expected during maneuvers)

---

## Future Extensions

### Town10HD_Opt (Parallel Parking)

Requires geometry audit:
- Slot locations and orientations
- Safe NPC exclusion zones
- Goal yaw for each slot
- Collision patterns during approach

**Not yet validated — don't collect at scale without audit.**

### Custom Maps

Add to `SLOT_GEOMETRY` dict:
```python
SLOT_GEOMETRY["YourMap"] = {
    "width_m": 3.0,
    "length_m": 5.5,
    "type": "perpendicular_forward"  # or "parallel", etc.
}
GOAL_YAW["YourMap"] = 180.0
```

Also update `slot_indices` per-map logic in `main()`.

