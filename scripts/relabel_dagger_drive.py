"""DAgger relabeling via the REAL collection stack (Auto_Park RS+MPC) — the production labeler.

Turns each logged DAgger off-path state into ONE training token whose future label is a TRAINING-
IDENTICAL speed-profiled trajectory: it spawns the ego at the logged pose and lets Auto_Park (the
ParkingScenes RS plan + MPC drive that produced the original 50k) drive ~3s toward the slot,
recording the DRIVEN poses every 0.5s. That driven trajectory IS the label — same accel/cruise/brake
distribution as training (unlike a flat constant-speed RS resample).

Emits the full package the model repo trains on (point --out at the model-repo processed dir):
  <out>                          cache: {token: gt_ego_lcf_feat/his/fut/slot_local/maneuver/side}
  <out_dir>/dagger_infos.pkl     per-token nuScenes infos (pose + cam paths + calibration)
  <out_dir>/dagger_conversations.json
  <out_dir>/dagger_frames/<token>/CAM_*.jpg   (the 6 logged images, copied)

Reuses generate_episodes.py UNCHANGED (imported as `ge`): WorldState, Auto_Park, the O(1) planner
patches, the plan cache, constants. Differences from collection: (1) spawn at the DAgger pose, not
the clean aisle spawn; (2) NO sensors (cameras already logged — skip rendering, the speed win);
(3) drive only ~3s and emit the driven poses as the label.

Runs against a LOCAL CARLA (collection ran CARLA locally too). No model server needed — give CARLA
the full GPU:
  ParkingScenes/carla/CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen >logs/carla_server.log 2>&1 &

Usage:
  # full run -> writes the package straight into the model repo:
  venv/bin/python scripts/relabel_dagger_drive.py --log-dir data/dagger_raw \
      --out ~/projects/openvla_nuscenes/data_carla/processed/dagger_cached.pkl
  # quick test (10 states), report timing + profile:
  venv/bin/python scripts/relabel_dagger_drive.py --limit 10
Resumable: re-running skips tokens already in <out> (so a crash/restart continues).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import shutil
import sys
import time
import types

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import carla
import generate_episodes as ge   # applies planner + plan-cache patches at import (collection stack)
from harness.model_server import live_prompt as lp   # validated global->ego-local transform
from build_infos_pkl import (_carla_to_nuscenes_pose, _carla_cam_extrinsic,
                             CAM_INTRINSIC, _CARLA_CAMERAS, CAMERA_NAMES)

VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 1.8
N_FUT = 7            # future points t=0..3.0s (incl. origin)
HIST_STEPS = 4
DRIVE_TICKS = 110    # ~3.7s @30Hz — enough margin to fill 6 future records past the origin
MIN_REACH = 0.5      # drop a drive that barely moved (Auto_Park stuck / infeasible from this pose)


# ── label + package builders ─────────────────────────────────────────────────

def driven_future(poses_carla, ex_n, ey_n, eyaw_n):
    """Driven CARLA poses [(x,y,yaw_deg)] -> (7,3) ego-local (right, forward, heading) relative to
    the logged START nuScenes pose. Same convention as the cache (global_to_local_xy + heading)."""
    fut = np.zeros((N_FUT, 3), dtype=np.float32)
    for j in range(N_FUT):
        xc, yc, yawd = poses_carla[min(j, len(poses_carla) - 1)]
        xn, yn, yawn = xc, -yc, -math.radians(yawd)        # CARLA -> nuScenes
        r, f = lp.global_to_local_xy(xn, yn, ex_n, ey_n, eyaw_n)
        fut[j] = [r, f, lp.normalize_angle(yawn - eyaw_n)]
    fut[0] = [0.0, 0.0, 0.0]
    return fut


def build_cache_entry(state, run_meta, fut):
    """Cache entry: driven future label + logged measured ego-state/history/slot (correct velocity)."""
    ego = state["ego"]
    ex, ey, eyaw = float(ego["x"]), float(ego["y"]), float(ego["yaw"])
    slot = run_meta["slot_global"]
    gx, gy, gyaw = float(slot["x"]), float(slot["y"]), float(slot["yaw"])

    hpts = [list(lp.global_to_local_xy(h["x"], h["y"], ex, ey, eyaw))
            for h in state.get("history", [])[-HIST_STEPS:]]
    while len(hpts) < HIST_STEPS:
        hpts.insert(0, list(hpts[0]) if hpts else [0.0, 0.0])
    hpts.append([0.0, 0.0])
    his = np.array(hpts, dtype=np.float32)
    his_diff = np.diff(his, axis=0).astype(np.float32)

    sr, sf = lp.global_to_local_xy(gx, gy, ex, ey, eyaw)
    slot_local = np.array([sr, sf, lp.normalize_angle(gyaw - eyaw)], dtype=np.float32)
    lcf = np.array([float(ego["fwd_v"]), float(ego["right_v"]), ex, ey, float(ego["yaw_rate"]),
                    VEHICLE_LENGTH, VEHICLE_WIDTH, float(ego.get("speed", 0.0)),
                    float(ego["steer"])], dtype=np.float32)
    return {
        "gt_ego_lcf_feat": lcf, "gt_ego_his_trajs": his, "gt_ego_his_diff": his_diff,
        "gt_ego_fut_cmd": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "gt_ego_fut_trajs": fut,
        "maneuver_type": run_meta["maneuver_type"], "side": run_meta["side"],
        "slot_local": slot_local,
    }


def build_infos_entry(state, run_meta, token, frames_dir, reverse):
    """nuScenes-style infos record (UniAD extraction consumes cams + pose + can_bus). Each DAgger
    token is its own 1-frame scene so UniAD resets BEV memory per state (independent snapshots)."""
    ego = state["ego"]
    ec = state["ego_carla"]
    yaw_deg = math.degrees(float(ec["yaw_rad"]))
    nx, ny, nz, quat = _carla_to_nuscenes_pose(
        float(ec["x"]), float(ec["y"]), float(ego.get("z", 0.0)), yaw_deg)
    cams = {}
    for cam_name in CAMERA_NAMES:
        R, T = _carla_cam_extrinsic(_CARLA_CAMERAS[cam_name])
        cams[cam_name] = {"data_path": str((frames_dir / token / f"{cam_name}.jpg").resolve()),
                          "cam_intrinsic": CAM_INTRINSIC,
                          "sensor2lidar_rotation": R, "sensor2lidar_translation": T}
    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[0], can_bus[1], can_bus[13] = nx, ny, float(ego.get("speed", 0.0))
    return {
        "token": token, "scene_token": token, "prev": "", "next": "", "frame_idx": 0,
        "timestamp": 0, "cams": cams,
        "ego2global_translation": [nx, ny, nz], "ego2global_rotation": quat, "can_bus": can_bus,
        "lidar_path": "", "sweeps": [],
        "lidar2ego_rotation": [1.0, 0.0, 0.0, 0.0], "lidar2ego_translation": np.array([0.0, 0.0, 0.0]),
        "reverse": bool(reverse),
        "ego_fwd_v": float(ego["fwd_v"]), "ego_right_v": float(ego["right_v"]),
        "ego_speed": float(ego.get("speed", 0.0)), "ego_yaw_rate": float(ego["yaw_rate"]),
        "ego_steer": float(ego["steer"]),
        "maneuver_type": run_meta["maneuver_type"], "side": run_meta["side"],
        "target_slot": run_meta.get("target_slot"),
        "gt_boxes": np.zeros((0, 7), dtype=np.float64), "gt_names": np.array([], dtype="<U32"),
        "gt_velocity": np.zeros((0, 2), dtype=np.float64), "gt_inds": np.zeros(0, dtype=np.int64),
        "gt_ins_tokens": np.array([], dtype="<U32"), "valid_flag": np.zeros(0, dtype=bool),
        "num_lidar_pts": np.zeros(0, dtype=np.int32),
        "fut_traj": np.zeros((0, 16, 2), dtype=np.float64),
        "fut_traj_valid_mask": np.zeros((0, 16, 2), dtype=np.float64),
    }


# ── CARLA driving (Auto_Park, no sensors) ────────────────────────────────────

def drive_one(world, slot_idx, spawn_tf, args):
    """Spawn ego at spawn_tf (NO sensors), run Auto_Park ~3s, return (driven CARLA poses, collided).
    Cameras+IMU+GNSS are disabled so there is no per-tick render / drain_sensors timeout."""
    ws = ge.WorldState(world)
    ws.index = slot_idx
    ws._attach_cameras = types.MethodType(lambda self: None, ws)
    ws._attach_misc_sensors = types.MethodType(lambda self: None, ws)
    ws.spawn_ego(spawn_tf)
    try:
        for _ in range(4):
            world.tick()
        controller = ge.Auto_Park(ws)
        poses = []
        for tick in range(DRIVE_TICKS + 1):
            world.tick()
            controller.main(None, ws, ge._CLOCK, slot_idx, args)
            if tick % ge.RECORD_EVERY == 0:
                tr = ws.player.get_transform()
                poses.append((tr.location.x, tr.location.y, tr.rotation.yaw))
                if len(poses) >= N_FUT:
                    break
            if ws._collision.hit:
                return poses, True
        return poses, False
    finally:
        ws.destroy_all()


def connect(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.load_world(args.map)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / ge.SIM_HZ
    world.apply_settings(s)
    return client, world


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="data/dagger_raw")
    ap.add_argument("--out", default="data/processed/dagger_cached.pkl",
                    help="cache path; infos/conversations/frames are written alongside it")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map", default="Town04_Opt")
    ap.add_argument("--limit", type=int, default=0, help="process only N states (0 = all)")
    ap.add_argument("--reload-every", type=int, default=300,
                    help="reload the CARLA world every N states to clear accumulated actor state "
                         "(0 = never)")
    ap.add_argument("--pedestrians", action="store_true",
                    help="spawn pedestrians like collection (default OFF: a walker in the path can "
                         "clip an otherwise-valuable recovery label)")
    args = ap.parse_args(argv)
    args.place_pedestrians = bool(args.pedestrians)
    args.verbose = False
    args.num_episodes = 1
    args.save_path = ""
    args.seed = 0

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / "dagger_frames"
    infos_path = out.parent / "dagger_infos.pkl"
    conv_path = out.parent / "dagger_conversations.json"

    # Resume: load any existing package so a re-run continues instead of redoing finished tokens.
    cache = pickle.load(open(out, "rb")) if out.exists() else {}
    infos = pickle.load(open(infos_path, "rb"))["infos"] if infos_path.exists() else []
    convs = json.loads(conv_path.read_text()) if conv_path.exists() else []
    done = set(cache)
    if done:
        print(f"[drive] resuming: {len(done)} tokens already done", flush=True)

    # gather pending jobs
    runs = sorted(p for p in pathlib.Path(args.log_dir).glob("*") if (p / "run_meta.json").exists())
    jobs = []
    for run in runs:
        rm = json.loads((run / "run_meta.json").read_text())
        for st in sorted(run.glob("state_*")):
            token = f"dagger_{run.name}_{st.name}"
            if token not in done:
                jobs.append((run, rm, st, token))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[drive] {len(jobs)} states to process\n", flush=True)

    print(f"[drive] connecting CARLA {args.host}:{args.port} ...", flush=True)
    client, world = connect(args)
    print("[drive] world ready", flush=True)

    def flush():
        pickle.dump(cache, open(out, "wb"))
        pickle.dump({"infos": infos, "metadata": {"version": "v1.0-carla-dagger"}}, open(infos_path, "wb"))
        conv_path.write_text(json.dumps(convs))

    kept = dropped = 0
    times = []
    t_start = time.time()
    for n, (run, rm, st, token) in enumerate(jobs):
        if args.reload_every and n > 0 and n % args.reload_every == 0:
            print(f"[drive] reloading world after {n} states ...", flush=True)
            client, world = connect(args)
        state = json.loads((st / "state.json").read_text())
        ec = state["ego_carla"]
        spawn_tf = carla.Transform(
            carla.Location(x=float(ec["x"]), y=float(ec["y"]), z=0.3),
            carla.Rotation(yaw=math.degrees(float(ec["yaw_rad"]))))
        t0 = time.time()
        try:
            poses, collided = drive_one(world, rm["slot_idx"], spawn_tf, args)
        except Exception as e:
            print(f"  DROP {token}: drive error ({e})", flush=True)
            dropped += 1
            continue
        times.append(time.time() - t0)
        ego = state["ego"]
        fut = driven_future(poses, float(ego["x"]), float(ego["y"]), float(ego["yaw"]))
        reach = float(np.max(np.hypot(fut[:, 0], fut[:, 1])))
        if collided or reach < MIN_REACH or len(poses) < 2:
            dropped += 1
            continue
        # accept: cache + cams + infos + conversation
        cache[token] = build_cache_entry(state, rm, fut)
        tok_frames = frames_dir / token
        tok_frames.mkdir(parents=True, exist_ok=True)
        for cam in rm["cam_order"]:
            src = st / f"{cam}.jpg"
            if src.exists():
                shutil.copy2(src, tok_frames / f"{cam}.jpg")
        reverse = float(np.sum(fut[:, 1])) < 0.0
        infos.append(build_infos_entry(state, rm, token, frames_dir, reverse))
        convs.append({"qa_id": f"{token}_trajectory", "sample_id": token,
                      "conversations": [{"from": "human", "value": ""},
                                        {"from": "gpt", "value": ""}]})
        kept += 1
        if kept % 50 == 0:
            flush()
            rate = sum(times) / len(times)
            print(f"  [{n+1}/{len(jobs)}] kept={kept} dropped={dropped} "
                  f"({rate:.1f}s/state, ETA {rate*(len(jobs)-n-1)/3600:.1f}h)", flush=True)

    flush()
    print(f"\n[drive] DONE: kept {kept}, dropped {dropped}, total {len(cache)} tokens in "
          f"{(time.time()-t_start)/3600:.2f}h")
    print(f"  cache         -> {out}")
    print(f"  infos         -> {infos_path}")
    print(f"  conversations -> {conv_path}")
    print(f"  frames        -> {frames_dir}/<token>/CAM_*.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
