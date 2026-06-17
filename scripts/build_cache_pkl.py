"""
Build cached_parking_info.pkl from raw episode data.

Equivalent to cached_nuscenes_info.pkl consumed by OpenDriveVLA.
Keyed by sample_token → ego state + history/future trajectories.

Coordinate convention (output):
  All trajectories are in ego-local (right, forward) frame.
  index 0 of future = current position = (0, 0).
  index 4 of history = current position = (0, 0).
  Negative forward = reverse motion.

Velocity storage:
  gt_ego_lcf_feat[0] = vx (raw m/s forward, NOT scaled by 0.5).
  The OpenDriveVLA prompt builder multiplies by 0.5 before printing.

Usage:
  python scripts/build_cache_pkl.py
  python scripts/build_cache_pkl.py --raw_dir data/raw --out data/processed/cached_parking_info.pkl
"""

import argparse
import json
import math
import pathlib
import pickle

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

VEHICLE_LENGTH = 4.5  # Tesla Model3 approx
VEHICLE_WIDTH  = 1.8


def _carla_world_to_ego_local(wx, wy, ego_x, ego_y, ego_yaw_deg):
    """Project world (wx, wy) into ego-local (right, forward).

    CARLA convention: yaw=0 → facing +x, yaw=90 → facing +y (left-hand Z-up).
    right   =  dx*sin(θ) - dy*cos(θ)   (positive = right of vehicle)
    forward =  dx*cos(θ) + dy*sin(θ)   (positive = ahead, negative = reverse)
    """
    theta = math.radians(ego_yaw_deg)
    dx = wx - ego_x
    dy = wy - ego_y
    forward = dx * math.cos(theta) + dy * math.sin(theta)
    right   = dx * math.sin(theta) - dy * math.cos(theta)
    return np.array([right, forward], dtype=np.float32)


def _signed_forward_speed(vx_world, vy_world, yaw_deg):
    """Project world velocity onto ego forward axis; negative when reversing."""
    theta = math.radians(yaw_deg)
    return vx_world * math.cos(theta) + vy_world * math.sin(theta)


def build_cache(raw_dir: pathlib.Path, out_path: pathlib.Path):
    cache = {}
    episode_dirs = sorted(raw_dir.glob('episode_*'))
    if not episode_dirs:
        raise RuntimeError(f"No episodes found in {raw_dir}")

    for ep_dir in episode_dirs:
        meta_path  = ep_dir / 'meta.json'
        poses_path = ep_dir / 'poses.json'
        if not meta_path.exists() or not poses_path.exists():
            print(f"Skipping {ep_dir.name}")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        with open(poses_path) as f:
            poses = json.load(f)

        scene_token = meta['episode_id']
        n = len(poses)

        for i, pose in enumerate(poses):
            token = f"{scene_token}_f{i:04d}"
            ex, ey = pose['x_world'], pose['y_world']
            eyaw = pose['yaw_deg']

            # ── Future trajectory: 7 points at t=0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s
            # At RECORD_HZ=2 Hz, 1 step = 0.5 s, so future index j maps to frame i+j
            fut_traj = np.zeros((7, 2), dtype=np.float32)
            for j in range(7):
                src_idx = min(i + j, n - 1)
                fp = poses[src_idx]
                fut_traj[j] = _carla_world_to_ego_local(
                    fp['x_world'], fp['y_world'], ex, ey, eyaw
                )
            # index 0 must be (0, 0) — ego position in ego frame
            assert np.allclose(fut_traj[0], 0.0, atol=1e-4), \
                f"fut_traj[0] != 0 for {token}: {fut_traj[0]}"

            # ── History trajectory: 5 points at t=-2, -1.5, -1, -0.5, 0 s
            # index 0 = t=-2s = frame i-4; index 4 = t=0 = frame i
            his_traj = np.zeros((5, 2), dtype=np.float32)
            for j in range(5):
                src_idx = max(0, i - (4 - j))
                hp = poses[src_idx]
                his_traj[j] = _carla_world_to_ego_local(
                    hp['x_world'], hp['y_world'], ex, ey, eyaw
                )
            # index 4 must be (0, 0)
            assert np.allclose(his_traj[4], 0.0, atol=1e-4), \
                f"his_traj[4] != 0 for {token}: {his_traj[4]}"

            his_diff = np.diff(his_traj, axis=0).astype(np.float32)  # shape (4, 2)

            # ── Ego speed in body frame
            vx = _signed_forward_speed(pose['vx_world'], pose['vy_world'], eyaw)
            vy = 0.0  # lateral speed negligible for parking

            # ── Command: always park ([0,0,1]) — forward slot used for parking
            fut_cmd = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # ── gt_ego_lcf_feat [9]
            lcf = np.zeros(9, dtype=np.float32)
            lcf[0] = vx                        # forward speed m/s (raw, NOT ×0.5)
            lcf[1] = vy                        # lateral speed m/s
            lcf[2] = pose['x_world']           # global x (pass-through)
            lcf[3] = pose['y_world']           # global y (pass-through)
            lcf[4] = pose.get('yaw_rate_rads', 0.0)  # yaw rate rad/s
            lcf[5] = VEHICLE_LENGTH
            lcf[6] = VEHICLE_WIDTH
            lcf[7] = vx                        # speed_head = same as vx
            lcf[8] = pose.get('steer_normalized', 0.0)

            cache[token] = {
                'gt_ego_lcf_feat':  lcf,
                'gt_ego_his_trajs': his_traj,
                'gt_ego_his_diff':  his_diff,
                'gt_ego_fut_cmd':   fut_cmd,
                'gt_ego_fut_trajs': fut_traj,
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"Wrote {len(cache)} cache entries from {len(episode_dirs)} episodes → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(_REPO_ROOT / 'data' / 'raw'))
    ap.add_argument('--out', default=str(_REPO_ROOT / 'data' / 'processed' / 'cached_parking_info.pkl'))
    args = ap.parse_args()
    build_cache(pathlib.Path(args.raw_dir), pathlib.Path(args.out))


if __name__ == '__main__':
    main()
