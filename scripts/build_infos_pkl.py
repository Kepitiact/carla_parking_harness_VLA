"""
Build parking_infos_temporal.pkl from raw episode data.

Equivalent to nuscenes_infos_temporal_mini.pkl consumed by OpenDriveVLA's
NuScenesE2EDataset loader.

All fields required by NuScenesE2EDataset.get_data_info and get_ann_info are
included.  Synthetic lidar stubs (empty, identity) let the loader run without
a physical lidar.  Camera sensor2lidar transforms use real nuScenes v1.0-mini
calibrations as a proxy (closest available reference for the model's training
distribution).

Usage:
  python scripts/build_infos_pkl.py
  python scripts/build_infos_pkl.py --raw_dir data/raw --out data/processed/parking_infos_temporal.pkl
  python scripts/build_infos_pkl.py --absolute-paths  # store absolute image paths
"""

import argparse
import math
import pathlib
import pickle

import numpy as np
from pyquaternion import Quaternion

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CAMERA_NAMES = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]

# Intrinsic for 1600×900 at FOV=70°
_f = 1600 / (2 * math.tan(math.radians(70 / 2)))
CAM_INTRINSIC = np.array([
    [_f,  0, 800],
    [ 0, _f, 450],
    [ 0,  0,   1],
], dtype=np.float64)

# Actual CARLA camera mount config (from data collection script).
# CARLA actor frame: x=forward, y=right, z=up (UE4 left-hand).
# yaw: positive = clockwise (right) in CARLA.
_CARLA_CAMERAS = {
    "CAM_FRONT":       {"x":  1.5, "y":  0.0, "z": 1.5, "yaw":   0},
    "CAM_FRONT_LEFT":  {"x":  1.0, "y": -0.9, "z": 1.5, "yaw": -55},
    "CAM_FRONT_RIGHT": {"x":  1.0, "y":  0.9, "z": 1.5, "yaw":  55},
    "CAM_BACK":        {"x": -1.5, "y":  0.0, "z": 1.5, "yaw": 180},
    "CAM_BACK_LEFT":   {"x": -1.0, "y": -0.9, "z": 1.5, "yaw":-110},
    "CAM_BACK_RIGHT":  {"x": -1.0, "y":  0.9, "z": 1.5, "yaw": 110},
}


def _carla_cam_extrinsic(cam):
    """Convert CARLA camera mount config to nuScenes sensor2lidar (sensor→ego).

    nuScenes ego: x=forward, y=left, z=up (right-hand).
    Camera optical: x=right, y=down, z=forward.

    Translation: flip y (CARLA y=right → nuScenes y=left).
    Rotation R satisfies P_ego = R @ P_cam.
      nusc_yaw θ = −carla_yaw (left-hand→right-hand flip):
        R = [[sinθ, 0, cosθ], [−cosθ, 0, sinθ], [0, −1, 0]]
    """
    theta = -math.radians(cam["yaw"])
    s, c = math.sin(theta), math.cos(theta)
    R = np.array([[ s, 0,  c],
                  [-c, 0,  s],
                  [ 0, -1, 0]], dtype=np.float64)
    T = np.array([cam["x"], -cam["y"], cam["z"]], dtype=np.float64)
    return R, T


def _carla_to_nuscenes_pose(x_carla, y_carla, z_carla, yaw_deg):
    """Convert CARLA world pose to nuScenes world frame.

    CARLA uses left-hand Z-up (y-axis flipped vs nuScenes right-hand).
    Flip: nx = cx, ny = -cy.  Yaw: nuscenes_yaw = -carla_yaw.
    """
    nx = x_carla
    ny = -y_carla
    nz = z_carla
    nuscenes_yaw_rad = -math.radians(yaw_deg)
    q = Quaternion(axis=[0, 0, 1], angle=nuscenes_yaw_rad)
    return nx, ny, nz, [q.w, q.x, q.y, q.z]


def build_infos(raw_dir: pathlib.Path, out_path: pathlib.Path,
                absolute_paths: bool = False):
    import json

    infos = []
    episode_dirs = sorted(raw_dir.glob('episode_*'))
    if not episode_dirs:
        raise RuntimeError(f"No episodes found in {raw_dir}")

    for ep_dir in episode_dirs:
        meta_path = ep_dir / 'meta.json'
        poses_path = ep_dir / 'poses.json'
        if not meta_path.exists() or not poses_path.exists():
            print(f"Skipping {ep_dir.name}: missing meta.json or poses.json")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        with open(poses_path) as f:
            poses = json.load(f)

        scene_token = meta['episode_id']
        n = len(poses)

        for i, pose in enumerate(poses):
            token = f"{scene_token}_f{i:04d}"
            prev_token = f"{scene_token}_f{(i-1):04d}" if i > 0 else ""
            next_token = f"{scene_token}_f{(i+1):04d}" if i < n - 1 else ""

            frame_dir = ep_dir / 'frames' / f'frame_{i:04d}'
            cams = {}
            for cam_name in CAMERA_NAMES:
                img_path = frame_dir / f'{cam_name}.jpg'
                path_str = (str(img_path.resolve()) if absolute_paths
                            else str(img_path.resolve().relative_to(_REPO_ROOT)))
                R, T = _carla_cam_extrinsic(_CARLA_CAMERAS[cam_name])
                cams[cam_name] = {
                    'data_path':                path_str,
                    'cam_intrinsic':            CAM_INTRINSIC,
                    'sensor2lidar_rotation':    R,
                    'sensor2lidar_translation': T,
                }

            nx, ny, nz, quat = _carla_to_nuscenes_pose(
                pose['x_world'], pose['y_world'], pose['z_world'], pose['yaw_deg']
            )

            # can_bus: 18 floats.  [0]=x, [1]=y, [13]=speed, rest=0
            can_bus = np.zeros(18, dtype=np.float64)
            can_bus[0]  = nx
            can_bus[1]  = ny
            can_bus[13] = pose['speed_ms']

            info = {
                'token':                  token,
                'scene_token':            scene_token,
                'prev':                   prev_token,
                'next':                   next_token,
                'frame_idx':              i,
                'timestamp':              pose['timestamp_us'],
                'cams':                   cams,
                'ego2global_translation': [nx, ny, nz],
                'ego2global_rotation':    quat,
                'can_bus':                can_bus,
                # Lidar stubs: no physical lidar; ego frame = lidar frame.
                'lidar_path':             '',
                'sweeps':                 [],
                'lidar2ego_rotation':     [1.0, 0.0, 0.0, 0.0],
                'lidar2ego_translation':  np.array([0.0, 0.0, 0.0]),
                # Reverse gear flag (used by cached info generator for command labeling).
                'reverse':             bool(pose['reverse']),
                # No other agents in the scene.
                'reverse':             bool(pose.get('reverse', False)),
                'gt_boxes':            np.zeros((0, 7),  dtype=np.float64),
                'gt_names':            np.array([],      dtype='<U32'),
                'gt_velocity':         np.zeros((0, 2),  dtype=np.float64),
                'gt_inds':             np.zeros(0,       dtype=np.int64),
                'gt_ins_tokens':       np.array([],      dtype='<U32'),
                'valid_flag':          np.zeros(0,       dtype=bool),
                'num_lidar_pts':       np.zeros(0,       dtype=np.int32),
                'fut_traj':            np.zeros((0, 16, 2), dtype=np.float64),
                'fut_traj_valid_mask': np.zeros((0, 16, 2), dtype=np.float64),
            }
            infos.append(info)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'infos': infos, 'metadata': {'version': 'v1.0-carla'}}, f)

    print(f"Wrote {len(infos)} info records from {len(episode_dirs)} episodes → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(_REPO_ROOT / 'data' / 'raw'))
    ap.add_argument('--out', default=str(_REPO_ROOT / 'data' / 'processed' / 'parking_infos_temporal.pkl'))
    ap.add_argument('--absolute-paths', action='store_true',
                    help='Store absolute image paths (required when pkl is used '
                         'from a different working directory)')
    args = ap.parse_args()
    build_infos(pathlib.Path(args.raw_dir), pathlib.Path(args.out),
                absolute_paths=args.absolute_paths)


if __name__ == '__main__':
    main()
