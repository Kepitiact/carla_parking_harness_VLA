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


def _carla_to_ego_velocity(vx_world, vy_world, yaw_deg):
    """Measured world velocity (CARLA m/s) -> ego frame [fwd_v, right_v], signed.

    CARLA->nuScenes world: nvx = vx_world, nvy = -vy_world; ego yaw th = -radians(yaw_deg).
    Rotate the world velocity into the ego frame; matches live_prompt.global_to_local_xy and
    the gt_ego_lcf_feat [fwd_v, right_v] convention (reverse => fwd_v negative).
    """
    th = -math.radians(yaw_deg)
    nvx = vx_world
    nvy = -vy_world
    c, s = math.cos(th), math.sin(th)
    fwd_v = c * nvx + s * nvy
    right_v = s * nvx - c * nvy
    return fwd_v, right_v


# ── Maneuver / target-slot labels (nuScenes global frame) ─────────────────────
# Mirrors generate_episodes.py.  New episodes record these in meta.json (already
# in nuScenes frame) → copied verbatim.  Pre-existing episodes lack them → derived
# from the final parked pose + approach geometry.

_DEFAULT_BAY_WIDTH_M = 3.0
_DEFAULT_BAY_LENGTH_M = 5.5


def _yaw_quat(heading_rad):
    """Quaternion [w, x, y, z] for a pure yaw rotation about +z."""
    return [math.cos(heading_rad / 2), 0.0, 0.0, math.sin(heading_rad / 2)]


def _make_target_slot(cx, cy, cz, heading_rad, width_m, length_m):
    """Build target_slot (polygon + pose) in the nuScenes global frame.

    Perpendicular bay rectangle: length along the parked heading, width across.
    Corners ordered front-left, front-right, rear-right, rear-left.
    """
    hl, hw = length_m / 2.0, width_m / 2.0
    fx, fy = math.cos(heading_rad), math.sin(heading_rad)    # forward unit
    lx, ly = -math.sin(heading_rad), math.cos(heading_rad)   # left unit
    polygon = [
        [cx + hl * fx + hw * lx, cy + hl * fy + hw * ly],
        [cx + hl * fx - hw * lx, cy + hl * fy - hw * ly],
        [cx - hl * fx - hw * lx, cy - hl * fy - hw * ly],
        [cx - hl * fx + hw * lx, cy - hl * fy + hw * ly],
    ]
    return {
        "polygon": polygon,
        "pose": {
            "translation": [cx, cy, cz],
            "rotation": _yaw_quat(heading_rad),
        },
    }


def _compute_side(ax, ay, ayaw, sx, sy):
    """Side the slot is on relative to the approach heading (nuScenes frame)."""
    dx, dy = sx - ax, sy - ay
    lateral = dx * math.sin(ayaw) - dy * math.cos(ayaw)  # d · right
    return "right" if lateral > 0 else "left"


# ── Actor GT: CARLA world 3D boxes -> per-frame lidar-frame gt_boxes ───────────
# gt_boxes[N,7] = (x, y, z, w, l, h, yaw) in the LIDAR frame. Lidar frame == ego
# frame (lidar2ego identity): x=forward, y=left, z=up. build_carla_nusc_tables.py
# must emit one sample_annotation per gt box so len(sample['anns'])==N per frame.

def _gt_from_actors(actor_frame, ego_pose):
    """Convert one frame's actor GT (CARLA world) to lidar-frame gt arrays.

    actor_frame: {'frame_idx', 'actors': [ {id,type_id,category,world_center,
                  yaw_deg,size_lwh,velocity}, ... ]} (from actors.json).
    ego_pose: the matching poses.json record (CARLA world x/y/z/yaw + velocity).

    Returns a dict of numpy arrays matching the infos GT schema.
    """
    actors = actor_frame.get('actors', []) if actor_frame else []

    # Ego pose in nuScenes world (flip y, negate yaw), lidar==ego.
    ex, ey = ego_pose['x_world'], -ego_pose['y_world']
    eyaw = -math.radians(ego_pose['yaw_deg'])
    ce, se = math.cos(eyaw), math.sin(eyaw)
    evx, evy = ego_pose.get('vx_world', 0.0), -ego_pose.get('vy_world', 0.0)

    boxes, names, velos, inds, ins_tokens, num_pts = [], [], [], [], [], []
    for a in actors:
        # Actor centre -> nuScenes world (flip y).
        wx, wy = a['world_center'][0], -a['world_center'][1]
        wz = a['world_center'][2]
        ayaw = -math.radians(a['yaw_deg'])
        # World -> ego/lidar: translate then rotate by -eyaw.
        dx, dy = wx - ex, wy - ey
        lx = ce * dx + se * dy      # forward (lidar x)
        ly = -se * dx + ce * dy     # left    (lidar y)
        lz = wz - ego_pose['z_world']
        lyaw = _norm_angle(ayaw - eyaw)

        # size_lwh = [length(x), width(y), height(z)] full dims. gt box wants
        # (w, l, h) per nuScenes convention (w across, l along heading, h up).
        length, width, height = a['size_lwh'][0], a['size_lwh'][1], a['size_lwh'][2]
        # Box centre lidar z is at the box CENTRE; nuScenes origin=(0.5,0.5,0.5)
        # is honoured by get_ann_info, so keep centre z.
        boxes.append([lx, ly, lz, width, length, height, lyaw])
        names.append(a['category'])

        # Actor velocity -> ego frame (relative to ego is NOT applied; nuScenes
        # gt_velocity is the object's own ground velocity in the ego/lidar frame).
        avx, avy = a.get('velocity', [0.0, 0.0])
        avx_n, avy_n = avx, -avy   # CARLA world -> nuScenes world
        vlx = ce * avx_n + se * avy_n
        vly = -se * avx_n + ce * avy_n
        velos.append([vlx, vly])

        inds.append(int(a['id']))
        ins_tokens.append(f"{a['id']:08d}")
        # No physical lidar; mark boxes as valid with a nominal point count so
        # use_valid_flag / num_lidar_pts>0 both keep them.
        num_pts.append(10)

    n = len(boxes)
    return {
        'gt_boxes':      np.array(boxes, dtype=np.float64).reshape(n, 7) if n else np.zeros((0, 7)),
        'gt_names':      np.array(names, dtype='<U32'),
        'gt_velocity':   np.array(velos, dtype=np.float64).reshape(n, 2) if n else np.zeros((0, 2)),
        'gt_inds':       np.array(inds, dtype=np.int64),
        'gt_ins_tokens': np.array(ins_tokens, dtype='<U32'),
        'valid_flag':    np.ones(n, dtype=bool),
        'num_lidar_pts': np.array(num_pts, dtype=np.int32),
    }


def _norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _maneuver_labels(meta, poses):
    """Return (maneuver_type, side, target_slot) for an episode.

    Uses meta.json fields when present (new collection format); otherwise derives
    them from the existing raw: reverse_perpendicular, target_slot from the final
    parked pose, side from the spawn→slot geometry.
    """
    if 'maneuver_type' in meta and 'side' in meta and 'target_slot' in meta:
        return meta['maneuver_type'], meta['side'], meta['target_slot']

    first, last = poses[0], poses[-1]
    sx_n, sy_n, _, _ = _carla_to_nuscenes_pose(
        first['x_world'], first['y_world'], first['z_world'], first['yaw_deg'])
    syaw_n = -math.radians(first['yaw_deg'])
    fx_n, fy_n, fz_n, _ = _carla_to_nuscenes_pose(
        last['x_world'], last['y_world'], last['z_world'], last['yaw_deg'])
    fheading_n = -math.radians(last['yaw_deg'])

    slot = meta.get('slot', {})
    width_m = slot.get('width_m', _DEFAULT_BAY_WIDTH_M)
    length_m = slot.get('length_m', _DEFAULT_BAY_LENGTH_M)

    target_slot = _make_target_slot(fx_n, fy_n, fz_n, fheading_n, width_m, length_m)
    side = _compute_side(sx_n, sy_n, syaw_n, fx_n, fy_n)
    return "reverse_perpendicular", side, target_slot


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

        # Per-frame actor GT (Task 3). Absent for pre-existing episodes -> empty GT.
        actors_path = ep_dir / 'actors.json'
        actor_frames = {}
        if actors_path.exists():
            with open(actors_path) as f:
                for rec in json.load(f):
                    actor_frames[rec['frame_idx']] = rec

        scene_token = meta['episode_id']
        n = len(poses)

        # Episode-constant maneuver labels (copied from meta or derived for old raw).
        maneuver_type, side, target_slot = _maneuver_labels(meta, poses)

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

            # Per-frame actor GT -> lidar-frame boxes (Task 3/6).
            gt = _gt_from_actors(actor_frames.get(i), pose)

            # can_bus: 18 floats.  [0]=x, [1]=y, [13]=speed, rest=0
            can_bus = np.zeros(18, dtype=np.float64)
            can_bus[0]  = nx
            can_bus[1]  = ny
            can_bus[13] = pose['speed_ms']

            # MEASURED ego-state (ego frame, nuScenes signs) for the cache generator to
            # use instead of the future-derived leak / hardcoded steering.
            ego_fwd_v, ego_right_v = _carla_to_ego_velocity(
                pose['vx_world'], pose['vy_world'], pose['yaw_deg'])

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
                # MEASURED ego-state (ego frame, m/s / rad·s⁻¹ / [-1,1]) — measured truth so
                # the cache generator can replace the future-derived ego-state + hardcoded steer.
                'ego_fwd_v':           ego_fwd_v,
                'ego_right_v':         ego_right_v,
                'ego_speed':           pose['speed_ms'],
                'ego_yaw_rate':        -pose['yaw_rate_rads'],
                'ego_steer':           pose['steer_normalized'],
                # Episode-level parking maneuver (conditions the VLA prompt).
                'maneuver_type':       maneuver_type,
                'side':                side,
                'target_slot':         target_slot,
                'gt_boxes':            gt['gt_boxes'],
                'gt_names':            gt['gt_names'],
                'gt_velocity':         gt['gt_velocity'],
                'gt_inds':             gt['gt_inds'],
                'gt_ins_tokens':       gt['gt_ins_tokens'],
                'valid_flag':          gt['valid_flag'],
                'num_lidar_pts':       gt['num_lidar_pts'],
                # Future/past agent trajectories are computed by the dataset's
                # traj_api from the DB sample_annotation chain (build_carla_nusc_tables);
                # store zero-motion placeholders sized to N so any direct consumer
                # sees the right shape (parked cars are static -> zero displacement).
                'fut_traj':            np.zeros((len(gt['gt_boxes']), 16, 2), dtype=np.float64),
                'fut_traj_valid_mask': np.ones((len(gt['gt_boxes']), 16, 2), dtype=np.float64),
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
