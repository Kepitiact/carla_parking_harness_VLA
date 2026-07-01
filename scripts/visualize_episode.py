"""
Visualize a generated episode: camera grid + ego trajectory.

Usage:
  # last saved episode (auto-detected):
  python scripts/visualize_episode.py

  # specific episode:
  python scripts/visualize_episode.py --episode episode_0003

  # also overlay cache trajectories:
  python scripts/visualize_episode.py --episode episode_0003 --cache
"""

import argparse
import json
import math
import pathlib
import pickle

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RAW = _REPO_ROOT / 'data' / 'raw'
_PROCESSED = _REPO_ROOT / 'data' / 'processed'

CAMERA_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]


def _load_image(path, scale=0.2):
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((90, 160, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def plot_bev(ep_dir: pathlib.Path, show=True):
    """Large standalone bird's-eye view of the whole maneuver.

    Shows the full driven trajectory, the obstacle-aware A* expert path, the goal
    slot, spawned obstacles (if recorded in meta), and start/final markers — big
    enough to actually read, unlike the cramped subplot in the camera grid.
    """
    with open(ep_dir / 'meta.json') as f:
        meta = json.load(f)
    with open(ep_dir / 'poses.json') as f:
        poses = json.load(f)

    xs = [p['x_world'] for p in poses]
    ys = [p['y_world'] for p in poses]
    yaws = [math.radians(p['yaw_deg']) for p in poses]

    fig, ax = plt.subplots(figsize=(11, 11))

    # Spawned obstacles (parked NPCs) the planner avoided, if the collector
    # recorded them in meta['obstacles_world'] (list of polygons or {'polygon':...}).
    for ob in meta.get('obstacles_world', []):
        poly = ob.get('polygon') if isinstance(ob, dict) else ob
        if poly:
            ax.add_patch(plt.Polygon(poly, closed=True, facecolor='0.6',
                                     edgecolor='k', alpha=0.6, zorder=1))

    # A* expert path in CARLA world frame (meta['astar_path_world'] = [[x, y], ...]).
    astar = meta.get('astar_path_world', [])
    if astar:
        ax.plot([p[0] for p in astar], [p[1] for p in astar], '-',
                color='royalblue', lw=2.5, zorder=3, label='A* expert path')

    # Driven trajectory, coloured by frame index.
    sc = ax.scatter(xs, ys, c=range(len(poses)), cmap='plasma', s=28, zorder=4)
    plt.colorbar(sc, ax=ax, label='frame index', shrink=0.8)

    # Ego heading arrows every few frames so the maneuver (fwd->reverse) is legible.
    step = max(1, len(poses) // 25)
    for i in range(0, len(poses), step):
        ax.arrow(xs[i], ys[i], 0.8 * math.cos(yaws[i]), 0.8 * math.sin(yaws[i]),
                 head_width=0.25, head_length=0.25, fc='dimgray', ec='dimgray',
                 alpha=0.6, zorder=5)

    # Goal slot rectangle (meta polygon is nuScenes frame: y flipped vs CARLA).
    slot_poly = meta.get('target_slot', {}).get('polygon')
    if slot_poly:
        ax.add_patch(plt.Polygon([(x, -y) for (x, y) in slot_poly], closed=True,
                                 facecolor='none', edgecolor='green', lw=2.5,
                                 zorder=6, label='goal slot'))
    ax.scatter([meta['slot']['cx_world']], [meta['slot']['cy_world']],
               c='lime', s=180, marker='*', zorder=7, label='slot center')
    ax.scatter([xs[0]], [ys[0]], c='cyan', s=120, marker='s', zorder=7, label='spawn')
    ax.scatter([xs[-1]], [ys[-1]], c='red', s=160, marker='X', zorder=7, label='final')

    final_dist = math.hypot(xs[-1] - meta['slot']['cx_world'],
                            ys[-1] - meta['slot']['cy_world'])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('CARLA x (m)')
    ax.set_ylabel('CARLA y (m)')
    ax.set_title(f"{meta['episode_id']} — {meta['map']} — obstacle-aware A* expert\n"
                 f"{len(poses)} frames, final dist to slot center: {final_dist:.2f} m")
    ax.legend(loc='best', fontsize=10)
    fig.tight_layout()
    out = ep_dir / 'bev.png'
    fig.savefig(str(out), dpi=140)
    print(f"Saved BEV → {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_episode(ep_dir: pathlib.Path, cache_path=None, frame_idx=None, show=True):
    with open(ep_dir / 'meta.json') as f:
        meta = json.load(f)
    with open(ep_dir / 'poses.json') as f:
        poses = json.load(f)

    n = len(poses)
    if frame_idx is None:
        frame_idx = n // 2  # middle frame

    print(f"Episode: {meta['episode_id']}")
    print(f"  Map:           {meta['map']}")
    print(f"  Type:          {meta['parking_type']}")
    print(f"  Total frames:  {n}")
    print(f"  Approach:      {meta['approach_mode']}")
    print(f"  Heading error: {math.degrees(meta['heading_error_at_spawn_rad']):.1f}°")
    print(f"  Slot:          ({meta['slot']['cx_world']:.1f}, {meta['slot']['cy_world']:.1f})")
    print(f"  Showing frame: {frame_idx}/{n-1}")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{meta['episode_id']}  |  {meta['parking_type']}  |  "
        f"{meta['approach_mode']} approach  |  frame {frame_idx}/{n-1}",
        fontsize=12,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.1)

    # ── Row 1: 6 camera images ────────────────────────────────────────────────
    cam_axes_positions = [
        (0, 0), (0, 1), (0, 2),  # front-left, front, front-right
        (1, 0), (1, 1), (1, 2),  # back-left, back, back-right
    ]
    frame_dir = ep_dir / 'frames' / f'frame_{frame_idx:04d}'
    for (row, col), cam_name in zip(cam_axes_positions, CAMERA_ORDER):
        ax = fig.add_subplot(gs[row, col])
        img_path = frame_dir / f'{cam_name}.jpg'
        img = _load_image(img_path, scale=0.18)
        ax.imshow(img)
        ax.set_title(cam_name.replace('CAM_', ''), fontsize=8)
        ax.axis('off')

    # ── Column 4: world-frame trajectory plot ─────────────────────────────────
    ax_world = fig.add_subplot(gs[:2, 3])
    xs = [p['x_world'] for p in poses]
    ys = [p['y_world'] for p in poses]

    # Obstacle-aware A* expert path in CARLA world frame (meta['astar_path_world']).
    astar = meta.get('astar_path_world', [])
    if astar:
        ax_a = [p[0] for p in astar]
        ay_a = [p[1] for p in astar]
        ax_world.plot(ax_a, ay_a, '-', color='royalblue', lw=2.0,
                      zorder=2, label='A* expert path')

    # Spawned obstacles (parked NPCs) the planner avoided, from meta if present.
    for ob in meta.get('obstacles_world', []):
        poly = ob.get('polygon') if isinstance(ob, dict) else ob
        if poly:
            ax_world.add_patch(plt.Polygon(poly, closed=True, facecolor='0.6',
                                           edgecolor='k', alpha=0.6, zorder=1))

    # colour by frame index
    sc = ax_world.scatter(xs, ys, c=range(n), cmap='plasma', s=20, zorder=3)
    ax_world.scatter([meta['slot']['cx_world']], [meta['slot']['cy_world']],
                     c='lime', s=120, marker='*', zorder=5, label='slot')
    ax_world.scatter([poses[0]['x_world']], [poses[0]['y_world']],
                     c='cyan', s=80, marker='s', zorder=5, label='spawn')
    ax_world.scatter([poses[frame_idx]['x_world']], [poses[frame_idx]['y_world']],
                     c='red', s=80, marker='o', zorder=5, label=f'frame {frame_idx}')
    plt.colorbar(sc, ax=ax_world, label='frame idx')
    ax_world.set_xlabel('X (world)')
    ax_world.set_ylabel('Y (world)')
    ax_world.set_title('World trajectory (A* expert vs driven)', fontsize=9)
    ax_world.legend(fontsize=7)
    ax_world.set_aspect('equal')
    ax_world.grid(True, alpha=0.3)

    # ── Row 3: ego-local trajectory from cache (if available) ─────────────────
    ax_ego = fig.add_subplot(gs[2, :2])
    ax_speed = fig.add_subplot(gs[2, 2:])

    if cache_path and cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        # Collect ego-local future trajs for all frames in this episode
        scene = meta['episode_id']
        fut_ys, fut_xs = [], []
        speeds = []
        for i in range(n):
            token = f"{scene}_f{i:04d}"
            if token in cache:
                e = cache[token]
                fut = e['gt_ego_fut_trajs']   # (7, 2) — (right, forward)
                fut_xs.append(fut[:, 0])       # right
                fut_ys.append(fut[:, 1])       # forward
                speeds.append(e['gt_ego_lcf_feat'][0])

        for i, (fx, fy) in enumerate(zip(fut_xs, fut_ys)):
            alpha = 0.3 if i != frame_idx else 1.0
            lw = 0.5 if i != frame_idx else 2.0
            ax_ego.plot(fx, fy, alpha=alpha, lw=lw, color='steelblue')

        ax_ego.axhline(0, color='gray', lw=0.5)
        ax_ego.axvline(0, color='gray', lw=0.5)
        ax_ego.set_xlabel('right (m)')
        ax_ego.set_ylabel('forward (m)')
        ax_ego.set_title('Ego-local future trajectories (all frames)', fontsize=9)
        ax_ego.set_aspect('equal')
        ax_ego.grid(True, alpha=0.3)

        # Speed profile
        ax_speed.plot(speeds, lw=1.5, color='darkorange')
        ax_speed.axhline(0, color='gray', lw=0.5)
        ax_speed.axvline(frame_idx, color='red', lw=1, linestyle='--', label=f'frame {frame_idx}')
        ax_speed.set_xlabel('frame')
        ax_speed.set_ylabel('vx (m/s)')
        ax_speed.set_title('Forward speed profile', fontsize=9)
        ax_speed.legend(fontsize=7)
        ax_speed.grid(True, alpha=0.3)
    else:
        ax_ego.text(0.5, 0.5, 'Run build_cache_pkl.py first\nto see ego-local trajectories',
                    ha='center', va='center', transform=ax_ego.transAxes, fontsize=9, color='gray')
        ax_speed.text(0.5, 0.5, 'No cache available',
                      ha='center', va='center', transform=ax_speed.transAxes, fontsize=9, color='gray')

    out_path = ep_dir / f'viz_frame{frame_idx:04d}.png'
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    print(f"Saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', default=None,
                    help='Episode dir name, e.g. episode_0003. Default: last episode.')
    ap.add_argument('--frame', type=int, default=None,
                    help='Frame index to show cameras for. Default: middle frame.')
    ap.add_argument('--root', default=None,
                    help='Episodes root dir (default: data/raw).')
    ap.add_argument('--bev', action='store_true',
                    help='Render only the large bird\'s-eye path view (bev.png).')
    ap.add_argument('--no-show', action='store_true',
                    help='Save figures without opening an interactive window.')
    ap.add_argument('--cache', action='store_true',
                    help='Overlay ego-local trajectories from cached_parking_info.pkl.')
    args = ap.parse_args()

    root = pathlib.Path(args.root) if args.root else _RAW
    if args.episode:
        ep_dir = root / args.episode
    else:
        episodes = sorted(root.glob('episode_*'))
        if not episodes:
            print(f"No episodes found in {root}")
            raise SystemExit(1)
        ep_dir = episodes[-1]
        print(f"Auto-selected: {ep_dir.name}")

    cache_path = _PROCESSED / 'cached_parking_info.pkl' if args.cache else None
    if args.bev:
        plot_bev(ep_dir, show=not args.no_show)
    else:
        plot_episode(ep_dir, cache_path=cache_path, frame_idx=args.frame,
                     show=not args.no_show)


if __name__ == '__main__':
    main()
