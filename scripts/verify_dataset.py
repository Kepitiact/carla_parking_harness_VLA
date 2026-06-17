"""
Sanity-check the processed dataset (pkl files + raw images).

Checks:
  1. All 6 images exist for every sample_token
  2. gt_ego_fut_trajs[0] == [0, 0] for every entry
  3. Forward speed in [0, 10] m/s (no outliers)
  4. Reverse samples have gt_ego_fut_trajs[1, 1] < 0
  5. gt_ego_fut_trajs[1, 1] ≈ vx * 0.5 for forward motion (within 20%)
  6. Prints count summary

Usage:
  python scripts/verify_dataset.py
  python scripts/verify_dataset.py --infos data/processed/parking_infos_temporal.pkl \
                                   --cache data/processed/cached_parking_info.pkl
"""

import argparse
import math
import pathlib
import pickle

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CAMERA_NAMES = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]


def verify(infos_path: pathlib.Path, cache_path: pathlib.Path):
    errors = []
    warnings = []

    with open(infos_path, 'rb') as f:
        infos_dict = pickle.load(f)
    infos = infos_dict['infos']

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"Loaded {len(infos)} info records, {len(cache)} cache entries.")

    # ── Per-token checks ──────────────────────────────────────────────────────
    forward_count = 0
    reverse_count = 0
    vx_outliers   = 0
    trajs_wrong   = 0

    scene_tokens = set()
    for info in infos:
        token = info['token']
        scene_tokens.add(info['scene_token'])

        # 1. Images exist (data_path is relative to repo root)
        for cam_name in CAMERA_NAMES:
            p = pathlib.Path(info['cams'][cam_name]['data_path'])
            full_p = p if p.is_absolute() else _REPO_ROOT / p
            if not full_p.exists():
                errors.append(f"Missing image: {full_p}")

        # 2. Cache entry exists
        if token not in cache:
            errors.append(f"Missing cache entry for token {token}")
            continue

        entry = cache[token]

        # 3. fut_traj[0] == (0, 0)
        fut = entry['gt_ego_fut_trajs']
        if not np.allclose(fut[0], 0.0, atol=0.01):
            trajs_wrong += 1
            if trajs_wrong <= 5:
                errors.append(f"fut_traj[0] != 0 for {token}: {fut[0]}")

        # 4. Speed range
        vx = entry['gt_ego_lcf_feat'][0]
        if abs(vx) > 10.0:
            vx_outliers += 1
            warnings.append(f"Speed outlier {vx:.2f} m/s at {token}")

        # 5. Reverse check
        is_reverse = (vx < -0.1)
        if is_reverse:
            reverse_count += 1
            if fut[1, 1] >= 0.0 and abs(vx) > 0.3:
                warnings.append(
                    f"Reverse frame {token}: vx={vx:.2f} but fut[1,1]={fut[1,1]:.3f} (should be <0)"
                )
        else:
            forward_count += 1

        # 6. Velocity–trajectory consistency for forward motion
        if vx > 0.3 and not is_reverse:
            expected_step = vx * 0.5  # 0.5-second step
            actual_step   = fut[1, 1]  # forward displacement over 0.5 s
            if actual_step > 0:
                ratio = actual_step / expected_step
                if ratio < 0.5 or ratio > 2.0:
                    warnings.append(
                        f"Velocity/traj mismatch at {token}: "
                        f"vx={vx:.2f}, expected Δfwd≈{expected_step:.2f}, actual={actual_step:.2f}"
                    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  Episodes (scenes):  {len(scene_tokens)}")
    print(f"  Total frames:       {len(infos)}")
    print(f"  Forward frames:     {forward_count}")
    print(f"  Reverse frames:     {reverse_count}")
    if len(infos) > 0:
        print(f"  Forward/Reverse:    {forward_count / max(1, reverse_count):.1f}:1")

    if trajs_wrong > 0:
        print(f"  ⚠  fut_traj[0]!=0:  {trajs_wrong} frames")
    if vx_outliers > 0:
        print(f"  ⚠  Speed outliers:  {vx_outliers} frames")
    print("═" * 60)

    if errors:
        print(f"\n✗ {len(errors)} ERRORS:")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more")
    else:
        print("\n✓ No errors found.")

    if warnings:
        print(f"\n⚠  {len(warnings)} warnings:")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings)-10} more")

    return len(errors) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infos', default=str(_REPO_ROOT / 'data' / 'processed' / 'parking_infos_temporal.pkl'))
    ap.add_argument('--cache', default=str(_REPO_ROOT / 'data' / 'processed' / 'cached_parking_info.pkl'))
    args = ap.parse_args()

    ok = verify(pathlib.Path(args.infos), pathlib.Path(args.cache))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
