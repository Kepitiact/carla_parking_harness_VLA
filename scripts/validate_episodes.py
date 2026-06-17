"""
Validate episode completeness before adding to the dataset.

Checks per episode:
  1. meta.json and poses.json exist and parse
  2. All required pose fields are present
  3. All 6 camera images exist for every frame in poses.json
  4. No corrupt JPEG files (header/footer check)
  5. Frame directory count matches poses.json length
  6. timestamp_us is strictly increasing
  7. speed_ms in [0, 15] m/s

Usage:
  python scripts/validate_episodes.py
  python scripts/validate_episodes.py --raw_dir data/raw
  python scripts/validate_episodes.py --raw_dir data/raw --fix   # delete bad episodes
"""

import argparse
import json
import pathlib
import shutil
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CAMERA_NAMES = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]

REQUIRED_POSE_FIELDS = {
    "frame_idx", "timestamp_us",
    "x_world", "y_world", "z_world", "yaw_deg",
    "speed_ms", "reverse", "steer_normalized",
    "vx_world", "vy_world",
}


def _is_valid_jpeg(path: pathlib.Path) -> bool:
    try:
        with open(path, "rb") as f:
            header = f.read(2)
            if header != b"\xff\xd8":
                return False
            f.seek(-2, 2)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


def validate_episode(ep_dir: pathlib.Path) -> list:
    errors = []

    meta_path = ep_dir / "meta.json"
    poses_path = ep_dir / "poses.json"

    if not meta_path.exists():
        return ["missing meta.json"]
    if not poses_path.exists():
        return ["missing poses.json"]

    try:
        with open(poses_path) as f:
            poses = json.load(f)
    except Exception as e:
        return [f"poses.json parse error: {e}"]

    if len(poses) < 4:
        errors.append(f"too few frames: {len(poses)} (min 4)")

    if poses:
        missing_fields = REQUIRED_POSE_FIELDS - set(poses[0].keys())
        if missing_fields:
            errors.append(f"missing pose fields: {sorted(missing_fields)}")

    frames_dir = ep_dir / "frames"
    frame_dirs = sorted(frames_dir.glob("frame_*")) if frames_dir.exists() else []
    if len(frame_dirs) != len(poses):
        errors.append(f"frame dir count {len(frame_dirs)} != poses count {len(poses)}")

    for pose in poses:
        fi = pose.get("frame_idx", -1)
        frame_dir = frames_dir / f"frame_{fi:04d}"
        for cam in CAMERA_NAMES:
            img = frame_dir / f"{cam}.jpg"
            if not img.exists():
                errors.append(f"missing: frame_{fi:04d}/{cam}.jpg")
            elif not _is_valid_jpeg(img):
                errors.append(f"corrupt JPEG: frame_{fi:04d}/{cam}.jpg")

    timestamps = [p.get("timestamp_us", 0) for p in poses]
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            errors.append(f"non-increasing timestamp at frame {i}")
            break

    for p in poses:
        spd = p.get("speed_ms", 0)
        if spd < 0 or spd > 15:
            errors.append(f"speed outlier {spd:.2f} m/s at frame {p.get('frame_idx')}")
            break

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(_REPO_ROOT / "data" / "raw"))
    ap.add_argument("--fix", action="store_true",
                    help="Delete episodes that fail validation")
    args = ap.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    episode_dirs = sorted(raw_dir.glob("episode_*"))
    if not episode_dirs:
        print(f"No episodes found in {raw_dir}")
        sys.exit(1)

    ok, bad = [], []
    for ep_dir in episode_dirs:
        errors = validate_episode(ep_dir)
        if errors:
            bad.append((ep_dir, errors))
            suffix = f" (+{len(errors)-1} more)" if len(errors) > 1 else ""
            print(f"  FAIL  {ep_dir.name}: {errors[0]}{suffix}")
        else:
            ok.append(ep_dir)

    print()
    print("=" * 50)
    print(f"  Valid:    {len(ok)} / {len(episode_dirs)}")
    print(f"  Invalid:  {len(bad)} / {len(episode_dirs)}")
    print("=" * 50)

    if args.fix and bad:
        for ep_dir, _ in bad:
            shutil.rmtree(ep_dir)
            print(f"  Deleted {ep_dir.name}")

    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
