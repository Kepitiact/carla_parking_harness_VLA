"""
Create a minimal nuScenes metadata stub so that NuScenes(version='v1.0-mini', dataroot=...)
can initialise without crashing, even though there is no real nuScenes database.

The loader uses these table files at startup but does not actually read their
records when infos come from a pre-built pkl (ann_file= pointing to our pkl).

Usage:
  python scripts/make_nuscenes_stub.py --data_root data/
"""

import argparse
import json
import pathlib

# Tables that nuScenes.__init__ opens and parses
EMPTY_TABLES = [
    'attribute', 'calibrated_sensor', 'category', 'ego_pose',
    'instance', 'log', 'map', 'sample', 'sample_annotation',
    'sample_data', 'scene', 'sensor', 'visibility',
]


def make_stub(data_root: pathlib.Path):
    stub_dir = data_root / 'v1.0-mini'
    stub_dir.mkdir(parents=True, exist_ok=True)

    for table in EMPTY_TABLES:
        path = stub_dir / f'{table}.json'
        if not path.exists():
            path.write_text('[]')

    # version.txt is also checked
    version_file = stub_dir / 'version.txt'
    if not version_file.exists():
        version_file.write_text('v1.0-mini\n')

    print(f"Created nuScenes stub at {stub_dir}")
    print(f"  Tables: {', '.join(EMPTY_TABLES)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/')
    args = ap.parse_args()
    make_stub(pathlib.Path(args.data_root).resolve())


if __name__ == '__main__':
    main()
