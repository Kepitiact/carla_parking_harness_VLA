"""
OpenDriveVLA config overrides for the CARLA parking dataset.

Copy or symlink this file into your OpenDriveVLA configs/ directory, then
pass it to the training script instead of the default nuScenes config.

NuScenes loader compatibility note:
  NuScenesE2EDataset calls NuScenes(version=..., dataroot=...) internally.
  For CARLA data, provide a minimal stub at data_root/v1.0-mini/ by running:
    python scripts/make_nuscenes_stub.py --data_root <DATA_ROOT>
  This creates empty JSON tables so the loader initialises without error.
"""

import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_DATA = _HERE.parent / 'data' / 'processed'

# ── Paths ─────────────────────────────────────────────────────────────────────
data_root      = str(_HERE.parent / 'data')          # parent of v1.0-mini stub
ann_file       = str(_DATA / 'parking_infos_temporal.pkl')
cache_path     = str(_DATA / 'cached_parking_info.pkl')

# ── NuScenes version string expected by the loader ────────────────────────────
version        = 'v1.0-mini'

# ── Planning / queue parameters (must match what OpenDriveVLA expects) ────────
planning_steps = 6   # future trajectory length (indices 1–6 of gt_ego_fut_trajs)
queue_length   = 5   # history length (frames 0–4 of gt_ego_his_trajs)

# ── Image config (match camera rig used during generation) ───────────────────
input_size     = (1600, 900)

# ── Training overrides ────────────────────────────────────────────────────────
# Set these to match the fine-tuning run (adjust as needed)
max_epochs     = 10
batch_size     = 1
num_workers    = 4
