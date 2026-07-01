"""Single source of truth for closed-loop harness paths, hosts, and tunables.

Nothing downstream hardcodes a path or host: point the harness at a different
checkpoint or CARLA server by editing the defaults here or passing CLI/env
overrides. Pure stdlib so BOTH venvs can import it unchanged:
  - Py3.8  CARLA client  (parking_data_gen/venv)
  - Py3.10 model server  (openvla_nuscenes/.venv)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

_HOME = Path.home()
_PROJECTS = _HOME / "projects"

# Defaults = the "Key paths" from CLOSED_LOOP_HARNESS_HANDOFF.md.
DEF_CHECKPOINT = _PROJECTS / "openvla_nuscenes/checkpoints/OpenDriveVLA-0.5B-carla/merged"
DEF_MODEL_REPO = _PROJECTS / "openvla_nuscenes/OpenDriveVLA"
DEF_MODEL_VENV = _PROJECTS / "openvla_nuscenes/.venv"
# UniAD config path, relative to the model repo root (the CARLA calibration rig).
DEF_UNIAD_CONFIG = "projects/configs/stage1_track_map/carla_parking.py"
DEF_OUT_DIR = _PROJECTS / "parking_data_gen/harness/runs"


@dataclass
class Config:
    # ── model server (runs in the Py3.10 venv, owns the GPU) ──
    checkpoint: Path = DEF_CHECKPOINT
    model_repo: Path = DEF_MODEL_REPO
    model_venv: Path = DEF_MODEL_VENV
    uniad_config: str = DEF_UNIAD_CONFIG
    # Known-good uniad_data fixture used to warm up the model (see live_uniad.warmup).
    # Generate once with harness/model_server/capture_reference.py.
    uniad_template: Path = DEF_OUT_DIR / "uniad_ref/uniad_data.pth"

    # ── bridge (localhost socket between CARLA client and model server) ──
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 5557

    # ── CARLA (Py3.8 client) ──
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    town: str = "Town04_Opt"

    # ── episode / loop tunables ──
    # Success tolerances vs the TARGET SLOT POSE (mirrors generate_episodes.py).
    goal_dist_m: float = 0.5
    goal_rot_deg: float = 3.5
    # Receding-horizon cadence. The model predicts a 3s/6-waypoint plan; replan once per horizon.
    replan_dt: float = 3.0
    timeout_s: float = 180.0

    # ── output ──
    out_dir: Path = DEF_OUT_DIR

    # -- helpers -------------------------------------------------------------
    def __post_init__(self) -> None:
        # Normalise path-typed fields (CLI/env may pass strings).
        for name in ("checkpoint", "model_repo", "model_venv", "out_dir", "uniad_template"):
            setattr(self, name, Path(getattr(self, name)).expanduser())

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Expose every field as an optional --flag; env vars (HARNESS_<FIELD>)
        fill in when a flag is absent."""
        defaults = cls()
        for f in fields(cls):
            env = os.environ.get(f"HARNESS_{f.name.upper()}")
            default = env if env is not None else getattr(defaults, f.name)
            argtype = int if f.type == "int" or isinstance(default, int) and not isinstance(default, bool) else (
                float if isinstance(default, float) else str)
            parser.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name,
                                default=default, type=argtype)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        kwargs = {f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)}
        return cls(**kwargs)


def parse(argv=None) -> Config:
    """Convenience: build a Config from CLI args (used by standalone entry points)."""
    p = argparse.ArgumentParser()
    Config.add_cli_args(p)
    return Config.from_args(p.parse_args(argv))
