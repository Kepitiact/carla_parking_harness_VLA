"""Rebuild the model's user-message prompt live, every control step.

The model was retrained on a maneuver-level mission goal + target slot. This module
reconstructs the exact `data_dict` that build_llava_conversation.generate_user_message
consumes, from LIVE CARLA state (no NuScenes DB), then reuses build_llava_conversation
itself so the assembled string byte-matches the training format by construction.

All poses are in the nuScenes/OpenDRIVE global frame (y_odr = -y_carla); the ego-local
output convention is x=right, y=forward (matching the model's waypoints + the controller).

Source of truth mirrored here (yaw-only forms, validated against cached_parking_info.pkl):
  scripts/generate_cached_nuscenes_info.py -> global_to_local_xy / slot_to_local / build_entry

The ego-state (gt_ego_lcf_feat[0,1,4,7,8] = fwd_v, right_v, yaw_rate, speed, steer) comes from the
MEASURED CARLA state passed in `ego`, the same per-frame measured source build_entry now reads
(build_infos_pkl ego_* fields). This matches the retrained checkpoint, which was trained on measured
ego-state + real steering — the earlier future-derived "next_step" reconstruction and the steer=0.0
workaround are gone (the leak they imitated no longer exists). Validated by tests/test_live_prompt_bytematch.
"""
from __future__ import annotations

import numpy as np

LENGTH_M = 4.5   # gt_ego_lcf_feat[5], matches build_entry
WIDTH_M = 1.8    # gt_ego_lcf_feat[6]
HISTORY_STEPS = 4  # 4 past @0.5s + current origin = 5-point history


def normalize_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def global_to_local_xy(gx, gy, ox, oy, oyaw):
    """World (OpenDRIVE) point -> ego-local [right, forward]. Yaw-only equivalent of
    generate_cached_nuscenes_info.global_to_local_xy (which uses Quaternion.rotation_matrix.T).
    Verified against cached slot_local: ego(285.6,232.65,90deg) + slot(290.94,226.39)
    -> right=5.34, forward=-6.26."""
    dx, dy = gx - ox, gy - oy
    c, s = np.cos(oyaw), np.sin(oyaw)
    forward = c * dx + s * dy
    right = s * dx - c * dy
    return float(right), float(forward)


def build_data_dict(ego: dict, ego_history, slot: dict, maneuver_type: str, side: str) -> dict:
    """Construct the generate_user_message data_dict from live MEASURED CARLA state.

    ego:          {x, y, yaw, fwd_v, right_v, yaw_rate, speed, steer} — pose (yaw in rad,
                  world/OpenDRIVE frame) plus the measured ego-state (ego frame, signed; same
                  convention as the build_infos_pkl ego_* fields build_entry reads). Fills
                  gt_ego_lcf_feat[0,1,4,7,8] = fwd_v, right_v, yaw_rate, speed, steer.
    ego_history:  list of (x, y) global, oldest->newest, the last <=4 poses @0.5s
                  (current pose excluded; it is the origin)
    slot:         {x, y, yaw}  target slot pose, global frame
    """
    ox, oy, oyaw = float(ego["x"]), float(ego["y"]), float(ego["yaw"])

    # gt_ego_lcf_feat[0,1,4,7,8] from the measured CARLA state, byte-for-byte the same source
    # generate_cached_nuscenes_info.build_entry reads (ego_fwd_v/right_v/yaw_rate/speed/steer).
    fwd_v = float(ego["fwd_v"])
    right_v = float(ego["right_v"])
    speed = float(ego.get("speed", np.hypot(fwd_v, right_v)))
    yaw_rate = float(ego["yaw_rate"])
    steer = float(ego.get("steer", 0.0))
    lcf = np.array(
        [fwd_v, right_v, ox, oy, yaw_rate,
         LENGTH_M, WIDTH_M, speed, steer],
        dtype=np.float32,
    )

    # History in ego-local; pad at the front with the oldest available (mirrors
    # collect_history_local), then append current origin -> 5 points.
    pts = [list(global_to_local_xy(hx, hy, ox, oy, oyaw)) for (hx, hy) in ego_history[-HISTORY_STEPS:]]
    if pts:
        while len(pts) < HISTORY_STEPS:
            pts.insert(0, list(pts[0]))
    else:
        pts = [[0.0, 0.0] for _ in range(HISTORY_STEPS)]
    pts.append([0.0, 0.0])
    his = np.array(pts, dtype=np.float32)
    his_diff = np.diff(his, axis=0).astype(np.float32)

    sr, sf = global_to_local_xy(float(slot["x"]), float(slot["y"]), ox, oy, oyaw)
    slot_local = np.array([sr, sf, normalize_angle(float(slot["yaw"]) - oyaw)], dtype=np.float32)

    return {
        "gt_ego_lcf_feat": lcf,
        "gt_ego_his_trajs": his,
        "gt_ego_his_diff": his_diff,
        "gt_ego_fut_trajs": np.zeros((7, 3), dtype=np.float32),  # (right,forward,heading) dummy;
        # unused at inference (assistant target) but generate_user_message indexes the heading col.
        "maneuver_type": maneuver_type,
        "side": side,
        "slot_local": slot_local,
    }


_LIVE_TOKEN = "__live__"


def build_question(ego: dict, ego_history, slot: dict, maneuver_type: str, side: str):
    """Return (user_message_string, data_dict). Reuses build_llava_conversation so the
    assembled prompt is byte-identical to the training/inference format."""
    from data_utils.build_llava_conversation import build_llava_conversation

    dd = build_data_dict(ego, ego_history, slot, maneuver_type, side)
    sample = {"sample_id": _LIVE_TOKEN,
              "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]}
    build_llava_conversation(sample, {_LIVE_TOKEN: dd})
    return sample["conversations"][0]["value"], dd
