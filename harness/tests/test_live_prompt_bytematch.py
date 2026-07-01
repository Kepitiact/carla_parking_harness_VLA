"""Byte-match the live prompt against an offline frame (handoff requirement).

Reconstructs a real frame's LIVE inputs (ego pose, last-2s history, target slot) from
parking_infos_temporal.pkl, runs live_prompt, and asserts:
  - the ego-local history trajectory matches the cached pipeline (geometry),
  - slot_local matches the cached pipeline (geometry),
  - the assembled "Mission goal:" and "Historical trajectory:" lines byte-match
    build_llava_conversation/generate_user_message for that token.

Post-retrain the live ego-state is fed from the recorded MEASURED fields (ego_fwd_v/ego_right_v/
ego_yaw_rate/ego_speed/ego_steer) — the same source generate_cached_nuscenes_info.build_entry
reads — so the "Ego states:" line byte-matches the regenerated cache too (the future-leak is gone).

Run in the model venv:
    ~/projects/openvla_nuscenes/.venv/bin/python harness/tests/test_live_prompt_bytematch.py
"""
import os
import pathlib
import pickle
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from harness import config as cfg_mod
from harness.model_server.model_runner import bootstrap_model_repo

PROC = pathlib.Path.home() / "projects/openvla_nuscenes/data_carla/processed"


def _line(msg, prefix):
    for ln in msg.split("\n"):
        if ln.startswith(prefix):
            return ln
    raise AssertionError(f"no line starting {prefix!r}")


def main() -> int:
    cfg = cfg_mod.Config()
    bootstrap_model_repo(cfg.model_repo)
    from pyquaternion import Quaternion
    from data_utils.build_llava_conversation import build_llava_conversation
    from harness.model_server import live_prompt

    infos = pickle.load(open(PROC / "parking_infos_temporal.pkl", "rb"))
    infos = infos["infos"] if isinstance(infos, dict) and "infos" in infos else infos
    by_tok = {e["token"]: e for e in infos}
    cached = pickle.load(open(PROC / "cached_parking_info.pkl", "rb"))

    # A frame deep enough to have a full 4-step history and a maneuver label.
    token = "episode_0000_f0010"
    assert token in by_tok and token in cached, token
    info = by_tok[token]

    def yaw_of(q):
        return float(Quaternion(q).yaw_pitch_roll[0])

    ex, ey = info["ego2global_translation"][:2]
    eyaw = yaw_of(info["ego2global_rotation"])

    # History: walk prev links 4x, collect oldest->newest.
    hist, tok = [], token
    for _ in range(live_prompt.HISTORY_STEPS):
        prev = by_tok[tok].get("prev")
        if not prev or prev not in by_tok:
            break
        hist.append(tuple(by_tok[prev]["ego2global_translation"][:2]))
        tok = prev
    hist = hist[::-1]

    spose = info["target_slot"]["pose"]
    slot = {"x": spose["translation"][0], "y": spose["translation"][1], "yaw": yaw_of(spose["rotation"])}

    # MEASURED ego-state (from the recorded ego_* fields) -> the live "Ego states:" line should
    # byte-match the regenerated cache, which build_entry now fills from the same measured source.
    ego = {"x": ex, "y": ey, "yaw": eyaw,
           "fwd_v": info["ego_fwd_v"], "right_v": info["ego_right_v"],
           "yaw_rate": info["ego_yaw_rate"], "speed": info["ego_speed"],
           "steer": info["ego_steer"]}
    live_msg, dd = live_prompt.build_question(
        ego, hist, slot, info["maneuver_type"], info["side"])

    # Reference offline message for this token.
    ref_sample = {"sample_id": token,
                  "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]}
    build_llava_conversation(ref_sample, cached)
    ref_msg = ref_sample["conversations"][0]["value"]

    c = cached[token]
    err_hist = float(np.max(np.abs(dd["gt_ego_his_trajs"] - c["gt_ego_his_trajs"])))
    err_slot = float(np.max(np.abs(dd["slot_local"] - c["slot_local"])))
    print(f"[bytematch] token={token}")
    print(f"  max |history error| = {err_hist:.4f} m")
    print(f"  max |slot_local err| = {err_slot:.4f}")
    assert err_hist < 0.02, f"history geometry drift {err_hist}"
    assert err_slot < 0.02, f"slot_local geometry drift {err_slot}"

    for prefix in ("Ego states:", "Mission goal:", "Historical trajectory"):
        lv, rf = _line(live_msg, prefix), _line(ref_msg, prefix)
        print(f"  live : {lv}")
        print(f"  offl : {rf}")
        assert lv == rf, f"{prefix} mismatch"

    assert live_msg == ref_msg, "full prompt mismatch"
    print("[bytematch] PASS — FULL prompt byte-matches offline (ego-state + geometry + goal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
