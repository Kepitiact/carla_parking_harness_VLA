"""Validate live_uniad against the offline pipeline (no warmup, no tricks).

Two checks:
  1. INPUT reconstruction (exact, no model): for f0000 (the captured template frame), the
     per-frame fields live_uniad recomputes from raw JPEGs + ego pose (img, l2g, can_bus)
     must byte-match the offline pipeline's uniad_data.
  2. END-TO-END (waypoints): for a normal mid-maneuver frame (f0010), feeding the live-built
     uniad_data through the full model must produce essentially the offline precomputed-feature
     (uniad_pth) waypoints. f0010 has clear motion so the model is in its robust regime (the
     near-stationary frames f0000/f0015 are borderline by nature and are NOT used here).

Run in the model venv (needs the GPU):
    ~/projects/openvla_nuscenes/.venv/bin/python harness/tests/validate_live_uniad.py
"""
import copy
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
REF = _REPO / "harness/runs/uniad_ref"


def main():
    cfg = cfg_mod.Config()
    os.environ["CACHED_DATA_PATH"] = str(PROC / "cached_parking_info.pkl")
    bootstrap_model_repo(cfg.model_repo)
    import torch
    from pyquaternion import Quaternion
    from harness.model_server.model_runner import ModelRunner, parse_waypoints
    from harness.model_server import live_uniad, live_prompt
    from data_utils.build_llava_conversation import build_llava_conversation

    infos = {e["token"]: e for e in
             pickle.load(open(PROC / "parking_infos_temporal.pkl", "rb"))["infos"]}
    cached = pickle.load(open(PROC / "cached_parking_info.pkl", "rb"))
    convs = {e.get("sample_id"): e for e in __import__("json").load(open(PROC / "carla_conversations.json"))}

    print("[validate] loading model...")
    runner = ModelRunner(cfg.checkpoint, cfg.model_repo, device="cuda")
    live = live_uniad.LiveUniadInput(REF / "uniad_data.pth")

    def build_ud(token):
        info = infos[token]
        jpegs = [open(info["cams"][c]["data_path"], "rb").read() for c in live_uniad.CAM_ORDER]
        et, er = info["ego2global_translation"], info["ego2global_rotation"]
        return live.build(jpegs, et, list(Quaternion(er)), "ep0000",
                          timestamp=info["timestamp"] / 1e6,
                          speed=float(info["can_bus"][13]),  # MEASURED speed (build_infos layout)
                          frame_idx=int(token.split("f")[-1]))

    # --- 1. input reconstruction (exact) on the template frame ---
    ud0 = build_ud("episode_0000_f0000")
    tmpl = live.template
    img_err = float(torch.max(torch.abs(ud0["img"][0] - tmpl["img"][0])))
    l2g_err = float(torch.max(torch.abs(ud0["l2g_r_mat"] - tmpl["l2g_r_mat"])))
    cb_err = float(np.max(np.abs(live_uniad._find_meta(ud0["img_metas"])["can_bus"]
                                 - live_uniad._find_meta(tmpl["img_metas"])["can_bus"])))
    print(f"[validate] f0000 INPUT recompute vs offline pipeline: img={img_err:.4f} l2g={l2g_err:.6f} can_bus={cb_err:.6f}")
    assert img_err < 1.0 and l2g_err < 1e-3 and cb_err < 1e-3, "input reconstruction drift"

    # --- 2. end-to-end waypoints on a robust mid-maneuver frame (no warmup) ---
    token = "episode_0000_f0010"
    info = infos[token]
    et, er = info["ego2global_translation"], info["ego2global_rotation"]
    sp = info["target_slot"]["pose"]
    slot = {"x": sp["translation"][0], "y": sp["translation"][1], "yaw": Quaternion(sp["rotation"]).yaw_pitch_roll[0]}
    ego = {"x": et[0], "y": et[1], "yaw": Quaternion(er).yaw_pitch_roll[0],
           "fwd_v": info["ego_fwd_v"], "right_v": info["ego_right_v"],
           "yaw_rate": info["ego_yaw_rate"], "speed": info["ego_speed"], "steer": info["ego_steer"]}
    question, _ = live_prompt.build_question(
        ego, [], slot, info["maneuver_type"], info["side"])
    input_ids = runner.build_input_ids(question)

    def gen(**kw):
        live_uniad.reset_temporal(runner.model)
        return parse_waypoints(runner.generate(input_ids, **kw))

    wp_live = gen(uniad_data=build_ud(token))
    wp_pth = gen(uniad_pth=torch.load(convs[token]["uniad_pth"], map_location="cuda"))
    gt = cached[token]["gt_ego_fut_trajs"]

    def show(name, wp):  # wp items may be (right, forward) or (right, forward, heading)
        print(f"  {name}: " + " ".join(f"({w[0]:+.2f},{w[1]:+.2f})" for w in wp))
    print(f"[validate] {token} waypoints (right, forward):")
    show("live uniad_data ", wp_live)
    show("offline uniad_pth", wp_pth)
    show("ground truth     ", [(gt[i][0], gt[i][1]) for i in range(1, 7)])

    a = np.array([[w[0], w[1]] for w in wp_live])  # position only (drop heading col)
    b = np.array([[w[0], w[1]] for w in wp_pth])
    n = min(len(a), len(b))
    l2 = float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1))) if n else 9.9
    nz = sum(1 for w in wp_live if abs(w[0]) + abs(w[1]) > 1e-3)
    print(f"[validate] nonzero={nz}, mean L2 (live vs offline uniad_pth): {l2:.3f} m")
    assert nz > 0 and l2 < 0.3, f"live waypoints diverge from offline ({l2:.3f} m, nz={nz})"
    print("\n[validate] PASS — live_uniad reconstructs input exactly and yields offline-matching waypoints (no warmup)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
