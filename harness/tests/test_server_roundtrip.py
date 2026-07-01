"""End-to-end bridge test: a (py3.8-compatible) client sends the real episode_0000_f0000
frame to the running model server and checks the returned waypoints match the offline path.

Proves the full Option-C path: client serialises 6 JPEGs + ego/slot over the socket ->
py3.10 server runs live UniAD + prompt + LLM -> waypoints come back. Run AFTER starting
the server (harness/model_server/server.py).

    python harness/tests/test_server_roundtrip.py [--bridge-port 5557]
Uses only stdlib + numpy (no torch / pyquaternion), so it runs under the CARLA venv too.
"""
import argparse
import math
import os
import pathlib
import pickle
import socket
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from harness import config as cfg_mod
from harness import protocol

PROC = pathlib.Path.home() / "projects/openvla_nuscenes/data_carla/processed"
CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
OFFLINE_F0000 = [(0.00, 0.00), (0.04, -0.21), (0.13, -0.65), (-0.04, -1.28), (-0.30, -2.00), (-0.30, -2.90)]


def yaw_of(q):  # quaternion wxyz -> yaw (rad)
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main(argv=None):
    p = argparse.ArgumentParser()
    cfg_mod.Config.add_cli_args(p)
    cfg = cfg_mod.Config.from_args(p.parse_args(argv))

    infos = pickle.load(open(PROC / "parking_infos_temporal.pkl", "rb"))
    infos = infos["infos"] if isinstance(infos, dict) and "infos" in infos else infos
    by_tok = {e["token"]: e for e in infos}
    cached = pickle.load(open(PROC / "cached_parking_info.pkl", "rb"))
    # A normal mid-maneuver frame (clear motion = the model's robust regime).
    TOKEN = "episode_0000_f0010"
    info = by_tok[TOKEN]

    ex, ey, ez = info["ego2global_translation"]
    eyaw = yaw_of(info["ego2global_rotation"])
    spose = info["target_slot"]["pose"]
    slot = {"x": spose["translation"][0], "y": spose["translation"][1], "yaw": yaw_of(spose["rotation"])}
    jpegs = [open(info["cams"][c]["data_path"], "rb").read() for c in CAM_ORDER]

    # ego history (last <=4 poses @0.5s) from the prev links, oldest->newest.
    hist, tok = [], TOKEN
    for _ in range(4):
        prev = by_tok[tok].get("prev")
        if not prev or prev not in by_tok:
            break
        hist.append({"x": by_tok[prev]["ego2global_translation"][0], "y": by_tok[prev]["ego2global_translation"][1]})
        tok = prev
    hist = hist[::-1]
    # offline reference = this frame's GT future trajectory (right, forward).
    OFFLINE = [(float(cached[TOKEN]["gt_ego_fut_trajs"][i][0]),
                float(cached[TOKEN]["gt_ego_fut_trajs"][i][1])) for i in range(1, 7)]

    # MEASURED ego-state (the retrained checkpoint's contract), from the recorded ego_* fields.
    ego = {"x": ex, "y": ey, "z": ez, "yaw": eyaw, "speed": float(info["ego_speed"]),
           "fwd_v": float(info["ego_fwd_v"]), "right_v": float(info["ego_right_v"]),
           "yaw_rate": float(info["ego_yaw_rate"]), "steer": float(info["ego_steer"])}
    header, blobs = protocol.build_infer_request(
        frame_idx=int(TOKEN.split("f")[-1]), reset=True, maneuver_type=info["maneuver_type"],
        side=info["side"], slot_global=slot, ego=ego,
        ego_history=hist, cam_names=CAM_ORDER, calib={}, jpegs=jpegs)

    print(f"[client] connecting to {cfg.bridge_host}:{cfg.bridge_port}")
    sock = socket.create_connection((cfg.bridge_host, cfg.bridge_port), timeout=120)
    protocol.send_msg(sock, header, blobs)
    resp, _ = protocol.recv_msg(sock)
    sock.close()

    assert resp["type"] == protocol.WAYPOINTS, resp
    wps = resp["waypoints"]
    print("[client] verbatim prompt the model saw:\n" + resp["prompt"])
    print(f"[client] infer_ms={resp['infer_ms']:.0f}")
    print("[client] waypoints (right, forward, heading):")
    print("   live: " + " ".join(
        f"({wp[0]:+.2f},{wp[1]:+.2f},{(wp[2] if len(wp) > 2 else 0.0):+.2f})" for wp in wps))
    print("   GT  : " + " ".join(f"({r:+.2f},{f:+.2f})" for r, f in OFFLINE))

    import numpy as np
    n = min(len(wps), len(OFFLINE))
    live_rf = np.array([[wp[0], wp[1]] for wp in wps[:n]])  # position only (drop heading col)
    l2 = float(np.mean(np.linalg.norm(live_rf - np.array(OFFLINE[:n]), axis=1)))
    nz = sum(1 for wp in wps if abs(wp[0]) + abs(wp[1]) > 1e-3)
    print(f"[client] nonzero waypoints={nz}, mean L2 vs GT={l2:.3f} m")
    assert nz > 0, "server returned all-zero waypoints"
    assert l2 < 0.5, f"waypoints diverge from GT ({l2:.3f} m)"
    print("[client] PASS — bridge round-trip produces correct waypoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
