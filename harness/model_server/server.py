"""Model server (Py3.10, owns the GPU).

Localhost socket bridge: each request = 6 live JPEGs + ego pose/history + maneuver/side/slot,
each reply = 6 ego-local waypoints + the verbatim prompt. Wraps model_runner + live_prompt +
live_uniad behind protocol.py. Warms up with the captured template at startup (live frames do
not self-bootstrap), and resets UniAD's temporal BEV state on each episode boundary.

Run (in the model venv):
    ~/projects/openvla_nuscenes/.venv/bin/python harness/model_server/server.py
    # optional overrides: --bridge-port 5557 --checkpoint <dir> --uniad-template <pth>
"""
from __future__ import annotations

import argparse
import math
import pathlib
import socket
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from harness import config as cfg_mod
from harness import protocol


def yaw_to_quat_wxyz(yaw: float):
    """Ego rotation is pure yaw about +z (flat parking lot), matching the training
    ego2global_rotation [cos(y/2),0,0,sin(y/2)]."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


class ModelServer:
    def __init__(self, cfg: cfg_mod.Config):
        from harness.model_server.model_runner import ModelRunner, parse_waypoints
        from harness.model_server import live_prompt, live_uniad

        self.cfg = cfg
        self._live_prompt = live_prompt
        self._live_uniad = live_uniad
        self._parse_waypoints = parse_waypoints

        print(f"[server] loading model: {cfg.checkpoint}", flush=True)
        self.runner = ModelRunner(cfg.checkpoint, cfg.model_repo, device="cuda")
        # The template supplies the constant calibration + GT-dummy structure live_uniad
        # clones each tick (NOT a warmup — there is none; see live_uniad).
        print(f"[server] loading UniAD structural template: {cfg.uniad_template}", flush=True)
        self.live = live_uniad.LiveUniadInput(cfg.uniad_template)
        self.episode = 0
        print("[server] ready.", flush=True)

    def handle(self, header, jpegs):
        if header.get("type") == protocol.PING:
            return {"type": protocol.PONG}, []

        ego = header["ego"]
        slot = header["slot_global"]
        hist_xy = [(h["x"], h["y"]) for h in header.get("ego_history", [])]

        if header.get("reset"):
            self.episode += 1
            self._live_uniad.reset_temporal(self.runner.model)
        scene_token = f"ep{self.episode:04d}"

        # 1) prompt (mission goal recomputed in the current ego frame every tick). The ego-state
        #    is filled from the MEASURED CARLA motion in `ego` (fwd_v/right_v/yaw_rate/speed/steer),
        #    matching the retrained checkpoint — see live_prompt.
        question, _ = self._live_prompt.build_question(
            ego, hist_xy, slot, header["maneuver_type"], header["side"])
        input_ids = self.runner.build_input_ids(question)

        # 2) UniAD input from the 6 live images (reordered to CAM_ORDER).
        idx = {c: i for i, c in enumerate(header["cam_names"])}
        ordered = [jpegs[idx[c]] for c in self._live_uniad.CAM_ORDER]
        ego_t = [ego["x"], ego["y"], ego.get("z", 0.0)]
        ud = self.live.build(
            ordered, ego_t, yaw_to_quat_wxyz(ego["yaw"]), scene_token,
            timestamp=header["frame_idx"] * 0.5, speed=ego.get("speed", 0.0),
            frame_idx=header["frame_idx"])

        # 3) UniAD perception (reported) -> LLM -> waypoints.
        t0 = time.time()
        answer, n_tracks = self.runner.perceive_and_generate(input_ids, ud)
        infer_ms = (time.time() - t0) * 1000.0
        wps = self._parse_waypoints(answer)
        print(f"[server] frame {header.get('frame_idx')}: UniAD detected {n_tracks} objects "
              f"-> wp0=({wps[0][0]:+.2f},{wps[0][1]:+.2f})" if wps else
              f"[server] frame {header.get('frame_idx')}: UniAD detected {n_tracks} objects (no wps)",
              flush=True)
        resp_h, resp_b = protocol.build_waypoints_response(
            waypoints=wps, prompt=question, raw_answer=answer, infer_ms=infer_ms)
        resp_h["n_tracks"] = n_tracks
        return resp_h, resp_b

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.cfg.bridge_host, self.cfg.bridge_port))
        srv.listen(1)
        print(f"[server] listening on {self.cfg.bridge_host}:{self.cfg.bridge_port}", flush=True)
        while True:
            conn, addr = srv.accept()
            print(f"[server] client connected: {addr}", flush=True)
            try:
                while True:
                    header, jpegs = protocol.recv_msg(conn)
                    try:
                        resp_h, resp_b = self.handle(header, jpegs)
                    except Exception as e:  # don't kill the server on a bad request
                        import traceback
                        traceback.print_exc()
                        resp_h, resp_b = {"type": protocol.ERROR, "message": str(e)}, []
                    protocol.send_msg(conn, resp_h, resp_b)
            except (ConnectionError, OSError) as e:
                print(f"[server] client disconnected ({e})", flush=True)
            finally:
                conn.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    cfg_mod.Config.add_cli_args(p)
    cfg = cfg_mod.Config.from_args(p.parse_args(argv))
    ModelServer(cfg).serve()


if __name__ == "__main__":
    main()
