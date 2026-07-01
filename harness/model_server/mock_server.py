"""Mock model server — returns canned waypoints heading toward the slot, so the closed-loop
orchestration can be validated WITHOUT the GPU/model (drop-in for server.py on the same port).

Uses only stdlib + numpy (no torch), so it runs in the CARLA venv too.

    venv/bin/python harness/model_server/mock_server.py [--bridge-port 5557]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import socket
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from harness import config as cfg_mod
from harness import protocol
from harness.model_server import live_prompt  # numpy-only helpers


def plan_toward_slot(slot_global, ego, step=0.55, n=6):
    """6 waypoints stepping straight toward the slot in the ego frame (right, forward).
    A non-holonomic car can't follow a pure strafe, but this is enough to exercise the
    loop: the controller curves toward it and the car drives into the slot region."""
    r, f = live_prompt.global_to_local_xy(slot_global["x"], slot_global["y"],
                                          ego["x"], ego["y"], ego["yaw"])
    dist = math.hypot(r, f)
    if dist < 1e-3:
        return [[0.0, 0.0]] * n
    ux, uy = r / dist, f / dist
    return [[ux * min(dist, step * i), uy * min(dist, step * i)] for i in range(1, n + 1)]


class MockServer:
    def __init__(self, cfg):
        self.cfg = cfg

    def handle(self, header, jpegs):
        if header.get("type") == protocol.PING:
            return {"type": protocol.PONG}, []
        wps = plan_toward_slot(header["slot_global"], header["ego"])
        prompt = (f"[MOCK] frame {header['frame_idx']} reset={header['reset']} "
                  f"side={header['side']} #imgs={len(jpegs)} -> heading to slot")
        return protocol.build_waypoints_response(waypoints=wps, prompt=prompt,
                                                 raw_answer="mock", infer_ms=1.0)

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.cfg.bridge_host, self.cfg.bridge_port))
        srv.listen(1)
        print(f"[mock] listening on {self.cfg.bridge_host}:{self.cfg.bridge_port}", flush=True)
        while True:
            conn, addr = srv.accept()
            print(f"[mock] client connected: {addr}", flush=True)
            try:
                while True:
                    header, jpegs = protocol.recv_msg(conn)
                    resp_h, resp_b = self.handle(header, jpegs)
                    protocol.send_msg(conn, resp_h, resp_b)
            except (ConnectionError, OSError):
                print("[mock] client disconnected", flush=True)
            finally:
                conn.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    cfg_mod.Config.add_cli_args(p)
    cfg = cfg_mod.Config.from_args(p.parse_args(argv))
    MockServer(cfg).serve()


if __name__ == "__main__":
    main()
