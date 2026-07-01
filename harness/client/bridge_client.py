"""Thin client to the model server (harness/model_server/server.py) over the localhost/
network socket. Runs in the CARLA venv (py3.8). Stateless except the TCP connection.

The model server may run on a DIFFERENT machine (set bridge_host to its IP) — the whole point
of the two-process design — so this never imports torch or the model.
"""
from __future__ import annotations

import socket

from harness import protocol


class BridgeClient:
    def __init__(self, host: str, port: int, timeout: float = 120.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection(self.addr, timeout=self.timeout)
        return self

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def infer(self, *, frame_idx, reset, maneuver_type, side, slot_global, ego,
              ego_history, cam_names, jpegs, calib=None):
        """Send one tick's data, return the server response dict
        {waypoints, prompt, raw_answer, infer_ms}."""
        header, blobs = protocol.build_infer_request(
            frame_idx=frame_idx, reset=reset, maneuver_type=maneuver_type, side=side,
            slot_global=slot_global, ego=ego, ego_history=ego_history,
            cam_names=cam_names, calib=calib or {}, jpegs=jpegs)
        protocol.send_msg(self.sock, header, blobs)
        resp, _ = protocol.recv_msg(self.sock)
        if resp.get("type") == protocol.ERROR:
            raise RuntimeError(f"model server error: {resp.get('message')}")
        return resp

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()
