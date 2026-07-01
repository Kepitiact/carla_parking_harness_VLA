"""Length-framed request/response wire format for the CARLA<->model bridge.

Deliberately pickle-free: a JSON header plus raw binary blobs (the 6 JPEGs).
That keeps it robust across the two interpreters (Py3.8 client, Py3.10 server),
trivially debuggable, and free of shared-class/version coupling. Pure stdlib.

Frame layout on the wire:
    [8-byte big-endian payload length]
    [4-byte big-endian header length][header JSON utf-8]
    [blob 0][blob 1]...        # sizes/order given by header["blob_sizes"]
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Dict, List, Tuple

# Message types (header["type"]).
INFER = "infer"          # client -> server: 6 imgs + ego/history + maneuver/slot/calib
WAYPOINTS = "waypoints"  # server -> client: 6 ego-local waypoints + verbatim prompt
ERROR = "error"          # server -> client: {"type":"error","message":...}
PING = "ping"
PONG = "pong"


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, header: Dict, blobs: List[bytes] = None) -> None:
    blobs = blobs or []
    header = dict(header)
    header["blob_sizes"] = [len(b) for b in blobs]
    hdr_bytes = json.dumps(header).encode("utf-8")
    payload = struct.pack(">I", len(hdr_bytes)) + hdr_bytes + b"".join(blobs)
    sock.sendall(struct.pack(">Q", len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Tuple[Dict, List[bytes]]:
    (payload_len,) = struct.unpack(">Q", _recv_exactly(sock, 8))
    payload = _recv_exactly(sock, payload_len)
    (hdr_len,) = struct.unpack(">I", payload[:4])
    header = json.loads(payload[4 : 4 + hdr_len].decode("utf-8"))
    blobs, off = [], 4 + hdr_len
    for size in header.get("blob_sizes", []):
        blobs.append(payload[off : off + size])
        off += size
    return header, blobs


# ── request/response builders (single place the schema is defined) ──────────

def build_infer_request(
    *, frame_idx: int, reset: bool, maneuver_type: str, side: str,
    slot_global: Dict, ego: Dict, ego_history: List[Dict],
    cam_names: List[str], calib: Dict, jpegs: List[bytes],
) -> Tuple[Dict, List[bytes]]:
    """Assemble the per-tick inference request. All poses are in the nuScenes/
    OpenDRIVE global frame (y_odr = -y_carla); the server converts to ego-local.

    slot_global:       {"x","y","yaw"}.
    ego:               {"x","y","yaw"} + the MEASURED ego-state {"z","speed","fwd_v","right_v",
                       "yaw_rate","steer"} (ego frame, signed) — the server fills gt_ego_lcf_feat
                       from this measured motion.
    ego_history:       oldest->newest list of {"x","y"} (~last 2s).
    calib[cam]:        {"intrinsic": 3x3 list, "cam2ego": 4x4 list} (optional; server uses
                       its own constant calibration template).
    jpegs:             JPEG bytes, one per cam, in cam_names order.
    """
    header = {
        "type": INFER,
        "frame_idx": frame_idx,
        "reset": bool(reset),
        "maneuver_type": maneuver_type,
        "side": side,
        "slot_global": slot_global,
        "ego": ego,
        "ego_history": ego_history,
        "cam_names": cam_names,
        "calib": calib,
    }
    return header, jpegs


def build_waypoints_response(
    *, waypoints: List[List[float]], prompt: str, raw_answer: str, infer_ms: float,
) -> Tuple[Dict, List[bytes]]:
    header = {
        "type": WAYPOINTS,
        "waypoints": waypoints,   # [[right, forward, heading], ...] x6, 0.5s spacing (heading rad,
                                  # relative to current ego frame; legacy 2-tuples also accepted)
        "prompt": prompt,         # verbatim assembled user message (for viewer + log)
        "raw_answer": raw_answer,
        "infer_ms": infer_ms,
    }
    return header, []
