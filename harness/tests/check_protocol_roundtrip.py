"""Standalone smoke test: config imports and a protocol frame round-trips over a
real socket. Run under BOTH venvs to prove the cross-version wire contract holds.

    <py3.8>  python harness/tests/check_protocol_roundtrip.py
    <py3.10> python harness/tests/check_protocol_roundtrip.py
"""
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness import config, protocol


def main() -> int:
    cfg = config.Config()
    assert str(cfg.checkpoint).endswith("merged"), cfg.checkpoint
    assert cfg.bridge_port == 5557

    # Build a representative inference request (3 fake "jpegs" stand in for 6 cams).
    jpegs = [b"\xff\xd8fakejpeg%d\xff\xd9" % i for i in range(6)]
    header, blobs = protocol.build_infer_request(
        frame_idx=0, reset=True, maneuver_type="reverse_perpendicular", side="right",
        slot_global={"x": 290.0, "y": 201.2, "yaw": 3.14159},
        ego={"x": 285.0, "y": 200.0, "yaw": 0.0, "vx": 1.0, "vy": 0.0, "yaw_rate": 0.0, "steer": 0.0},
        ego_history=[{"x": 284.0, "y": 200.0, "yaw": 0.0, "t": -0.5}],
        cam_names=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"],
        calib={"CAM_FRONT": {"intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "cam2ego": [[1, 0, 0, 0]] * 4}},
        jpegs=jpegs,
    )

    a, b = socket.socketpair()
    out = {}

    def server():
        h, bl = protocol.recv_msg(b)
        out["h"], out["bl"] = h, bl
        resp_h, resp_b = protocol.build_waypoints_response(
            waypoints=[[0.0, 0.0], [0.1, 0.6], [0.2, 1.2], [0.3, 1.8], [0.4, 2.4], [0.5, 3.0]],
            prompt="Mission goal: reverse-perpendicular park, right side, into slot at (5.00,1.20,3.14)",
            raw_answer="[(0.00,0.00),...]", infer_ms=42.0,
        )
        protocol.send_msg(b, resp_h, resp_b)

    t = threading.Thread(target=server)
    t.start()
    protocol.send_msg(a, header, blobs)
    resp_h, _ = protocol.recv_msg(a)
    t.join()
    a.close(); b.close()

    assert out["h"]["type"] == protocol.INFER
    assert out["bl"] == jpegs, "blob round-trip mismatch"
    assert out["h"]["maneuver_type"] == "reverse_perpendicular"
    assert len(resp_h["waypoints"]) == 6
    print(f"OK  py{sys.version_info.major}.{sys.version_info.minor}  "
          f"6 blobs round-tripped, {len(resp_h['waypoints'])} waypoints back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
