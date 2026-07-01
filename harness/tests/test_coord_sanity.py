"""Static coordinate sanity check (no CARLA, no GPU) — the handoff's explicit ask.

Verifies the controller-side ego-local<->world transform (harness/client/transforms.py) is
the exact inverse of the VALIDATED server-side live_prompt.global_to_local_xy, plus a couple
of hand-computed known cases. If this passes, waypoints map to the right CARLA world points.

    python harness/tests/test_coord_sanity.py
"""
import math
import pathlib
import random
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from harness.client import transforms as tf
from harness.model_server import live_prompt as lp


def main() -> int:
    # 1. Round-trip: ego_local_to_odr must invert global_to_local_xy for random poses/points.
    random.seed(0)
    worst = 0.0
    for _ in range(2000):
        ex, ey = random.uniform(-300, 300), random.uniform(-300, 300)
        eyaw = random.uniform(-math.pi, math.pi)
        right, forward = random.uniform(-20, 20), random.uniform(-20, 20)
        wx, wy = tf.ego_local_to_odr(right, forward, ex, ey, eyaw)
        r2, f2 = lp.global_to_local_xy(wx, wy, ex, ey, eyaw)
        worst = max(worst, abs(r2 - right), abs(f2 - forward))
    print(f"[coord] round-trip max error over 2000 cases: {worst:.2e}")
    assert worst < 1e-9, f"transform not invertible ({worst})"

    # 2. Known case: ego at origin facing +x (yaw_n=0). A point 2 right, 5 forward.
    wx, wy = tf.ego_local_to_odr(2.0, 5.0, 0.0, 0.0, 0.0)
    assert abs(wx - 5.0) < 1e-9 and abs(wy + 2.0) < 1e-9, (wx, wy)  # forward->+x, right->-y
    print(f"[coord] ego@origin yaw0: (right=2,fwd=5) -> world ({wx:.1f},{wy:.1f})  [forward=+x, right=-y] OK")

    # 3. CARLA-frame waypoint mapping: ego facing CARLA yaw=0 (=odr yaw 0).
    #    A 'forward' waypoint must move +x in CARLA; a 'right' waypoint must move +y in CARLA
    #    (CARLA y points south, so ego-right = +y_carla when facing +x).
    cx, cy = tf.ego_local_to_carla(0.0, 3.0, 10.0, 20.0, 0.0)  # 3m forward
    assert abs(cx - 13.0) < 1e-9 and abs(cy - 20.0) < 1e-9, (cx, cy)
    cx, cy = tf.ego_local_to_carla(3.0, 0.0, 10.0, 20.0, 0.0)  # 3m right
    assert abs(cx - 10.0) < 1e-9 and abs(cy - 23.0) < 1e-9, (cx, cy)
    print("[coord] CARLA-frame: forward->+x, right->+y (facing +x) OK")

    # 4. waypoints_to_carla_path prepends the ego origin and maps each point.
    path = tf.waypoints_to_carla_path([(0.0, 1.0), (0.0, 2.0)], 5.0, 5.0, 0.0)
    assert path[0] == (5.0, 5.0) and abs(path[1][0] - 6.0) < 1e-9 and abs(path[2][0] - 7.0) < 1e-9, path
    print("[coord] waypoints_to_carla_path origin + mapping OK")

    print("[coord] PASS — controller transform is the exact inverse of the validated server transform")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
