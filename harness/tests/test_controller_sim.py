"""Kinematic-sim validation of the controller adapter (no CARLA, no GPU).

For both a feasible FORWARD arc and a feasible REVERSE arc:
  - generate a ground-truth bicycle-model path (constant steer + gear),
  - drop the car onto the path start with a perturbed heading,
  - each step feed the upcoming path points (expressed in the car's ego frame, @~0.5s spacing)
    to ControllerAdapter, integrate the bicycle model,
  - assert the controller detects the right gear and the cross-track error converges toward 0.

    python harness/tests/test_controller_sim.py
"""
import math
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from harness.client.controller_adapter import ControllerAdapter, WB, MAX_STEER_RAD


def norm(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def bicycle_step(x, y, yaw, v, steer_norm, dt):
    """Kinematic bicycle in CARLA's steer convention: +steer_norm turns toward the ego's
    RIGHT. In the std math frame (yaw CCW+) a right turn is -yaw, hence the minus. v is
    signed (reverse < 0), which naturally inverts reverse steering. steer_norm in [-1,1]."""
    yaw = norm(yaw - (v / WB) * math.tan(steer_norm * MAX_STEER_RAD) * dt)
    x += v * math.cos(yaw) * dt
    y += v * math.sin(yaw) * dt
    return x, y, yaw


def gen_path(steer_norm, reverse, n=60, dt=0.1, speed=1.2):
    v = -speed if reverse else speed
    x = y = yaw = 0.0
    pts = [(0.0, 0.0, 0.0)]   # (x, y, yaw)
    for _ in range(n):
        x, y, yaw = bicycle_step(x, y, yaw, v, steer_norm, dt)
        pts.append((x, y, yaw))
    return pts


def world_to_ego(px, py, x, y, yaw):
    dx, dy = px - x, py - y
    c, s = math.cos(yaw), math.sin(yaw)
    return (s * dx - c * dy, c * dx + s * dy)  # (right, forward); matches global_to_local_xy


def nearest_idx(path, x, y):
    return min(range(len(path)), key=lambda i: math.hypot(path[i][0] - x, path[i][1] - y))


def cross_track(path, x, y):
    return min(math.hypot(px - x, py - y) for px, py, _ in path)


def track(label, steer_norm, reverse, perturb_yaw):
    path = gen_path(steer_norm, reverse)
    ctrl = ControllerAdapter()
    x, y = path[0][0], path[0][1]
    yaw = perturb_yaw                 # GT started at yaw=0; perturb to test feedback
    v_mag, dt = 1.2, 0.1
    errs, gear_ok = [], True
    for _ in range(400):
        i = nearest_idx(path, x, y)
        seg = path[i + 5:i + 35:5]    # ~6 waypoints at ~0.5s spacing ahead along the path
        if len(seg) < 2:
            break
        # (right, forward, head_err) — head_err = the path's heading at the point minus the car's,
        # exactly what remaining_ego_path feeds the controller live.
        wps = [(*world_to_ego(px, py, x, y, yaw), norm(pyaw - yaw)) for (px, py, pyaw) in seg]
        while len(wps) < 6:
            wps.append(wps[-1])
        thr, brk, steer_norm, rev = ctrl.control(wps, v_mag)
        gear_ok = gear_ok and (rev == reverse)
        v = (-v_mag if rev else v_mag)
        x, y, yaw = bicycle_step(x, y, yaw, v, steer_norm, dt)
        errs.append(cross_track(path, x, y))
        if math.hypot(path[-1][0] - x, path[-1][1] - y) < 0.5:
            break
    converged = sum(errs[-10:]) / min(10, len(errs))
    reached = math.hypot(path[-1][0] - x, path[-1][1] - y)
    print(f"[ctrl] {label}: gear_ok={gear_ok} start_err={errs[0]:.2f} "
          f"final_cross_track={converged:.3f}m dist_to_end={reached:.2f}m")
    assert gear_ok, f"{label}: wrong gear detected"
    assert converged < 0.25, f"{label}: did not converge ({converged:.3f} m)"
    # Stops ~1 lookahead short of the end (test artifact: no more waypoints to feed);
    # the real loop keeps predicting toward the slot.
    assert reached < 1.5, f"{label}: did not reach path end ({reached:.2f} m)"


def heading_sign():
    """Collinear plan (right=0) so pure pursuit gives ~0 steer; the model's heading must drive the
    rotation. Checks the steer SIGN matches the bicycle model (the live reverse-perp case)."""
    # Each maneuver gets a FRESH controller: the gear now has commit-tick hysteresis, so a single
    # controller can't flip forward<->reverse within one call (by design). In real use a forward and
    # a reverse maneuver are many ticks apart; here we model that as independent controller instances.
    # collinear (right=0) so pure pursuit ~0; heading ROTATES across the plan (0.08..0.48 rad).
    rev_plan = [(0.0, -0.5 * k, 0.08 * k) for k in range(1, 7)]   # backing straight, want +yaw
    _, _, steer_r, rev = ControllerAdapter().control(rev_plan, 1.0)
    assert rev and steer_r > 0.05, f"reverse heading sign wrong: rev={rev} steer={steer_r:+.3f}"
    fwd_plan = [(0.0, 0.5 * k, 0.08 * k) for k in range(1, 7)]    # going straight, want +yaw
    _, _, steer_f, revf = ControllerAdapter().control(fwd_plan, 1.0)
    assert (not revf) and steer_f < -0.05, f"forward heading sign wrong: rev={revf} steer={steer_f:+.3f}"
    # k_heading=0 must fall back to position-only (~no steer on a collinear plan).
    _, _, steer_off, _ = ControllerAdapter(k_heading=0.0).control(rev_plan, 1.0)
    assert abs(steer_off) < 0.05, f"k_heading=0 should be position-only: steer={steer_off:+.3f}"
    print(f"[ctrl] heading-sign: reverse+(+yaw) -> steer {steer_r:+.2f} (right); "
          f"forward+(+yaw) -> steer {steer_f:+.2f} (left); k_heading=0 -> {steer_off:+.2f}  OK")


def main():
    # steer_norm 0.4 -> feasible arc (radius = WB/tan(0.4*70deg) ~ 2.5m > min 0.9m).
    track("forward arc (perturbed)", steer_norm=0.4, reverse=False, perturb_yaw=math.radians(12))
    track("reverse arc (perturbed)", steer_norm=0.4, reverse=True, perturb_yaw=math.radians(12))
    track("forward straight", steer_norm=0.0, reverse=False, perturb_yaw=math.radians(15))
    heading_sign()
    print("[ctrl] PASS — controller tracks forward + reverse arcs, recovers heading error, and "
          "the heading channel supplies rotation with the correct sign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
