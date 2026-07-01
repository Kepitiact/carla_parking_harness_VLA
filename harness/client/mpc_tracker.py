"""MPC trajectory tracker for the closed-loop harness.

Drop-in alternative to ControllerAdapter (same control() signature). Instead of hand-tuned
pure-pursuit it tracks the model's predicted trajectory with a linearized-bicycle MPC — the same
controller family the data collection used (ParkingScenes/tool/MotionPlanning/Control/MPC.py).

The win for reverse-perpendicular parking: the MPC tracks POSITION and HEADING jointly in one QP
(state z = [x, y, v, yaw], with a real yaw weight in Q — the collection's Q had yaw=0 because it
tracked dense Hybrid A* paths). The model carries the reverse rotation in its per-waypoint heading
channel, so the MPC nulls the heading error natively instead of via an open-loop curvature hack.

Frame: the QP runs in an ego-local "MPC frame" rebuilt every tick — car at the origin, forward = +x,
left = +y, yaw = 0 (standard right-handed, yaw CCW+). A model waypoint (right, forward, head_err)
maps to (x, y, yaw) = (forward, -right, head_err). The MPC steering angle delta (CCW+, left turn for
forward motion) maps to CARLA steer as steer = -delta / CARLA_STEER_MAX (CARLA +steer = right).
Signs validated in harness/tests/test_controller_sim.py.

Only the STEERING comes from the MPC; the longitudinal command reuses the same arc-length target
speed + throttle/brake as ControllerAdapter, so this change is isolated to the rotation problem.
"""
from __future__ import annotations

import math

import cvxpy
import numpy as np

from harness.client.controller_adapter import ControllerAdapter

# ── model / MPC config (mirrors ParkingScenes MPC.P, with a yaw weight added) ──
NX, NU, T = 4, 2, 6              # state [x, y, v, yaw], input [accel, steer], horizon length
DT = 0.2                        # MPC time step [s]
WB = 2.5                        # wheelbase [m]
MPC_STEER_MAX = math.radians(60.0)
STEER_RATE_MAX = math.radians(45.0)   # per second
ACCEL_MAX = 2.0
SPEED_MAX, SPEED_MIN = 3.0, -3.0
ITER_MAX, DU_RES = 3, 0.1
REF_DS = 0.15                   # reference resampling spacing [m]

# State penalties: x, y, v, YAW. The yaw weight is what makes the reverse-perp rotation tracked.
Q = np.diag([2.0, 2.0, 0.5, 4.0])
QF = np.diag([2.0, 2.0, 0.5, 4.0])
R = np.diag([0.01, 0.1])        # input penalty (accel, steer)
RD = np.diag([0.01, 1.0])       # input-rate penalty (smooth steering)

CARLA_STEER_MAX = math.radians(70.0)  # CARLA front-wheel max (matches controller_adapter)
DT_WP = 0.5                     # model waypoint spacing [s]

# MPC tracks tighter than pure-pursuit, so the car works the cusp zone more actively and the
# leading-segment net forward crosses zero more often -> needs STRONGER gear hysteresis than
# pure-pursuit to avoid chattering. Wider deadband + more commit ticks (vs the pursuit 0.3 / 4).
MPC_GEAR_DEADBAND = 0.6         # m; net |forward| the new segment must exceed to even be "wanted"
MPC_GEAR_COMMIT_TICKS = 9       # consecutive wanted ticks before the switch commits (~0.3s @30Hz)

GOAL_SLOW_M = 3.0               # start tapering speed when within this distance of the slot [m]
GOAL_STOP_M = 0.6               # inside this distance, cap speed below the is_parked threshold so
GOAL_STOP_SPEED = 0.12          # the car ARRIVES parked (<0.2 m/s) instead of blasting through


def _pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _linear_model(v, phi, delta):
    """Linearized discrete bicycle (z = [x, y, v, yaw]) about (v, phi, delta)."""
    A = np.array([[1.0, 0.0, DT * math.cos(phi), -DT * v * math.sin(phi)],
                  [0.0, 1.0, DT * math.sin(phi),  DT * v * math.cos(phi)],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, DT * math.tan(delta) / WB, 1.0]])
    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [DT, 0.0],
                  [0.0, DT * v / (WB * math.cos(delta) ** 2)]])
    C = np.array([DT * v * math.sin(phi) * phi,
                  -DT * v * math.cos(phi) * phi,
                  0.0,
                  -DT * v * delta / (WB * math.cos(delta) ** 2)])
    return A, B, C


def _resample(pts, ds=REF_DS):
    """Linear-resample a polyline [(x, y, yaw), ...] to ~ds spacing (yaw interpolated unwrapped)."""
    xs, ys, yaws = [pts[0][0]], [pts[0][1]], [pts[0][2]]
    for (x0, y0, a0), (x1, y1, a1) in zip(pts, pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg / ds))
        da = _pi(a1 - a0)
        for k in range(1, n + 1):
            t = k / n
            xs.append(x0 + t * (x1 - x0))
            ys.append(y0 + t * (y1 - y0))
            yaws.append(a0 + t * da)
    return xs, ys, yaws


def _ref_in_horizon(z0, cx, cy, cyaw, sp, ind_old, v_adv):
    """Build the T-step reference z_ref [x, y, v, yaw] by advancing along the path by v_adv*dt.
    v_adv is floored at the planned speed (not the current speed) so the horizon still looks ahead
    from a standstill start — otherwise the reference collapses to the origin and the MPC steers 0."""
    n = len(cx)
    d = [math.hypot(z0[0] - x, z0[1] - y) for x, y in zip(cx, cy)]
    ind = max(ind_old, int(np.argmin(d)))
    z_ref = np.zeros((NX, T + 1))
    moved = 0.0
    for i in range(T + 1):
        moved += abs(v_adv) * DT
        j = min(ind + int(round(moved / REF_DS)), n - 1)
        z_ref[:, i] = [cx[j], cy[j], sp[j], cyaw[j]]
    return z_ref, ind


def _solve(z_ref, z_bar, z0, d_bar):
    z = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))
    cost, cons = 0.0, []
    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], Q)
        A, B, C = _linear_model(z_bar[2, t], z_bar[3, t], d_bar[t])
        cons += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]
        if t < T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], RD)
            cons += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= STEER_RATE_MAX * DT]
    cost += cvxpy.quad_form(z_ref[:, T] - z[:, T], QF)
    cons += [z[:, 0] == z0,
             z[2, :] <= SPEED_MAX, z[2, :] >= SPEED_MIN,
             cvxpy.abs(u[0, :]) <= ACCEL_MAX, cvxpy.abs(u[1, :]) <= MPC_STEER_MAX]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), cons)
    prob.solve(solver=cvxpy.OSQP, warm_start=True)
    if z.value is None:
        return None, None
    return u.value[0, :], u.value[1, :]


def _iterative(z_ref, z0, a_old, d_old):
    if a_old is None:
        a_old, d_old = [0.0] * T, [0.0] * T
    a, d = a_old, d_old
    for _ in range(ITER_MAX):
        # Linearize about the REFERENCE trajectory (its planned speed/heading), not the current
        # state — so the model "knows" it will be moving and steering affects yaw even from a
        # standstill start (the bicycle's d(yaw)/d(steer) term is proportional to speed).
        z_bar = z_ref.copy()
        a_new, d_new = _solve(z_ref, z_bar, z0, d)
        if a_new is None:
            break
        du = max(max(abs(x - y) for x, y in zip(a_new, a)),
                 max(abs(x - y) for x, y in zip(d_new, d)))
        a, d = list(a_new), list(d_new)
        if du < DU_RES:
            break
    return a, d


class MpcTracker:
    """Same interface as ControllerAdapter: control(waypoints, speed, force_reverse=None)."""

    def __init__(self, target_speed=1.6, max_speed=1.8, goal_eps=0.15):
        self.target_speed = target_speed
        self.max_speed = max_speed
        self.goal_eps = goal_eps
        self._a_old = None
        self._d_old = None
        self._gear = None        # latched gear (False=forward, True=reverse) for cusp hysteresis
        self._switch_votes = 0   # consecutive ticks the opposite gear has been committed-and-wanted

    # Reuse the validated cusp split + gear naming from ControllerAdapter so the MPC tracks the
    # same single-gear segments and decides gear identically (DRY; signs already validated there).
    first_segment = staticmethod(ControllerAdapter.first_segment)

    @staticmethod
    def gear_is_reverse(waypoints) -> bool:
        return sum(wp[1] for wp in waypoints) < 0.0

    def _decide_gear(self, waypoints, force_reverse):
        """Two-stage gear hysteresis (deadband + commit-ticks), identical to ControllerAdapter, so
        the MPC can't flicker forward<->reverse at a cusp vertex."""
        if force_reverse is not None:
            return force_reverse
        seg_net = sum(wp[1] for wp in waypoints)
        want_reverse = seg_net < 0.0
        if self._gear is None:
            self._gear = want_reverse
            self._switch_votes = 0
        elif want_reverse != self._gear and abs(seg_net) > MPC_GEAR_DEADBAND:
            self._switch_votes += 1
            if self._switch_votes >= MPC_GEAR_COMMIT_TICKS:
                self._gear = want_reverse
                self._switch_votes = 0
        else:
            self._switch_votes = 0
        return self._gear

    def control(self, waypoints, speed, force_reverse=None, goal_dist=None):
        # Track only the leading single-gear segment (cusp tail driven after the car passes it).
        waypoints = self.first_segment(waypoints) or waypoints
        reach = max(math.hypot(wp[0], wp[1]) for wp in waypoints) if waypoints else 0.0
        if reach < self.goal_eps:
            self._a_old = self._d_old = None
            return 0.0, 1.0, 0.0, False
        reverse = self._decide_gear(waypoints, force_reverse)
        direction = -1.0 if reverse else 1.0

        # Model waypoints (right, forward, head_err) -> MPC frame (x=forward, y=-right, yaw=head_err),
        # prepend the current pose (origin). Resample to a fine reference.
        pts = [(0.0, 0.0, 0.0)] + [
            (wp[1], -wp[0], wp[2] if len(wp) > 2 else 0.0) for wp in waypoints]
        cx, cy, cyaw = _resample(pts)
        sp = [direction * self.target_speed] * len(cx)
        sp[-1] = 0.0

        v0 = direction * float(speed)
        z0 = [0.0, 0.0, v0, 0.0]
        v_adv = direction * max(abs(v0), self.target_speed)  # look ahead at the planned speed
        z_ref, _ = _ref_in_horizon(z0, cx, cy, cyaw, sp, 0, v_adv)
        a, d = _iterative(z_ref, z0, self._a_old, self._d_old)
        self._a_old, self._d_old = a, d
        delta = d[0] if d else 0.0
        # MPC frame steer (delta, CCW+) -> CARLA steer (+ = right): steer = -delta / CARLA max.
        steer_norm = max(-1.0, min(1.0, -delta / CARLA_STEER_MAX))

        # Longitudinal: same arc-length target speed + throttle/brake as ControllerAdapter (the
        # MPC's accel isn't used — steering is the part that was failing).
        arc, prev = 0.0, (0.0, 0.0)
        for wp in waypoints:
            arc += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
            prev = (wp[0], wp[1])
        target_speed = min(self.max_speed, arc / (len(waypoints) * DT_WP))
        target_speed *= max(0.7, 1.0 - 0.3 * abs(steer_norm))
        # Goal taper: slow down as the car nears the slot so it ARRIVES at <0.2 m/s and is_parked()
        # can latch — instead of blasting through the perfect pose at ~1.6 m/s and overshooting
        # (which forces the forward/reverse jockey-back chatter). Linear taper from GOAL_SLOW_M,
        # then a hard cap below the is_parked speed threshold inside GOAL_STOP_M.
        if goal_dist is not None:
            if goal_dist < GOAL_SLOW_M:
                target_speed = min(target_speed, self.max_speed * (goal_dist / GOAL_SLOW_M))
            if goal_dist < GOAL_STOP_M:
                target_speed = min(target_speed, GOAL_STOP_SPEED)
        err = target_speed - speed
        throttle = max(0.0, min(0.6, 0.6 * err + 0.25))
        brake = 0.0
        if err < -0.3:
            throttle, brake = 0.0, max(0.0, min(0.5, -0.5 * err))
        return throttle, brake, steer_norm, reverse
