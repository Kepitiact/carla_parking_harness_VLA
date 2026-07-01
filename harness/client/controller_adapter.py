"""Track the model's 6 ego-local waypoints with a pure-pursuit controller.

Input each tick: the model's waypoints as (right, forward) in the CURRENT ego frame, plus the
ego's current speed. Output: throttle / brake / steer / reverse for carla.VehicleControl.

Pure pursuit is done DIRECTLY in the ego-local frame (the waypoints already live there), so no
world transform is needed for control. Gear is read from the trajectory's net forward motion
(reverse_perpendicular trajectories have negative forward) — the ARCHITECTURE.md gear/reverse
requirement. Signs were validated against a kinematic bicycle sim (test_controller_sim.py):
a forward arc and a REVERSE arc both converge onto the path.

Self-contained (no carla import) so it unit-tests without a sim; `to_vehicle_control` wraps
the tuple into carla.VehicleControl when carla is available.
"""
from __future__ import annotations

import math

WB = 2.5                       # wheelbase [m] (matches MPC P.WB)
MAX_STEER_RAD = math.radians(70)  # CARLA front-wheel max
DT_WP = 0.5                    # waypoint spacing [s]
TURN_GAP = 0.12                # rad; if the model's heading delta across the plan exceeds the
                               # position polyline's own tangent turn by more than this, the
                               # positions under-specify the rotation (reverse-perpendicular case)
                               # and the heading channel supplies the missing curvature
GEAR_DEADBAND = 0.3            # m; near a cusp the leading segment shrinks to ~0 net forward and
                               # its sign chatters. Hold the current gear until the new leading
                               # segment is COMMITTED (|net forward| exceeds this), so the gear
                               # can't flicker tick-to-tick at the cusp vertex.
GEAR_COMMIT_TICKS = 4          # consecutive ticks the OPPOSITE gear must be committed-and-wanted
                               # before the controller actually switches. The deadband alone leaves
                               # brief flicker where the leading segment swings across +-deadband as
                               # remaining_ego_path trims points; requiring N consistent ticks turns
                               # those momentary excursions into no-ops. At 30Hz, 4 ticks ~= 0.13s.
GOAL_SLOW_M = 3.0              # start tapering speed when within this distance of the slot [m]
GOAL_STOP_M = 0.6             # inside this distance, cap speed below the is_parked threshold so the
GOAL_STOP_SPEED = 0.12        # car ARRIVES parked (<0.2 m/s) instead of blasting through the slot


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class ControllerAdapter:
    def __init__(self, wheelbase=WB, lookahead_min=1.8, lookahead_k=0.8,
                 max_speed=1.8, max_steer_rad=MAX_STEER_RAD, goal_eps=0.15, k_heading=1.0):
        self.WB = wheelbase
        self.ld_min = lookahead_min
        self.ld_k = lookahead_k
        self.max_speed = max_speed
        self.max_steer = max_steer_rad
        self.goal_eps = goal_eps  # if the whole prediction is within this, brake (model says stay)
        # How much to follow the model's emitted per-waypoint heading when the (near-collinear)
        # position waypoints under-specify the reverse-perpendicular rotation. 0 = position-only
        # (legacy), 1 = fully supply the missing rotation. Tunable live via HEADING_GAIN.
        self.k_heading = k_heading
        self._gear = None        # latched gear (False=forward, True=reverse) for cusp hysteresis
        self._switch_votes = 0   # consecutive ticks the opposite gear has been committed-and-wanted

    @staticmethod
    def gear_is_reverse(waypoints) -> bool:
        """Reverse gear when the trajectory's net forward motion is backward.
        Waypoints may be (right, forward) or (right, forward, heading); only forward is used."""
        return sum(wp[1] for wp in waypoints) < 0.0

    @staticmethod
    def first_segment(waypoints):
        """Leading single-gear segment of the plan: waypoints up to (and including) the first
        CUSP — the point where the plan's forward MOTION reverses direction. A reverse-
        perpendicular park is planned forward-then-reverse in one 6-waypoint horizon; tracking the
        whole plan at once collapses it to sum(forward) and the car only ever executes HALF the
        maneuver, see-sawing between replans (the classifier's 'context' failures). Driving the
        first segment to the cusp, then — once those points are trimmed away as the car passes
        them — the next segment, executes BOTH phases. No cusp -> the whole plan is one segment."""
        if len(waypoints) < 2:
            return list(waypoints)
        prev_f = 0.0
        first_dir = 0
        cut = len(waypoints)
        for i, wp in enumerate(waypoints):
            df = wp[1] - prev_f
            d = 1 if df > 1e-3 else (-1 if df < -1e-3 else 0)
            if d != 0:
                if first_dir == 0:
                    first_dir = d
                elif d != first_dir:      # motion reversed here: cut so the cusp vertex is last
                    cut = i
                    break
            prev_f = wp[1]
        return list(waypoints[:cut])

    @staticmethod
    def _path_tangent_turn(waypoints) -> float:
        """Net signed rotation (rad) of the position polyline's tangent (origin -> wp1 -> ...).
        ~0 when the waypoints are collinear (rotation is carried only by the heading channel);
        large on a genuinely curved path, where pure pursuit already tracks the rotation."""
        pts = [(0.0, 0.0)] + [(wp[0], wp[1]) for wp in waypoints]
        segs = [(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
        segs = [s for s in segs if math.hypot(*s) > 1e-6]
        turn = 0.0
        for a, b in zip(segs, segs[1:]):
            turn += math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
        return turn

    def _lookahead_point(self, waypoints, ld):
        """Walk the polyline origin->wp1->... and return the (right, forward, head_err) point ~ld
        along it (by arc length); head_err is the model's heading at that point, linearly
        interpolated. Falls back to the farthest waypoint if the path is shorter than ld."""
        prev = (0.0, 0.0, 0.0)
        acc = 0.0
        for wp in waypoints:
            r, f = wp[0], wp[1]
            h = wp[2] if len(wp) > 2 else 0.0
            seg = math.hypot(r - prev[0], f - prev[1])
            if acc + seg >= ld and seg > 1e-6:
                t = (ld - acc) / seg
                return (prev[0] + t * (r - prev[0]), prev[1] + t * (f - prev[1]),
                        prev[2] + t * (h - prev[2]))
            acc += seg
            prev = (r, f, h)
        return prev  # farthest point

    def control(self, waypoints, speed, force_reverse=None, goal_dist=None):
        """waypoints: [(right, forward), ...]; speed: current |ego speed| m/s.
        force_reverse: if not None, force this gear instead of inferring it from the plan
        (tests/back-compat). Otherwise the gear comes from the leading single-gear SEGMENT of the
        plan (up to the first cusp), so a forward-then-reverse plan is executed forward first, then
        reverse once the forward points are trimmed away — not flattened.
        goal_dist: distance (m) to the slot, used to taper speed near the goal so the car arrives
        slow enough to latch is_parked() instead of overshooting. Returns
        (throttle, brake, steer_norm, reverse)."""
        # Track only the leading single-gear segment; the cusp tail is driven after the car passes
        # the cusp (remaining_ego_path trims the consumed points, exposing the next segment).
        waypoints = self.first_segment(waypoints) or waypoints
        # Path essentially empty -> the model wants to hold position: brake.
        reach = max(math.hypot(wp[0], wp[1]) for wp in waypoints) if waypoints else 0.0
        if reach < self.goal_eps:
            return 0.0, 1.0, 0.0, False

        if force_reverse is not None:
            reverse = force_reverse
        else:
            # Gear from the leading segment, with TWO-STAGE hysteresis so the gear can't human-jiggle
            # at the cusp:
            #   1) deadband — a switch is only "wanted" when the new segment's net forward motion is
            #      committed (|net| > GEAR_DEADBAND); tiny ~0 sign flips at the cusp are ignored.
            #   2) commit-ticks — even a wanted switch must persist GEAR_COMMIT_TICKS consecutive
            #      ticks before it takes effect; a momentary excursion (the leading segment briefly
            #      crossing the deadband as points are trimmed) decays back to 0 votes and is a no-op.
            # First call adopts immediately (no gear to hold yet).
            seg_net = sum(wp[1] for wp in waypoints)
            want_reverse = seg_net < 0.0
            if self._gear is None:
                self._gear = want_reverse
                self._switch_votes = 0
            elif want_reverse != self._gear and abs(seg_net) > GEAR_DEADBAND:
                self._switch_votes += 1
                if self._switch_votes >= GEAR_COMMIT_TICKS:
                    self._gear = want_reverse
                    self._switch_votes = 0
            else:
                self._switch_votes = 0   # agree with current gear (or in deadband) -> reset vote
            reverse = self._gear
        # Total remaining arc length of the plan (used for both the heading curvature and speed).
        arc, prev = 0.0, (0.0, 0.0)
        for wp in waypoints:
            arc += math.hypot(wp[0] - prev[0], wp[1] - prev[1])
            prev = (wp[0], wp[1])

        # Look further ahead on a long, gently-curving plan so pure pursuit reacts to the
        # PATH's net curvature instead of slamming onto the curved near-segment (the 2*WB gain
        # over-reacts to a close lookahead, full-locks, and — with no heading damping — overshoots
        # past alignment into a spiral). Scaling with 'reach' keeps the long reverse arcs gentle.
        ld = max(self.ld_min, self.ld_k * speed, 0.5 * reach)
        rl, fl, _ = self._lookahead_point(waypoints, ld)
        ld_eff = max(math.hypot(rl, fl), 1e-3)
        sign_v = -1.0 if reverse else 1.0

        # Pure-pursuit curvature in the ego frame (x=right, y=forward). Reverse measures the
        # lookahead from -forward; the rear-axle geometry keeps the SAME steer sign as forward
        # (validated against the reverse-arc kinematic sim).
        alpha = math.atan2(rl, fl if not reverse else -fl)
        kappa_pp = 2.0 * math.sin(alpha) / ld_eff
        # The model expresses rotation in BOTH the positions and the heading channel. Compare how
        # much the (remaining) plan rotates in each: the position polyline's own tangent turn vs the
        # heading delta ACROSS the plan (last - first head_err — the difference cancels the car's
        # current heading offset, so on an ordinary arc the two agree and the gap is ~0). On a
        # reverse-perpendicular arc the positions barely turn while the heading wants a big rotation
        # -> a positive gap -> supply the missing curvature (gap spread over the plan's arc). Sign
        # (-sign_v): forward +yaw needs left steer, reverse +yaw needs right (per the bicycle sim).
        pos_turn = self._path_tangent_turn(waypoints)
        plan_turn = (waypoints[-1][2] - waypoints[0][2]) if len(waypoints[0]) > 2 else 0.0
        # Engage only when the heading wants MORE rotation (in magnitude, either direction) than the
        # positions provide — the under-rotating reverse-perp case. If the positions already turn as
        # much or more, trust pure pursuit. Supply the signed gap, spread over the plan's arc.
        if abs(plan_turn) - abs(pos_turn) > TURN_GAP:
            kappa = kappa_pp + self.k_heading * (-sign_v) * (plan_turn - pos_turn) / max(arc, 1e-3)
        else:
            kappa = kappa_pp
        steer_norm = _clamp(math.atan(self.WB * kappa) / self.max_steer, -1.0, 1.0)
        # Near the goal the plan is short and trimming leaves only the lateral end-waypoint, so
        # pure pursuit demands full lock and the car circles in the slot instead of backing
        # straight in. Damp the steer as the remaining plan shrinks below one lookahead: when
        # we're basically there, small corrections, not full-lock loops.
        steer_norm *= _clamp(reach / self.ld_min, 0.35, 1.0)

        # Longitudinal: match the speed the model's OWN plan implies. The waypoints are spaced
        # at DT_WP (0.5s), so the plan's arc length over its horizon is the trained speed
        # (~1.7 m/s for these reverse maneuvers). Using the whole-horizon ARC LENGTH (not the
        # euclidean 'reach', which under-reads a curling reverse arc) keeps the car fast enough
        # to actually traverse a long reverse plan within the 3s replan window — at half speed
        # it only ever executes the first half of each plan and never reaches the slot. The arc
        # shrinks to ~0 near the goal so the car slows; the reach<goal_eps brake stops it.
        target_speed = min(self.max_speed, arc / (len(waypoints) * DT_WP))
        # Mild slowdown for very sharp turns only (guards overshoot on short tight arcs).
        target_speed *= max(0.7, 1.0 - 0.3 * abs(steer_norm))
        # Goal taper: slow as the car nears the slot so it ARRIVES at <0.2 m/s and is_parked()
        # latches, instead of blasting through the slot at speed and overshooting (which forces the
        # forward/reverse jockey-back chatter). Linear from GOAL_SLOW_M, hard cap inside GOAL_STOP_M.
        if goal_dist is not None:
            if goal_dist < GOAL_SLOW_M:
                target_speed = min(target_speed, self.max_speed * (goal_dist / GOAL_SLOW_M))
            if goal_dist < GOAL_STOP_M:
                target_speed = min(target_speed, GOAL_STOP_SPEED)
        err = target_speed - speed
        throttle = _clamp(0.6 * err + 0.25, 0.0, 0.6)
        brake = 0.0
        if err < -0.3:                            # overspeed -> brake instead
            throttle, brake = 0.0, _clamp(-0.5 * err, 0.0, 0.5)
        return throttle, brake, steer_norm, reverse

    def to_vehicle_control(self, waypoints, speed):
        import carla
        thr, brk, steer, rev = self.control(waypoints, speed)
        return carla.VehicleControl(throttle=float(thr), brake=float(brk),
                                    steer=float(steer), reverse=bool(rev))
