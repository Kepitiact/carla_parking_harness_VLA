"""Ego-local <-> world frame transforms for the CARLA client (controller + viewer).

Conventions (see ../../ARCHITECTURE.md):
  - CARLA world:            left-handed, y points south.
  - nuScenes/OpenDRIVE (n): y_n = -y_carla ;  yaw_n = -yaw_carla.
  - Ego-local (model + controller): x = right, y = forward.

The model outputs waypoints as (right, forward) in the ego frame. To drive the controller
they must become CARLA world points. `ego_local_to_odr` is the exact algebraic inverse of
the VALIDATED live_prompt.global_to_local_xy (the one that byte-matched the offline cache),
so the round-trip is guaranteed — see harness/tests/test_coord_sanity.py.
"""
import math


def carla_to_odr_pose(x_c, y_c, yaw_c_rad):
    return x_c, -y_c, -yaw_c_rad


def odr_to_carla_pose(x_n, y_n, yaw_n_rad):
    return x_n, -y_n, -yaw_n_rad


def ego_local_to_odr(right, forward, ex_n, ey_n, eyaw_n):
    """Ego-local (right, forward) -> world (nuScenes/OpenDRIVE) (x, y). Inverse of
    live_prompt.global_to_local_xy: there forward=c*dx+s*dy, right=s*dx-c*dy (c=cos,s=sin
    of the ego yaw); solving for (dx,dy) gives the expressions below."""
    c, s = math.cos(eyaw_n), math.sin(eyaw_n)
    dx = c * forward + s * right
    dy = s * forward - c * right
    return ex_n + dx, ey_n + dy


def ego_local_to_carla(right, forward, ex_c, ey_c, eyaw_c_rad):
    """Ego-local (right, forward) waypoint at a CARLA ego pose -> CARLA world (x, y)."""
    ex_n, ey_n, eyaw_n = carla_to_odr_pose(ex_c, ey_c, eyaw_c_rad)
    wn_x, wn_y = ego_local_to_odr(right, forward, ex_n, ey_n, eyaw_n)
    return wn_x, -wn_y  # nuScenes -> CARLA


def waypoints_to_carla_path(waypoints, ex_c, ey_c, eyaw_c_rad):
    """6 ego-local (right, forward) waypoints -> CARLA world polyline [(x,y),...].
    The current ego position is prepended as the path origin."""
    pts = [(ex_c, ey_c)]
    for wp in waypoints:  # wp = (right, forward[, heading]); heading unused for the path
        pts.append(ego_local_to_carla(wp[0], wp[1], ex_c, ey_c, eyaw_c_rad))
    return pts


def path_headings(pts):
    """Tangent yaw (CARLA rad) for each point of a polyline; last repeats the previous."""
    yaws = []
    for i in range(len(pts) - 1):
        yaws.append(math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]))
    yaws.append(yaws[-1] if yaws else 0.0)
    return yaws
