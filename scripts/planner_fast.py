"""Priority-1 speed patch for the ParkingScenes Hybrid A* planner.

Profiling (scripts/planner_profile.py) showed the planner is already fast on a
SOLVABLE case (~34 expansions, <1s) but blows up on UNREACHABLE / hard cases,
marching toward _MAX_ASTAR_NODES (=2,000,000) which takes hours. Two costs
dominate each expansion:
  1. calc_node_heuristic runs a Reeds-Shepp curve for every child node
     (steering_angle_num*2 = 22 RS solves per expansion).
  2. try_reach_goal runs an RS-curve + full collision check for EVERY popped
     node, because flag_radius (18 m) covers the whole parking lot.

This module patches the (already O(1)-patched) hybrid_a_star to:
  * hard-cap expansions at FAST_MAX_NODES so a stuck plan fails in seconds, not
    hours (raises PlanningTimeout);
  * only pay the RS heuristic when the node is within HEURISTIC_RS_RADIUS of the
    goal (far away, the O(1) Dijkstra grid distance + Euclidean floor guides the
    search); this keeps the search admissible-enough to stay collision-free
    while cutting most RS solves;
  * only attempt the RS goal-connection within FAST_FLAG_RADIUS of the goal.

None of these change collision checking, so produced paths remain collision-free
against the case obstacles (correctness preserved; the path found may differ but
is still a valid obstacle-avoiding trajectory).

Import order matters: apply this AFTER the O(1) patches (generate_episodes.py's
_patch_planners, or planner_profile.apply_current_patches).
"""
import math
import time

# Tunables --------------------------------------------------------------------
# A solvable park needs <100 expansions; a hard-but-solvable case (tight/narrow
# aisle, multi-point turn) can need tens of thousands. We bound the search by a
# WALL-CLOCK budget (primary) plus a node cap (safety), whichever hits first, so
# a genuinely stuck plan gives up in ~5 min instead of the old multi-hour hang —
# while still letting hard-but-solvable scenes (valuable training data) finish.
FAST_MAX_SECONDS = 300.0     # ~5 minute search budget per plan
FAST_MAX_NODES = 80000       # safety cap (~5 min worth of expansions)
# Only run the expensive RS goal-connection attempt this close to the goal.
FAST_FLAG_RADIUS = 8.0
# Only fold the RS length into the heuristic this close to the goal.
HEURISTIC_RS_RADIUS = 10.0


class PlanningTimeout(RuntimeError):
    """Raised when a plan exceeds the time budget / node cap (unreachable/hard)."""


_counter = {"n": 0, "t0": 0.0}


def get_last_expansions():
    return _counter["n"]


def _obstacle_positions(park_map):
    """Obstacle cell (x,y) coordinates, computed ONCE per Map and cached.

    The stock collision checker re-runs np.where(cost_map==255) over the whole
    grid on EVERY check (thousands of times per plan). The obstacle set is fixed
    for a plan, so we cache it. Pure speed change — identical cells returned.
    """
    import numpy as np
    cache = getattr(park_map, "_obs_pos_cache", None)
    if cache is None:
        idx = np.where(park_map.cost_map == 255)
        ox = park_map.map_position[0][idx[0]]
        oy = park_map.map_position[1][idx[1]]
        cache = (ox, oy)
        park_map._obs_pos_cache = cache
    return cache


def _patch_collision_cache():
    """Make distance_checker.get_near_obstacles use the cached obstacle cells."""
    import numpy as np
    from collision_check import collision_check as _cc

    if getattr(_cc, "_fast_collision_patched", False):
        return

    def get_near_obstacles(self, node_x, node_y, theta):
        vehicle_boundary = self.vehicle.create_anticlockpoint(
            x=node_x, y=node_y, theta=theta, config=self.config)
        x_max = max(vehicle_boundary[:, 0]); x_min = min(vehicle_boundary[:, 0])
        y_max = max(vehicle_boundary[:, 1]); y_min = min(vehicle_boundary[:, 1])
        ox, oy = _obstacle_positions(self.map)
        mx = (ox >= x_min) & (ox <= x_max)
        near_x = ox[mx]; near_y = oy[mx]
        my = (near_y >= y_min) & (near_y <= y_max)
        return [near_x[my], near_y[my]], vehicle_boundary

    _cc.collision_checker.get_near_obstacles = get_near_obstacles

    # two_circle_checker.check inlines the same full-grid scan; give it the cache too.
    def circle_check(self, node_x, node_y, theta):
        v = self.vehicle
        Rd = 0.5 * np.sqrt(((v.lr + v.lw + v.lf) / 2) ** 2 + (v.lb ** 2))
        front_circle = (node_x + 1 / 4 * (3 * v.lw + 3 * v.lf - v.lr) * np.cos(theta),
                        node_y + 1 / 4 * (3 * v.lw + 3 * v.lf - v.lr) * np.sin(theta))
        rear_circle = (node_x + 1 / 4 * (v.lw + v.lf - 3 * v.lr) * np.cos(theta),
                       node_y + 1 / 4 * (v.lw + v.lf - 3 * v.lr) * np.sin(theta))
        left = min(front_circle[0], rear_circle[0]) - Rd
        right = max(front_circle[0], rear_circle[0]) + Rd
        down = min(front_circle[1], rear_circle[1]) - Rd
        upper = max(front_circle[1], rear_circle[1]) + Rd
        ox, oy = _obstacle_positions(self.map)
        mx = (ox > left) & (ox < right)
        nx = ox[mx]; ny = oy[mx]
        my = (ny > down) & (ny < upper)
        nx = nx[my]; ny = ny[my]
        for x, y in zip(nx, ny):
            if np.hypot(x - front_circle[0], y - front_circle[1]) <= Rd:
                return True
            if np.hypot(x - rear_circle[0], y - rear_circle[1]) <= Rd:
                return True
        return False

    _cc.two_circle_checker.check = circle_check
    _cc._fast_collision_patched = True


def apply_fast_patches(max_nodes=FAST_MAX_NODES,
                       max_seconds=FAST_MAX_SECONDS,
                       flag_radius=FAST_FLAG_RADIUS,
                       heuristic_rs_radius=HEURISTIC_RS_RADIUS):
    from path_plan import hybrid_a_star as _ha
    from path_plan import rs_curve as _rs

    _patch_collision_cache()

    orig_init = _ha.hybrid_a_star.__init__
    orig_expand = _ha.hybrid_a_star.expand_node
    orig_reach = _ha.hybrid_a_star.try_reach_goal

    def _init(self, config, park_map, vehicle):
        orig_init(self, config, park_map, vehicle)
        _counter["n"] = 0
        _counter["t0"] = time.perf_counter()

    def _calc_h(self, current_node):
        # O(1) Dijkstra grid distance (metres).
        gid = self.park_map.convert_position_to_index(current_node.x, current_node.y)
        h_grid = self._h_dict.get(gid, 0) / 100.0
        dx = current_node.x - self.goal_node.x
        dy = current_node.y - self.goal_node.y
        euclid = math.hypot(dx, dy)
        # Far from goal: cheap heuristic (no RS solve).
        if h_grid > heuristic_rs_radius and euclid > heuristic_rs_radius:
            return max(h_grid, euclid)
        # Near goal: pay for the kinematically-aware RS length.
        max_c = 1 / self.vehicle.min_radius_turn
        rs_path = _rs.calc_optimal_path(
            sx=current_node.x, sy=current_node.y, syaw=current_node.theta,
            gx=self.goal_node.x, gy=self.goal_node.y, gyaw=self.goal_node.theta, maxc=max_c)
        return max(h_grid, rs_path.L)

    def _expand(self, current_node):
        _counter["n"] += 1
        if _counter["n"] > max_nodes:
            raise PlanningTimeout(
                f"A* exceeded {max_nodes} expansions — unreachable/hard case")
        if (_counter["n"] & 0x3FF) == 0 and (time.perf_counter() - _counter["t0"]) > max_seconds:
            raise PlanningTimeout(
                f"A* exceeded {max_seconds:.0f}s search budget — unreachable/hard case")
        return orig_expand(self, current_node)

    def _reach(self, current_node):
        # Skip the RS-to-goal attempt (and its collision check) when far away.
        dist = math.hypot(current_node.x - self.goal_node.x,
                          current_node.y - self.goal_node.y)
        if dist > flag_radius:
            return None, False, {"in_radius": False, "collision_position": None}
        return orig_reach(self, current_node)

    _ha.hybrid_a_star.__init__ = _init
    _ha.hybrid_a_star.expand_node = _expand
    _ha.hybrid_a_star.try_reach_goal = _reach
    _ha.hybrid_a_star.calc_node_heuristic = _calc_h
