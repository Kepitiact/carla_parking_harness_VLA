"""Standalone profiler / benchmark for the ParkingScenes Hybrid A* planner.

Runs WITHOUT CARLA and WITHOUT the collector: it builds a representative
Town04 reverse-perpendicular parking case (start in the aisle, goal flanked by
parked cars) directly as a BenchmarkCases-style CSV, then times the planner.

Two modes let us measure the effect of the Priority-1 speed fix:
  --variant current  : reproduce the O(1) monkey-patches that generate_episodes.py
                       applies today (baseline).
  --variant fast     : additionally apply the lot-bounded search + cheap heuristic
                       (the Priority-1 fix, imported from planner_fast when present).

Usage:
  venv/bin/python scripts/planner_profile.py --variant current --profile
  venv/bin/python scripts/planner_profile.py --variant fast
"""
import argparse
import cProfile
import io
import math
import os
import pstats
import sys
import time
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = pathlib.Path(__file__).resolve().parent.parent
_AVP = _REPO / "ParkingScenes" / "tool" / "AutomatedValetParking"
sys.path.insert(0, str(_AVP))


# ── Representative case builders ──────────────────────────────────────────────
def _box(cx, cy, half_len=2.4, half_wid=1.0):
    """A car-footprint rectangle centred at (cx,cy), long axis along x (row car)."""
    return [
        (cx + half_len, cy - half_wid),
        (cx + half_len, cy + half_wid),
        (cx - half_len, cy + half_wid),
        (cx - half_len, cy - half_wid),
    ]


def build_town04_case(n_obstacles="full", block_goal=False):
    """Return (start, goal, obstacles) for a Town04 reverse-perpendicular park.

    Geometry mirrors parking_position.parking_vehicle_locations_Town04: row-1 at
    x=298.5, row-2 at x=290.9, row-3 at x=280.0; aisle centreline x=285.6.
    Goal = a row-2 slot; obstacles = parked cars in the neighbouring bays.
    block_goal=True walls the aisle side of the goal bay so the goal is
    unreachable — this reproduces the pathological "explore everything" blow-up.
    """
    goal = (290.9, -235.73, math.pi)          # back into row-2 bay, facing -x
    start = (285.6, -228.0, -math.pi / 2)      # in the aisle, facing -y

    rows_x = [298.5, 290.9, 280.0]
    ys = [-238.9, -235.73, -232.73, -229.53, -226.43, -223.43, -240.0, -220.2]
    obstacles = []
    for rx in rows_x:
        for py in ys:
            if abs(rx - goal[0]) < 0.5 and abs(py - goal[1]) < 0.5:
                continue  # leave the goal bay empty
            obstacles.append(_box(rx, py))
    if block_goal:
        # cars parked immediately in front of the goal bay (aisle side, x=287.5)
        obstacles.append(_box(287.3, goal[1], half_len=0.6, half_wid=1.4))
    if n_obstacles != "full":
        obstacles = obstacles[: int(n_obstacles)]
    return start, goal, obstacles


def write_case_csv(path, start, goal, obstacles):
    vals = [start[0], start[1], start[2], goal[0], goal[1], goal[2], len(obstacles)]
    vals += [len(o) for o in obstacles]
    for o in obstacles:
        for (x, y) in o:
            vals += [x, y]
    path.write_text(",".join(str(v) for v in vals))


# ── Current-behaviour O(1) patches (copied from generate_episodes._patch_planners) ──
class _NodeCap(RuntimeError):
    pass


def apply_current_patches(max_nodes=2_000_000):
    import queue as _q
    import math as _math
    import numpy as _np
    from path_plan import hybrid_a_star as _ha
    from path_plan import compute_h as _ch
    from path_plan import rs_curve as _rs

    _orig_dijk_init = _ch.Dijkstra.__init__

    def _dijk_init(self, map):
        _orig_dijk_init(self, map)
        self.openlist_index = set()
        self.openlist_dict = {}

    def _dijk_add(self, gridx, gridy, priority, father_id):
        index = self.map.convert_position_to_index(gridx, gridy)
        if index in self.openlist_index:
            node = self.openlist_dict[index]
            if node.distance > priority:
                node.distance = priority
                node.father_id = father_id
        else:
            grid_node = _ch.Grid(grid_id=index, grid_x=gridx, grid_y=gridy,
                                 distance=priority, father_id=father_id)
            self.open_list.put(grid_node)
            self.openlist_index.add(index)
            self.openlist_dict[index] = grid_node

    _ch.Dijkstra.__init__ = _dijk_init
    _ch.Dijkstra.add_grid_to_openlist = _dijk_add

    def _astar_key(x, y, theta, ds=0.1, dth=0.05):
        return (round(x / ds), round(y / ds), round(theta / dth))

    _orig_ha_init = _ha.hybrid_a_star.__init__
    stats = {"expand": 0, "reach_goal_calls": 0}

    def _ha_init(self, config, park_map, vehicle):
        _orig_ha_init(self, config, park_map, vehicle)
        self.goal_node.theta = _rs.pi_2_pi(self.goal_node.theta)
        self.initial_node.theta = _rs.pi_2_pi(self.initial_node.theta)
        self._h_dict = {g.grid_id: g.distance for g in self.h_value_list}
        self._closed_set = {_astar_key(n.x, n.y, n.theta) for n in self.closed_list}
        self._open_dict = {}

    def _ha_calc_h(self, current_node):
        _grid_id = self.park_map.convert_position_to_index(current_node.x, current_node.y)
        h1 = self._h_dict.get(_grid_id, 0) / 100
        max_c = 1 / self.vehicle.min_radius_turn
        rs_path = _rs.calc_optimal_path(
            sx=current_node.x, sy=current_node.y, syaw=current_node.theta,
            gx=self.goal_node.x, gy=self.goal_node.y, gyaw=self.goal_node.theta, maxc=max_c)
        return max(h1, rs_path.L)

    def _ha_expand(self, current_node):
        stats["expand"] += 1
        if stats["expand"] > max_nodes:
            raise _NodeCap(f"exceeded {max_nodes} expansions")
        child_group = _q.PriorityQueue()
        next_index = int(2 * self.config['steering_angle_num'])
        for i in range(next_index):
            steering_angle = self.steering_angle[i % self.config['steering_angle_num']]
            is_forward = i < next_index / 2
            speed = self.vehicle.max_v if is_forward else -self.vehicle.max_v
            travel = speed * self.dt
            theta_ = _rs.pi_2_pi(current_node.theta +
                (self.vehicle.max_v * _np.tan(steering_angle)) / self.vehicle.lw * self.dt)
            x_ = current_node.x + travel * _math.cos(theta_)
            y_ = current_node.y + travel * _math.sin(theta_)
            if (x_ > self.park_map.boundary[1] or x_ < self.park_map.boundary[0] or
                    y_ > self.park_map.boundary[3] or y_ < self.park_map.boundary[2]):
                continue
            key = _astar_key(x_, y_, theta_)
            if key in self._closed_set:
                continue
            if key in self._open_dict:
                child_node = self._open_dict[key]
                find_open = True
            else:
                find_open = False
                child_node = _ha.Node(x=x_, y=y_, theta=theta_,
                                      index=self.global_index + i + 1,
                                      parent_index=current_node.index,
                                      is_forward=is_forward, steering_angle=steering_angle)
                collision = False
                for j in range(_math.ceil(self.dt / self.ddt)):
                    tj = speed * self.ddt * (j + 1)
                    thj = _rs.pi_2_pi(current_node.theta +
                        (self.vehicle.max_v * _np.tan(steering_angle)) / self.vehicle.lw * self.ddt * (j + 1))
                    xj = current_node.x + tj * _math.cos(thj)
                    yj = current_node.y + tj * _math.sin(thj)
                    collision = self.collision_checker.check(node_x=xj, node_y=yj, theta=thj)
                    if collision:
                        self.closed_list.append(child_node)
                        self._closed_set.add(key)
                        child_node.in_closed = True
                        break
                if not collision:
                    child_node.g = self.calc_node_cost(child_node,
                        father_theta=current_node.theta, father_gear=current_node.forward)
                    child_node.h = self.calc_node_heuristic(child_node)
                    child_node.f = child_node.g + child_node.h
                    self.open_list.put(child_node)
                    self._open_dict[key] = child_node
                    child_node.in_open = True
            if find_open:
                new_g = self.calc_node_cost(child_node, father_theta=current_node.theta,
                                            father_gear=current_node.forward)
                new_h = self.calc_node_heuristic(child_node)
                new_f = new_g + new_h
                if new_f < child_node.f:
                    child_node.f, child_node.g, child_node.h = new_f, new_g, new_h
                    child_node.parent_index = current_node.index
                    child_node.forward = is_forward
                    child_node.steering_angle = steering_angle
            if not child_node.in_closed and child_node.in_open:
                child_group.put(child_node)
        current_node.in_closed = True
        current_node.in_open = False
        self.closed_list.append(current_node)
        self._closed_set.add(_astar_key(current_node.x, current_node.y, current_node.theta))
        self.global_index += next_index
        return child_group

    _orig_reach = _ha.hybrid_a_star.try_reach_goal

    def _reach(self, current_node):
        stats["reach_goal_calls"] += 1
        return _orig_reach(self, current_node)

    _ha.hybrid_a_star.__init__ = _ha_init
    _ha.hybrid_a_star.expand_node = _ha_expand
    _ha.hybrid_a_star.calc_node_heuristic = _ha_calc_h
    _ha.hybrid_a_star.try_reach_goal = _reach
    return stats


def run(case_csv, profile=False):
    from config import read_config
    from map import costmap
    from path_plan import path_planner

    config = read_config.read_config(config_name="config")
    park_map = costmap.Map(file=str(case_csv), discrete_size=config["map_discrete_size"])
    vehicle = costmap.Vehicle()

    gx = int((park_map.boundary[1] - park_map.boundary[0]) / park_map._discrete_x)
    gy = int((park_map.boundary[3] - park_map.boundary[2]) / park_map._discrete_y)
    n_obs_cells = int((park_map.cost_map == 255).sum())
    print(f"[MAP] boundary={park_map.boundary.tolist()} grid={gx}x{gy}={gx*gy} cells | obstacle_cells={n_obs_cells}")

    planner = path_planner.PathPlanner(config=config, map=park_map, vehicle=vehicle)

    t0 = time.perf_counter()
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    final_path = []
    split = []
    capped = False
    try:
        final_path, path_info, split = planner.path_planning()
    except RuntimeError as e:
        capped = True
        print(f"[RESULT] SEARCH ABORTED: {e}")
    if profile:
        pr.disable()
    dt = time.perf_counter() - t0

    if not capped:
        print(f"[RESULT] path_planning: {dt:.2f}s | path_pts={len(final_path)} | segments={len(split)}")
    else:
        print(f"[RESULT] gave up after {dt:.2f}s (unreachable/hard case)")
    if profile:
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
        print(s.getvalue())
    return dt, final_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["current", "fast"], default="current")
    ap.add_argument("--obstacles", default="full")
    ap.add_argument("--block-goal", action="store_true")
    ap.add_argument("--max-nodes", type=int, default=200000)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    stats = apply_current_patches(max_nodes=args.max_nodes)
    if args.variant == "fast":
        import planner_fast  # noqa: applies the Priority-1 speed patch on import
        planner_fast.apply_fast_patches()

    start, goal, obstacles = build_town04_case(args.obstacles, block_goal=args.block_goal)
    case_csv = _AVP / "BenchmarkCases" / "PROFILE_CASE.csv"
    case_csv.parent.mkdir(parents=True, exist_ok=True)
    write_case_csv(case_csv, start, goal, obstacles)
    print(f"[CASE] start={start} goal={goal} obstacles={len(obstacles)}")

    dt, path = run(case_csv, profile=args.profile)
    print(f"[STATS] expansions={stats['expand']} try_reach_goal_calls={stats['reach_goal_calls']}")


if __name__ == "__main__":
    main()
