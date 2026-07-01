"""Shared obstacle-aware expert planner for BOTH collectors.

Used by scripts/generate_episodes.py (episode collection) and the harness /
DAgger relabel path. It replaces the old Reeds-Shepp bypass with the REAL
ParkingScenes Hybrid A* (kinematic bicycle-model primitives + Reeds-Shepp
goal curves + collision checking against the spawned obstacles), sped up by
scripts/planner_fast.py, and writes the resulting collision-free trajectory
straight to solution/CARLA.csv in the exact format the MPC tracker consumes —
bypassing the slow/fragile pyomo OCP stage.

Why bypass the OCP:
  * The Hybrid A* path is already collision-free (every edge is collision-checked
    against the case obstacles) AND kinematically feasible (edges are max-steer
    bicycle-model arcs at the vehicle's min turn radius); the MPC tracks it
    directly, exactly as it tracked the RS bypass before.
  * The OCP (path_optimazition + pyomo/ipopt) is the main source of the original
    multi-hour / infeasible-constraint failures.

Cache key = md5 of the FULL BenchmarkCases/CARLA.csv (start + goal + EVERY
obstacle vertex). This fixes the correctness bug where the old cache keyed only
on start/goal, so a plan computed for one spawned-obstacle subset was wrongly
reused for a different subset — letting an "expert" path clip a rendered car.

Solution CSV format (tab-delimited, consumed by MotionPlanning/Control/MPC.py
read_specific_rows_from_csv): [idx, x, y, yaw, v, sin(yaw), cos(yaw)]
  * columns 1,2,3 = rear-axle x, y, yaw (planning frame -> CARLA frame here)
  * column 4 = v: SIGN encodes gear (v>0 forward, v<0 reverse). Row 0 v=0.0 is
    skipped by the reader (documented behaviour), so it seeds no direction.
"""
import csv
import hashlib
import math
import os
import pathlib
import shutil
import sys


class PlanRejected(RuntimeError):
    """Raised when no acceptable collision-free plan could be produced."""


def _obstacle_aware_key(bench_csv: pathlib.Path, index) -> str:
    """md5 over the whole case file (start+goal+obstacles) => obstacle-aware."""
    raw = bench_csv.read_bytes()
    return "astar_" + hashlib.md5((f"slot{index}_").encode() + raw).hexdigest()[:12]


def _gear_signs(path):
    """Per-waypoint velocity sign from motion-vector vs heading (MPC semantics).

    Mirrors MotionPlanning/Control/MPC.calc_speed_profile: at each point, if the
    step direction opposes the heading it is a reverse (v<0) sample, else forward.
    """
    n = len(path)
    signs = [1.0] * n
    for i in range(n - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        yaw = path[i][2]
        move_ang = math.atan2(dy, dx)
        d = abs(_pi_2_pi(move_ang - yaw))
        signs[i] = -1.0 if d >= math.pi / 2 else 1.0
    if n >= 2:
        signs[-1] = signs[-2]
    return signs


def _pi_2_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _count_gear_changes(signs):
    changes = 0
    prev = None
    for s in signs:
        cur = 1 if s > 0 else -1
        if prev is not None and cur != prev:
            changes += 1
        prev = cur
    return changes


def write_astar_solution(path, coordinate_rotation, out_csv: pathlib.Path):
    """Write a Hybrid A* [x,y,theta] path to solution/CARLA.csv (MPC format).

    When coordinate_rotation=True the path is in the rotated planning frame; we
    apply the same inverse rotation plan.py uses:
        carla_x = plan_y,  carla_y = -plan_x,  carla_yaw = plan_yaw - pi/2
    """
    signs = _gear_signs(path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for i, (x, y, yaw) in enumerate(path):
            if coordinate_rotation:
                ox, oy, oyaw = y, -x, yaw - math.pi / 2
            else:
                ox, oy, oyaw = x, y, yaw
            oyaw = _pi_2_pi(oyaw)
            v = 0.0 if i == 0 else signs[i]
            w.writerow([i, ox, oy, oyaw, v, math.sin(oyaw), math.cos(oyaw)])
    return _count_gear_changes(signs[1:])


def run_astar_path(bench_csv: pathlib.Path):
    """Run the REAL Hybrid A* on the case file; return final [x,y,theta] path.

    Assumes the AutomatedValetParking dir is already on sys.path and that the
    O(1) + fast speed patches have been applied (see ensure_patched)."""
    from config import read_config
    from map import costmap
    from path_plan import path_planner

    config = read_config.read_config(config_name="config")
    park_map = costmap.Map(file=str(bench_csv), discrete_size=config["map_discrete_size"])
    vehicle = costmap.Vehicle()
    planner = path_planner.PathPlanner(config=config, map=park_map, vehicle=vehicle)
    final_path, _info, _split = planner.path_planning()
    return final_path, planner


def _verify_collisionfree(planner, path):
    checker = planner.collision_checker
    return [p for p in path if checker.check(node_x=p[0], node_y=p[1], theta=p[2])]


_PATCHED = {"done": False}


def ensure_patched(scripts_dir: pathlib.Path):
    """Apply the fast speed patch (node cap + lazy RS + obstacle-cell cache) once.

    The O(1) base patch is applied separately by generate_episodes._patch_planners;
    this adds the Priority-1 speed layer. Import path must already include the
    AutomatedValetParking dir.
    """
    if _PATCHED["done"]:
        return
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import planner_fast
    planner_fast.apply_fast_patches()
    _PATCHED["done"] = True


def make_cached_plan(repo_root: pathlib.Path, orig_plan=None,
                     max_gear_changes=8, logger=print):
    """Build a plan(coordinate_rotation, index) replacement for the collectors.

    Drop-in for ParkingScenes' plan(): reads BenchmarkCases/CARLA.csv (written by
    Auto_Park.perception from the LIVE spawned obstacles), runs the fast
    obstacle-aware Hybrid A*, and writes solution/CARLA.csv. Results are cached by
    obstacle-aware key. `orig_plan` (the stock plan()) is used as a last-resort
    fallback if provided.
    """
    avp = repo_root / "ParkingScenes" / "tool" / "AutomatedValetParking"
    bench_csv = avp / "BenchmarkCases" / "CARLA.csv"
    sol_csv = avp / "solution" / "CARLA.csv"
    cache_dir = repo_root / "data" / "plan_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = repo_root / "scripts"

    def _plan(coordinate_rotation, index):
        ensure_patched(scripts_dir)
        if not bench_csv.exists():
            raise PlanRejected("no BenchmarkCases/CARLA.csv (perception did not run)")
        key = _obstacle_aware_key(bench_csv, index)
        cache_file = cache_dir / f"plan_{key}.csv"

        if cache_file.exists():
            logger(f"[EXPERT] cache hit {key}")
            sol_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(cache_file), str(sol_csv))
            return

        logger(f"[EXPERT] cache miss {key} — running obstacle-aware Hybrid A*...")
        import time as _t
        t0 = _t.perf_counter()
        try:
            path, planner = run_astar_path(bench_csv)
        except Exception as e:  # PlanningTimeout or solver error
            logger(f"[EXPERT] A* failed: {e}")
            raise PlanRejected(f"A* failed: {e}")
        dt = _t.perf_counter() - t0

        hits = _verify_collisionfree(planner, path)
        if hits:
            logger(f"[EXPERT] REJECT: path collides at {len(hits)} pts ({dt:.1f}s)")
            raise PlanRejected(f"A* path collides at {len(hits)} points")

        gc = write_astar_solution(path, coordinate_rotation, sol_csv)
        logger(f"[EXPERT] A* ok: {dt:.1f}s, {len(path)} pts, {gc} gear changes")
        if gc > max_gear_changes:
            logger(f"[EXPERT] REJECT: {gc} gear changes > {max_gear_changes}")
            raise PlanRejected(f"{gc} gear changes > {max_gear_changes}")

        shutil.copy2(str(sol_csv), str(cache_file))
        logger(f"[EXPERT] cached {key}")

    return _plan
