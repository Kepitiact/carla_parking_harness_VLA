"""Standalone correctness + speed check for the obstacle-aware Hybrid A*.

Runs WITHOUT CARLA. Builds several Town04 reverse-perpendicular parking cases
with realistic parked-car obstacle geometry, runs the REAL ParkingScenes Hybrid
A* (with the O(1) patches + planner_fast speed patch), and asserts that the
returned trajectory is collision-free against those obstacles using the planner's
own collision checker. Also reports wall-clock time per case.

This is the Priority-1 deliverable that is verifiable off-simulator: fast +
collision-free expert trajectory GT.

Usage: venv/bin/python scripts/test_planner_collisionfree.py
"""
import math
import os
import sys
import time
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")
_REPO = pathlib.Path(__file__).resolve().parent.parent
_AVP = _REPO / "ParkingScenes" / "tool" / "AutomatedValetParking"
sys.path.insert(0, str(_AVP))
sys.path.insert(0, str(_REPO / "scripts"))

import planner_profile as pp   # reuse case builder + O(1) patches
import planner_fast


def check_path_collisionfree(planner, path):
    """Return list of colliding (x,y,theta) waypoints using the planner's checker."""
    checker = planner.collision_checker
    hits = []
    for (x, y, theta) in path:
        if checker.check(node_x=x, node_y=y, theta=theta):
            hits.append((x, y, theta))
    return hits


def run_case(name, start, goal, obstacles):
    from config import read_config
    from map import costmap
    from path_plan import path_planner

    case_csv = _AVP / "BenchmarkCases" / "TEST_CASE.csv"
    case_csv.parent.mkdir(parents=True, exist_ok=True)
    pp.write_case_csv(case_csv, start, goal, obstacles)

    config = read_config.read_config(config_name="config")
    park_map = costmap.Map(file=str(case_csv), discrete_size=config["map_discrete_size"])
    vehicle = costmap.Vehicle()
    planner = path_planner.PathPlanner(config=config, map=park_map, vehicle=vehicle)

    t0 = time.perf_counter()
    try:
        final_path, _info, split = planner.path_planning()
    except planner_fast.PlanningTimeout as e:
        dt = time.perf_counter() - t0
        print(f"[{name}] TIMEOUT after {dt:.2f}s: {e}")
        return False
    dt = time.perf_counter() - t0

    hits = check_path_collisionfree(planner, final_path)
    ok = len(hits) == 0
    status = "OK" if ok else f"COLLISION x{len(hits)}"
    print(f"[{name}] {dt:.2f}s | pts={len(final_path)} segs={len(split)} | {status}")
    if hits:
        for h in hits[:3]:
            print(f"    collide @ ({h[0]:.2f},{h[1]:.2f}) th={math.degrees(h[2]):.0f}")
    return ok


def main():
    # Apply the same O(1) patches the collector uses, then the speed patch.
    pp.apply_current_patches(max_nodes=2_000_000)
    planner_fast.apply_fast_patches()

    all_ok = True

    # Case 1: neighbours on both sides of the goal bay (tight reverse park).
    s, g, obs = pp.build_town04_case("full")
    all_ok &= run_case("full_lot", s, g, obs)

    # Case 2: sparse lot (only 6 parked cars).
    s, g, obs = pp.build_town04_case("6")
    all_ok &= run_case("sparse", s, g, obs)

    # Case 3: a car parked directly beside the goal bay on the aisle-entry side.
    s, g, obs = pp.build_town04_case("full")
    obs.append(pp._box(288.6, g[1] + 3.0, half_len=0.6, half_wid=1.4))
    all_ok &= run_case("tight_neighbour", s, g, obs)

    print("\nRESULT:", "ALL COLLISION-FREE" if all_ok else "SOME PATHS COLLIDE")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
