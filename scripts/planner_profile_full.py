"""Time the FULL ParkingScenes plan() pipeline (A* + path-opt + OCP), no CARLA.

Switching the collector from the RS bypass back to Hybrid A* re-enables the whole
optimization pipeline in plan.py (path_optimazition + interpolation + velocity +
pyomo OCP). This measures that total per-plan cost on a realistic case so we know
whether it is "fast enough" for collection.
"""
import os
import sys
import time
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")
_REPO = pathlib.Path(__file__).resolve().parent.parent
_AVP = _REPO / "ParkingScenes" / "tool" / "AutomatedValetParking"
sys.path.insert(0, str(_AVP))
sys.path.insert(0, str(_REPO / "ParkingScenes"))
sys.path.insert(0, str(_REPO / "scripts"))

import planner_profile as pp
import planner_fast

pp.apply_current_patches(max_nodes=2_000_000)
planner_fast.apply_fast_patches()

start, goal, obstacles = pp.build_town04_case("full")
case_csv = _AVP / "BenchmarkCases" / "CARLA.csv"
case_csv.parent.mkdir(parents=True, exist_ok=True)
pp.write_case_csv(case_csv, start, goal, obstacles)

sys.argv[1:] = []  # plan() runs its own argparse
from tool.AutomatedValetParking import plan as plan_mod

t0 = time.perf_counter()
plan_mod.plan(coordinate_rotation=False, index=17)
dt = time.perf_counter() - t0

sol = _AVP / "solution" / "CARLA.csv"
n = sum(1 for _ in open(sol)) if sol.exists() else 0
print(f"[FULL PIPELINE] {dt:.2f}s | solution rows={n}")
