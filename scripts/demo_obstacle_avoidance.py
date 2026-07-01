"""Demonstrate that the obstacle-aware Hybrid A* SEES and AVOIDS parked cars.

Runs WITHOUT CARLA. Builds Town04 reverse-perpendicular parking scenes with
controlled obstacle occupancy, runs the REAL ParkingScenes Hybrid A* (the same
planner the collector uses) against those obstacles, and renders a bird's-eye PNG
showing the parked-car footprints the planner saw + the resulting collision-free
path. This is the geometry the planner actually plans against (the case file its
collision checker reads), so it proves the path threads between the cars.

Scenarios:
  neighbours_both : parked cars in BOTH bays flanking the goal -> path must arc
                    into the narrow gap, cannot cut across the neighbours.
  narrow_aisle    : neighbours_both PLUS a row of cars on the opposite side of the
                    aisle (row-3) -> the ego cannot swing wide; it must do a tight
                    (often multi-point) maneuver inside the narrow lane.
  boxed_in        : neighbours + a car parked across the aisle mouth of the goal
                    bay -> goal unreachable, planner fails fast (no path).
  open            : goal bay clear, sparse lot -> easy reference path.

Usage:
  venv/bin/python scripts/demo_obstacle_avoidance.py
  venv/bin/python scripts/demo_obstacle_avoidance.py --scenario narrow_aisle
"""
import argparse
import math
import os
import sys
import time
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

_REPO = pathlib.Path(__file__).resolve().parent.parent
_AVP = _REPO / "ParkingScenes" / "tool" / "AutomatedValetParking"
sys.path.insert(0, str(_AVP))
sys.path.insert(0, str(_REPO / "scripts"))

import planner_profile as pp
import planner_fast


def _car(cx, cy, half_len=2.4, half_wid=1.0):
    """Parked-car footprint centred at (cx,cy), long axis along x (a row car)."""
    return [
        (cx + half_len, cy - half_wid),
        (cx + half_len, cy + half_wid),
        (cx - half_len, cy + half_wid),
        (cx - half_len, cy - half_wid),
    ]


def build_scene(scenario):
    """Return (start, goal, obstacles) in CARLA-like planning coordinates.

    Row-2 bays at x=290.9, aisle centreline x=285.6. Goal bay at y=-235.73 with
    its two neighbours at y=-232.73 and y=-238.9.
    """
    goal = (290.9, -235.73, math.pi)          # reverse into row-2 bay, facing -x
    start = (285.6, -229.0, -math.pi / 2)      # in the aisle, facing -y
    gy = goal[1]

    obstacles = []
    # A few unrelated parked cars elsewhere in row 2 (context, not blocking).
    for py in (-223.43, -226.43, -220.2):
        obstacles.append(_car(290.9, py))
    # Cars across the aisle in row 1 (x=298.5) — bound the far side.
    for py in (-232.73, -235.73, -238.9):
        obstacles.append(_car(298.5, py))

    if scenario in ("neighbours_both", "boxed_in"):
        # BOTH bays flanking the goal are occupied.
        obstacles.append(_car(290.9, gy + 3.0))   # neighbour on the +y side
        obstacles.append(_car(290.9, gy - 3.17))  # neighbour on the -y side
    if scenario == "narrow_aisle":
        # Neighbours occupied AND the opposite side of the aisle (row-3, x=280.0)
        # is lined with parked cars, so the ego cannot swing wide into open space —
        # it must maneuver within the narrow lane between rows 2 and 3.
        obstacles.append(_car(290.9, gy + 3.0))
        obstacles.append(_car(290.9, gy - 3.17))
        for py in (gy + 3.0, gy, gy - 3.0, gy + 6.0, gy - 6.0):
            obstacles.append(_car(280.0, py))      # row-3 wall across the aisle
    if scenario == "boxed_in":
        # A car parked across the aisle mouth of the goal bay -> no way in.
        obstacles.append(_car(287.0, gy, half_len=0.6, half_wid=1.4))

    return start, goal, obstacles


def run_and_plot(scenario, out_png):
    from config import read_config
    from map import costmap
    from path_plan import path_planner

    start, goal, obstacles = build_scene(scenario)
    case_csv = _AVP / "BenchmarkCases" / "DEMO_CASE.csv"
    case_csv.parent.mkdir(parents=True, exist_ok=True)
    pp.write_case_csv(case_csv, start, goal, obstacles)

    config = read_config.read_config(config_name="config")
    park_map = costmap.Map(file=str(case_csv), discrete_size=config["map_discrete_size"])
    vehicle = costmap.Vehicle()
    planner = path_planner.PathPlanner(config=config, map=park_map, vehicle=vehicle)

    t0 = time.perf_counter()
    path = []
    reached = True
    try:
        path, _info, _split = planner.path_planning()
    except planner_fast.PlanningTimeout:
        reached = False
    dt = time.perf_counter() - t0

    # Collision check the produced path against the very obstacles the planner saw.
    hits = [p for p in path if planner.collision_checker.check(node_x=p[0], node_y=p[1], theta=p[2])]

    fig, ax = plt.subplots(figsize=(10, 10))
    # Obstacles the planner's collision checker read from the case.
    for ob in obstacles:
        ax.add_patch(plt.Polygon(ob, closed=True, facecolor="0.6",
                                 edgecolor="k", alpha=0.8, zorder=1, label="_"))
    # Goal bay outline (5.5 x 3.0 perpendicular bay along -x heading).
    hl, hw = 5.5 / 2, 3.0 / 2
    gx, gyv, gyaw = goal
    fx, fy = math.cos(gyaw), math.sin(gyaw)
    lx, ly = -math.sin(gyaw), math.cos(gyaw)
    bay = [(gx + hl*fx + hw*lx, gyv + hl*fy + hw*ly),
           (gx + hl*fx - hw*lx, gyv + hl*fy - hw*ly),
           (gx - hl*fx - hw*lx, gyv - hl*fy - hw*ly),
           (gx - hl*fx + hw*lx, gyv - hl*fy + hw*ly)]
    ax.add_patch(plt.Polygon(bay, closed=True, facecolor="none",
                             edgecolor="green", lw=2.5, zorder=2, label="goal bay"))

    if path:
        ax.plot([p[0] for p in path], [p[1] for p in path], "-",
                color="royalblue", lw=2.5, zorder=4, label="A* expert path")
        # heading arrows so the reverse arc is legible
        step = max(1, len(path) // 20)
        for i in range(0, len(path), step):
            x, y, th = path[i]
            ax.arrow(x, y, 0.6*math.cos(th), 0.6*math.sin(th), head_width=0.2,
                     fc="dimgray", ec="dimgray", alpha=0.6, zorder=5)
    ax.scatter([start[0]], [start[1]], c="cyan", s=140, marker="s", zorder=6, label="start")
    ax.scatter([goal[0]], [goal[1]], c="lime", s=200, marker="*", zorder=6, label="goal")
    if hits:
        ax.scatter([h[0] for h in hits], [h[1] for h in hits], c="red", s=40,
                   marker="x", zorder=7, label="COLLISION")

    status = ("REACHED, collision-free" if (reached and path and not hits)
              else "UNREACHABLE (planner refused)" if not path
              else f"path has {len(hits)} collisions")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Obstacle avoidance demo — scenario '{scenario}'\n{status}  ({dt:.2f}s)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[{scenario}] {status} | {dt:.2f}s | path_pts={len(path)} | saved {out_png}")
    return reached, path, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all",
                    choices=["all", "open", "neighbours_both", "narrow_aisle", "boxed_in"])
    ap.add_argument("--outdir", default=str(_REPO / "data" / "planner_demos"))
    args = ap.parse_args()

    pp.apply_current_patches(max_nodes=2_000_000)
    planner_fast.apply_fast_patches()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = ["open", "neighbours_both", "narrow_aisle", "boxed_in"] if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        run_and_plot(sc, outdir / f"demo_{sc}.png")


if __name__ == "__main__":
    main()
