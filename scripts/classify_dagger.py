"""DAgger triage — classify each logged state as MODEL error vs CONTEXT (controller) error.

For every state logged by `run_episode.py --log-dagger`, compare TWO plans at the SAME pose:
  - what the MODEL did      -> the logged `model_wps` (its real closed-loop output), and
  - what the EXPERT would do -> a Reeds-Shepp plan from that pose to the goal slot.

If the model's plan DISAGREES with the expert (wrong direction / wrong gear / large L2), the model
itself is wrong at that state -> DAgger (which supplies the expert plan as the label) is the fix.
If the model's plan AGREES with the expert yet the car still drifted off-path over the run, the
model planned correctly and the divergence came from execution -> a CONTROLLER/tracking fix is the
lever, and more data won't help that state.

This needs NEITHER CARLA NOR the model server (model_wps are already saved; RS is pure geometry),
so it can run while collection is still going. It only reads finished run folders.

Usage:
  venv/bin/python scripts/classify_dagger.py --log-dir data/dagger_raw
  # options: --l2 1.5 (disagreement threshold m), --near 0.6 (expert reach below this = near-goal,
  #          reported separately), --margin 1.0 (NPC collision margin for expert validity)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import math
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "ParkingScenes" / "tool" / "AutomatedValetParking"))

from harness.model_server import live_prompt as lp

# RS-geometry helpers (self-contained — used here as a cheap EXPERT reference to classify a state
# as model-error vs context, NOT as the training label; the production labeler is relabel_dagger_
# drive.py which DRIVES Auto_Park for the real speed profile).
DT_WP = 0.5
N_FUT = 7
MAXC = 0.2          # 1 / 5m min turn radius (matches generate_episodes RS call)


def _rs_path(ex, ey, eyaw, gx, gy, gyaw):
    """Reeds-Shepp dense path (nuScenes frame) from ego pose to goal. Returns (xs, ys, yaws)."""
    from path_plan import rs_curve as rs
    path = rs.calc_optimal_path(ex, ey, eyaw, gx, gy, gyaw, MAXC)
    return list(path.x), list(path.y), list(path.yaw)


def _resample_by_arclen(xs, ys, yaws, step_dists):
    """Sample (x, y, yaw) at the given cumulative arc-length distances along the polyline."""
    seg = [0.0]
    for i in range(1, len(xs)):
        seg.append(seg[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
    total = seg[-1]
    out, k = [], 0
    for d in step_dists:
        d = min(d, total)
        while k + 1 < len(seg) and seg[k + 1] < d:
            k += 1
        if k + 1 < len(seg) and seg[k + 1] > seg[k]:
            t = (d - seg[k]) / (seg[k + 1] - seg[k])
            x = xs[k] + t * (xs[k + 1] - xs[k])
            y = ys[k] + t * (ys[k + 1] - ys[k])
        else:
            x, y = xs[k], ys[k]
        out.append((x, y, yaws[min(k, len(yaws) - 1)]))
    return out


def _hits_npc(xs, ys, yaws, npcs, margin):
    """True if any dense path point lands inside an NPC box (expanded by margin)."""
    for px, py, pyaw in zip(xs, ys, yaws):
        for b in npcs:
            dx, dy = px - b["x"], py - b["y"]
            c, s = math.cos(-b["yaw"]), math.sin(-b["yaw"])
            xl, yl = c * dx - s * dy, s * dx + c * dy
            if abs(xl) <= b["ext_x"] + margin and abs(yl) <= b["ext_y"] + margin:
                return True
    return False


def seg_total(xs, ys):
    return sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))



def _model_future(model_wps):
    """Logged model output -> (6,2) ego-local future positions (drop origin, drop heading col)."""
    pts = [(float(w[0]), float(w[1])) for w in model_wps]
    return np.array(pts[1:N_FUT], dtype=np.float32) if len(pts) >= 2 else np.zeros((0, 2), np.float32)


def _expert_future(ego, slot, speed, margin, npcs):
    """RS expert plan from ego pose to slot, resampled at `speed` -> (6,2) ego-local, + collide flag."""
    ex, ey, eyaw = float(ego["x"]), float(ego["y"]), float(ego["yaw"])
    gx, gy, gyaw = float(slot["x"]), float(slot["y"]), float(slot["yaw"])
    xs, ys, yaws = _rs_path(ex, ey, eyaw, gx, gy, gyaw)
    collide = _hits_npc(xs, ys, yaws, npcs, margin)
    step_dists = [speed * DT_WP * j for j in range(N_FUT)]
    samples = _resample_by_arclen(xs, ys, yaws, step_dists)
    fut = np.zeros((N_FUT - 1, 2), dtype=np.float32)
    for j, (px, py, _) in enumerate(samples[1:], start=0):
        r, f = lp.global_to_local_xy(px, py, ex, ey, eyaw)
        fut[j] = [r, f]
    return fut, collide


def classify_state(state, run_meta, l2_thresh, near_reach, margin, min_speed):
    """Return a dict with the per-state verdict."""
    ego = state["ego"]
    slot = run_meta["slot_global"]
    speed = max(float(ego.get("speed", 0.0)), min_speed)
    expert, collide = _expert_future(ego, slot, speed, margin, run_meta.get("npcs", []))
    model = _model_future(state["model_wps"])

    n = min(len(expert), len(model))
    if n == 0:
        return {"verdict": "empty"}
    l2 = float(np.mean(np.linalg.norm(model[:n] - expert[:n], axis=1)))
    model_fwd = float(np.sum(model[:, 1])) if len(model) else 0.0
    expert_fwd = float(np.sum(expert[:, 1])) if len(expert) else 0.0
    gear_match = (model_fwd < 0) == (expert_fwd < 0)
    expert_reach = float(np.max(np.linalg.norm(expert, axis=1))) if len(expert) else 0.0

    if collide:
        verdict = "no_label"          # expert path clips an NPC -> can't trust the reference
    elif expert_reach < near_reach:
        verdict = "near_goal"         # expert says basically "you're here" -> finishing/heading regime
    elif (not gear_match) or l2 > l2_thresh:
        verdict = "model_error"       # model disagrees with expert -> DAgger target
    else:
        verdict = "context_ok"        # model agrees with expert -> any failure is execution/controller
    return {"verdict": verdict, "l2": l2, "gear_match": gear_match,
            "model_fwd": model_fwd, "expert_fwd": expert_fwd, "expert_reach": expert_reach}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="data/dagger_raw")
    ap.add_argument("--l2", type=float, default=1.5, help="model-vs-expert L2 disagreement threshold (m)")
    ap.add_argument("--near", type=float, default=0.6, help="expert reach below this = near-goal regime")
    ap.add_argument("--margin", type=float, default=1.0, help="NPC collision margin for expert validity")
    ap.add_argument("--min-speed", type=float, default=1.0, help="resample speed floor for expert plan")
    ap.add_argument("--per-run", action="store_true", help="also print a one-line summary per run")
    args = ap.parse_args(argv)

    runs = sorted(p for p in pathlib.Path(args.log_dir).glob("*") if (p / "run_meta.json").exists())
    if not runs:
        raise SystemExit(f"no runs under {args.log_dir}")

    tally = {"model_error": 0, "context_ok": 0, "near_goal": 0, "no_label": 0, "empty": 0}
    total = 0
    for run in runs:
        try:
            run_meta = json.loads((run / "run_meta.json").read_text())
        except Exception:
            continue
        rt = {k: 0 for k in tally}
        for st_path in sorted(glob.glob(str(run / "state_*" / "state.json"))):
            try:
                state = json.loads(pathlib.Path(st_path).read_text())
            except Exception:
                continue            # in-progress / partial state -> skip
            v = classify_state(state, run_meta, args.l2, args.near, args.margin, args.min_speed)["verdict"]
            tally[v] += 1
            rt[v] += 1
            total += 1
        if args.per_run:
            print(f"  {run.name:<34} slot{run_meta['slot_idx']:>2} {run_meta['side']:<5} "
                  f"model={rt['model_error']:2d} ctx={rt['context_ok']:2d} near={rt['near_goal']:2d} "
                  f"nolbl={rt['no_label']:2d}")

    print(f"\n=== DAgger triage over {total} states in {len(runs)} runs ===")
    labelled = tally["model_error"] + tally["context_ok"]
    for k in ("model_error", "context_ok", "near_goal", "no_label", "empty"):
        pct = 100 * tally[k] / total if total else 0
        print(f"  {k:<12} {tally[k]:5d}  ({pct:4.1f}%)")
    if labelled:
        print(f"\n  Of decisively-labelled states ({labelled}): "
              f"{100*tally['model_error']/labelled:.0f}% MODEL (DAgger-fixable), "
              f"{100*tally['context_ok']/labelled:.0f}% CONTEXT (controller-fixable)")
    print("\n  model_error = model disagrees with expert -> DAgger relabel targets these")
    print("  context_ok  = model matches expert but car diverged -> controller/tracking is the lever")
    print("  near_goal   = finishing/heading regime (separate near-slot handling)")
    print("  no_label    = expert RS clips an NPC -> reference unreliable, excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
