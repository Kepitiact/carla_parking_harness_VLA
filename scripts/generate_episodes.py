"""
Generate CARLA parking episodes in OpenDriveVLA-compatible format.

Reuses ParkingScenes' Hybrid A* + MPC planner (Auto_Park) without modification.
Adds 6 nuScenes-style cameras (1600×900, FOV=70°) in place of ParkingScenes' 4-camera rig.

Output per episode:
  data/raw/episode_XXXX/
    frames/frame_NNNN/CAM_FRONT.jpg  CAM_FRONT_LEFT.jpg  ... (6 cameras)
    poses.json    — per-frame ego pose, velocity, control, IMU, GNSS
    meta.json     — slot geometry, spawn, A* path summary, episode metadata

Usage:
  python scripts/generate_episodes.py --map Town04_Opt --num_episodes 50
  python scripts/generate_episodes.py --map Town10HD_Opt --num_episodes 20
"""

import sys
import os
import pathlib
import math
import json
import random
import logging
import argparse
import time
from queue import Queue, Empty

import numpy as np
import cv2

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'ParkingScenes'))
sys.path.insert(0, str(_REPO_ROOT / 'scripts'))

import carla

# Force non-interactive matplotlib backend before any ParkingScenes import
# triggers 'import matplotlib.pyplot' (costmap.py line 50).
os.environ['MPLBACKEND'] = 'Agg'

# pygame needed by Auto_Park for event polling
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import pygame
print("[INIT] pygame imported", flush=True)
pygame.init()
print("[INIT] pygame.init done", flush=True)
pygame.font.init()
print("[INIT] pygame.font.init done", flush=True)
_DISPLAY = pygame.display.set_mode((1, 1))
print("[INIT] pygame display done", flush=True)
_CLOCK = pygame.time.Clock()

from tool import parking_position
print("[INIT] parking_position imported", flush=True)

import carla_actor_gt  # shared per-frame actor GT export (Task 3)

# ── Monkey-patch ParkingScenes planners for O(1) lookup ──────────────────────
# Both Dijkstra (compute_h.py) and Hybrid A* (hybrid_a_star.py) use plain
# Python lists with O(n) linear search → O(n²) total → hangs on CARLA world
# coordinates. We replace list membership with sets/dicts for O(1) lookup.
def _patch_planners():
    print("[INIT] _patch_planners start", flush=True)
    _avp = str(_REPO_ROOT / 'ParkingScenes' / 'tool' / 'AutomatedValetParking')
    if _avp not in sys.path:
        sys.path.insert(0, _avp)

    import queue as _q
    import math as _math
    import numpy as _np
    print("[INIT] importing hybrid_a_star...", flush=True)
    from path_plan import hybrid_a_star as _ha
    print("[INIT] importing compute_h...", flush=True)
    from path_plan import compute_h as _ch
    print("[INIT] importing rs_curve...", flush=True)
    from path_plan import rs_curve as _rs
    print("[INIT] planner modules imported", flush=True)

    # ── Fix 1: Dijkstra openlist_index list → set ──────────────────────
    _orig_dijk_init = _ch.Dijkstra.__init__

    def _dijk_init(self, map):
        _orig_dijk_init(self, map)
        self.openlist_index = set()   # was list; count() → O(n), now O(1)
        self.openlist_dict = {}       # grid_id → Grid for O(1) priority update

    def _dijk_add(self, gridx, gridy, priority, father_id):
        index = self.map.convert_position_to_index(gridx, gridy)
        if index in self.openlist_index:
            node = self.openlist_dict[index]   # O(1) dict lookup, no queue scan
            if node.distance > priority:
                node.distance = priority
                node.father_id = father_id
        else:
            grid_node = _ch.Grid(grid_id=index, grid_x=gridx,
                                  grid_y=gridy, distance=priority,
                                  father_id=father_id)
            self.open_list.put(grid_node)
            self.openlist_index.add(index)
            self.openlist_dict[index] = grid_node

    _orig_compute_path = _ch.Dijkstra.compute_path

    def _dijk_compute(self, node_x, node_y):
        print("[PATCH] Dijkstra.compute_path start", flush=True)
        result = _orig_compute_path(self, node_x, node_y)
        print(f"[PATCH] Dijkstra.compute_path done ({len(self.closedlist)} cells)", flush=True)
        return result

    _ch.Dijkstra.__init__ = _dijk_init
    _ch.Dijkstra.add_grid_to_openlist = _dijk_add
    _ch.Dijkstra.compute_path = _dijk_compute

    # ── Fix 2: Hybrid A* closed_list linear scan → closed_set dict ─────
    def _astar_key(x, y, theta, ds=0.1, dth=0.05):
        return (round(x / ds), round(y / ds), round(theta / dth))

    _orig_ha_init = _ha.hybrid_a_star.__init__

    def _ha_init(self, config, park_map, vehicle):
        _orig_ha_init(self, config, park_map, vehicle)
        _expand_count[0] = 0  # reset per-instance so multi-episode counts are correct
        # Normalize goal angle to [-π, π] — Pyomo optimizer has hard bounds there;
        # unnormalized angles from CARLA yaw conversion cause infeasible variable errors.
        self.goal_node.theta = _rs.pi_2_pi(self.goal_node.theta)
        self.initial_node.theta = _rs.pi_2_pi(self.initial_node.theta)
        print(f"[PATCH] A* init done, building h_dict from {len(self.h_value_list)} entries...", flush=True)
        # O(1) heuristic lookup: grid_id → distance from precomputed Dijkstra list
        self._h_dict = {g.grid_id: g.distance for g in self.h_value_list}
        # O(1) closed/open lookup for A*
        self._closed_set = {_astar_key(n.x, n.y, n.theta) for n in self.closed_list}
        self._open_dict  = {}

    def _ha_calc_h(self, current_node):
        _grid_id = self.park_map.convert_position_to_index(current_node.x, current_node.y)
        h1 = self._h_dict.get(_grid_id, 0) / 100  # O(1) instead of O(n) scan + Dijkstra
        max_c = 1 / self.vehicle.min_radius_turn
        rs_path = _rs.calc_optimal_path(
            sx=current_node.x, sy=current_node.y, syaw=current_node.theta,
            gx=self.goal_node.x, gy=self.goal_node.y, gyaw=self.goal_node.theta,
            maxc=max_c)
        return max(h1, rs_path.L)

    _MAX_ASTAR_NODES = 2_000_000

    class PlanningTimeout(RuntimeError):
        pass

    _expand_count = [0]

    def _ha_expand(self, current_node):
        _expand_count[0] += 1
        if _expand_count[0] % 2000 == 0:
            deg = _math.degrees(current_node.theta) % 360
            h_covered = len(self._h_dict)
            print(f"[ASTAR] {_expand_count[0]} nodes | pos=({current_node.x:.1f},{current_node.y:.1f}) θ={deg:.0f}° | goal=({self.goal_node.x:.1f},{self.goal_node.y:.1f}) | h_map={h_covered}", flush=True)
        if _expand_count[0] > _MAX_ASTAR_NODES:
            raise PlanningTimeout(f"A* exceeded {_MAX_ASTAR_NODES} nodes — skipping slot")
        child_group = _q.PriorityQueue()
        next_index = int(2 * self.config['steering_angle_num'])
        for i in range(next_index):
            steering_angle = self.steering_angle[i % self.config['steering_angle_num']]
            is_forward = i < next_index / 2
            speed = self.vehicle.max_v if is_forward else -self.vehicle.max_v
            travel = speed * self.dt
            theta_ = _rs.pi_2_pi(
                current_node.theta +
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
                                      is_forward=is_forward,
                                      steering_angle=steering_angle)
                collision = False
                for j in range(_math.ceil(self.dt / self.ddt)):
                    tj = speed * self.ddt * (j + 1)
                    thj = _rs.pi_2_pi(
                        current_node.theta +
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
                                                        father_theta=current_node.theta,
                                                        father_gear=current_node.forward)
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

    _ha.hybrid_a_star.__init__ = _ha_init
    _ha.hybrid_a_star.expand_node = _ha_expand
    _ha.hybrid_a_star.calc_node_heuristic = _ha_calc_h
    logging.info("Planner O(1) patch applied (Dijkstra + A*)")

_patch_planners()


def _patch_plan_cache():
    """Monkey-patch plan() to cache A* solutions by rounded start+goal.

    A* for CARLA world coordinates takes 30+ min in Python. Each slot has a
    fixed goal and near-fixed spawn, so caching the solution CSV means we only
    plan once per unique (start, goal) pair — every subsequent call is instant.
    """
    import hashlib as _hl
    import shutil as _sh
    _avp = str(_REPO_ROOT / 'ParkingScenes' / 'tool' / 'AutomatedValetParking')
    if _avp not in sys.path:
        sys.path.insert(0, _avp)

    import importlib as _il
    _plan_mod = _il.import_module('tool.AutomatedValetParking.plan')
    _orig_plan = _plan_mod.plan

    _cache_dir = _REPO_ROOT / 'data' / 'plan_cache'
    _cache_dir.mkdir(parents=True, exist_ok=True)

    _bench_csv = pathlib.Path(_avp) / 'BenchmarkCases' / 'CARLA.csv'
    _sol_csv   = pathlib.Path(_avp) / 'solution' / 'CARLA.csv'

    _MAX_GEAR_CHANGES = 20  # plans with more gear changes are unexecutable in 5 min

    def _count_gear_changes(csv_path):
        """Count direction changes in column 4 (velocity sign) of the plan CSV."""
        import csv as _csv_mod
        prev_sign = None
        changes = 0
        with open(csv_path, newline='') as f:
            for row in _csv_mod.reader(f, delimiter='\t'):
                try:
                    v = float(row[4])
                except (IndexError, ValueError):
                    continue
                if v == 0.0:
                    continue
                sign = 1 if v > 0 else -1
                if prev_sign is not None and sign != prev_sign:
                    changes += 1
                prev_sign = sign
        return changes

    def _write_rs_solution(path, coordinate_rotation, out_csv):
        """Write RS curve path to solution/CARLA.csv in the format MPC expects.

        The path is in planning frame when coordinate_rotation=True.
        plan.py's csv_coordinate_rotation() maps planning→CARLA as:
          carla_x = plan_y,  carla_y = -plan_x,  carla_yaw = plan_yaw - π/2
        We apply the same transform here so the cached CSV is always in CARLA frame.
        Row 0 has v=0.0; MPC's read_specific_rows_from_csv skips it (vs[-2] IndexError
        on empty list) and initialises its Node at cx[1].
        """
        import math as _m
        import csv as _csv_mod
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = _csv_mod.writer(f, delimiter='\t')
            for i, (x, y, yaw, d) in enumerate(zip(path.x, path.y, path.yaw, path.directions)):
                v = 0.0 if i == 0 else float(d)
                if coordinate_rotation:
                    out_x, out_y, out_yaw = y, -x, yaw - _m.pi / 2
                else:
                    out_x, out_y, out_yaw = x, y, yaw
                writer.writerow([i, out_x, out_y, out_yaw, v,
                                 _m.sin(out_yaw), _m.cos(out_yaw)])

    # Build the shared obstacle-aware Hybrid A* expert planner (Priority-1).
    # Imported lazily so a missing module degrades to the RS/stock path instead
    # of breaking collection.
    try:
        import carla_expert_planner as _cep
        _astar_plan = _cep.make_cached_plan(_REPO_ROOT, orig_plan=_orig_plan,
                                            max_gear_changes=_MAX_GEAR_CHANGES)
    except Exception as _e:
        print(f"[EXPERT] could not init A* expert planner: {_e}", flush=True)

        def _astar_plan(coordinate_rotation, index):
            raise RuntimeError('carla_expert_planner unavailable')

    def _cached_plan(coordinate_rotation, index):
        # ── Priority-1: obstacle-aware Hybrid A* expert (default) ─────────────
        # Runs the REAL ParkingScenes Hybrid A* against the LIVE spawned obstacles
        # (BenchmarkCases/CARLA.csv, written by Auto_Park.perception), verifies the
        # path is collision-free, and writes it straight to solution/CARLA.csv —
        # bypassing the slow/fragile pyomo OCP. Cache key hashes the whole case
        # file so a plan is never reused across a different obstacle subset.
        # Set CARLA_EXPERT_PLANNER=rs to fall back to the old Reeds-Shepp bypass.
        if os.environ.get('CARLA_EXPERT_PLANNER', 'astar').lower() != 'rs':
            try:
                _astar_plan(coordinate_rotation, index)
                return
            except Exception as _e:
                print(f"[EXPERT] A* unavailable ({_e}) — falling back to RS/stock", flush=True)

        # Cache key = "v3" + slot_index + rounded start/goal (1m precision).
        # "v3" invalidates v2 caches (start=rear-axle, goal=center, no WB on goal).
        # Start/goal are unique per map so switching maps gives a cache miss.
        key = "miss"
        sx = sy = syaw = gx = gy = gyaw = None
        if _bench_csv.exists():
            raw = _bench_csv.read_text()
            tokens = raw.split(',')
            vals = [f"v3_slot{index}"]
            for tok in tokens[:6]:
                try:
                    vals.append(round(float(tok), 1))
                except ValueError:
                    vals.append(tok)
            key = _hl.md5(str(vals).encode()).hexdigest()[:12]
            try:
                sx, sy, syaw = float(tokens[0]), float(tokens[1]), float(tokens[2])
                gx, gy, gyaw = float(tokens[3]), float(tokens[4]), float(tokens[5])
            except (IndexError, ValueError):
                pass

        cache_file = _cache_dir / f"plan_{key}.csv"
        if cache_file.exists():
            gc = _count_gear_changes(cache_file)
            if gc > _MAX_GEAR_CHANGES:
                print(f"[CACHE] plan {key} has {gc} gear changes — deleting bad cache, skipping slot", flush=True)
                cache_file.unlink()
                raise RuntimeError(f"Plan {key} rejected: {gc} gear changes > {_MAX_GEAR_CHANGES}")
            print(f"[CACHE] loading plan {key} ({gc} gear changes)", flush=True)
            _sol_csv.parent.mkdir(parents=True, exist_ok=True)
            _sh.copy2(str(cache_file), str(_sol_csv))
            return

        # RS curve bypass: ~1ms vs A*'s hours.
        # BenchmarkCases/CARLA.csv has start/goal in planning frame (when
        # coordinate_rotation=True); _write_rs_solution applies the inverse
        # rotation so the cached solution is always in CARLA world frame.
        if sx is not None:
            try:
                from path_plan import rs_curve as _rs
                import math as _m_rs
                maxc = 0.2  # 1 / 5.0m min-turn-radius (costmap.Vehicle)
                # BenchmarkCases CSV has center positions; MPC receives rear-axle
                # positions (auto_park_1.py move_back=1.4m). Offset START to rear-axle
                # so cx[0] aligns with where the MPC actually starts, preventing the
                # premature gear-change caused by nearest_index jumping ahead 2-3 steps.
                # Goal stays at slot CENTER (no offset): over=True triggers ~1.4m before
                # the center, and the car's braking momentum lands it at the slot center.
                WB = 1.4
                sx_r = sx - WB * _m_rs.cos(syaw)
                sy_r = sy - WB * _m_rs.sin(syaw)
                path = _rs.calc_optimal_path(sx_r, sy_r, syaw, gx, gy, gyaw, maxc)
                _write_rs_solution(path, coordinate_rotation, _sol_csv)
                gc = _count_gear_changes(_sol_csv)
                print(f"[RS] path L={path.L:.1f}m, {gc} gear changes", flush=True)
                if gc <= _MAX_GEAR_CHANGES:
                    _sh.copy2(str(_sol_csv), str(cache_file))
                    print(f"[RS] saved plan {key}", flush=True)
                    return
                print(f"[RS] {gc} gear changes > {_MAX_GEAR_CHANGES} — falling back to A*", flush=True)
            except Exception as e:
                print(f"[RS] failed: {e} — falling back to A*", flush=True)

        print(f"[CACHE] no cache for {key}, running A* (this takes a while)...", flush=True)
        _orig_plan(coordinate_rotation, index)

        # Save solution only if quality is acceptable
        if _sol_csv.exists():
            gc = _count_gear_changes(_sol_csv)
            print(f"[CACHE] plan quality: {gc} gear changes", flush=True)
            if gc <= _MAX_GEAR_CHANGES:
                _sh.copy2(str(_sol_csv), str(cache_file))
                print(f"[CACHE] saved plan {key}", flush=True)
            else:
                print(f"[CACHE] plan {key} rejected ({gc} gear changes) — skipping slot", flush=True)
                raise RuntimeError(f"Plan {key} rejected: {gc} gear changes > {_MAX_GEAR_CHANGES}")

    # auto_park_1.py does `from tool.AutomatedValetParking.plan import plan`
    # which creates its own local binding — we must patch sys.modules to override it
    _plan_mod.plan = _cached_plan
    if 'tool.auto_park_1' in sys.modules:
        sys.modules['tool.auto_park_1'].plan = _cached_plan

_patch_plan_cache()
print("[INIT] patch done", flush=True)
# ─────────────────────────────────────────────────────────────────────────────

print("[INIT] importing Auto_Park...", flush=True)
from tool.auto_park_1 import Auto_Park
print("[INIT] Auto_Park imported", flush=True)

# ── Camera config (nuScenes-compatible) ──────────────────────────────────────
CAMERAS_6 = {
    "CAM_FRONT":       {"x": 1.5,  "y":  0.0, "z": 1.5, "pitch": 0, "yaw":   0},
    "CAM_FRONT_LEFT":  {"x": 1.0,  "y": -0.9, "z": 1.5, "pitch": 0, "yaw": -55},
    "CAM_FRONT_RIGHT": {"x": 1.0,  "y":  0.9, "z": 1.5, "pitch": 0, "yaw":  55},
    "CAM_BACK":        {"x": -1.5, "y":  0.0, "z": 1.5, "pitch": 0, "yaw": 180},
    "CAM_BACK_LEFT":   {"x": -1.0, "y": -0.9, "z": 1.5, "pitch": 0, "yaw":-110},
    "CAM_BACK_RIGHT":  {"x": -1.0, "y":  0.9, "z": 1.5, "pitch": 0, "yaw": 110},
}
CAM_W, CAM_H, CAM_FOV = 1600, 900, 70
# Optional camera sensor_tick (sim-seconds between renders). 0.0 = every tick (collection
# default, unchanged). The closed-loop harness sets this >0 to render/stream the 6 cameras
# only at its planning cadence — drastically less network load when CARLA runs on another box.
CAMERA_SENSOR_TICK = 0.0

SIM_HZ     = 30          # CARLA fixed timestep
RECORD_HZ  = 2           # frames saved per second
RECORD_EVERY = SIM_HZ // RECORD_HZ  # = 15 ticks between recordings

GOAL_DIST_M  = 0.5   # distance threshold to consider "in goal"
GOAL_ROT_DEG = 3.5   # rotation threshold (degrees)
GOAL_HOLD_FRAMES = 30  # must stay in goal for this many 30Hz ticks
TIMEOUT_TICKS = 5 * 60 * SIM_HZ  # 5-minute hard limit per episode

# ── Parking slot geometry by map ─────────────────────────────────────────────
SLOT_GEOMETRY = {
    "Town04_Opt": {"width_m": 3.0, "length_m": 5.5, "type": "perpendicular_reverse"},
    "Town10HD_Opt": {"width_m": 2.5, "length_m": 6.0, "type": "parallel"},
}

# Goal yaw (degrees) for a successfully parked vehicle
GOAL_YAW = {
    "Town04_Opt": 180.0,  # slots 17-32 park facing -x (goal_yaw=180°)
    "Town10HD_Opt": 90.0, # facing +y (parallel slot orientation)
}

# Episode-level maneuver label per map. A string so future scenarios add methods
# (forward_perpendicular / parallel / angled / multi_point) without code changes.
MANEUVER_TYPE = {
    "Town04_Opt": "reverse_perpendicular",
    "Town10HD_Opt": "parallel",
}


# ── Maneuver / target-slot labels (nuScenes global frame) ─────────────────────
# These mirror build_infos_pkl.py so the recorded meta.json is already in the
# frame the pkl/model consume (nuScenes: y-left, right-hand). CARLA→nuScenes:
# flip y, negate yaw.

def _carla_xy_to_nuscenes(x, y):
    return x, -y


def _yaw_quat(heading_rad):
    """Quaternion [w, x, y, z] for a pure yaw rotation about +z."""
    return [math.cos(heading_rad / 2), 0.0, 0.0, math.sin(heading_rad / 2)]


def _make_target_slot(cx, cy, cz, heading_rad, width_m, length_m):
    """Build target_slot (polygon + pose) in the nuScenes global frame.

    Perpendicular bay rectangle: length along the parked heading, width across.
    Corners ordered front-left, front-right, rear-right, rear-left.
    """
    hl, hw = length_m / 2.0, width_m / 2.0
    fx, fy = math.cos(heading_rad), math.sin(heading_rad)    # forward unit
    lx, ly = -math.sin(heading_rad), math.cos(heading_rad)   # left unit
    polygon = [
        [cx + hl * fx + hw * lx, cy + hl * fy + hw * ly],
        [cx + hl * fx - hw * lx, cy + hl * fy - hw * ly],
        [cx - hl * fx - hw * lx, cy - hl * fy - hw * ly],
        [cx - hl * fx + hw * lx, cy - hl * fy + hw * ly],
    ]
    return {
        "polygon": polygon,
        "pose": {
            "translation": [cx, cy, cz],
            "rotation": _yaw_quat(heading_rad),
        },
    }


def _compute_side(ax, ay, ayaw, sx, sy):
    """Side the slot is on relative to the approach heading (nuScenes frame)."""
    dx, dy = sx - ax, sy - ay
    lateral = dx * math.sin(ayaw) - dy * math.cos(ayaw)  # d · right
    return "right" if lateral > 0 else "left"


class _CollisionFlag:
    def __init__(self):
        self.hit = False
        self.actor = None
    def on_event(self, event):
        self.hit = True
        try:
            other = event.other_actor
            self.actor = f"{other.type_id}(id={other.id})"
        except Exception:
            self.actor = "unknown"


class WorldState:
    """Minimal duck-typed wrapper around a CARLA world + ego vehicle.
    Provides the interface expected by Auto_Park without pulling in the full
    ParkingScenes World class.
    """
    def __init__(self, carla_world):
        self._carla_world = carla_world
        self._player = None
        self._sensor_list = []
        self._sensor_queue = Queue()
        self._collision = _CollisionFlag()

        # Auto_Park reads/writes these directly
        self.is_restart = True
        self.over = False
        self.need_init_ego_state = True
        self.index = 0          # current parking slot index

        self._actor_list = []   # static NPCs

    # ── Properties expected by Auto_Park ─────────────────────────────────────

    @property
    def player(self):
        return self._player

    @property
    def world(self):
        return self._carla_world

    # ── CARLA setup / teardown ────────────────────────────────────────────────

    def spawn_ego(self, transform):
        bp = self._carla_world.get_blueprint_library().find('vehicle.tesla.model3')
        self._player = self._carla_world.spawn_actor(bp, transform)
        self._attach_cameras()
        self._attach_misc_sensors()

    def _attach_cameras(self):
        bp_lib = self._carla_world.get_blueprint_library()
        for cam_name, spec in CAMERAS_6.items():
            bp = bp_lib.find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', str(CAM_W))
            bp.set_attribute('image_size_y', str(CAM_H))
            bp.set_attribute('fov', str(CAM_FOV))
            if CAMERA_SENSOR_TICK > 0:  # harness-only: render less often (see CAMERA_SENSOR_TICK)
                bp.set_attribute('sensor_tick', str(CAMERA_SENSOR_TICK))
            tf = carla.Transform(
                carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
                carla.Rotation(pitch=spec['pitch'], yaw=spec['yaw']),
            )
            cam = self._carla_world.spawn_actor(
                bp, tf, attach_to=self._player,
                attachment_type=carla.AttachmentType.Rigid,
            )
            # capture cam_name in closure
            cam.listen(lambda img, n=cam_name: self._sensor_queue.put((img, n)))
            self._sensor_list.append(cam)

    def _attach_misc_sensors(self):
        bp_lib = self._carla_world.get_blueprint_library()
        for bp_id, queue_name in [('sensor.other.imu', 'imu'),
                                   ('sensor.other.gnss', 'gnss')]:
            bp = bp_lib.find(bp_id)
            s = self._carla_world.spawn_actor(
                bp, carla.Transform(), attach_to=self._player,
                attachment_type=carla.AttachmentType.Rigid,
            )
            s.listen(lambda d, n=queue_name: self._sensor_queue.put((d, n)))
            self._sensor_list.append(s)

        bp_col = bp_lib.find('sensor.other.collision')
        col = self._carla_world.spawn_actor(
            bp_col, carla.Transform(), attach_to=self._player,
        )
        col.listen(self._collision.on_event)
        self._sensor_list.append(col)

    def spawn_pedestrian(self, ego_transform, map_name):
        """Spawn one walking pedestrian near the parking area (mirrors ParkingScenes)."""
        bp_lib = self._carla_world.get_blueprint_library()
        pedestrian_bps = list(bp_lib.filter('walker.pedestrian.*'))
        walker_bp = random.choice(pedestrian_bps)
        walker_bp.set_attribute('is_invincible', 'false')

        if map_name == 'Town04_Opt':
            random_y = random.uniform(-30.0, -10.0)
            start_loc = ego_transform.location + carla.Location(x=2.0, y=random_y, z=0.3)
            if start_loc.y < -239.8:
                start_loc.y = -239.8
            direction = carla.Vector3D(0, 1, 0).make_unit_vector()
        elif map_name == 'Town10HD_Opt':
            random_x = random.uniform(10.0, 30.0)
            start_loc = ego_transform.location + carla.Location(x=random_x, y=3.0, z=0.3)
            direction = carla.Vector3D(-1, 0, 0).make_unit_vector()
        else:
            return

        spawn_tf = carla.Transform(start_loc, carla.Rotation(yaw=90))
        walker = self._carla_world.try_spawn_actor(walker_bp, spawn_tf)
        if walker:
            ctrl = carla.WalkerControl()
            ctrl.direction = direction
            ctrl.speed = 1.2
            walker.apply_control(ctrl)
            self._actor_list.append(walker)

    def spawn_static_npcs(self, slot_idx, map_name, seed, max_npcs=None):
        random.seed(seed)
        if map_name == 'Town04_Opt':
            spawn_pts = parking_position.parking_vehicle_locations_Town04
        elif map_name == 'Town10HD_Opt':
            spawn_pts = parking_position.parking_vehicle_locations_Town10
        else:
            return

        target = spawn_pts[slot_idx]
        blueprints = [
            bp for bp in self._carla_world.get_blueprint_library().filter('vehicle.*')
            if int(bp.get_attribute('number_of_wheels')) == 4
            and 'firetruck' not in bp.id
            and 'cybertruck' not in bp.id
            and 'ambulance' not in bp.id
        ]
        pts_shuffled = list(spawn_pts)
        random.shuffle(pts_shuffled)
        spawned = 0
        for sp in pts_shuffled:
            if max_npcs is not None and spawned >= max_npcs:  # harness: cap NPC count
                break
            if sp == target:
                continue
            # Town04_Opt: skip row-3 (x≈280, west of aisle) — the ego's RS forward
            # phase swings west and clips those vehicles during the approach arc.
            if map_name == 'Town04_Opt' and sp.x < 285.0:
                continue
            rot = carla.Rotation(yaw=random.choice([0.0, 180.0]))
            npc = self._carla_world.try_spawn_actor(
                random.choice(blueprints),
                carla.Transform(sp, rot),
            )
            if npc:
                npc.set_simulate_physics(False)
                self._actor_list.append(npc)
                spawned += 1

    def drain_sensors(self):
        """Read one frame's worth of sensor data from the queue (blocking)."""
        n_sensors = len(CAMERAS_6) + 2  # 6 cams + IMU + GNSS (collision has no queue entry)
        collected = {}
        for _ in range(n_sensors):
            try:
                data, name = self._sensor_queue.get(block=True, timeout=2.0)
                collected[name] = data
            except Empty:
                logging.warning("Sensor timeout — missed data this tick")
        return collected

    def reset_vehicle(self, transform):
        self._player.set_transform(transform)
        self._player.apply_control(carla.VehicleControl())
        self._player.set_target_velocity(carla.Vector3D(0, 0, 0))
        self._collision.hit = False; self._collision.actor = None
        self.need_init_ego_state = True
        # drain any stale queue entries
        while not self._sensor_queue.empty():
            try:
                self._sensor_queue.get_nowait()
            except Empty:
                break

    def destroy_all(self):
        for sensor in self._sensor_list:
            try:
                sensor.stop()
                sensor.destroy()
            except Exception:
                pass
        self._sensor_list.clear()
        if self._player:
            self._player.destroy()
            self._player = None
        for actor in self._actor_list:
            try:
                actor.destroy()
            except Exception:
                pass
        self._actor_list.clear()
        self._sensor_queue = Queue()
        self._collision.hit = False; self._collision.actor = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _carla_image_to_bgr(carla_img):
    arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_img.height, carla_img.width, 4))
    return arr[:, :, :3]  # drop alpha (BGRA → BGR)


def _rotation_error_deg(yaw_deg, map_name):
    goal_yaw = GOAL_YAW.get(map_name, 0.0)
    diff = abs(yaw_deg - goal_yaw) % 360
    return min(diff, 360 - diff)


def _get_slot_info(slot_idx, map_name):
    if map_name == 'Town04_Opt':
        loc = parking_position.parking_vehicle_locations_Town04[slot_idx]
        heading_rad = 0.0
    elif map_name == 'Town10HD_Opt':
        loc = parking_position.parking_vehicle_locations_Town10[slot_idx]
        heading_rad = math.pi / 2
    else:
        raise ValueError(f"Unsupported map: {map_name}")
    return loc, heading_rad


def _get_spawn_transform(slot_location, slot_heading_rad, map_name):
    """Return a random ego spawn transform near the parking area.

    For Town04: randomly approach from north (yaw=-90) or south (yaw=+90).
    These produce opposite RS arc geometries — reverse-right vs reverse-left —
    adding trajectory variety to the dataset. Cache keys include syaw so each
    direction gets its own cached plan.

    For Town10HD: randomly approach from west (yaw=180) or east (yaw=0),
    spawning on the matching side of the target slot.
    """
    if map_name == 'Town04_Opt':
        # Fixed x=285.6 is the aisle centreline (between row-2 at x=290.9 and
        # row-3 at x=280.0). Approach direction randomised ±90° for path variety.
        spawn_x = 285.6
        spawn_y = slot_location.y + random.uniform(-8.0, 8.0)
        spawn_yaw = random.choice([-90.0, 90.0])
    elif map_name == 'Town10HD_Opt':
        # Aisle runs east-west at y≈129.6. Randomly approach from west or east.
        if random.random() < 0.5:
            spawn_x = slot_location.x + random.uniform(-8.0, -1.0)  # west of slot
            spawn_yaw = 0.0    # facing east
        else:
            spawn_x = slot_location.x + random.uniform(1.0, 8.0)    # east of slot
            spawn_yaw = 180.0  # facing west
        spawn_y = 129.6 + random.uniform(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported map: {map_name}")
    return carla.Transform(
        carla.Location(x=spawn_x, y=spawn_y, z=0.3),
        carla.Rotation(yaw=spawn_yaw),
    )


def _actor_footprints(actor_gt_frames):
    """Static-obstacle footprint polygons (CARLA world x,y) from frame-0 actor GT.

    For the BEV overlay in visualize_episode.py. Each polygon is the 4 corners of
    the actor's world-frame 3D box projected to x,y. Uses the first recorded
    frame (parked cars are static; a pedestrian's frame-0 pose is representative).
    """
    if not actor_gt_frames:
        return []
    out = []
    for a in actor_gt_frames[0].get('actors', []):
        cx, cy = a['world_center'][0], a['world_center'][1]
        length, width = a['size_lwh'][0], a['size_lwh'][1]
        yaw = math.radians(a['yaw_deg'])
        hl, hw = length / 2.0, width / 2.0
        fx, fy = math.cos(yaw), math.sin(yaw)     # forward (length axis)
        lx, ly = -math.sin(yaw), math.cos(yaw)    # left (width axis)
        corners = [
            [cx + hl * fx + hw * lx, cy + hl * fy + hw * ly],
            [cx + hl * fx - hw * lx, cy + hl * fy - hw * ly],
            [cx - hl * fx - hw * lx, cy - hl * fy - hw * ly],
            [cx - hl * fx + hw * lx, cy - hl * fy + hw * ly],
        ]
        out.append({'polygon': corners, 'category': a['category'], 'id': a['id']})
    return out


def _read_astar_path(coordinate_rotation):
    """Read the last-generated expert solution path as CARLA world-frame waypoints.

    solution/CARLA.csv is ALWAYS written in CARLA world frame (the expert planner
    and the RS writer both apply the inverse planning rotation; the stock plan.py
    rotates the file in place via csv_coordinate_rotation). The MPC tracks it
    directly in CARLA coordinates, so we read x,y as-is. `coordinate_rotation` is
    kept for signature compatibility but no longer re-applied (that was a
    double-transform bug that mirrored the path onto the wrong side of the lot).
    This path is used only for meta['astar_path_world'] (visualization).
    """
    current_dir = pathlib.Path(__file__).resolve().parent.parent
    csv_path = current_dir / 'ParkingScenes' / 'tool' / 'AutomatedValetParking' / 'solution' / 'CARLA.csv'
    if not csv_path.exists():
        return []
    path = []
    with open(csv_path) as f:
        import csv as _csv
        reader = _csv.reader(f, delimiter='\t')
        for row in reader:
            try:
                x, y = float(row[1]), float(row[2])
                path.append([x, y])
            except (IndexError, ValueError):
                continue
    return path


# ── Main episode runner ───────────────────────────────────────────────────────

def run_episode(carla_world, args, slot_idx, episode_id, save_root):
    """Run one parking episode and save to save_root/episode_{episode_id:04d}/."""
    import shutil

    slot_location, slot_heading_rad = _get_slot_info(slot_idx, args.map)
    spawn_transform = _get_spawn_transform(slot_location, slot_heading_rad, args.map)

    heading_error_rad = abs(
        math.radians(spawn_transform.rotation.yaw) - slot_heading_rad
    )
    while heading_error_rad > math.pi:
        heading_error_rad -= 2 * math.pi
    heading_error_rad = abs(heading_error_rad)

    approach_mode = "forward" if heading_error_rad < math.radians(30) else "reverse"

    # Create episode dir upfront so images can be streamed to disk immediately.
    # Deleted on failure to avoid leaving partial data.
    ep_dir = save_root / f'episode_{episode_id:04d}'
    ep_dir.mkdir(parents=True, exist_ok=True)

    print("[EP] spawning ego...", flush=True)
    ws = WorldState(carla_world)
    ws.index = slot_idx
    ws.spawn_ego(spawn_transform)
    print("[EP] spawning NPCs...", flush=True)
    ws.spawn_static_npcs(slot_idx, args.map, seed=random.randint(0, 9999))
    if args.place_pedestrians:
        print("[EP] spawning pedestrian...", flush=True)
        ws.spawn_pedestrian(spawn_transform, args.map)

    weather_presets = [p for p in dir(carla.WeatherParameters)
                       if p[0].isupper() and 'Rain' not in p and 'Night' not in p]
    preset = getattr(carla.WeatherParameters, random.choice(weather_presets))
    carla_world.set_weather(preset)

    print("[EP] creating Auto_Park controller...", flush=True)
    controller = Auto_Park(ws)
    print("[EP] controller ready, entering tick loop", flush=True)

    poses = []       # pose records only — images written to disk per-tick
    actor_gt_frames = []  # per-frame actor GT (parked cars / pedestrian boxes + classes)
    frame_count = 0
    tick_count = 0
    goal_hold = 0
    success = False
    _planning_done = False

    logging.info("Episode %04d: slot %d, spawn (%.1f, %.1f, yaw %.0f°)",
                 episode_id, slot_idx,
                 spawn_transform.location.x, spawn_transform.location.y,
                 spawn_transform.rotation.yaw)

    try:
        while tick_count < TIMEOUT_TICKS:
            if tick_count == 0:
                print("[EP] first tick...", flush=True)
            carla_world.tick()
            if tick_count == 0:
                print("[EP] tick done, draining sensors...", flush=True)
            sensor_data = ws.drain_sensors()
            if tick_count == 0:
                print("[EP] sensors drained", flush=True)

            t = ws.player.get_transform()
            v = ws.player.get_velocity()
            c = ws.player.get_control()

            if not _planning_done:
                print("[EP] calling controller.main (planning)...", flush=True)
            controller.main(None, ws, _CLOCK, slot_idx, args)
            if not _planning_done:
                print("[EP] plan done, executing MPC", flush=True)
                _planning_done = True
            elif tick_count == 1:
                print(f"[EP] first MPC tick: pos=({t.location.x:.1f},{t.location.y:.1f}) "
                      f"yaw={t.rotation.yaw:.0f}° speed={math.sqrt(v.x**2+v.y**2):.2f}m/s "
                      f"dist_to_goal={t.location.distance(slot_location):.1f}m over={ws.over}",
                      flush=True)

            dist = t.location.distance(slot_location)
            rot_err = _rotation_error_deg(t.rotation.yaw, args.map)
            if ws.over or (dist < GOAL_DIST_M and rot_err < GOAL_ROT_DEG):
                goal_hold += 1
            else:
                goal_hold = 0

            # Early-tick diagnostic: print every tick for the first 10, then every 30
            if tick_count > 0 and (tick_count <= 10 or tick_count % 30 == 0):
                speed_now = math.sqrt(v.x**2 + v.y**2)
                print(f"[TICK {tick_count:4d}] pos=({t.location.x:.2f},{t.location.y:.2f}) "
                      f"yaw={t.rotation.yaw:.1f}° spd={speed_now:.2f} dist={dist:.2f} "
                      f"over={ws.over} goal_hold={goal_hold} "
                      f"ctrl=(thr={c.throttle:.2f} brk={c.brake:.2f} steer={c.steer:.2f} rev={c.reverse})",
                      flush=True)

            if tick_count % 150 == 0 and tick_count > 0:
                logging.info("Episode %04d: tick %d/%d, dist=%.1fm, over=%s, pos=(%.1f,%.1f) yaw=%.0f°",
                             episode_id, tick_count, TIMEOUT_TICKS,
                             t.location.distance(slot_location),
                             ws.over, t.location.x, t.location.y, t.rotation.yaw)

            if ws._collision.hit:
                print(f"[EP] COLLISION at tick {tick_count} with {ws._collision.actor} — aborting", flush=True)
                logging.info("Episode %04d: collision at tick %d — aborting",
                             episode_id, tick_count)
                break

            # Stream images to disk immediately — avoids buffering GB of CARLA
            # Image objects in RAM for the duration of the episode.
            if tick_count % RECORD_EVERY == 0:
                frame_idx = frame_count
                frame_dir = ep_dir / 'frames' / f'frame_{frame_idx:04d}'
                frame_dir.mkdir(parents=True, exist_ok=True)

                for cam_name in CAMERAS_6:
                    img_data = sensor_data.get(cam_name)
                    if img_data is not None:
                        bgr = _carla_image_to_bgr(img_data)
                        cv2.imwrite(
                            str(frame_dir / f'{cam_name}.jpg'), bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 90],
                        )

                imu = sensor_data.get('imu')
                speed_ms = math.sqrt(v.x**2 + v.y**2)
                poses.append({
                    'frame_idx':        frame_idx,
                    'timestamp_us':     int(time.time() * 1e6) + frame_idx * 500_000,
                    'x_world':          t.location.x,
                    'y_world':          t.location.y,
                    'z_world':          t.location.z,
                    'yaw_deg':          t.rotation.yaw,
                    'pitch_deg':        t.rotation.pitch,
                    'roll_deg':         t.rotation.roll,
                    'vx_world':         v.x,
                    'vy_world':         v.y,
                    'vz_world':         v.z,
                    'speed_ms':         speed_ms,
                    'yaw_rate_rads':    imu.gyroscope.z if imu else 0.0,
                    'steer_normalized': c.steer,
                    'throttle':         c.throttle,
                    'brake':            c.brake,
                    'reverse':          c.reverse,
                })
                # Per-frame actor GT: parked NPCs + pedestrian boxes + classes,
                # in CARLA world frame (build_infos_pkl converts to lidar). Static
                # cars repeat each frame; a walking pedestrian moves per frame.
                ego_id = ws.player.id if ws.player else None
                actor_gt_frames.append({
                    'frame_idx': frame_idx,
                    'actors': carla_actor_gt.collect_actor_gt(ws._actor_list, ego_id=ego_id),
                })
                frame_count += 1

            tick_count += 1

            if goal_hold >= GOAL_HOLD_FRAMES:
                success = True
                logging.info("Episode %04d: goal reached at tick %d (%d frames recorded)",
                             episode_id, tick_count, frame_count)
                break

    finally:
        ws.destroy_all()
        if not success:
            shutil.rmtree(ep_dir, ignore_errors=True)

    if not success:
        reason = "collision" if ws._collision.hit else f"timeout at tick {tick_count}"
        print(f"[EP] FAILED ({reason}) frames={frame_count}", flush=True)
        return False

    if frame_count < 4:
        print(f"[EP] SUCCESS but only {frame_count} frames recorded (<4) — discarding", flush=True)
        shutil.rmtree(ep_dir, ignore_errors=True)
        return False

    with open(ep_dir / 'poses.json', 'w') as f:
        json.dump(poses, f, indent=2)

    with open(ep_dir / 'actors.json', 'w') as f:
        json.dump(actor_gt_frames, f, indent=2)

    astar_path = _read_astar_path(getattr(controller, '_coordinate_rotation', False))

    geom = SLOT_GEOMETRY.get(args.map, {})
    slot_w = geom.get('width_m', 2.8)
    slot_l = geom.get('length_m', 5.5)

    # Episode-level maneuver label from the scenario's known ground truth: the
    # slot it placed + the parked heading (GOAL_YAW) it drove to. Written in the
    # nuScenes global frame so build_infos_pkl can copy it verbatim.
    goal_heading = -math.radians(GOAL_YAW.get(args.map, 0.0))
    slot_cx_n, slot_cy_n = _carla_xy_to_nuscenes(slot_location.x, slot_location.y)
    spawn_x_n, spawn_y_n = _carla_xy_to_nuscenes(
        spawn_transform.location.x, spawn_transform.location.y)
    spawn_yaw_n = -math.radians(spawn_transform.rotation.yaw)

    meta = {
        'episode_id':   f'episode_{episode_id:04d}',
        'map':          args.map,
        'parking_type': geom.get('type', 'unknown'),
        'maneuver_type': MANEUVER_TYPE.get(args.map, 'reverse_perpendicular'),
        'side':          _compute_side(spawn_x_n, spawn_y_n, spawn_yaw_n,
                                       slot_cx_n, slot_cy_n),
        'target_slot':   _make_target_slot(slot_cx_n, slot_cy_n, slot_location.z,
                                           goal_heading, slot_w, slot_l),
        'slot': {
            'cx_world':    slot_location.x,
            'cy_world':    slot_location.y,
            'heading_rad': slot_heading_rad,
            'width_m':     slot_w,
            'length_m':    slot_l,
        },
        'spawn': {
            'x_world':    spawn_transform.location.x,
            'y_world':    spawn_transform.location.y,
            'heading_rad': math.radians(spawn_transform.rotation.yaw),
        },
        'heading_error_at_spawn_rad': heading_error_rad,
        'approach_mode':  approach_mode,
        'astar_path_world': astar_path[:200],
        'obstacles_world': _actor_footprints(actor_gt_frames),
        'total_frames':   frame_count,
        'sample_rate_hz': RECORD_HZ,
    }
    with open(ep_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Generate CARLA parking episodes')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--map', default='Town04_Opt',
                    choices=['Town04_Opt', 'Town10HD_Opt'])
    ap.add_argument('--num_episodes', type=int, default=50)
    ap.add_argument('--save_path', default=str(_REPO_ROOT / 'data' / 'raw'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--place_pedestrians', action='store_true', default=True,
                    help='Spawn a walking pedestrian each episode (default: True)')
    ap.add_argument('--no_pedestrians', dest='place_pedestrians', action='store_false')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    # plan.py inside ParkingScenes also calls parse_args() on sys.argv; clear
    # leftover args so it falls back to its own defaults.
    sys.argv[1:] = []

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    save_root = pathlib.Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    print(f"[MAIN] loading world {args.map} (may take 60-120s)...", flush=True)
    carla_world = client.load_world(args.map)
    print("[MAIN] world loaded", flush=True)
    carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

    settings = carla_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / SIM_HZ
    carla_world.apply_settings(settings)
    print("[MAIN] sync mode enabled, starting episodes", flush=True)

    if args.map == 'Town04_Opt':
        # Slots 17-32 span both rows (row 2: 17-31, row 3: 32) and all have
        # goal_yaw=180° (reverse park facing -x). Slots with goal_yaw=0° (slot 16,
        # slots 33-47) cause the RS+MPC to drive the car into adjacent NPCs.
        slot_indices = list(range(17, 33))
    else:
        slot_indices = list(range(len(parking_position.parking_vehicle_locations_Town10)))

    # Count existing episodes to continue numbering
    existing = sorted(save_root.glob('episode_*'))
    episode_id = int(existing[-1].name.split('_')[1]) + 1 if existing else 0

    collected = 0
    attempts = 0
    while collected < args.num_episodes:
        slot_idx = slot_indices[(episode_id + attempts) % len(slot_indices)]
        try:
            ok = run_episode(carla_world, args, slot_idx, episode_id, save_root)
        except Exception as e:
            logging.error("Episode %04d failed with exception: %s", episode_id, e)
            ok = False

        if ok:
            logging.info("Saved episode %04d (%d/%d)", episode_id, collected + 1, args.num_episodes)
            episode_id += 1
            collected += 1
            attempts = 0
        else:
            attempts += 1
            if attempts > 10:
                logging.error("Too many consecutive failures — stopping")
                break

    # Restore async mode before exit
    settings.synchronous_mode = False
    carla_world.apply_settings(settings)
    logging.info("Done. Collected %d episodes.", collected)


if __name__ == '__main__':
    main()
