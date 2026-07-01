"""Closed-loop orchestrator: spawn a CARLA scene, run the model (live, over the bridge) in the
loop, and measure the real outcome (success / collision / final-pose error / gear changes /
timeout). Runs in the CARLA venv (py3.8).

The model server (real harness/model_server/server.py, or the GPU-free harness/model_server/
mock_server.py) may run anywhere — set --bridge-host/--bridge-port.

    # GPU-free smoke (mock server in another shell on :5557):
    venv/bin/python harness/model_server/mock_server.py &
    venv/bin/python harness/run_episode.py --slot-idx 24 --seed 7

Loop, in CARLA synchronous mode @30Hz:
  every tick   : grab 6 imgs + ego pose, track the current 3s plan with the controller, step;
  every 3s (v0): re-plan (send to the model server, receive 6 waypoints) — receding horizon.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys
import time
from queue import Empty

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO))

import carla
import generate_episodes as ge
import carla_actor_gt  # shared per-frame actor GT export (Task 3, same as generate_episodes)
from harness import config as cfg_mod
from harness.client import scene as scene_mod
from harness.client import transforms as tf
from harness.client.bridge_client import BridgeClient
from harness.client.controller_adapter import ControllerAdapter
from harness.client.live_viewer import LiveViewer
from harness.client.metrics import EpisodeMetrics
from harness.model_server import live_prompt
from harness.model_server.live_uniad import CAM_ORDER

import cv2

# Per-tick controller telemetry (CTRL_DEBUG=1 to enable). Prints the command actually applied,
# the re-projected waypoints, the gear-deciding fwd_sum, and the pose error to the slot — so we
# can see whether the gear oscillates / the car cuts the corner while tracking a fixed plan.
CTRL_DEBUG = bool(os.environ.get("CTRL_DEBUG"))
CTRL_DEBUG_EVERY = int(os.environ.get("CTRL_DEBUG_EVERY", "5"))
HISTORY_EVERY = ge.SIM_HZ // 2   # record an ego pose every 0.5s for the 2s history


def clear_queue(ego):
    while not ego._sensor_queue.empty():
        try:
            ego._sensor_queue.get_nowait()
        except Empty:
            break


def drain_update(ego, latest):
    """Non-blocking: pull everything queued, keep the newest camera frame per name, discard
    IMU/GNSS. With cameras on a sensor_tick (render only at replan), this keeps the queue tiny
    and `latest` holding the most recent frames."""
    while True:
        try:
            data, name = ego._sensor_queue.get_nowait()
        except Empty:
            break
        if name in CAM_ORDER:
            latest[name] = data
        elif name == "imu":            # keep newest IMU for the measured yaw-rate (gyro.z)
            latest["imu"] = data


def ensure_cams(ego, latest, timeout=30.0):
    """Block until all 6 cameras are present in `latest` (first frame, or a fresh render at a
    replan tick). Generous timeout for WiFi-streamed frames."""
    deadline = time.time() + timeout
    while not all(c in latest for c in CAM_ORDER) and time.time() < deadline:
        try:
            data, name = ego._sensor_queue.get(timeout=2.0)
            if name in CAM_ORDER:
                latest[name] = data
            elif name == "imu":
                latest["imu"] = data
        except Empty:
            break
    return latest


def encode_jpegs(sensors):
    """6 CARLA images -> JPEG bytes in CAM_ORDER (BGR, quality 90 — matches collection)."""
    jpegs = []
    for cam in CAM_ORDER:
        bgr = ge._carla_image_to_bgr(sensors[cam])
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        jpegs.append(buf.tobytes())
    return jpegs


def remaining_ego_path(world_path, ex_c, ey_c, ex_n, ey_n, eyaw_n):
    """CARLA-world polyline -> ego-local (right, forward) waypoints, DROPPING the path points
    the car has already driven past. Without this the full 6-waypoint plan is re-projected
    every tick, so the early points slide in front of the car as it moves; their forward
    component flips sign and corrupts the gear decision (sum(forward)) and the lookahead. Keep
    only the points beyond the one nearest the car (always at least the final point so there's
    a target to pursue)."""
    if not world_path:
        return []
    nearest = min(range(len(world_path)),
                  key=lambda i: math.hypot(world_path[i][0] - ex_c, world_path[i][1] - ey_c))
    remaining = world_path[nearest + 1:] or world_path[-1:]
    out = []
    for px_c, py_c, tgt_yaw_n in remaining:
        right, forward = live_prompt.global_to_local_xy(px_c, -py_c, ex_n, ey_n, eyaw_n)
        # head_err = how much the car must still rotate (nuScenes frame) to match the model's
        # planned heading at this point. Recomputed each tick from the live pose, so it shrinks
        # to 0 as the car aligns. Third element of the controller's (right, forward, head_err).
        head_err = live_prompt.normalize_angle(tgt_yaw_n - eyaw_n)
        out.append((right, forward, head_err))
    return out


def _chase_cam(spectator, tr):
    """Point the CARLA spectator at the ego (3rd-person chase cam) so the CARLA window
    follows the car instead of sitting on a random road."""
    yaw = math.radians(tr.rotation.yaw)
    loc = carla.Location(tr.location.x - 10.0 * math.cos(yaw),
                         tr.location.y - 10.0 * math.sin(yaw), tr.location.z + 8.0)
    spectator.set_transform(carla.Transform(loc, carla.Rotation(pitch=-30.0, yaw=tr.rotation.yaw)))


def run(cfg, slot_idx=None, seed=None, max_episode_s=None, save_name="episode", no_view=False,
        no_npcs=False, max_npcs=None, start_pose=None, log_dagger=None, leash_m=None,
        controller=None):
    max_ticks = int((max_episode_s or cfg.timeout_s) * ge.SIM_HZ)
    replan_every = max(1, int(cfg.replan_dt * ge.SIM_HZ))

    print(f"[loop] connecting CARLA {cfg.carla_host}:{cfg.carla_port}, loading {cfg.town}...")
    ge.CAMERA_SENSOR_TICK = cfg.replan_dt  # cameras render/stream only at the planning cadence
    client, world = scene_mod.connect(cfg)
    scene = None
    try:
        scene = scene_mod.spawn_random_scene(world, cfg.town, slot_idx=slot_idx, seed=seed,
                                             npcs=not no_npcs, max_npcs=max_npcs,
                                             start_pose=start_pose)
        ego = scene.ego
        slot_global = scene.target["slot_global"]
        slot_polygon = scene.target["target_slot"]["polygon"]  # 4 corners, global frame
        maneuver, side = scene.target["maneuver_type"], scene.target["side"]
        print(f"[loop] slot {scene.slot_idx}: {maneuver} side={side} "
              f"goal_local={scene_mod.slot_local(ego, slot_global)}")

        bridge = BridgeClient(cfg.bridge_host, cfg.bridge_port).connect()
        # HEADING_GAIN tunes how much the controller follows the model's emitted per-waypoint
        # heading on near-collinear reverse arcs (1=full, 0=position-only/legacy). Live-tunable.
        # CONTROLLER selects the tracker: 'mpc' (linearized-bicycle QP, collection-grade — default,
        # better waypoint following + joint pos/yaw) or 'pursuit' (legacy pure-pursuit).
        controller_kind = (controller or os.environ.get("CONTROLLER", "mpc")).lower()
        if controller_kind == "pursuit":
            ctrl = ControllerAdapter(k_heading=float(os.environ.get("HEADING_GAIN", "1.0")))
            print("[loop] controller: pure-pursuit")
        else:
            from harness.client.mpc_tracker import MpcTracker
            ctrl = MpcTracker()
            print("[loop] controller: MPC (linearized-bicycle QP)")
        met = EpisodeMetrics(slot_global, cfg.goal_dist_m, cfg.goal_rot_deg)
        # Offline history = 4 PREVIOUS poses (t-2.0..t-0.5) + current origin (collect_history_local,
        # history_steps=4). We append the current pose each 0.5s and pass list(history)[:-1] to drop
        # it, so the deque must hold 5 to leave 4 historical. maxlen=4 left only 3 -> build_data_dict
        # padded the missing oldest by duplication -> a fake zero-velocity history step (OOD). The
        # offline-frame tests miss this (they build their own 4-pose history, bypassing this deque).
        history = collections.deque(maxlen=5)
        world_path = None
        plan_reverse = None   # gear latched once per replan (see remaining_ego_path / fix)
        trail = []
        last_prompt, last_infer_ms, last_rev, last_n_tracks = "", 0.0, False, -1
        cams_bgr = None
        out_dir = cfg.out_dir / "episodes"
        out_dir.mkdir(parents=True, exist_ok=True)
        viewer = None if no_view else LiveViewer(save_path=str(out_dir / f"{save_name}_replay.mp4"))
        spectator = world.get_spectator()
        latest = {}                  # newest camera frame per name
        t_start = time.time()
        drain_update(ego, latest)    # seed from the settle-tick camera frames

        # --- DAgger Phase-1 state logger ---------------------------------------------------
        # Records the REAL states the model drives into (cams + pose + velocity + goal), so an
        # OFFLINE pass (scripts/relabel_dagger.py) can label each with the expert RS planner and
        # emit standard episodes. No CARLA needed in that second phase. Static NPC footprints are
        # captured once here so the offline RS path can be collision-checked. This is TRUE DAgger:
        # the logged states come from the model's own (off-path) trajectory distribution.
        dagger_dir = None
        dagger_state_idx = 0
        if log_dagger:
            dagger_dir = pathlib.Path(log_dagger) / save_name
            dagger_dir.mkdir(parents=True, exist_ok=True)
            npc_boxes = []
            actors_world = []   # shared rich actor GT (class + id + full 3D box, CARLA world)
            for actor in world.get_actors().filter("vehicle.*"):
                if ego._player is not None and actor.id == ego._player.id:
                    continue
                at = actor.get_transform()
                bb = actor.bounding_box
                ax_n, ay_n, ayaw_n = tf.carla_to_odr_pose(
                    at.location.x, at.location.y, math.radians(at.rotation.yaw))
                npc_boxes.append({"x": ax_n, "y": ay_n, "yaw": ayaw_n,
                                  "ext_x": bb.extent.x, "ext_y": bb.extent.y})
                try:
                    actors_world.append(carla_actor_gt.extract_actor_gt(actor))
                except Exception:
                    pass
            (dagger_dir / "run_meta.json").write_text(json.dumps({
                "save_name": save_name, "slot_idx": scene.slot_idx,
                "maneuver_type": maneuver, "side": side,
                "slot_global": slot_global,                 # nuScenes {x,y,yaw}
                "slot_polygon": slot_polygon,               # nuScenes 4 corners
                "npcs": npc_boxes,                          # nuScenes centre+yaw+half-extent (legacy)
                "actors_world": actors_world,               # CARLA-world 3D box GT + class + id (shared)
                "cam_order": list(CAM_ORDER),
            }, indent=2))
            print(f"[dagger] logging states to {dagger_dir} "
                  f"({len(npc_boxes)} NPC boxes, {len(actors_world)} classed actor GT)")

        def render_view(stage, pred):
            if viewer is None:
                return
            sr, sf = live_prompt.global_to_local_xy(slot_global["x"], slot_global["y"], ex_n, ey_n, eyaw_n)
            sdh = live_prompt.normalize_angle(slot_global["yaw"] - eyaw_n)
            # The real slot rectangle (4 corners, global frame) projected into the ego frame, so
            # the viewer can show the slot's true footprint + heading and you can see whether the
            # car actually matches it (not just the centre point).
            slot_poly = [live_prompt.global_to_local_xy(px, py, ex_n, ey_n, eyaw_n)
                         for (px, py) in slot_polygon]
            viewer.set_stage(stage)
            viewer.render(cams=cams_bgr, prompt=last_prompt, pred_wps=pred, slot_local=(sr, sf, sdh),
                          slot_poly=slot_poly,
                          trail_world=trail, ego_n=(ex_n, ey_n, eyaw_n),
                          pose_err=met.pose_error(ex_n, ey_n, eyaw_n), tick=tick,
                          infer_ms=last_infer_ms, gear=last_rev, n_tracks=last_n_tracks)

        for tick in range(max_ticks):
            world.tick()
            drain_update(ego, latest)   # keep newest cam frames; cams render only at replan
            if viewer and all(c in latest for c in CAM_ORDER):
                cams_bgr = {c: ge._carla_image_to_bgr(latest[c]) for c in CAM_ORDER}
            tr = ego._player.get_transform()
            v = ego._player.get_velocity()
            _chase_cam(spectator, tr)  # CARLA window follows the car
            ex_c, ey_c, eyaw_c = tr.location.x, tr.location.y, math.radians(tr.rotation.yaw)
            ex_n, ey_n, eyaw_n = tf.carla_to_odr_pose(ex_c, ey_c, eyaw_c)
            speed = math.hypot(v.x, v.y)
            met.ticks = tick + 1
            trail.append((ex_n, ey_n))

            if tick % HISTORY_EVERY == 0:
                history.append({"x": ex_n, "y": ey_n})

            # --- termination ---
            if ego._collision.hit:
                met.collision, met.collision_actor = True, ego._collision.actor
                print(f"[loop] COLLISION at tick {tick}: {met.collision_actor}")
                break
            if met.is_parked(ex_n, ey_n, eyaw_n, speed):
                print(f"[loop] PARKED at tick {tick}")
                break
            # Leash: the model has driven hopelessly far from the slot (covariate-shift runaway).
            # Stop early so a batch doesn't waste sim time on a fled car; the states already
            # logged up to here are still valid DAgger data (RS will plan the long recovery).
            if leash_m is not None:
                d_slot = math.hypot(ex_n - slot_global["x"], ey_n - slot_global["y"])
                if d_slot > leash_m:
                    print(f"[loop] LEASH at tick {tick}: {d_slot:.1f}m > {leash_m}m from slot — stopping")
                    break

            # --- re-plan (receding horizon) ---
            cur = remaining_ego_path(world_path, ex_c, ey_c, ex_n, ey_n, eyaw_n) if world_path else []
            if tick % replan_every == 0:
                ensure_cams(ego, latest)  # block for a full fresh set of 6 cams
                render_view("capturing 6 cams -> UniAD + LLM running... (waiting)", cur)
                # MEASURED CARLA ego-state for the prompt. Velocity is rotated CARLA-world -> ego
                # frame exactly like build_infos_pkl._carla_to_ego_velocity (nvx=v.x, nvy=-v.y,
                # rotate by the nuScenes yaw eyaw_n); yaw-rate from the IMU gyro.z with the nuScenes
                # sign flip; steering straight off the vehicle control.
                # NOTE: global_to_local_xy returns (right, forward) — so unpack as (right_v, fwd_v),
                # NOT (fwd_v, right_v). The earlier swap put the car's forward/reverse speed into the
                # lateral field (fwd_v stuck ~0 while moving) — an OOD ego-state vs training, which
                # builds gt_ego_lcf_feat[0]=fwd_v, [1]=right_v from this measured motion.
                right_v, fwd_v = live_prompt.global_to_local_xy(v.x, -v.y, 0.0, 0.0, eyaw_n)
                imu = latest.get("imu")
                yaw_rate = -float(imu.gyroscope.z) if imu is not None else 0.0
                ego_payload = {"x": ex_n, "y": ey_n, "z": tr.location.z, "yaw": eyaw_n,
                               "speed": speed, "fwd_v": fwd_v, "right_v": right_v,
                               "yaw_rate": yaw_rate, "steer": float(ego._player.get_control().steer)}
                resp = bridge.infer(
                    frame_idx=tick, reset=(tick == 0), maneuver_type=maneuver, side=side,
                    slot_global=slot_global, ego=ego_payload,
                    ego_history=list(history)[:-1],
                    cam_names=CAM_ORDER, jpegs=encode_jpegs(latest))
                wps = resp["waypoints"]
                world_path = tf.waypoints_to_carla_path(wps, ex_c, ey_c, eyaw_c)
                # Carry each waypoint's TARGET nuScenes yaw (current ego yaw + model heading delta)
                # alongside the world path so the controller can track rotation, not just position.
                # waypoints_to_carla_path prepends the origin, which keeps the current heading.
                path_yaws = [eyaw_n] + [live_prompt.normalize_angle(eyaw_n + (wp[2] if len(wp) > 2 else 0.0))
                                        for wp in wps]
                world_path = [(x, y, yw) for (x, y), yw in zip(world_path, path_yaws)]
                plan_reverse = ctrl.gear_is_reverse(ctrl.first_segment(wps))  # leading-segment gear (cold-start creep dir)
                last_prompt, last_infer_ms = resp.get("prompt", ""), resp.get("infer_ms", 0.0)
                last_n_tracks = resp.get("n_tracks", -1)
                cur = remaining_ego_path(world_path, ex_c, ey_c, ex_n, ey_n, eyaw_n)
                render_view("received waypoints -> controller", cur)
                # Does the MODEL'S OWN plan reach the slot? Compare its last waypoint to the slot,
                # both in the current ego frame. Small endplan_err => model is planning to the
                # slot (any miss is controller tracking); large => the model's plan itself misses.
                slot_r, slot_f = live_prompt.global_to_local_xy(
                    slot_global["x"], slot_global["y"], ex_n, ey_n, eyaw_n)
                end_r, end_f = wps[-1][0], wps[-1][1]
                endplan_err = math.hypot(end_r - slot_r, end_f - slot_f)
                ego_line = next((ln for ln in last_prompt.split("\n")
                                 if ln.startswith("Ego states:")), "")
                print(f"[loop] tick {tick}: replan ({last_infer_ms:.0f}ms) "
                      f"UniAD_objects={last_n_tracks} wp0=({wps[0][0]:+.2f},{wps[0][1]:+.2f}) "
                      f"reach={max(math.hypot(wp[0], wp[1]) for wp in wps):.2f}m "
                      f"gear={'R' if plan_reverse else 'F'} "
                      f"plan_end=({end_r:+.2f},{end_f:+.2f}) slot=({slot_r:+.2f},{slot_f:+.2f}) "
                      f"endplan_err={endplan_err:.2f}m")
                print(f"[loop]   measured ego: fwd_v={fwd_v:+.3f} right_v={right_v:+.3f} "
                      f"yaw_rate={yaw_rate:+.3f} steer={ego_payload['steer']:+.3f} "
                      f"speed={speed:.2f}m/s | {ego_line}", flush=True)

                # DAgger: dump THIS visited state (cams + measured ego + history) for offline
                # expert relabelling. We save the cams the model actually saw and the real
                # moving history (list(history)[:-1]) so the offline label gets a true history.
                if dagger_dir is not None:
                    st_dir = dagger_dir / f"state_{dagger_state_idx:04d}"
                    st_dir.mkdir(parents=True, exist_ok=True)
                    for cam, blob in zip(CAM_ORDER, encode_jpegs(latest)):
                        (st_dir / f"{cam}.jpg").write_bytes(blob)
                    (st_dir / "state.json").write_text(json.dumps({
                        "tick": tick,
                        "ego": ego_payload,                         # nuScenes pose + measured ego-state
                        "ego_carla": {"x": ex_c, "y": ey_c, "yaw_rad": eyaw_c},
                        "history": list(history)[:-1],              # nuScenes [{x,y}], oldest->newest
                        "model_wps": [[float(a) for a in wp] for wp in wps],
                        "endplan_err": endplan_err,
                    }, indent=2))
                    dagger_state_idx += 1

            # --- track the current plan ---
            if cur:
                # Gear is decided per-tick from the leading single-gear SEGMENT of the (trimmed)
                # plan, NOT latched for the whole window: as the car passes the forward segment,
                # remaining_ego_path drops those points and the cusp's reverse tail becomes the new
                # leading segment -> the controller switches gear mid-window and completes the
                # forward-then-reverse maneuver instead of see-sawing between replans.
                goal_dist = math.hypot(ex_n - slot_global["x"], ey_n - slot_global["y"])
                thr, brk, steer, rev = ctrl.control(cur, speed, force_reverse=None, goal_dist=goal_dist)
                # Creep ONLY fires far from the slot (a genuine standstill start). Near the slot
                # the model correctly emits a small plan when it thinks it has arrived; if the
                # heading is still off, is_parked() fails -> WITHOUT this gate the creep would
                # nudge the car around the slot forever (the near-goal creep loop). Near the slot
                # with a stop command and not parked: just brake and let it settle.
                le_, lo_, _ = met.pose_error(ex_n, ey_n, eyaw_n)
                near_slot = math.hypot(le_, lo_) < cfg.goal_dist_m
                cold_start = (brk > 0.5 and speed < 0.1 and not near_slot
                              and not met.is_parked(ex_n, ey_n, eyaw_n, speed))
                # cold start: model says "stay" but we're not parked and stopped -> creep in the
                # planned gear (NOT hard-forward, which would shove a reverse maneuver outward).
                if cold_start:
                    thr, brk, steer, rev = 0.3, 0.0, 0.0, bool(plan_reverse)
                last_rev = rev
                met.note_gear(rev)
                ego._player.apply_control(carla.VehicleControl(
                    throttle=float(thr), brake=float(brk), steer=float(steer), reverse=bool(rev)))
                if CTRL_DEBUG and tick % CTRL_DEBUG_EVERY == 0:
                    reach_c = max(math.hypot(wp[0], wp[1]) for wp in cur) if cur else 0.0
                    fwd_sum = sum(wp[1] for wp in cur)
                    wp_dump = " ".join(f"({wp[0]:+.2f},{wp[1]:+.2f},h{wp[2]:+.2f})" for wp in cur)
                    le, lo, hd = met.pose_error(ex_n, ey_n, eyaw_n)
                    print(f"[ctrl] t{tick:4d} spd={speed:5.2f} reach={reach_c:5.2f} "
                          f"fwd_sum={fwd_sum:+6.2f} gear={'R' if rev else 'F'} "
                          f"thr={thr:.2f} brk={brk:.2f} steer={steer:+.2f}"
                          f"{' COLD' if cold_start else ''} "
                          f"| to_slot lat={le:.2f} lon={lo:.2f} head={hd:.1f} "
                          f"| wps {wp_dump}", flush=True)
            render_view(f"tracking plan -> steer/throttle{' [REVERSE]' if last_rev else ''}", cur)

        timeout = met.ticks >= max_ticks
        tr = ego._player.get_transform()
        ex_n, ey_n, eyaw_n = tf.carla_to_odr_pose(tr.location.x, tr.location.y,
                                                  math.radians(tr.rotation.yaw))
        summary = met.summary(ex_n, ey_n, eyaw_n, timeout)
        summary.update(wall_s=round(time.time() - t_start, 1), last_infer_ms=round(last_infer_ms, 1),
                       slot_idx=scene.slot_idx, maneuver=maneuver, side=side)
        bridge.close()
        if viewer is not None:
            viewer.close()

        (out_dir / f"{save_name}_summary.json").write_text(json.dumps(summary, indent=2))
        _save_trajectory(scene, trail, ex_n, ey_n, out_dir / f"{save_name}_traj.png")
        print(f"[loop] DONE: {json.dumps(summary)}")
        print(f"[loop] saved {out_dir}/{save_name}_summary.json + _traj.png + _replay.avi")
        return summary
    finally:
        # If the simulator died mid-run, don't hang 120s on each cleanup RPC — fail fast.
        try:
            client.set_timeout(8.0)
        except Exception:
            pass
        try:
            if scene is not None and scene.ego is not None:
                scene.ego.destroy_all()
        except Exception as e:
            print(f"[loop] cleanup: destroy skipped (simulator unreachable: {e})")
        try:  # restore async mode so the sim isn't left frozen for the spectator
            s = world.get_settings()
            s.synchronous_mode = False
            world.apply_settings(s)
        except Exception:
            pass


def _save_trajectory(scene, trail, fx, fy, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    fig, ax = plt.subplots(figsize=(9, 9))
    for c in scene.candidates:
        chosen = c["slot_idx"] == scene.slot_idx
        ax.add_patch(MplPoly(c["target_slot"]["polygon"], closed=True, fill=chosen,
                             edgecolor="green" if chosen else "gray",
                             facecolor="green" if chosen else "none", alpha=0.35 if chosen else 1.0))
    if trail:
        xs, ys = zip(*trail)
        ax.plot(xs, ys, "b-", linewidth=1.5, label="ego trail")
        ax.plot(xs[0], ys[0], "go", markersize=8, label="start")
    ax.plot(fx, fy, "r*", markersize=14, label="final")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(f"slot {scene.slot_idx} trajectory")
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    cfg_mod.Config.add_cli_args(p)
    p.add_argument("--slot-idx", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-s", type=float, default=None, help="episode time cap (default cfg.timeout_s)")
    p.add_argument("--name", default="episode")
    p.add_argument("--no-view", action="store_true", help="disable the live viewer (headless batch)")
    p.add_argument("--no-npcs", action="store_true", help="skip parked NPC cars (clean maneuver test)")
    p.add_argument("--max-npcs", type=int, default=None, help="cap number of parked NPC cars (fewer = easier)")
    p.add_argument("--start-token", default=None,
                   help="spawn the ego at a recorded frame's exact ego pose (e.g. "
                        "episode_0000_f0000) for an on-distribution closed-loop test")
    p.add_argument("--log-dagger", default=None,
                   help="directory to log the model's visited states (cams + pose + history + "
                        "NPC boxes) for offline expert relabelling (true DAgger Phase 1)")
    p.add_argument("--leash", type=float, default=None,
                   help="stop the episode early once the car gets this many metres from the slot "
                        "(avoids wasting sim time on a runaway; logged states stay valid)")
    p.add_argument("--controller", choices=["pursuit", "mpc"], default=None,
                   help="trajectory tracker: 'mpc' (linearized-bicycle QP, default — better "
                        "waypoint following) or 'pursuit' (legacy pure-pursuit)")
    args = p.parse_args(argv)
    cfg = cfg_mod.Config.from_args(args)

    start_pose = None
    if args.start_token:
        import pickle
        proc = pathlib.Path.home() / "projects/openvla_nuscenes/data_carla/processed"
        infos = pickle.load(open(proc / "parking_infos_temporal.pkl", "rb"))
        infos = infos["infos"] if isinstance(infos, dict) and "infos" in infos else infos
        info = next(e for e in infos if e["token"] == args.start_token)
        ex, ey = info["ego2global_translation"][:2]
        q = info["ego2global_rotation"]
        eyaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                          1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
        start_pose = (ex, ey, eyaw)
        print(f"[loop] start-token {args.start_token}: ego @ ({ex:.2f},{ey:.2f},yaw={eyaw:.3f})")

    run(cfg, slot_idx=args.slot_idx, seed=args.seed, max_episode_s=args.max_s,
        save_name=args.name, no_view=args.no_view, no_npcs=args.no_npcs, max_npcs=args.max_npcs,
        start_pose=start_pose, log_dagger=args.log_dagger, leash_m=args.leash,
        controller=args.controller)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
