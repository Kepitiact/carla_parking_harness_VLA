"""Spawn a random CARLA parking scene and identify the target slot — reusing the
collection pipeline (scripts/generate_episodes.py) so the ego rig, slot geometry, and
maneuver/side labels are IDENTICAL to what the model trained on.

Runs in the CARLA venv (py3.8). Needs a running CARLA server:
    ParkingScenes/carla/CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen &

Standalone demo (spawns a random scene, draws the slots, saves a top-down PNG):
    venv/bin/python harness/client/scene.py                 # random slot
    venv/bin/python harness/client/scene.py --slot-idx 20   # specific bay

All slot/ego poses are returned in the nuScenes/OpenDRIVE global frame (y_odr = -y_carla),
the frame the model + live_prompt consume.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))   # import the collection script as a library
sys.path.insert(0, str(_REPO))

import carla
import generate_episodes as ge  # reuse WorldState (6-cam rig), slot helpers, constants
from harness import config as cfg_mod


class Scene:
    """Result of spawning a scene: the CARLA world + ego, the chosen target slot, and the
    candidate bays (all in the nuScenes/OpenDRIVE frame). `ego` is a ge.WorldState."""
    def __init__(self, world, ego, target, candidates, spawn_tf, slot_idx):
        self.world = world
        self.ego = ego
        self.target = target          # dict: slot_global {x,y,yaw}, side, maneuver_type, polygon, ...
        self.candidates = candidates  # list of the same dict for every valid bay
        self.spawn_tf = spawn_tf
        self.slot_idx = slot_idx


def connect(cfg: cfg_mod.Config):
    client = carla.Client(cfg.carla_host, cfg.carla_port)
    client.set_timeout(120.0)
    world = client.load_world(cfg.town)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 1.0 / ge.SIM_HZ
    world.apply_settings(s)
    # Match the training distribution: collection used random CLEAR daytime presets (no
    # rain/night). Fix a representative in-distribution preset for reproducible eval.
    world.set_weather(carla.WeatherParameters.ClearNoon)
    return client, world


def valid_slots(town: str):
    """Perpendicular bays the collection used (Town04: reverse-park, goal_yaw=180)."""
    if town == "Town04_Opt":
        return list(range(17, 33))
    return list(range(len(ge.parking_position.parking_vehicle_locations_Town10)))


def slot_target(slot_idx: int, spawn_tf, town: str) -> dict:
    """Reproduce run_episode's target-slot + side construction (nuScenes global frame)."""
    slot_loc, slot_heading_rad = ge._get_slot_info(slot_idx, town)
    goal_heading = -math.radians(ge.GOAL_YAW.get(town, 0.0))
    scx, scy = ge._carla_xy_to_nuscenes(slot_loc.x, slot_loc.y)
    spx, spy = ge._carla_xy_to_nuscenes(spawn_tf.location.x, spawn_tf.location.y)
    spyaw = -math.radians(spawn_tf.rotation.yaw)
    geom = ge.SLOT_GEOMETRY[town]
    return {
        "slot_idx": slot_idx,
        "maneuver_type": ge.MANEUVER_TYPE.get(town, "reverse_perpendicular"),
        "side": ge._compute_side(spx, spy, spyaw, scx, scy),
        "slot_global": {"x": scx, "y": scy, "yaw": goal_heading},
        "target_slot": ge._make_target_slot(scx, scy, slot_loc.z, goal_heading,
                                            geom["width_m"], geom["length_m"]),
        "carla_xy": (slot_loc.x, slot_loc.y),
    }


def ego_nuscenes_pose(ego):
    """ego (ge.WorldState) -> (x, y, yaw) in the nuScenes/OpenDRIVE frame (y_odr=-y_carla)."""
    t = ego._player.get_transform()
    return t.location.x, -t.location.y, -math.radians(t.rotation.yaw)


def slot_local(ego, slot_global: dict):
    """The slot in the CURRENT ego frame [right, forward, dheading] — i.e. the (R,F,H) the
    model actually receives in its Mission goal. Reuses the VALIDATED live_prompt transform
    (same one that byte-matched the offline cache). This is what to compare to training."""
    from harness.model_server import live_prompt
    ex, ey, eyaw = ego_nuscenes_pose(ego)
    r, f = live_prompt.global_to_local_xy(slot_global["x"], slot_global["y"], ex, ey, eyaw)
    dh = live_prompt.normalize_angle(slot_global["yaw"] - eyaw)
    return r, f, dh


def spawn_random_scene(world, town: str, slot_idx=None, seed=None, pedestrians=False,
                       npcs=True, max_npcs=None, start_pose=None) -> Scene:
    if seed is not None:
        random.seed(seed)
    slots = valid_slots(town)
    if slot_idx is None:
        slot_idx = random.choice(slots)
    slot_loc, slot_heading_rad = ge._get_slot_info(slot_idx, town)
    spawn_tf = ge._get_spawn_transform(slot_loc, slot_heading_rad, town)
    if start_pose is not None:
        # Override the computed spawn with an exact recorded pose (x, y, yaw in the nuScenes/
        # OpenDRIVE frame) -> CARLA transform (x=x_n, y=-y_n, yaw=-yaw_n). Used to start an
        # episode at its recorded f0000 ego pose for an on-distribution closed-loop test.
        sx, sy, syaw = start_pose
        spawn_tf = carla.Transform(
            carla.Location(x=float(sx), y=-float(sy), z=spawn_tf.location.z),
            carla.Rotation(yaw=-math.degrees(float(syaw))))

    ego = ge.WorldState(world)
    ego.index = slot_idx
    ego.spawn_ego(spawn_tf)
    if npcs:  # parked neighbour cars; disable (npcs=False) or cap (max_npcs) for testing
        ego.spawn_static_npcs(slot_idx, town, seed=random.randint(0, 9999), max_npcs=max_npcs)
    if pedestrians:
        ego.spawn_pedestrian(spawn_tf, town)
    for _ in range(4):  # let physics settle the parked NPCs
        world.tick()

    target = slot_target(slot_idx, spawn_tf, town)
    candidates = [slot_target(i, spawn_tf, town) for i in slots]
    return Scene(world, ego, target, candidates, spawn_tf, slot_idx)


def render_topdown(scene: Scene, out_path: pathlib.Path):
    """Top-down PNG (nuScenes frame): candidate bays, occupied ones, ego, chosen target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    fig, ax = plt.subplots(figsize=(9, 9))
    # candidate bays (gray) + chosen (green)
    for c in scene.candidates:
        poly = c["target_slot"]["polygon"]
        chosen = c["slot_idx"] == scene.slot_idx
        ax.add_patch(MplPoly(poly, closed=True, fill=chosen, alpha=0.35 if chosen else 1.0,
                             edgecolor="green" if chosen else "gray",
                             facecolor="green" if chosen else "none", linewidth=2 if chosen else 1))
        cx, cy = c["slot_global"]["x"], c["slot_global"]["y"]
        ax.text(cx, cy, str(c["slot_idx"]), fontsize=6, ha="center", va="center",
                color="green" if chosen else "gray")

    # other vehicles (NPCs) in nuScenes frame
    for v in scene.world.get_actors().filter("vehicle.*"):
        if scene.ego._player is not None and v.id == scene.ego._player.id:
            continue
        t = v.get_transform()
        ax.plot(t.location.x, -t.location.y, "rs", markersize=8, alpha=0.6)

    # ego (blue arrow) in nuScenes frame
    et = scene.ego._player.get_transform()
    ex, ey = et.location.x, -et.location.y
    eyaw = -math.radians(et.rotation.yaw)
    ax.arrow(ex, ey, 2.5 * math.cos(eyaw), 2.5 * math.sin(eyaw), head_width=1.0,
             color="blue", length_includes_head=True)
    ax.plot(ex, ey, "bo", markersize=6)

    tgt = scene.target
    ax.set_title(f"slot {scene.slot_idx} | {tgt['maneuver_type']} | side={tgt['side']} | "
                 f"goal=({tgt['slot_global']['x']:.1f},{tgt['slot_global']['y']:.1f},"
                 f"{tgt['slot_global']['yaw']:.2f})")
    ax.set_aspect("equal")
    ax.set_xlabel("x_odr (=x_carla)")
    ax.set_ylabel("y_odr (=-y_carla)")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    cfg_mod.Config.add_cli_args(p)
    p.add_argument("--slot-idx", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)
    cfg = cfg_mod.Config.from_args(args)

    print(f"[scene] connecting to CARLA {cfg.carla_host}:{cfg.carla_port}, loading {cfg.town} (60-120s)...")
    client, world = connect(cfg)
    print("[scene] world loaded; spawning scene...")
    try:
        scene = spawn_random_scene(world, cfg.town, slot_idx=args.slot_idx, seed=args.seed)
        t = scene.target
        print(f"[scene] spawn @ ({scene.spawn_tf.location.x:.1f},{scene.spawn_tf.location.y:.1f}, "
              f"yaw={scene.spawn_tf.rotation.yaw:.0f})")
        print(f"[scene] TARGET slot {scene.slot_idx}: maneuver={t['maneuver_type']} side={t['side']}")
        print(f"[scene]   slot_GLOBAL (world frame, for drawing only): x={t['slot_global']['x']:.2f} "
              f"y={t['slot_global']['y']:.2f} yaw={t['slot_global']['yaw']:.3f}")
        r, f, h = slot_local(scene.ego, t["slot_global"])
        print(f"[scene]   slot_LOCAL (what the model sees, right/forward/heading): "
              f"({r:.2f}, {f:.2f}, {h:.2f})   <- compare to training ~5-7m")
        print(f"[scene]   {len(scene.candidates)} candidate bays")
        out = cfg.out_dir / "scenes" / f"scene_slot{scene.slot_idx}.png"
        render_topdown(scene, out)
        print(f"[scene] top-down saved: {out}")
    finally:
        if "scene" in dir() and scene.ego is not None:
            scene.ego.destroy_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
