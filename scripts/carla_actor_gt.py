"""Shared per-frame actor ground-truth export for BOTH collectors.

Used by scripts/generate_episodes.py and the harness/DAgger collector so both
datasets carry identical actor GT. Extracts, from a CARLA actor, the world-frame
3D box + a stable id + a nuScenes-10 class label, ready for build_infos_pkl.py to
convert (world -> lidar) into gt_boxes/gt_names/gt_inds/gt_velocity.

nuScenes-10 classes (must match UniAD stage1 class_names):
  car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle,
  pedestrian, traffic_cone

CARLA blueprints expose:
  * base_type: car / truck / van / Bus / motorcycle / bicycle   (may be empty)
  * special_type: emergency / electric / taxi / motorcycle / ...
  * number_of_wheels
We map with base_type first, then special_type / name / wheel-count fallbacks.
CARLA has no construction_vehicle / trailer / traffic_cone vehicle blueprints, so
those classes only appear if a future scenario spawns static props with an
explicit category (see PROP_CLASS_HINTS).
"""
import math

NUSC_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]

# CARLA base_type -> nuScenes class. 'van' has no nuScenes class; nuScenes folds
# vans/pickups into 'truck'.
_BASE_TYPE_MAP = {
    "car": "car",
    "truck": "truck",
    "van": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}

# Name substrings for blueprints whose base_type is empty or misleading.
_NAME_HINTS = [
    ("firetruck", "truck"),
    ("ambulance", "truck"),
    ("carlacola", "truck"),
    ("cybertruck", "truck"),
    ("sprinter", "truck"),
    ("t2", "truck"),
    ("fuso", "bus"),
    ("rosa", "bus"),
    ("harley", "motorcycle"),
    ("vespa", "motorcycle"),
    ("yamaha", "motorcycle"),
    ("kawasaki", "motorcycle"),
    ("ninja", "motorcycle"),
    ("omafiets", "bicycle"),
    ("crossbike", "bicycle"),
    ("century", "bicycle"),
    ("diamondback", "bicycle"),
    ("gazelle", "bicycle"),
]

# Optional: static props (barriers/cones) a future scenario may spawn, keyed by
# blueprint-id substring -> nuScenes class.
PROP_CLASS_HINTS = [
    ("cone", "traffic_cone"),
    ("trafficcone", "traffic_cone"),
    ("barrier", "barrier"),
    ("barrel", "barrier"),
    ("streetbarrier", "barrier"),
]


def blueprint_to_class(type_id, base_type="", special_type="", number_of_wheels=0):
    """Map a CARLA blueprint (+ its attributes) to a nuScenes-10 class string."""
    tid = (type_id or "").lower()
    if tid.startswith("walker.") or "pedestrian" in tid:
        return "pedestrian"

    for sub, cls in PROP_CLASS_HINTS:
        if sub in tid:
            return cls

    bt = (base_type or "").strip().lower()
    if bt in _BASE_TYPE_MAP:
        return _BASE_TYPE_MAP[bt]

    st = (special_type or "").strip().lower()
    if st == "motorcycle":
        return "motorcycle"

    for sub, cls in _NAME_HINTS:
        if sub in tid:
            return cls

    if number_of_wheels == 2:
        return "motorcycle"
    return "car"


def _attr(bp_or_actor, name, default=""):
    try:
        if bp_or_actor.has_attribute(name):
            return bp_or_actor.get_attribute(name).as_str()
    except Exception:
        pass
    return default


def classify_actor(actor):
    """nuScenes class for a live CARLA actor (reads its type_id + attributes)."""
    type_id = getattr(actor, "type_id", "")
    base = _attr(actor, "base_type")
    special = _attr(actor, "special_type")
    try:
        nw = actor.get_attribute("number_of_wheels").as_int() \
            if actor.has_attribute("number_of_wheels") else 0
    except Exception:
        nw = 0
    return blueprint_to_class(type_id, base, special, nw)


def extract_actor_gt(actor):
    """World-frame 3D box GT for one CARLA actor.

    Returns a JSON-serializable dict (all lengths in metres, FULL box dimensions,
    yaw in degrees, world frame). build_infos_pkl.py converts this to the lidar
    frame per the recorded ego pose.
      id            stable per-episode actor id (CARLA actor.id)
      type_id       CARLA blueprint id (provenance)
      category      nuScenes-10 class string
      world_center  [x, y, z] box centre in CARLA world coords
      yaw_deg       box yaw in CARLA world (deg)
      size_lwh      [length(x), width(y), height(z)] FULL dims (2*extent)
      velocity      [vx, vy] world (m/s) — nonzero for pedestrians
    """
    tf = actor.get_transform()
    bb = actor.bounding_box
    # Box centre in world = actor transform applied to the bbox local centre.
    center = tf.transform(bb.location)
    try:
        vel = actor.get_velocity()
        velocity = [vel.x, vel.y]
    except Exception:
        velocity = [0.0, 0.0]
    return {
        "id": int(actor.id),
        "type_id": actor.type_id,
        "category": classify_actor(actor),
        "world_center": [center.x, center.y, center.z],
        "yaw_deg": tf.rotation.yaw + bb.rotation.yaw,
        "size_lwh": [2.0 * bb.extent.x, 2.0 * bb.extent.y, 2.0 * bb.extent.z],
        "velocity": velocity,
    }


def collect_actor_gt(actor_list, ego_id=None):
    """Per-frame actor GT list from a collector's actor list (skips the ego)."""
    out = []
    for a in actor_list:
        try:
            if ego_id is not None and int(a.id) == int(ego_id):
                continue
            if not a.is_alive:
                continue
            out.append(extract_actor_gt(a))
        except Exception:
            continue
    return out
