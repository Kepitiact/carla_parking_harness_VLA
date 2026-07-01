"""Project infos gt_boxes onto the episode camera images (GT-on-camera overlay).

Validation deliverable: draws each lidar-frame GT box (from parking_infos_temporal.pkl)
into every camera using the SAME intrinsics + sensor2lidar extrinsics the infos
carry, so we can eyeball that boxes land on the actual parked cars — especially
close cars (<3 m) that UniAD misses. Saves a 6-camera composite PNG per frame.

Usage:
  venv/bin/python scripts/overlay_gt_on_camera.py \
      --infos data/processed_test/parking_infos_temporal.pkl \
      --frames 0 10 20 --out data/gt_overlays
"""
import argparse
import os
import pathlib
import pickle

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import cv2

CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]

CLASS_COLOR = {
    "car": (0, 200, 0), "truck": (0, 128, 255), "bus": (255, 128, 0),
    "pedestrian": (0, 0, 255), "motorcycle": (255, 0, 255),
    "bicycle": (255, 255, 0), "construction_vehicle": (128, 128, 255),
    "trailer": (0, 255, 255), "barrier": (200, 200, 200), "traffic_cone": (255, 0, 128),
}


def box_corners(b):
    """8 corners of a lidar-frame box (x,y,z,w,l,h,yaw), origin at box centre."""
    x, y, z, w, l, h, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    # local corners: l along x (heading), w along y, h along z
    xs = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    ys = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    zs = np.array([h, h, h, h, -h, -h, -h, -h]) / 2
    pts = np.stack([xs, ys, zs], axis=1)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    world = pts @ R.T + np.array([x, y, z])
    return world  # (8,3) in lidar frame


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def project_box(corners_lidar, R_s2l, T_s2l, K):
    """Lidar-frame corners -> image pixels for one camera. Returns pts or None.

    Drops the box entirely if ANY corner is at/behind the near plane — a partially
    behind-camera box projects to wild off-screen lines otherwise.
    """
    # lidar = R_s2l @ cam + T_s2l  =>  cam = R_s2l^T @ (lidar - T_s2l)
    cam = (corners_lidar - T_s2l) @ R_s2l  # == R_s2l^T applied on the right
    # camera optical frame: x=right, y=down, z=forward
    if np.any(cam[:, 2] <= 0.5):
        return None
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv


def draw_frame(info, out_png):
    K = np.array(info["cams"][CAM_ORDER[1]]["cam_intrinsic"])
    tiles = []
    for cam in CAM_ORDER:
        c = info["cams"][cam]
        img = cv2.imread(c["data_path"])
        if img is None:
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
        R = np.array(c["sensor2lidar_rotation"])
        T = np.array(c["sensor2lidar_translation"])
        Kc = np.array(c["cam_intrinsic"])
        for b, name in zip(info["gt_boxes"], info["gt_names"]):
            corners = box_corners(b)
            uv = project_box(corners, R, T, Kc)
            if uv is None:
                continue
            col = CLASS_COLOR.get(str(name), (255, 255, 255))
            for (i, j) in EDGES:
                p1 = tuple(np.round(uv[i]).astype(int))
                p2 = tuple(np.round(uv[j]).astype(int))
                if all(-2000 < v < 4000 for v in (*p1, *p2)):
                    cv2.line(img, p1, p2, col, 2)
        cv2.putText(img, cam, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (533, 300)))
    top = np.hstack(tiles[:3])
    bot = np.hstack(tiles[3:])
    cv2.imwrite(str(out_png), np.vstack([top, bot]))
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--frames", type=int, nargs="*", default=[0])
    ap.add_argument("--out", default="data/gt_overlays")
    args = ap.parse_args()

    d = pickle.load(open(args.infos, "rb"))
    infos = d["infos"]
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for fi in args.frames:
        if fi < len(infos):
            draw_frame(infos[fi], outdir / f"gt_overlay_f{fi:04d}.png")


if __name__ == "__main__":
    main()
