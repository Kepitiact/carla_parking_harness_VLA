#!/usr/bin/env python3
"""
Build a nuScenes-compatible BEV map PNG from a CARLA OpenDRIVE (.xodr) file.

Coordinate conventions
----------------------
OpenDRIVE world : right-hand, x=east, y=north
CARLA world     : left-hand,  x=east, y=south  (cy = -xodr_y)
nuScenes world  : right-hand, x=east, y=north  (ny = xodr_y = -cy)
=> OpenDRIVE coords == nuScenes coords: no horizontal transform needed.

PNG pixel layout
----------------
  origin (px=0, py=0) = top-left corner = (x_min, y_max) in world
  px = (x - x_min) / RESOLUTION
  py = (y_max - y) / RESOLUTION        # y-flipped: pixels go south

Output PNG (RGB, 8-bit per channel):
  R = 255 → drivable area  (driving + shoulder lanes, filled)
  G = 255 → lane dividers  (road-mark lines between lanes)
  B = 255 → road edges     (outermost lane boundary / curb lines)

Usage
-----
  python scripts/build_carla_map.py \\
      --xodr ParkingScenes/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04_Opt.xodr \\
      --out-dir data/nuscenes/maps \\
      --map-json data/nuscenes/v1.0-mini/map.json
"""

import argparse
import json
import math
import pathlib
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

# ── Constants ─────────────────────────────────────────────────────────────────

RESOLUTION  = 0.1   # metres per pixel
SAMPLE_STEP = 0.5   # metres between reference-line samples
MARK_PX     = 2     # lane-mark line width in pixels
PAD_M       = 20.0  # world padding around road extent

DRIVABLE_TYPES = {"driving", "shoulder", "parking"}
SKIP_MARKS     = {"none", "no mark", ""}


# ── OpenDRIVE geometry sampling ───────────────────────────────────────────────

def _sample_line(x0, y0, hdg, length):
    n  = max(2, int(length / SAMPLE_STEP) + 2)
    ts = np.linspace(0.0, length, n)
    return ts, x0 + ts * math.cos(hdg), y0 + ts * math.sin(hdg), np.full(n, hdg)


def _sample_arc(x0, y0, hdg, length, curv):
    if abs(curv) < 1e-9:
        return _sample_line(x0, y0, hdg, length)
    n    = max(2, int(length / SAMPLE_STEP) + 2)
    ts   = np.linspace(0.0, length, n)
    dth  = curv * ts
    r    = 1.0 / curv
    xs   = x0 + r * (np.sin(hdg + dth) - math.sin(hdg))
    ys   = y0 - r * (np.cos(hdg + dth) - math.cos(hdg))
    return ts, xs, ys, hdg + dth


def _sample_geom(gelem):
    x0  = float(gelem.get("x"))
    y0  = float(gelem.get("y"))
    hdg = float(gelem.get("hdg"))
    ln  = float(gelem.get("length"))
    tag = gelem[0].tag if len(gelem) else "line"
    if tag == "arc":
        return _sample_arc(x0, y0, hdg, ln, float(gelem[0].get("curvature", 0)))
    return _sample_line(x0, y0, hdg, ln)


# ── Cubic-polynomial helpers ──────────────────────────────────────────────────

def _parse_polys(parent, tag, s_attr="sOffset"):
    """Return sorted list of (s, a, b, c, d) tuples."""
    out = []
    for el in parent.findall(tag):
        s = float(el.get("s", el.get(s_attr, 0)))
        out.append((s, float(el.get("a", 0)), float(el.get("b", 0)),
                       float(el.get("c", 0)), float(el.get("d", 0))))
    return sorted(out)


def _eval_poly(entries, ds_global):
    """Evaluate the polynomial whose s ≤ ds_global."""
    best = entries[0] if entries else None
    for e in entries:
        if ds_global >= e[0]:
            best = e
    if best is None:
        return 0.0
    ds = ds_global - best[0]
    return best[1] + best[2]*ds + best[3]*ds**2 + best[4]*ds**3


# ── Lane-section geometry ─────────────────────────────────────────────────────

def _parse_lane_dict(side_elem):
    """Return {lane_id: (type, width_polys, mark_types)}."""
    d = {}
    if side_elem is None:
        return d
    for lane in side_elem.findall("lane"):
        lid   = int(lane.get("id", 0))
        ltype = lane.get("type", "none")
        wps   = _parse_polys(lane, "width", "sOffset")
        marks = [m.get("type", "none") for m in lane.findall("roadMark")]
        d[lid] = (ltype, wps, marks)
    return d


def _build_lane_polys(ref_x, ref_y, nx, ny, ss_local,
                      lanes, sorted_ids, sign):
    """
    Walk from the reference line outward, one lane at a time.

    Returns list of (lane_type, inner_x, inner_y, outer_x, outer_y, mark_type).
    sign = +1 for left lanes, -1 for right lanes.
    """
    results = []
    cum = np.zeros(len(ref_x))
    prev_x, prev_y = ref_x.copy(), ref_y.copy()

    for lid in sorted_ids:
        if lid not in lanes:
            continue
        ltype, wps, marks = lanes[lid]
        w   = np.array([_eval_poly(wps, ds) for ds in ss_local])
        cum = cum + w
        ox  = ref_x + sign * cum * nx
        oy  = ref_y + sign * cum * ny
        mark = marks[0] if marks else "none"
        results.append((ltype, prev_x.copy(), prev_y.copy(), ox, oy, mark))
        prev_x, prev_y = ox, oy

    return results


def _make_poly(ix, iy, ox, oy):
    pts = list(zip(ix, iy)) + list(zip(ox[::-1], oy[::-1]))
    if len(pts) < 3:
        return None
    try:
        p = Polygon(pts)
        return p.buffer(0) if not p.is_valid else p
    except Exception:
        return None


# ── Per-road processing ───────────────────────────────────────────────────────

def process_road(road_elem):
    """
    Parse one <road> and return:
      drivable_polys : list of shapely Polygon
      divider_lines  : list of shapely LineString  (lane marks between lanes)
      edge_lines     : list of shapely LineString  (outermost road boundary)
    """
    drivable_polys, divider_lines, edge_lines = [], [], []

    pv = road_elem.find("planView")
    if pv is None:
        return drivable_polys, divider_lines, edge_lines

    road_length = float(road_elem.get("length", 0))

    # ── Sample reference line (concatenate all geometry segments) ──────────
    all_s, all_x, all_y, all_h = [], [], [], []
    s_cum = 0.0
    for gelem in pv.findall("geometry"):
        ts, xs, ys, hs = _sample_geom(gelem)
        all_s.append(ts + s_cum)
        all_x.append(xs); all_y.append(ys); all_h.append(hs)
        s_cum += float(gelem.get("length", 0))

    if not all_s:
        return drivable_polys, divider_lines, edge_lines

    s_arr = np.concatenate(all_s)
    rx    = np.concatenate(all_x)
    ry    = np.concatenate(all_y)
    rh    = np.concatenate(all_h)

    # Apply laneOffset (lateral shift of reference line)
    lo_entries = _parse_polys(road_elem, ".//laneOffset", "s")
    if lo_entries:
        lo = np.array([_eval_poly(lo_entries, s) for s in s_arr])
        rx -= lo * np.sin(rh)
        ry += lo * np.cos(rh)

    # Left-pointing normal
    nx_arr = -np.sin(rh)
    ny_arr =  np.cos(rh)

    # ── Process each lane section ──────────────────────────────────────────
    sections = road_elem.findall(".//laneSection")
    for idx, ls in enumerate(sections):
        s_start = float(ls.get("s", 0))
        next_s  = [float(n.get("s", 0)) for n in sections
                   if float(n.get("s", 0)) > s_start]
        s_end   = min(next_s) if next_s else road_length

        mask = (s_arr >= s_start - 1e-3) & (s_arr <= s_end + 1e-3)
        if mask.sum() < 2:
            continue

        srx, sry = rx[mask], ry[mask]
        snx, sny = nx_arr[mask], ny_arr[mask]
        ss_loc   = s_arr[mask] - s_start

        left_d  = _parse_lane_dict(ls.find("left"))
        right_d = _parse_lane_dict(ls.find("right"))

        left_ids  = sorted(k for k in left_d  if k > 0)
        right_ids = sorted((k for k in right_d if k < 0), key=abs)

        for side_lanes, sorted_ids, sign in (
            (left_d,  left_ids,  +1),
            (right_d, right_ids, -1),
        ):
            lane_info = _build_lane_polys(
                srx, sry, snx, sny, ss_loc, side_lanes, sorted_ids, sign)

            for i, (ltype, ix, iy, ox, oy, mark) in enumerate(lane_info):
                is_last = (i == len(lane_info) - 1)

                if ltype in DRIVABLE_TYPES:
                    p = _make_poly(ix, iy, ox, oy)
                    if p is not None and p.area > 0.1:
                        drivable_polys.append(p)

                if mark not in SKIP_MARKS:
                    ls_geom = LineString(zip(ox, oy))
                    if is_last:
                        edge_lines.append(ls_geom)
                    else:
                        divider_lines.append(ls_geom)
                elif is_last:
                    # Still record outermost edge even without a painted mark
                    edge_lines.append(LineString(zip(ox, oy)))

    return drivable_polys, divider_lines, edge_lines


# ── World → pixel helpers ─────────────────────────────────────────────────────

def world_to_px(xs, ys, x_min, y_max, res):
    px = ((xs - x_min) / res).astype(int)
    py = ((y_max - ys) / res).astype(int)
    return px, py


def poly_to_px_coords(poly, x_min, y_max, res):
    xs, ys = poly.exterior.coords.xy
    return list(zip(
        ((np.array(xs) - x_min) / res).tolist(),
        ((y_max - np.array(ys)) / res).tolist(),
    ))


def line_to_px_coords(line, x_min, y_max, res):
    xs, ys = line.coords.xy
    return list(zip(
        ((np.array(xs) - x_min) / res).tolist(),
        ((y_max - np.array(ys)) / res).tolist(),
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xodr",     required=True,
                    help="Path to the .xodr OpenDRIVE file")
    ap.add_argument("--out-dir",  required=True,
                    help="Output directory for the map PNG")
    ap.add_argument("--map-json", default=None,
                    help="Path to map.json to update (optional)")
    ap.add_argument("--log-token", default=None,
                    help="Log token(s) to associate in map.json (comma-separated)")
    args = ap.parse_args()

    xodr_path = pathlib.Path(args.xodr)
    out_dir   = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_name = xodr_path.stem   # e.g. "Town04_Opt"
    png_path = out_dir / f"{map_name}.png"
    meta_path = out_dir / f"{map_name}_meta.json"

    print(f"[MAP] Parsing {xodr_path.name} ...")
    tree  = ET.parse(xodr_path)
    root  = tree.getroot()
    roads = root.findall("road")
    print(f"[MAP] {len(roads)} roads")

    # ── Collect all geometry ───────────────────────────────────────────────
    all_drivable, all_dividers, all_edges = [], [], []
    for i, road in enumerate(roads):
        dp, dl, el = process_road(road)
        all_drivable.extend(dp)
        all_dividers.extend(dl)
        all_edges.extend(el)
        if (i + 1) % 50 == 0:
            print(f"[MAP]   processed {i+1}/{len(roads)} roads "
                  f"({len(all_drivable)} polys so far)")

    print(f"[MAP] Total: {len(all_drivable)} drivable polys, "
          f"{len(all_dividers)} dividers, {len(all_edges)} edges")

    if not all_drivable:
        raise RuntimeError("No drivable polygons extracted — check the .xodr file")

    # ── Compute world extent from road geometry ────────────────────────────
    # Use xodr header as a fallback, but prefer actual geometry bounds
    hdr = root.find("header")
    hdr_xmin = float(hdr.get("west",  -1000))
    hdr_xmax = float(hdr.get("east",   1000))
    hdr_ymin = float(hdr.get("south", -1000))
    hdr_ymax = float(hdr.get("north",  1000))

    # Tight bounds from the drivable geometry
    union = unary_union(all_drivable)
    bx, by, bx2, by2 = union.bounds   # (minx, miny, maxx, maxy)

    x_min = max(hdr_xmin, bx - PAD_M)
    x_max = min(hdr_xmax, bx2 + PAD_M)
    y_min = max(hdr_ymin, by - PAD_M)
    y_max = min(hdr_ymax, by2 + PAD_M)

    W = int(math.ceil((x_max - x_min) / RESOLUTION))
    H = int(math.ceil((y_max - y_min) / RESOLUTION))
    print(f"[MAP] World extent: x=[{x_min:.1f}, {x_max:.1f}]  "
          f"y=[{y_min:.1f}, {y_max:.1f}]")
    print(f"[MAP] Canvas: {W} × {H} px  "
          f"({W*RESOLUTION:.0f} m × {H*RESOLUTION:.0f} m)")

    # ── Rasterize ─────────────────────────────────────────────────────────
    img_r = Image.new("L", (W, H), 0)   # drivable area
    img_g = Image.new("L", (W, H), 0)   # lane dividers
    img_b = Image.new("L", (W, H), 0)   # road edges

    dr_r = ImageDraw.Draw(img_r)
    dr_g = ImageDraw.Draw(img_g)
    dr_b = ImageDraw.Draw(img_b)

    print("[MAP] Rasterizing drivable polygons ...")
    for poly in all_drivable:
        coords = poly_to_px_coords(poly, x_min, y_max, RESOLUTION)
        if len(coords) >= 3:
            dr_r.polygon(coords, fill=255)

    print("[MAP] Rasterizing lane dividers ...")
    for line in all_dividers:
        coords = line_to_px_coords(line, x_min, y_max, RESOLUTION)
        if len(coords) >= 2:
            dr_g.line(coords, fill=255, width=MARK_PX)

    print("[MAP] Rasterizing road edges ...")
    for line in all_edges:
        coords = line_to_px_coords(line, x_min, y_max, RESOLUTION)
        if len(coords) >= 2:
            dr_b.line(coords, fill=255, width=MARK_PX)

    # Merge into RGB
    print(f"[MAP] Saving PNG → {png_path}")
    rgb = Image.merge("RGB", (img_r, img_g, img_b))
    rgb.save(png_path)

    # ── Save coordinate metadata ───────────────────────────────────────────
    meta = {
        "map_name":   map_name,
        "filename":   f"maps/{map_name}.png",
        "resolution": RESOLUTION,
        "width_px":   W,
        "height_px":  H,
        "origin_world": {
            "x": x_min,
            "y": y_max,
            "note": "pixel (0,0) top-left = this world (x,y); y_pixel = (y_max - y_world) / res"
        },
        "world_bounds": {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
        },
        "coord_note": (
            "Coords are in OpenDRIVE / nuScenes frame (x=east, y=north). "
            "CARLA world: cx=x, cy=-y."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[MAP] Metadata → {meta_path}")

    # ── Update map.json ────────────────────────────────────────────────────
    if args.map_json:
        map_json_path = pathlib.Path(args.map_json)
        existing = []
        if map_json_path.exists():
            existing = json.loads(map_json_path.read_text())

        # Remove stale entry for this map if present
        existing = [e for e in existing if e.get("filename") != f"maps/{map_name}.png"]

        new_entry = {
            "category": "semantic_prior",
            "filename":  f"maps/{map_name}.png",
            "token":     str(uuid.uuid4()).replace("-", ""),
        }
        if args.log_token:
            new_entry["log_tokens"] = [t.strip() for t in args.log_token.split(",")]

        existing.append(new_entry)
        map_json_path.parent.mkdir(parents=True, exist_ok=True)
        map_json_path.write_text(json.dumps(existing, indent=2))
        print(f"[MAP] Updated {map_json_path}  (token={new_entry['token'][:8]}...)")

    print("[MAP] Done.")


if __name__ == "__main__":
    main()
