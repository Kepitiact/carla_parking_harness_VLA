"""Live watch window for the closed-loop harness (runs in the CARLA venv, py3.8).

Composites one debug frame per tick with cv2:
  - top:    the 6 live cameras (2x3 grid)
  - bottom-left:  BEV (ego frame) — predicted waypoints (cyan), executed trail (blue),
                  target slot (green), gear, pose error to slot
  - bottom-right: the VERBATIM prompt the model saw (Mission goal highlighted)
  - status bar:   the current PIPELINE STAGE (your addition) — e.g. "UniAD+LLM running...",
                  "converting waypoints -> steer/throttle" — so you can see what's happening live

Shows a live window (cv2.imshow) when a display is available, and/or writes an MP4 replay.
Standalone self-test (canned data, headless PNG):  python harness/client/live_viewer.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from harness.model_server import live_prompt  # global_to_local_xy (numpy only)

W, H = 1280, 920
CAM_W, CAM_H = 426, 240         # per-camera cell in the 2x3 grid
GRID_H = CAM_H * 2              # 480
MID_Y0, MID_Y1 = GRID_H, 880   # middle band (BEV + prompt)
BEV_W = 480
BAR_Y0 = MID_Y1                # status bar
CAM_GRID = [["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]]
FONT = cv2.FONT_HERSHEY_SIMPLEX


class LiveViewer:
    def __init__(self, save_path=None, show=None, fps=10):
        self.save_path = save_path
        self.writer = None
        self.fps = fps
        self.stage = "init"
        # auto-detect a display; allow override
        self.show = (show if show is not None else bool(os.environ.get("DISPLAY")))

    def set_stage(self, s):
        self.stage = s

    def close(self):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ── drawing helpers ─────────────────────────────────────────────────────
    def _cam_grid(self, canvas, cams):
        for r, row in enumerate(CAM_GRID):
            for c, name in enumerate(row):
                x0, y0 = c * CAM_W, r * CAM_H
                img = cams.get(name) if cams else None
                if img is None:
                    cell = np.full((CAM_H, CAM_W, 3), 40, np.uint8)
                else:
                    cell = cv2.resize(img, (CAM_W, CAM_H))
                canvas[y0:y0 + CAM_H, x0:x0 + CAM_W] = cell
                cv2.rectangle(canvas, (x0, y0), (x0 + CAM_W - 1, y0 + CAM_H - 1), (70, 70, 70), 1)
                cv2.putText(canvas, name, (x0 + 5, y0 + 18), FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _bev(self, canvas, pred_wps, slot_local, trail_ego, pose_err, gear, slot_poly=None):
        x0, y0 = 0, MID_Y0
        cv2.rectangle(canvas, (x0, y0), (x0 + BEV_W, MID_Y1), (25, 25, 25), -1)
        cv2.putText(canvas, "BEV (ego frame, forward=up)", (x0 + 8, y0 + 18), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cx, cy, scale = x0 + BEV_W // 2, y0 + 260, 11.0   # px/m

        def P(right, forward):
            return int(cx + right * scale), int(cy - forward * scale)

        for m in range(-15, 16, 5):                       # light grid
            cv2.line(canvas, P(m, -10), P(m, 24), (45, 45, 45), 1)
            cv2.line(canvas, P(-18, m), P(18, m), (45, 45, 45), 1)
        # executed trail (blue)
        pts = [P(r, f) for (r, f) in trail_ego]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, (255, 140, 0), 2)
        # target slot footprint (green rectangle, real 4 corners) + heading arrow, so you can
        # check the car's pose AND heading against the actual bay (not just the centre point).
        if slot_poly:
            corners = [P(r, f) for (r, f) in slot_poly]
            cv2.polylines(canvas, [np.array(corners, np.int32)], True, (0, 230, 0), 2, cv2.LINE_AA)
            mcx = sum(r for r, _ in slot_poly) / len(slot_poly)
            mcf = sum(f for _, f in slot_poly) / len(slot_poly)
            # polygon order is front-left, front-right, rear-right, rear-left -> front edge mid
            fmx = (slot_poly[0][0] + slot_poly[1][0]) / 2.0
            fmf = (slot_poly[0][1] + slot_poly[1][1]) / 2.0
            cv2.arrowedLine(canvas, P(mcx, mcf), P(fmx, fmf), (0, 230, 0), 2, cv2.LINE_AA, tipLength=0.3)
        # target slot centre (small green marker)
        if slot_local is not None:
            sr, sf, _ = slot_local
            cv2.drawMarker(canvas, P(sr, sf), (0, 230, 0), cv2.MARKER_STAR, 8, 1)
            cv2.putText(canvas, "slot", (P(sr, sf)[0] + 6, P(sr, sf)[1]), FONT, 0.4, (0, 230, 0), 1, cv2.LINE_AA)
        # predicted waypoints (cyan); items may be (right, forward) or (right, forward, head_err)
        ppts = [P(0, 0)] + [P(wp[0], wp[1]) for wp in pred_wps]
        for a, b in zip(ppts, ppts[1:]):
            cv2.line(canvas, a, b, (255, 255, 0), 2)
        for p in ppts[1:]:
            cv2.circle(canvas, p, 3, (255, 255, 0), -1)
        # ego (red triangle pointing up) + footprint rectangle, so you can compare the CAR's
        # rectangle (always axis-aligned here: forward=up) against the green slot rectangle to
        # see if the heading actually matches when the car is inside the bay.
        hl, hw = 4.5 / 2.0, 1.8 / 2.0   # ego length/width (matches live_prompt LENGTH_M/WIDTH_M)
        ego_rect = [P(-hw, hl), P(hw, hl), P(hw, -hl), P(-hw, -hl)]
        cv2.polylines(canvas, [np.array(ego_rect, np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, P(0, 0), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
        # text
        if pose_err is not None:
            lat, lon, head = pose_err
            cv2.putText(canvas, f"to slot: lat {lat:.2f}m lon {lon:.2f}m head {head:.1f}deg",
                        (x0 + 8, MID_Y1 - 26), FONT, 0.42, (0, 230, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"gear: {'REVERSE' if gear else 'forward'}",
                    (x0 + 8, MID_Y1 - 8), FONT, 0.42, (0, 200, 255) if gear else (200, 200, 200), 1, cv2.LINE_AA)

    def _prompt(self, canvas, prompt, tick, infer_ms):
        x0, y0 = BEV_W, MID_Y0
        cv2.rectangle(canvas, (x0, y0), (W, MID_Y1), (15, 15, 15), -1)
        cv2.putText(canvas, f"PROMPT  (tick {tick}, infer {infer_ms:.0f}ms)",
                    (x0 + 8, y0 + 18), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y = y0 + 42
        for line in (prompt or "").split("\n"):
            for chunk in _wrap(line, 64):
                hi = chunk.strip().startswith("Mission goal")
                cv2.putText(canvas, chunk, (x0 + 10, y), FONT, 0.42,
                            (0, 255, 120) if hi else (180, 180, 180), 1, cv2.LINE_AA)
                y += 18
                if y > MID_Y1 - 8:
                    return

    def _bar(self, canvas):
        cv2.rectangle(canvas, (0, BAR_Y0), (W, H), (0, 40, 70), -1)
        cv2.putText(canvas, f">> STAGE: {self.stage}", (12, BAR_Y0 + 28), FONT, 0.7,
                    (0, 220, 255), 2, cv2.LINE_AA)

    # ── main ────────────────────────────────────────────────────────────────
    def render(self, *, cams=None, prompt="", pred_wps=None, slot_local=None,
               trail_world=None, ego_n=None, pose_err=None, tick=0, infer_ms=0.0, gear=False,
               n_tracks=-1, slot_poly=None):
        # executed trail -> ego frame
        trail_ego = []
        if trail_world and ego_n is not None:
            ex, ey, eyaw = ego_n
            trail_ego = [live_prompt.global_to_local_xy(px, py, ex, ey, eyaw) for (px, py) in trail_world[-40:]]

        canvas = np.full((H, W, 3), 10, np.uint8)
        self._cam_grid(canvas, cams)
        self._bev(canvas, pred_wps or [], slot_local, trail_ego, pose_err, gear, slot_poly=slot_poly)
        self._prompt(canvas, prompt, tick, infer_ms)
        self._bar(canvas)
        if n_tracks >= 0:  # UniAD perception count — orange if it's basically blind
            col = (0, 220, 0) if n_tracks >= 3 else (0, 120, 255)
            cv2.putText(canvas, f"UniAD sees: {n_tracks} objects", (8, MID_Y1 - 44),
                        FONT, 0.55, col, 2, cv2.LINE_AA)

        if self.save_path:
            if self.writer is None:
                # OpenCV here has no H.264/avc1 encoder; mp4v writes a file but many players can't
                # open it. MJPG in an .avi container is universally playable, so force that.
                out = str(self.save_path)
                if out.lower().endswith(".mp4"):
                    out = out[:-4] + ".avi"
                self.writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (W, H))
                if not self.writer.isOpened():
                    print(f"[viewer] WARNING: could not open video writer for {out}")
            if self.writer is not None:
                self.writer.write(canvas)
        if self.show:
            try:
                cv2.imshow("CARLA closed-loop harness", canvas)
                cv2.waitKey(1)
            except cv2.error:
                self.show = False
        return canvas


def _wrap(text, n):
    if not text:
        return [""]
    return [text[i:i + n] for i in range(0, len(text), n)]


# ── standalone self-test (canned data, headless PNG) ────────────────────────
def _selftest():
    import math
    cams = {}
    for i, name in enumerate(sum(CAM_GRID, [])):
        img = np.full((900, 1600, 3), 30 + i * 12, np.uint8)
        cv2.putText(img, name, (40, 120), FONT, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
        cams[name] = img
    pred = [(0.0, -0.6), (0.1, -1.2), (0.0, -1.9), (-0.3, -2.5), (-0.4, -3.0), (-0.3, -3.4)]
    trail = [(285.6 + 0.05 * t, 213.5 - 0.03 * t) for t in range(30)]
    prompt = ("Scene information: <scene_start><SCENE><scene_end>\n"
              "Ego states: - Velocity (vx,vy): (-0.43,-0.14) - Heading Angular Velocity (v_yaw): (-2.81) ...\n"
              "Historical trajectory (last 2 seconds): [(2.32,2.96),(1.55,2.42),(0.88,1.71),(0.36,0.90)]\n"
              "Mission goal: reverse-perpendicular park, right side, into slot at (5.30,-2.82,1.57)\n"
              "Planning trajectory: <trajectory>")
    v = LiveViewer(save_path=None, show=False)
    v.set_stage("UniAD + LLM running... (waiting ~1.7s)")
    canvas = v.render(cams=cams, prompt=prompt, pred_wps=pred, slot_local=(5.3, -2.8, 1.57),
                      trail_world=trail, ego_n=(286.0, 213.0, math.pi / 2),
                      pose_err=(0.42, 3.0, 35.6), tick=120, infer_ms=1710, gear=True)
    out = os.path.join(os.path.dirname(__file__), "..", "runs", "live_viewer_selftest.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, canvas)
    print(f"[viewer] self-test composite saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    _selftest()
