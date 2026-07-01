"""Per-episode closed-loop metrics: success, collision, final-pose error, gear changes,
ticks/timeout. Scores against the TARGET SLOT POSE (not the GT path). nuScenes frame.
"""
from __future__ import annotations

import math

from harness.model_server import live_prompt  # global_to_local_xy / normalize_angle


class EpisodeMetrics:
    def __init__(self, slot_global: dict, goal_dist_m: float, goal_rot_deg: float):
        self.slot = slot_global
        self.goal_dist_m = goal_dist_m
        self.goal_rot_deg = goal_rot_deg
        self.gear_changes = 0
        self.collision = False
        self.collision_actor = None
        self.ticks = 0
        self._last_gear = None

    def note_gear(self, reverse: bool):
        if self._last_gear is not None and reverse != self._last_gear:
            self.gear_changes += 1
        self._last_gear = reverse

    def pose_error(self, ex_n, ey_n, eyaw_n):
        """Return (lateral, longitudinal, heading_deg) of the ego vs the slot pose."""
        r, f = live_prompt.global_to_local_xy(self.slot["x"], self.slot["y"], ex_n, ey_n, eyaw_n)
        head = abs(live_prompt.normalize_angle(self.slot["yaw"] - eyaw_n))
        return abs(r), abs(f), math.degrees(head)

    def is_parked(self, ex_n, ey_n, eyaw_n, speed):
        lat, lon, head = self.pose_error(ex_n, ey_n, eyaw_n)
        return (math.hypot(lat, lon) < self.goal_dist_m
                and head < self.goal_rot_deg and abs(speed) < 0.2)

    def summary(self, ex_n, ey_n, eyaw_n, timeout: bool):
        lat, lon, head = self.pose_error(ex_n, ey_n, eyaw_n)
        success = (not self.collision and not timeout
                   and math.hypot(lat, lon) < self.goal_dist_m and head < self.goal_rot_deg)
        return {
            "success": bool(success),
            "collision": bool(self.collision),
            "collision_actor": self.collision_actor,
            "timeout": bool(timeout),
            "ticks": self.ticks,
            "gear_changes": self.gear_changes,
            "lateral_err_m": round(lat, 3),
            "longitudinal_err_m": round(lon, 3),
            "heading_err_deg": round(head, 2),
        }
