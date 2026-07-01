"""Build UniAD's `uniad_data` input from 6 LIVE CARLA images — no NuScenes DB.

This is the keystone of the closed-loop harness. Offline, the mmdet3d pipeline reads a
registered sample from the nuScenes DB (calibration, can_bus, temporal links) and turns it
into the dict the UniAD vision tower consumes. Live there is no DB, so we synthesize that
exact dict every tick from: 6 JPEGs + ego global pose + a per-episode scene_token.

Approach (deliberately robust): a real `uniad_data` captured once by capture_reference.py
is used as a STRUCTURAL TEMPLATE — it carries the constant calibration (`lidar2img`, the
rigid 6-cam rig), the dummy GT tensors the full UniAD forward_test still indexes, and the
exact (DataContainer) wrapping. Each tick we deep-copy it and overwrite ONLY the per-frame
fields, recomputed from live state to byte-match get_data_info:
  - img:        decode 6 JPEGs (BGR, like mmcv.imread) -> normalize(-mean) -> pad 900->928 -> (1,6,3,928,1600)
  - can_bus:    [0:3]=translation, [3:7]=quat(wxyz), [-2]=yaw_rad, [-1]=yaw_deg   (nuscenes_e2e get_data_info)
  - l2g_r_mat:  ego2global_rot.T ;  l2g_t: ego2global_translation   (lidar2ego is identity for CARLA)
  - scene_token: per-episode constant -> UniAD auto-resets its BEV memory at episode boundaries.

Temporal state lives INSIDE the UniAD model (prev_frame_info). Call reset_temporal(model)
at the first tick of each episode for a clean start (zeroes the can_bus delta + prev_bev).

Reusable for the model repo: this is a general "raw images + pose -> UniAD input" path.
"""
from __future__ import annotations

import copy

import numpy as np

# nuScenes order the infos store cameras in (== order images stack in the img tensor).
CAM_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
IMG_MEAN_BGR = np.array([103.530, 116.280, 123.675], dtype=np.float32)  # img_norm_cfg, std=1, to_rgb=False
PAD_H, PAD_W = 928, 1600  # PadMultiViewImage size_divisor=32: 900->928


def _find_meta(img_metas):
    """Unwrap DataContainer / list nesting to the single img_metas dict (has 'can_bus')."""
    x = img_metas
    while type(x).__name__ == "DataContainer":
        x = x.data
    while isinstance(x, (list, tuple)):
        x = x[0]
    assert isinstance(x, dict) and "can_bus" in x, f"unexpected img_metas: {type(x)}"
    return x


def reset_temporal(model) -> None:
    """Clear UniAD's stateful BEV memory so a new episode starts clean (no carryover
    prev_bev, and the first can_bus delta is zeroed). Call at each episode's first tick."""
    det = model.get_vision_tower().vision_tower.vision_model
    det.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}


class LiveUniadInput:
    def __init__(self, template_path):
        import torch
        self._torch = torch
        self.template = torch.load(template_path, map_location="cpu")
        _find_meta(self.template["img_metas"])  # validate template structure early

    def build(self, jpegs, ego_translation, ego_rotation_wxyz, scene_token,
              timestamp, speed=0.0, frame_idx=0, prev_idx="", next_idx=""):
        """jpegs: list of 6 JPEG byte strings in CAM_ORDER. ego_* in the nuScenes/OpenDRIVE
        global frame; speed = |ego velocity| m/s. Returns a uniad_data dict for vision_tower(...)."""
        import cv2
        from pyquaternion import Quaternion
        from nuscenes.eval.common.utils import quaternion_yaw
        torch = self._torch

        assert len(jpegs) == 6, f"need 6 images in CAM_ORDER, got {len(jpegs)}"
        data = copy.deepcopy(self.template)

        # img: decode BGR (matches LoadMultiViewImageFromFilesInCeph), normalize, pad, CHW, stack.
        chw = []
        for b in jpegs:
            arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
            arr -= IMG_MEAN_BGR
            h, w = arr.shape[:2]
            padded = np.zeros((PAD_H, PAD_W, 3), dtype=np.float32)
            padded[:h, :w] = arr
            chw.append(padded.transpose(2, 0, 1))
        img = np.stack(chw, axis=0)[None]  # (1, 6, 3, 928, 1600)
        data["img"] = [torch.from_numpy(img).float()]

        # l2g: lidar==ego for CARLA (lidar2ego identity), so l2g_r = e2g_r.T, l2g_t = e2g_t.
        rot = Quaternion(ego_rotation_wxyz)
        e2g_r_mat = rot.rotation_matrix
        data["l2g_r_mat"] = torch.from_numpy(e2g_r_mat.T.astype(np.float32))[None]  # (1,3,3)
        data["l2g_t"] = torch.from_numpy(np.asarray(ego_translation, np.float32))[None]  # (1,3)

        # can_bus = build_infos_pkl layout ([13]=speed) + get_data_info overwrites.
        can_bus = np.zeros(18, dtype=np.float64)
        can_bus[13] = speed
        can_bus[:3] = ego_translation
        # MIRROR get_data_info EXACTLY: assigning a pyquaternion to a 4-slot numpy slice
        # makes numpy treat it as a scalar and BROADCAST it (so [3:7] all = w, not w,x,y,z).
        # That quirk is baked into the training data; reproduce it byte-for-byte.
        can_bus[3:7] = rot
        patch_angle = quaternion_yaw(rot) / np.pi * 180.0
        if patch_angle < 0:
            patch_angle += 360.0
        can_bus[-2] = patch_angle / 180.0 * np.pi
        can_bus[-1] = patch_angle

        meta = _find_meta(data["img_metas"])
        meta["can_bus"] = can_bus
        meta["scene_token"] = scene_token
        meta["sample_idx"] = f"{scene_token}_f{frame_idx:04d}"
        meta["prev_idx"] = prev_idx
        meta["next_idx"] = next_idx

        data["timestamp"] = [torch.tensor([float(timestamp)], dtype=torch.float64)]
        return data
