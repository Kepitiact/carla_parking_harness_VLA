"""Loads the OpenDriveVLA maneuver checkpoint and turns (prompt + perception) into
6 ego-local waypoints. The reusable core of server.py.

Runs ONLY in the model venv (~/projects/openvla_nuscenes/.venv, Py3.10): it imports
torch + the llava/drivevla stack. Mirrors the offline path in
inference_drivevla.inference_data and nuscenes_llava_dataset._get_llava_test_data,
minus DeepSpeed/DDP (plain model.generate works for single-sample inference).

Standalone smoke (de-risks the whole stack on a SAVED frame, using a precomputed
uniad_pth so live-UniAD is not yet required):

    ~/projects/openvla_nuscenes/.venv/bin/python harness/model_server/model_runner.py \
        --conversations ~/projects/openvla_nuscenes/data_carla/processed/carla_conversations.json \
        --cached        ~/projects/openvla_nuscenes/data_carla/processed/cached_parking_info.pkl
"""
from __future__ import annotations

import os
import pathlib
import re
import sys


def bootstrap_model_repo(model_repo) -> None:
    """Put the llava/projects/mmdet3d packages on sys.path and satisfy DeepSpeed's
    nvcc probe — same recipe as drivevla/extract_uniad_features.py, but the repo
    root comes from config instead of __file__ (this file lives outside the repo)."""
    model_repo = pathlib.Path(model_repo).resolve()
    # drivevla/ on path so `import data_utils.*` resolves, matching the convention
    # in drivevla/extract_uniad_features.py (run as `python drivevla/<script>.py`).
    for p in (model_repo / "third_party" / "mmdetection3d_1_0_0rc6", model_repo, model_repo / "drivevla"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    shim = model_repo / ".cache" / "fake_cuda"
    if not (shim / "bin" / "nvcc").exists():
        (shim / "bin").mkdir(parents=True, exist_ok=True)
        (shim / "bin" / "nvcc").write_text(
            '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.1, V12.1.0"\n')
        os.chmod(shim / "bin" / "nvcc", 0o755)
    os.environ.setdefault("CUDA_HOME", str(shim))
    os.environ["PATH"] = f"{shim / 'bin'}:{os.environ.get('PATH', '')}"
    # The UniAD vision tower loads its config via cwd-relative paths
    # ('projects/configs/...'), so the process must run from the repo root.
    os.chdir(model_repo)


def _slim_result(res):
    """Keep only the fields the LLM consumes (mirrors extract_uniad_features.py)."""
    rt = res.get("result_track", {}) or {}
    rs = res.get("result_seg", {}) or {}
    return {
        "scene_token": res.get("scene_token"),
        "sample_token": res.get("sample_token"),
        "result_track": {
            "track_query_embeddings": rt.get("track_query_embeddings"),
            "img_feat_2D": rt.get("img_feat_2D"),
            "track_gt_inds_to_embed_idx": rt.get("track_gt_inds_to_embed_idx"),
        },
        "result_seg": {
            "chosen_output_query_things": rs.get("chosen_output_query_things"),
            "output_query_stuff": rs.get("output_query_stuff"),
        },
        "planning_gt": res.get("planning_gt"),
    }


_TUPLE_RE = re.compile(
    r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*(?:,\s*(-?\d+\.?\d*)\s*)?\)")


def parse_waypoints(answer: str):
    """Model answer -> [[right, forward, heading], ...].

    The retrained model emits 3-tuples '(r,f,h)': (right, forward) position plus a per-waypoint
    heading h (radians, RELATIVE to the current ego frame; gt_ego_fut_trajs index 2 — the
    future yaw the car should reach, an OUTPUT label, never a future-derived INPUT, so no leak).
    Legacy 2-tuple answers '(r,f)' parse with heading 0.0. Position stays index [0:2] so existing
    pure-pursuit (which slices r,f) is unchanged; heading is carried for the controller."""
    return [[float(a), float(b), float(c) if c else 0.0]
            for a, b, c in _TUPLE_RE.findall(answer)]


class ModelRunner:
    def __init__(self, checkpoint, model_repo, device: str = "cuda"):
        bootstrap_model_repo(model_repo)
        import torch
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_uniad_token

        disable_torch_init()
        overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            str(checkpoint), model_base=None, model_name="llava_qwen",
            device_map=device, multimodal=True, attn_implementation="eager",
            overwrite_config=overwrite_config,
        )
        self.model.eval()
        self.device = device
        self._torch = torch
        self._conv_templates = conv_templates
        self._tok_uniad = tokenizer_uniad_token

    def build_input_ids(self, question_value: str):
        """Assemble the chat-template prompt + tokenize (mirrors _get_llava_test_data)."""
        conv = self._conv_templates["qwen_planning_oriented_vlm"].copy()
        conv.clear_conversation()
        conv.append_message(conv.roles[0], question_value)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        return self._tok_uniad(
            prompt_question, self.tokenizer, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

    def _to_device(self, x):
        """Move tensors to self.device; leave numpy/DataContainer/scalars untouched
        (img_metas carries a numpy can_bus that UniAD mutates in place)."""
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: self._to_device(v) for k, v in x.items()}
        if isinstance(x, list):
            return [self._to_device(v) for v in x]
        return x

    def perceive_and_generate(self, input_ids, uniad_data):
        """Run UniAD ONCE (so temporal state advances once), report how many objects it
        detected, then plan the LLM from those features (the offline extract->infer path).
        Returns (answer_text, n_tracks)."""
        torch = self._torch
        uniad_data = self._to_device(uniad_data)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                res = self.model.get_vision_tower()(uniad_data)
                slim = _slim_result(res)
                cont = self.model.generate(
                    input_ids, uniad_pth=slim, uniad_data=None,
                    do_sample=False, temperature=0, max_new_tokens=512, num_beams=1)
        answer = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        tqe = (res.get("result_track", {}) or {}).get("track_query_embeddings")
        n_tracks = int(tqe.shape[0]) if tqe is not None else 0
        return answer, n_tracks

    def generate(self, input_ids, uniad_pth=None, uniad_data=None) -> str:
        torch = self._torch
        if uniad_data is not None:
            uniad_data = self._to_device(uniad_data)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                cont = self.model.generate(
                    input_ids, uniad_data=uniad_data, uniad_pth=uniad_pth,
                    do_sample=False, temperature=0, max_new_tokens=512, num_beams=1,
                )
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]


# ── standalone offline smoke ────────────────────────────────────────────────

def _smoke(argv=None):
    import argparse
    import json
    import pickle

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from harness import config as cfg_mod

    ap = argparse.ArgumentParser()
    cfg_mod.Config.add_cli_args(ap)
    proc = pathlib.Path.home() / "projects/openvla_nuscenes/data_carla/processed"
    ap.add_argument("--conversations", default=str(proc / "carla_conversations.json"))
    ap.add_argument("--cached", default=str(proc / "cached_parking_info.pkl"))
    args = ap.parse_args(argv)
    cfg = cfg_mod.Config.from_args(args)

    print(f"[smoke] loading cached info: {args.cached}")
    cached = pickle.load(open(args.cached, "rb"))
    print(f"[smoke] loading conversations: {args.conversations}")
    convs = json.load(open(args.conversations))

    # Pick the first frame that has a precomputed uniad_pth AND a maneuver-level
    # cached entry (so we exercise the real maneuver+slot prompt the model trained on).
    entry = None
    for e in convs:
        tok = e.get("sample_id") or e.get("qa_id", "").replace("_trajectory", "")
        c = cached.get(tok)
        if c is not None and c.get("maneuver_type") and os.path.exists(e.get("uniad_pth", "")):
            entry = e
            break
    if entry is None:
        print("[smoke] FAIL: no frame with both maneuver cache + uniad_pth found")
        return 1
    tok = entry.get("sample_id")
    print(f"[smoke] using token: {tok}  ({cached[tok]['maneuver_type']}, {cached[tok].get('side')})")

    bootstrap_model_repo(cfg.model_repo)
    from data_utils.build_llava_conversation import build_llava_conversation

    print(f"[smoke] loading model: {cfg.checkpoint}")
    runner = ModelRunner(cfg.checkpoint, cfg.model_repo, device="cuda")

    entry = build_llava_conversation(entry, cached)
    question = entry["conversations"][0]["value"]
    print("\n===== ASSEMBLED PROMPT =====\n" + question + "\n============================")

    input_ids = runner.build_input_ids(question)
    uniad_pth = runner._torch.load(entry["uniad_pth"], map_location="cuda")
    answer = runner.generate(input_ids, uniad_pth=uniad_pth)
    print("\n===== MODEL ANSWER =====\n" + answer)
    wps = parse_waypoints(answer)
    print(f"\n[smoke] parsed {len(wps)} waypoints (right, forward, heading):")
    for i, wp in enumerate(wps, 1):
        r, f, h = wp[0], wp[1], wp[2]
        print(f"   t+{i*0.5:.1f}s: ({r:+.2f}, {f:+.2f}, h={h:+.2f})")
    print("\n[smoke] GT future (from cache, right/forward/heading):")
    gt = cached[tok]["gt_ego_fut_trajs"]
    for i in range(1, len(gt)):
        h = gt[i][2] if len(gt[i]) > 2 else 0.0
        print(f"   t+{i*0.5:.1f}s: ({gt[i][0]:+.2f}, {gt[i][1]:+.2f}, h={h:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
