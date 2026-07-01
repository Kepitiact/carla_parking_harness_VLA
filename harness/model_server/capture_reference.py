"""Capture ONE real `uniad_data` example from the offline pipeline as ground truth
for building live_uniad.py, and save it as a fixture.

This is the careful, no-guessing approach: instead of inferring what the UniAD vision
tower needs, we dump the exact structure the proven offline pipeline produces (keys,
shapes, dtypes, the full img_metas incl. lidar2img / can_bus / scene_token, and the
shapes of the GT-eval tensors we must dummy live). We also run the vision tower on it
and save the slim result, so live_uniad can be validated against a known-good output.

Run in the model venv:
    ~/projects/openvla_nuscenes/.venv/bin/python harness/model_server/capture_reference.py
Outputs (under harness/runs/uniad_ref/):
    structure.txt        human-readable structure dump
    uniad_data.pth       the captured input (for live validation / shape reference)
    results_for_vlm.pth  vision-tower output on that input (known-good reference)
"""
import os
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from harness import config as cfg_mod
from harness.model_server.model_runner import bootstrap_model_repo

PROC = pathlib.Path.home() / "projects/openvla_nuscenes/data_carla/processed"


def describe(x, indent=0, name=""):
    """Recursively describe a (possibly nested / DataContainer / tensor) object."""
    pad = "  " * indent
    t = type(x).__name__
    try:
        import torch
        import numpy as np
        # mmcv DataContainer
        if hasattr(x, "data") and t == "DataContainer":
            return f"{pad}{name}: DataContainer(stack={getattr(x,'stack',None)})\n" + describe(x.data, indent + 1, "data")
        if isinstance(x, torch.Tensor):
            return f"{pad}{name}: Tensor {tuple(x.shape)} {x.dtype}\n"
        if isinstance(x, np.ndarray):
            return f"{pad}{name}: ndarray {x.shape} {x.dtype}\n"
        if isinstance(x, dict):
            s = f"{pad}{name}: dict[{len(x)}]\n"
            for k, v in x.items():
                s += describe(v, indent + 1, str(k))
            return s
        if isinstance(x, (list, tuple)):
            s = f"{pad}{name}: {t}[{len(x)}]\n"
            for i, v in enumerate(x[:3]):
                s += describe(v, indent + 1, f"[{i}]")
            if len(x) > 3:
                s += f"{pad}  ... (+{len(x)-3} more)\n"
            return s
    except Exception as e:
        return f"{pad}{name}: <describe error {e}>\n"
    val = repr(x)
    if len(val) > 120:
        val = val[:120] + "..."
    return f"{pad}{name}: {t} = {val}\n"


def main():
    cfg = cfg_mod.Config()
    os.environ["CACHED_DATA_PATH"] = str(PROC / "cached_parking_info.pkl")
    bootstrap_model_repo(cfg.model_repo)

    import torch
    from mmengine import Config
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.train.train import DataArguments
    from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
    from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset
    from data_utils.nuscenes_llava_distributed_sampler import ContinuousSceneDistributedSampler
    from torch.utils.data import DataLoader

    out = _REPO / "harness/runs/uniad_ref"
    out.mkdir(parents=True, exist_ok=True)

    disable_torch_init()
    overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
    tokenizer, model, _, _ = load_pretrained_model(
        str(cfg.checkpoint), model_base=None, model_name="llava_qwen",
        device_map="cuda", multimodal=True, attn_implementation="eager",
        overwrite_config=overwrite_config,
    )
    model.eval()
    vision_tower = model.get_vision_tower()

    uniad_cfg = Config.fromfile(str(pathlib.Path(cfg.model_repo) / cfg.uniad_config))
    data_args = DataArguments(data_path=str(PROC / "carla_conversations.json"),
                              lazy_preprocess=True, frames_upbound=32)
    dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test,
                                   llava_test_mode=True, use_uniad_pth=False)
    sampler = ContinuousSceneDistributedSampler(dataset, num_replicas=1, rank=0,
                                                shuffle=False, drop_last=False)
    collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, collate_fn=collator)

    batch = next(iter(loader))
    uniad_data = batch["uniad_data"]
    sample_id = batch["id"][0] if isinstance(batch["id"], list) else batch["id"]

    # Structure dump.
    struct = f"sample_id: {sample_id}\n\n=== uniad_data ===\n" + describe(uniad_data, 0, "uniad_data")
    # Pull out the img_metas dict explicitly (the part live_uniad must synthesize).
    def unwrap(x):
        return unwrap(x.data) if type(x).__name__ == "DataContainer" else x
    im = unwrap(uniad_data.get("img_metas"))
    while isinstance(im, (list, tuple)):
        im = im[0]
    if isinstance(im, dict):
        struct += "\n=== img_metas[0][0] keys ===\n"
        for k, v in im.items():
            import numpy as np
            if isinstance(v, np.ndarray):
                struct += f"  {k}: ndarray {v.shape} {v.dtype}\n"
            elif k in ("lidar2img", "can_bus", "scene_token", "sample_idx", "img_shape",
                       "ori_shape", "pad_shape", "box_type_3d", "prev_idx", "next_idx"):
                sval = repr(v)
                struct += f"  {k}: {type(v).__name__} = {sval[:200]}\n"
            else:
                struct += f"  {k}: {type(v).__name__}\n"

    (out / "structure.txt").write_text(struct)
    print(struct)

    # Run the vision tower and save input + known-good output for live validation.
    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to("cuda")
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_dev(v) for v in x]
        return x

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            res = vision_tower(to_dev(uniad_data))

    def to_cpu(x):
        if isinstance(x, torch.Tensor):
            return x.cpu()
        if isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_cpu(v) for v in x]
        return x

    torch.save(to_cpu(uniad_data), out / "uniad_data.pth")
    torch.save(to_cpu(res), out / "results_for_vlm.pth")
    rt = res.get("result_track", {})
    print(f"\n[capture] saved fixture for {sample_id}")
    print(f"[capture] result_track.track_query_embeddings: "
          f"{tuple(rt['track_query_embeddings'].shape) if rt.get('track_query_embeddings') is not None else None}")
    print(f"[capture] wrote {out}/structure.txt, uniad_data.pth, results_for_vlm.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
