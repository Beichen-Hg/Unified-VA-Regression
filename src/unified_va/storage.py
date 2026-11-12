"""Lossless updated-tensor checkpoints over the pinned local base model."""

from pathlib import Path
import copy
import os
import shutil
import torch

FORMAT = "unified_va_overlay_v2"


def stored_keys(model):
    params = dict(model.named_parameters())
    return {k for k in model.state_dict() if k not in params or params[k].requires_grad}


def payload(model, cfg, norm, update, epoch):
    state = model.state_dict()
    return dict(
        format=FORMAT,
        config=copy.deepcopy(cfg),
        normalization=dict(norm),
        global_update=update,
        epoch=epoch,
        state={k: state[k].detach().cpu().clone() for k in stored_keys(model)},
    )


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = sum(t.numel() * t.element_size() for t in value["state"].values())
    if shutil.disk_usage(path.parent).free - size - 64 * 1024**2 < 40 * 1024**3:
        raise RuntimeError(
            "Checkpoint write would breach the 40 GiB free-space reserve"
        )
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load(path):
    value = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if value.get("format") != FORMAT:
        raise ValueError(
            "Checkpoint format does not match this reference implementation"
        )
    return value


def restore(model, value, cfg):
    if value["format"] != FORMAT or set(value["state"]) != stored_keys(model):
        raise ValueError("Missing or unexpected updated tensors in checkpoint")
    for key in ("stage", "split"):
        if value["config"]["experiment"][key] != cfg["experiment"][key]:
            raise ValueError("Checkpoint stage or partition differs")
    if value["config"]["seed"] != cfg["seed"]:
        raise ValueError("Checkpoint seed differs")
    model.load_state_dict(value["state"], strict=False)
    return value["normalization"]


def warm_start(model, value, cfg, norm):
    parent = value["config"]
    if parent["experiment"]["stage"] != "stage1" or parent["seed"] != cfg["seed"]:
        raise ValueError("Stage2 requires its own seed's selected Stage1 model")
    if (
        parent["experiment"]["split"] != cfg["experiment"]["split"]
        or value["normalization"] != norm
    ):
        raise ValueError("Stage1 partition or normalization differs")
    for key in ("head_type", "dual_head_mlp", "max_len", "freeze_until"):
        if parent.get(key) != cfg.get(key):
            raise ValueError("Stage1 model setting differs: " + key)
    for key in ("pooling", "prompt_template", "revision"):
        if parent["small_llm"][key] != cfg["small_llm"][key]:
            raise ValueError("Stage1 input/base setting differs: " + key)
    state = model.state_dict()
    if any(
        k not in state or state[k].shape != v.shape for k, v in value["state"].items()
    ):
        raise ValueError("Stage1 tensor structure differs from Stage2")
    model.load_state_dict(value["state"], strict=False)
