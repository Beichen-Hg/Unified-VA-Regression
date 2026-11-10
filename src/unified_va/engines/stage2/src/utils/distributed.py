"""Minimal distributed/FSDP2 helpers for training."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from src.utils.device import setup_device

try:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
except ImportError:  # pragma: no cover - depends on torch version
    MixedPrecisionPolicy = None
    fully_shard = None
    StateDictOptions = None
    get_model_state_dict = None
    init_device_mesh = None


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str = "nccl"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed_context(cfg_device: str | None = None, backend: str = "nccl") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return DistributedContext(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=setup_device(cfg_device),
            backend=backend,
        )

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def distributed_barrier(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()


def broadcast_path(path: str | Path | None, ctx: DistributedContext) -> Path:
    if not ctx.enabled:
        if path is None:
            raise ValueError("path must not be None when distributed is disabled")
        return Path(path)
    obj_list = [str(path) if ctx.is_main and path is not None else None]
    dist.broadcast_object_list(obj_list, src=0)
    value = obj_list[0]
    if value is None:
        raise RuntimeError("failed to broadcast run dir")
    return Path(value)


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def apply_fsdp2_to_backbone(
    model,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype | None = None,
    root_reshard_after_forward: bool = False,
):
    if fully_shard is None or init_device_mesh is None or MixedPrecisionPolicy is None:
        raise RuntimeError("FSDP2 helpers are unavailable in this torch build")
    if not dist.is_initialized():
        raise RuntimeError("dist.init_process_group must be called before FSDP2 sharding")

    layers = model._resolve_decoder_layers() if hasattr(model, "_resolve_decoder_layers") else None
    if layers is None:
        raise ValueError("failed to resolve decoder layers for FSDP2 sharding")

    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype or param_dtype,
        output_dtype=param_dtype,
        cast_forward_inputs=True,
    )
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(
        model.backbone,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=root_reshard_after_forward,
    )
    return mesh


def gather_full_state_dict(module) -> dict:
    if get_model_state_dict is None or StateDictOptions is None:
        raise RuntimeError("distributed checkpoint state-dict helpers are unavailable")
    return get_model_state_dict(
        module,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
