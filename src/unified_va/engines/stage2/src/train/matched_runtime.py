"""Fail-closed runtime support for the new, opt-in matched experiment protocol.

These helpers require no model download and can be tested with tiny CPU modules.
They do not certify the provenance of historical experiments.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import torch

from src.train.macro_selection import macro_dev_selection_metric_name
from src.train.checkpoint_storage import reconstruct_overlay_state, restore_overlay_buffers, uses_base_overlay

REQUIRED_DEV_METRICS = (
    "rmse_va", "rmse_v", "rmse_a", "pcc_macro", "pcc_v", "pcc_a",
    "ccc_macro", "ccc_v", "ccc_a", "pred_std_v", "pred_std_a",
    "true_std_v", "true_std_a", "std_gap_v", "std_gap_a",
)


def validate_matched_protocol(cfg: dict) -> dict:
    protocol = dict(cfg.get("matched_protocol", {}) or {})
    if not protocol.get("enabled", False):
        return {}
    if protocol.get("arm") not in {"S", "C", "M", "T"}:
        raise ValueError("matched_protocol.arm must be S/C/M/T")
    train = protocol.get("training_targets", [])
    held = protocol.get("held_out_targets", [])
    if len(train) != 12 or len(set(train)) != 12 or len(held) != 3 or len(set(held)) != 3:
        raise ValueError("matched protocol requires 12 unique training and 3 unique held-out targets")
    if set(train) & set(held):
        raise ValueError("training and held-out targets overlap")
    for key in ("require_complete_dev", "save_evaluation_checkpoints"):
        if protocol.get(key) is not True:
            raise ValueError(f"matched_protocol.{key} must be true")
    if protocol.get("optimizer_policy") != "reset":
        raise ValueError("matched protocol supports explicit weights-only continuation with optimizer_policy=reset")
    expected_world_size = int(protocol.get("world_size", 1))
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world_size:
        raise ValueError("runtime WORLD_SIZE differs from frozen matched protocol")
    # The initial matched protocol is single-process. Distributed restoration
    # needs a separate real-GPU acceptance run, rather than inferred readiness.
    if expected_world_size != 1:
        raise ValueError("matched distributed restoration is not accepted by this P0 implementation")
    if isinstance(cfg.get("sampler_seed"), bool) or not isinstance(cfg.get("sampler_seed"), int):
        raise ValueError("matched protocol requires an integer sampler_seed independent of model initialization")
    for key in ("resume_from", "resume_from_checkpoint", "resume"):
        if cfg.get(key):
            raise ValueError("matched protocol resets optimizer/scheduler; full resume is not permitted")
    if protocol["arm"] == "S" and cfg.get("pretrain_from"):
        raise ValueError("S starts from the pinned official base, not pretrain_from")
    if protocol["arm"] != "S" and not cfg.get("pretrain_from"):
        raise ValueError("C/M/T require the selected shared Stage-1 checkpoint")
    if not cfg.get("save_full_backbone_checkpoint", False) and not uses_base_overlay(cfg):
        raise ValueError("matched protocol requires full-backbone or pinned-base-overlay recoverable checkpoints")
    if cfg.get("group_balanced_sampling", {}).get("enabled", False) or (
        cfg.get("error_aware_sampling", {}).get("enabled", False)
        and cfg.get("error_aware_sampling", {}).get("use_sampler", True)
    ):
        raise ValueError("matched protocol uses the fixed independently seeded sampler, not weighted sampling")
    if cfg.get("alternating_optimization", {}).get("enabled", False):
        raise ValueError("matched protocol does not permit changing trainable sets during training")
    macro = cfg.get("macro_dev_eval", {})
    if not macro.get("enabled") or not macro.get("use_for_best", True):
        raise ValueError("matched protocol requires macro-dev selection")
    macro_names = [Path(str(x)).stem if str(x).endswith((".yaml", ".yml")) else str(x) for x in macro.get("targets", [])]
    if set(macro_names) != set(train) or len(macro_names) != 12:
        raise ValueError("macro-dev targets must equal the 12 training targets exactly")
    if macro_dev_selection_metric_name(macro.get("selection")) != "macro_dev_rmse_va":
        # This isolated reference restores the original DP selector, rather than
        # silently substituting the later RMSE-only comparison protocol.
        if (protocol.get("protocol_version"), protocol.get("arm")) not in {
                ("original_stage2_reference_20261104", "M"),
                ("original_source_stage2_textfix_20261104", "T")} :
            raise ValueError("primary matched comparison must select by RMSE-only; DP is offline on the same archive")
        import yaml
        historical = Path(__file__).resolve().parents[2] / "protocol/historical_stage2_config.yaml"
        expected = yaml.safe_load(historical.read_text())["macro_dev_eval"]["selection"]
        if macro.get("selection") != expected:
            raise ValueError("reference Stage 2 must use the exact recovered historical DP selector")
    if cfg.get("checkpoint_canary", {}).get("use_for_best", False):
        raise ValueError("matched primary checkpoint selection cannot use a canary adjustment")
    if protocol.get("protocol_version") == "method_validation_v2":
        from src.train.validation_v2 import validate_v2_config
        validate_v2_config(cfg)
    return protocol


def resolve_pretrain_checkpoint(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if path.is_dir() or path.suffix not in {".pt", ".pth", ".bin"}:
        path = path / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Required pretrain checkpoint is missing: {path}")
    return path.resolve()


def _stage1_backbone_plan(backbone, incoming: dict, allow_new_adapters: bool):
    if not isinstance(incoming, dict) or not incoming:
        raise ValueError("required backbone state is absent or empty")
    target = backbone.state_dict()
    allowed_new = set()
    wrapper_paths = []
    if allow_new_adapters:
        # Enumerate actual supported wrapper modules, not arbitrary keys containing
        # the word 'adapter'. A missing common parameter can never pass this rule.
        from src.models.backbone_conditional_adapter import (
            ConditionalDecoderLayerWrapper, FactorizedConditionalResidualAdapter,
        )
        for name, module in backbone.named_modules():
            if isinstance(module, ConditionalDecoderLayerWrapper):
                if module.repair_adapter is not None:
                    raise ValueError("new repair adapters are outside the S/C/M/T continuation contract")
                wrapper_paths.append(name)
                if module.adapter is not None:
                    if not isinstance(module.adapter, FactorizedConditionalResidualAdapter):
                        raise ValueError("unsupported newly introduced adapter module")
                    allowed_new.update(f"{name}.adapter.{key}" for key in module.adapter.state_dict())
    remapped = {}
    remap_report = {}
    for key, value in incoming.items():
        mapped = key
        if key not in target:
            candidates = []
            for path in wrapper_paths:
                prefix = path + "."
                if key.startswith(prefix):
                    candidate = prefix + "layer." + key[len(prefix):]
                    if candidate in target:
                        candidates.append(candidate)
            if len(candidates) == 1:
                mapped = candidates[0]
        if mapped in remapped:
            raise ValueError(f"checkpoint key remapping collision: {mapped}")
        remapped[mapped] = value
        if mapped != key:
            remap_report[key] = mapped
    _validate_state("backbone", target, remapped, allowed_new)
    return remapped, allowed_new, remap_report


def _validate_state(name, target, incoming, allowed_missing=()):
    if not isinstance(incoming, dict) or not incoming:
        raise ValueError(f"required {name} state is absent or empty")
    missing = set(target) - set(incoming)
    unexpected = set(incoming) - set(target)
    forbidden = missing - set(allowed_missing)
    mismatched = [key for key in set(target) & set(incoming)
                  if tuple(target[key].shape) != tuple(incoming[key].shape)]
    if forbidden or unexpected or mismatched:
        raise ValueError(f"incomplete {name} restoration: missing={sorted(forbidden)[:8]}, "
                         f"unexpected={sorted(unexpected)[:8]}, shape_mismatch={sorted(mismatched)[:8]}")


def restore_matched_checkpoint(model, payload: dict, cfg: dict, *, from_stage1=False,
                               extra_modules=None) -> dict:
    """Validate every common tensor before mutating; only real new adapters may miss.

    Stage-1 continuation intentionally resets optimizer/scheduler/RNG update state.
    Exact-model evaluation reload also requires all auxiliary heads to be restored.
    """
    checkpoint_type = payload.get("checkpoint_type")
    if checkpoint_type not in {"full_backbone", "pinned_base_overlay"}:
        raise ValueError("matched restoration requires a full-backbone or pinned-base-overlay checkpoint")
    if from_stage1 and payload.get("matched_protocol", {}).get("arm") != "S":
        raise ValueError("shared continuation checkpoint must identify a new-protocol S run")
    if from_stage1 and cfg.get("matched_protocol", {}).get("protocol_version") == "method_validation_v2":
        from src.train.validation_v2 import validate_stage1_parent
        validate_stage1_parent(payload, cfg)
    if payload.get("head_type") != cfg.get("head_type"):
        raise ValueError("shared checkpoint head_type differs from effective configuration")
    pooling = cfg.get("small_llm", {}).get("pooling", getattr(model, "pooling", None))
    if payload.get("pooling") != pooling:
        raise ValueError("shared checkpoint pooling differs from effective configuration")
    backbone = payload.get("backbone")
    omitted = set()
    if checkpoint_type == "pinned_base_overlay":
        backbone = reconstruct_overlay_state(model, payload, cfg)
        omitted = set(payload["checkpoint_storage"]["omitted_frozen_base_keys"])
    plan, allowed, remaps = _stage1_backbone_plan(model.backbone, backbone, from_stage1)
    if omitted:
        target_state = model.backbone.state_dict()
        if any(value.dtype != target_state[key].dtype for key, value in plan.items()):
            raise ValueError("base overlay dtype differs from the effective target model")
    _validate_state("head", model.head.state_dict(), payload.get("head"))
    pool_gate = getattr(model, "pool_gate", None)
    if pool_gate is not None:
        _validate_state("pool_gate", pool_gate.state_dict(), payload.get("pool_gate"))
    elif payload.get("pool_gate"):
        raise ValueError("checkpoint pool_gate has no corresponding target module")
    extra_plan = {}
    for name, module in (extra_modules or {}).items():
        if module is None:
            continue
        incoming = payload.get(name)
        if from_stage1 and name == "soft_condition_label_head" and incoming is None:
            continue  # Explicit newly introduced training supervision head in T.
        _validate_state(name, module.state_dict(), incoming)
        extra_plan[name] = incoming
    # Omitted frozen parameters already come from the retained pinned base.
    # Avoid no-op writes to them, preserving the mutation guard's versions.
    to_load = {key: value for key, value in plan.items() if key not in omitted}
    model.backbone.load_state_dict(to_load, strict=not bool(allowed or omitted))
    model.head.load_state_dict(payload["head"], strict=True)
    if pool_gate is not None:
        pool_gate.load_state_dict(payload["pool_gate"], strict=True)
    for name, state in extra_plan.items():
        extra_modules[name].load_state_dict(state, strict=True)
    if checkpoint_type == "pinned_base_overlay":
        restore_overlay_buffers(model, payload)
    return {"strict_common_restoration": True, "backbone_loaded_keys": len(to_load),
            "checkpoint_type": checkpoint_type, "pinned_base_keys_reused": len(omitted),
            "new_adapter_keys": sorted(allowed - set(plan)), "remapped_keys": remaps,
            "head_loaded_keys": len(payload["head"]), "optimizer_policy": "reset",
            "optimizer_scheduler_or_rng_restored": False,
            "extra_modules_restored": sorted(extra_plan)}


def validate_complete_dev(per_target: dict, expected_targets: list[str]) -> None:
    if set(per_target) != set(expected_targets) or len(per_target) != len(expected_targets):
        raise ValueError(f"incomplete development targets: expected={sorted(expected_targets)}, got={sorted(per_target)}")
    for target, metrics in per_target.items():
        missing = set(REQUIRED_DEV_METRICS) - set(metrics)
        if missing:
            raise ValueError(f"{target}: missing development metrics {sorted(missing)}")
        for name in REQUIRED_DEV_METRICS:
            if not math.isfinite(float(metrics[name])):
                raise ValueError(f"{target}: nonfinite development metric {name}")
        if int(metrics.get("num_examples", 0)) <= 0:
            raise ValueError(f"{target}: development evaluation has no examples")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def archive_evaluation(run_dir: Path, *, event: str, epoch: int, update: int,
                       per_target: dict, cfg: dict, checkpoint_writer, is_main=True,
                       barrier=lambda: None) -> Path:
    """Call the collective checkpoint writer on every rank, JSON writer on rank 0.

    A step and an epoch-end evaluation at the same update have separate immutable
    names. An archive record is published only after its weight file exists.
    """
    if event not in {"step", "epoch"}:
        raise ValueError("archive event must be step or epoch")
    expected = cfg["matched_protocol"]["training_targets"]
    validate_complete_dev(per_target, expected)
    archive = run_dir / "evaluation_archive"
    stem = f"eval_{event}_epoch{epoch:03d}_update{update:08d}"
    checkpoint = archive / (stem + ".pt")
    record = archive / (stem + ".json")
    if is_main:
        archive.mkdir(parents=True, exist_ok=True)
        if checkpoint.exists() or record.exists():
            raise FileExistsError(f"refusing to overwrite an evaluation candidate: {stem}")
    barrier()
    checkpoint_writer(archive, checkpoint.name)
    if is_main:
        if not checkpoint.is_file():
            raise RuntimeError("evaluation checkpoint writer did not create its linked artifact")
        write_json(record, {"event": event, "epoch": int(epoch), "global_update": int(update),
                            "checkpoint": checkpoint.name, "checkpoint_bytes": checkpoint.stat().st_size,
                            "targets": list(expected), "per_target": per_target,
                            "selection_cfg": copy.deepcopy(cfg["macro_dev_eval"].get("selection", {})),
                            "inference_config": copy.deepcopy(cfg),
                            "prediction_precision": ("original Python two decimals before metrics" if cfg.get("legacy_prediction_rounding") == "python_two_decimals_before_metrics"
                                                     else "continuous, official-scale clamped; no two-decimal rounding"),
                            "archive_role": "all scheduled development candidates; never sensitivity-pruned"})
    barrier()
    return record


def parameter_manifest(model, extra_modules, cfg: dict, optimizer=None) -> dict:
    rows = []
    seen = set()
    parameter_names = {}
    for prefix, module in [("model", model), *list((extra_modules or {}).items())]:
        if module is None:
            continue
        for name, param in module.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            parameter_names[id(param)] = f"{prefix}.{name}"
            rows.append({"name": f"{prefix}.{name}", "shape": list(param.shape), "numel": param.numel(),
                         "dtype": str(param.dtype), "requires_grad": bool(param.requires_grad)})
    optimizer_groups = []
    for index, group in enumerate(getattr(optimizer, "param_groups", [])):
        optimizer_groups.append({
            "index": index, "lr": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0)),
            "parameter_names": [parameter_names.get(id(param), "UNREGISTERED_OPTIMIZER_PARAMETER") for param in group["params"]],
        })
    return {"arm": cfg["matched_protocol"]["arm"], "optimizer_policy": "reset",
            "optimizer_groups": optimizer_groups,
            "optimizer_state_restored": False, "scheduler_state_restored": False,
            "initial_global_update": 0, "sampler_seed": cfg["sampler_seed"],
            "parameters": rows, "trainable_numel": sum(x["numel"] for x in rows if x["requires_grad"])}


def batch_order_record(batch: dict, train_data: list[dict], *, epoch: int, update: int,
                       microbatch: int, rank: int) -> dict:
    indices = batch.get("idx")
    if indices is None:
        raise ValueError("training order audit requires actual batch dataset indices")
    if torch.is_tensor(indices):
        indices = indices.tolist()
    samples = []
    for index in indices:
        sample = train_data[int(index)]
        samples.append({"dataset_index": int(index), "id": str(sample["id"]),
                        "aspect_index": int(sample["aspect_index"]),
                        "target": f"{sample.get('lang')}_{sample.get('domain')}"})
    return {"epoch": int(epoch), "optimizer_update": int(update), "microbatch": int(microbatch),
            "rank": int(rank), "samples": samples}
