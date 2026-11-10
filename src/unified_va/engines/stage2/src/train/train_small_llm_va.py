"""Training entry for decoder-only small LLM VA regression."""
from __future__ import annotations

import copy
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.utils.data import WeightedRandomSampler

try:
    import deepspeed
except ImportError:  # pragma: no cover - optional dependency
    deepspeed = None

from src.data.datamodule import build_dimasr_dataloader
from src.models.text_only_protocol import (
    PROTOCOL, export_soft_condition_classifier_config, validate_text_only_cfg, validate_text_only_model, validate_normalization,
)
from src.data.dimabsa2026 import expand_to_aspect_level, load_samples, load_splits
from src.data.sample_keys import make_sample_key
from src.data.translation_sidecar import audit_translation_coverage, load_translation_pivot_map
from src.eval.metrics_va import pcc_a, pcc_v, rmse_a, rmse_v, rmse_va
from src.eval.writer_va import write_predictions
from src.methods.D6_teacher.teacher_artifacts import (
    join_teacher_to_samples,
    load_teacher_map,
    save_teacher_audit,
)
from src.models.heads import (
    FactorizedOutputResidualHead,
    SoftConditionGroupClassifier,
    VAUncertaintyHead,
    infer_group_values_from_cfg,
)
from src.models.losses import (
    arousal_anti_collapse_loss,
    combine_primary_aux_losses,
    gaussian_nll_va,
    masked_weighted_mean,
    teacher_va_distill_loss,
    va_decorrelation_loss,
    va_loss,
)
from src.models.valence_sharpening import valence_sharpening_loss
from src.methods.D37_variance.variance_regularization import variance_regularization_loss
from src.models.va_region_weights import compute_va_region_weights
from src.models.small_llm_regressor import build_small_llm_regressor, freeze_small_llm_layers
from src.train.calibration import make_calibrator
from src.train.checkpoint_storage import export_base_overlay, initialize_checkpoint_storage, uses_base_overlay
from src.train.macro_selection import compute_macro_dev_selection, macro_dev_selection_metric_name
from src.train.matched_runtime import (
    archive_evaluation, batch_order_record, parameter_manifest,
    resolve_pretrain_checkpoint, restore_matched_checkpoint,
    validate_complete_dev, validate_matched_protocol, write_json,
)
from src.train.recipe import build_llrd_param_groups
from src.train.tta import disable_dropout_for_inference, enable_dropout_for_inference
from src.utils.distributed import (
    DistributedContext,
    apply_fsdp2_to_backbone,
    broadcast_path,
    distributed_barrier,
    gather_full_state_dict,
    init_distributed_context,
)
from src.utils.io import make_run_dir, save_config, save_metrics
from src.utils.logging import add_file_handler, get_logger
from src.utils.seed import set_seed, set_cuda_seed

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_TARGET_REGISTRY = {
    'jpn_hotel': 'configs/methods/small_llm_qwen3_8b_base_jpn_hotel.yaml',
    'jpn_finance': 'configs/methods/small_llm_qwen3_8b_base_jpn_finance.yaml',
    'eng_restaurant': 'configs/methods/small_llm_qwen3_8b_base_eng_restaurant.yaml',
    'deu_politics': 'scripts/experiments/baseline_pure/deu_politics_baseline_pure.yaml',
    'eng_laptop': 'scripts/experiments/baseline_pure/eng_laptop_baseline_pure.yaml',
    'zho_laptop': 'configs/methods/small_llm_qwen3_8b_base_zho_laptop.yaml',
    'zho_finance': 'configs/methods/small_llm_qwen3_8b_base_zho_finance.yaml',
    'ukr_restaurant': 'scripts/experiments/baseline_pure/ukr_restaurant_baseline_pure.yaml',
    'zho_restaurant': 'scripts/experiments/baseline_pure/zho_restaurant_baseline_pure.yaml',
    'eng_environmental_protection': 'scripts/experiments/baseline_pure/eng_env_baseline_pure.yaml',
    'rus_restaurant': 'configs/methods/small_llm_qwen3_8b_base_rus_restaurant.yaml',
    'tat_restaurant': 'configs/methods/small_llm_qwen3_8b_base_tat_restaurant.yaml',
    'pcm_politics': 'configs/methods/small_llm_qwen3_8b_base_pcm_politics.yaml',
    'swa_politics': 'configs/methods/small_llm_qwen3_8b_base_swa_politics.yaml',
    'zho_environmental_protection': 'configs/methods/small_llm_qwen3_8b_base_zho_environmental_protection.yaml',
}


def _deepspeed_cfg(cfg: dict) -> dict:
    return dict(cfg.get("deepspeed", {}) or {})


def _deepspeed_enabled(cfg: dict, dist_ctx: DistributedContext) -> bool:
    return bool(_deepspeed_cfg(cfg).get("enabled", False)) and dist_ctx.enabled


def _patch_deepspeed_count_used_parameters_in_backward() -> None:
    if deepspeed is None:
        return
    try:
        from deepspeed.runtime import utils as ds_utils
        from deepspeed.runtime.zero import stage3 as ds_stage3
    except Exception:
        return

    if getattr(ds_utils, "_proposal_safe_count_used_patch", False):
        return

    def _safe_count_used_parameters_in_backward(parameters):
        assert ds_utils.check_internal_apis_for_count_used_parameters(), (
            "count_used_parameters_in_backward requires internal PyTorch APIs that are not available "
            "in this PyTorch build."
        )
        from torch.autograd.graph import _get_grad_fn_or_grad_acc
        if torch._C._current_graph_task_id() == -1:
            raise RuntimeError("count_used_parameters_in_backward must be called during backward execution")

        seen_nodes = set()
        for param in parameters:
            if not isinstance(param, torch.Tensor) or not param.requires_grad:
                continue
            try:
                grad_fn = _get_grad_fn_or_grad_acc(param)
            except AttributeError:
                continue
            if grad_fn is None or grad_fn in seen_nodes:
                continue
            seen_nodes.add(grad_fn)

        if not seen_nodes:
            return 0
        return sum(map(torch._C._will_engine_execute_node, seen_nodes))

    ds_utils.count_used_parameters_in_backward = _safe_count_used_parameters_in_backward
    ds_stage3.count_used_parameters_in_backward = _safe_count_used_parameters_in_backward
    ds_utils._proposal_safe_count_used_patch = True


def _build_deepspeed_runtime_config(cfg: dict, dist_ctx: DistributedContext) -> dict:
    ds_cfg = _deepspeed_cfg(cfg)
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = max(int(cfg.get("grad_accum", 1)), 1)
    grad_clip = float(cfg.get("grad_clip", 0.0) or 0.0)
    amp_dtype = str(cfg.get("amp_dtype", "bf16")).lower()
    zero_stage = int(ds_cfg.get("zero_stage", 3))
    zero_cfg = {
        "stage": zero_stage,
        "contiguous_gradients": bool(ds_cfg.get("contiguous_gradients", True)),
        "overlap_comm": bool(ds_cfg.get("overlap_comm", True)),
        "reduce_scatter": bool(ds_cfg.get("reduce_scatter", True)),
        "allgather_partitions": bool(ds_cfg.get("allgather_partitions", True)),
        "stage3_gather_16bit_weights_on_model_save": bool(
            ds_cfg.get("stage3_gather_16bit_weights_on_model_save", True)
        ),
        "ignore_unused_parameters": bool(ds_cfg.get("ignore_unused_parameters", True)),
    }
    offload_optimizer_device = ds_cfg.get("offload_optimizer_device")
    if offload_optimizer_device:
        zero_cfg["offload_optimizer"] = {
            "device": str(offload_optimizer_device),
            "pin_memory": bool(ds_cfg.get("offload_optimizer_pin_memory", False)),
        }
    if ds_cfg.get("stage3_prefetch_bucket_size") is not None:
        zero_cfg["stage3_prefetch_bucket_size"] = int(ds_cfg["stage3_prefetch_bucket_size"])
    if ds_cfg.get("stage3_param_persistence_threshold") is not None:
        zero_cfg["stage3_param_persistence_threshold"] = int(ds_cfg["stage3_param_persistence_threshold"])
    if ds_cfg.get("stage3_max_live_parameters") is not None:
        zero_cfg["stage3_max_live_parameters"] = int(ds_cfg["stage3_max_live_parameters"])
    if ds_cfg.get("stage3_max_reuse_distance") is not None:
        zero_cfg["stage3_max_reuse_distance"] = int(ds_cfg["stage3_max_reuse_distance"])
    runtime_cfg = {
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": batch_size * max(int(dist_ctx.world_size), 1) * grad_accum,
        "gradient_clipping": grad_clip,
        "zero_optimization": zero_cfg,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": int(ds_cfg.get("steps_per_print", 2000)),
        "wall_clock_breakdown": bool(ds_cfg.get("wall_clock_breakdown", False)),
    }
    if amp_dtype == "bf16":
        runtime_cfg["bf16"] = {"enabled": True}
    else:
        runtime_cfg["fp16"] = {"enabled": True}
    return runtime_cfg


def _extract_submodule_state_dict(state_dict: dict[str, object] | None, prefix: str) -> dict[str, object]:
    if not state_dict:
        return {}
    prefix_with_dot = f"{prefix}."
    return {
        key[len(prefix_with_dot):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }


def _build_eval_diagnostics(vp: np.ndarray, ap: np.ndarray, vt: np.ndarray, at: np.ndarray) -> dict:
    return {
        "pred_mean_v": float(np.mean(vp)),
        "pred_std_v": float(np.std(vp)),
        "true_mean_v": float(np.mean(vt)),
        "true_std_v": float(np.std(vt)),
        "pred_mean_a": float(np.mean(ap)),
        "pred_std_a": float(np.std(ap)),
        "true_mean_a": float(np.mean(at)),
        "true_std_a": float(np.std(at)),
        "std_gap_v": float(abs(np.std(vp) - np.std(vt))),
        "std_gap_a": float(abs(np.std(ap) - np.std(at))),
    }


def _ccc_np(pred: np.ndarray, gold: np.ndarray) -> float:
    if len(pred) == 0 or len(gold) == 0:
        return 0.0
    pred = np.asarray(pred, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    pred_var = float(np.var(pred))
    gold_var = float(np.var(gold))
    pred_mean = float(np.mean(pred))
    gold_mean = float(np.mean(gold))
    cov = float(np.mean((pred - pred_mean) * (gold - gold_mean)))
    denom = pred_var + gold_var + (pred_mean - gold_mean) ** 2
    if denom < 1e-12:
        return 0.0
    return float(2.0 * cov / denom)


def _uses_uncertainty_head(model) -> bool:
    head = getattr(model, "head", None)
    if isinstance(head, VAUncertaintyHead):
        return True
    return isinstance(getattr(head, "base_head", None), VAUncertaintyHead)


def _sample_group_name(sample: dict) -> str:
    return f"{sample.get('lang', 'unknown')}_{sample.get('domain', 'unknown')}"


DEFAULT_LANGFAMILY_ALIAS = {
    "eng": "germanic",
    "deu": "germanic",
    "pcm": "germanic",
    "jpn": "east_asian",
    "zho": "east_asian",
    "ukr": "slavic",
    "rus": "slavic",
    "tat": "turkic",
    "swa": "bantu",
}


def _canonicalize_group_value(value: str, alias_map: dict | None = None) -> str:
    value = str(value or "unknown").strip()
    alias_map = alias_map or {}
    return str(alias_map.get(value, value))


def _sample_group_value(sample: dict, group_by: str, alias_map: dict | None = None) -> str:
    group_by = str(group_by or "domain").strip().lower()
    if group_by in {"lang_domain", "group", "pair", "lang+domain"}:
        return _sample_group_name(sample)
    if group_by in {"langfamily", "lang_family", "family"}:
        return _canonicalize_group_value(sample.get("lang", "unknown"), alias_map or DEFAULT_LANGFAMILY_ALIAS)
    if group_by in {"lang", "language"}:
        return _canonicalize_group_value(sample.get("lang", "unknown"), alias_map)
    return _canonicalize_group_value(sample.get("domain", "unknown"), alias_map)


def _batch_group_values(batch: dict, group_by: str, alias_map: dict | None = None) -> list[str]:
    if group_by in {"lang_domain", "group", "pair", "lang+domain"}:
        return [f"{lang}_{domain}" for lang, domain in zip(batch["lang"], batch["domain"])]
    if group_by in {"langfamily", "lang_family", "family"}:
        return [_canonicalize_group_value(v, alias_map or DEFAULT_LANGFAMILY_ALIAS) for v in batch["lang"]]
    if group_by in {"lang", "language"}:
        return [_canonicalize_group_value(v, alias_map) for v in batch["lang"]]
    return [_canonicalize_group_value(v, alias_map) for v in batch["domain"]]


def _resolve_group_values_from_cfg(section_cfg: dict, cfg: dict, key: str) -> list[str]:
    values = [str(v).strip() for v in section_cfg.get(f"{key}s", []) if str(v).strip()]
    if values:
        return values
    return infer_group_values_from_cfg(cfg, key)


def _build_group_balanced_sampler(cfg: dict, train_data: list[dict], run_dir: Path):
    sampler_cfg = cfg.get("group_balanced_sampling", {})
    if not sampler_cfg.get("enabled", False):
        return None, None

    group_by = str(sampler_cfg.get("group_by", "domain")).strip().lower()
    alias_map = sampler_cfg.get("alias_map")
    if group_by in {"langfamily", "lang_family", "family"} and alias_map is None:
        alias_map = DEFAULT_LANGFAMILY_ALIAS

    power = float(sampler_cfg.get("power", 1.0))
    min_weight = float(sampler_cfg.get("min_weight", 0.5))
    max_weight = float(sampler_cfg.get("max_weight", 5.0))
    replacement = bool(sampler_cfg.get("replacement", True))

    group_values = [_sample_group_value(sample, group_by, alias_map) for sample in train_data]
    counts = Counter(group_values)
    weights = []
    for value in group_values:
        w = (1.0 / max(counts[value], 1)) ** power
        w = min(max(w, min_weight), max_weight)
        weights.append(w)

    weights_np = np.asarray(weights, dtype=np.float64)
    if weights_np.mean() > 0:
        weights_np = weights_np / weights_np.mean()
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights_np, dtype=torch.double),
        num_samples=len(train_data),
        replacement=replacement,
    )
    audit = {
        "group_by": group_by,
        "power": power,
        "replacement": replacement,
        "num_groups": len(counts),
        "group_counts": dict(sorted(counts.items())),
        "mean_weight": float(weights_np.mean()) if len(weights_np) else 0.0,
        "min_weight": float(weights_np.min()) if len(weights_np) else 0.0,
        "max_weight": float(weights_np.max()) if len(weights_np) else 0.0,
    }
    (run_dir / "group_balanced_sampling_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Group-balanced sampling enabled: group_by=%s groups=%d mean=%.4f range=[%.4f, %.4f]",
        group_by,
        audit["num_groups"],
        audit["mean_weight"],
        audit["min_weight"],
        audit["max_weight"],
    )
    return sampler, audit


def _group_dro_loss(
    per_sample_loss: torch.Tensor,
    group_values: list[str],
    *,
    sample_weights: torch.Tensor | None = None,
    top_k: int = 1,
) -> torch.Tensor:
    if per_sample_loss.numel() == 0 or not group_values:
        return torch.tensor(0.0, device=per_sample_loss.device)

    unique = list(dict.fromkeys(group_values))
    group_losses = []
    for value in unique:
        mask = torch.tensor([1.0 if g == value else 0.0 for g in group_values], device=per_sample_loss.device)
        group_losses.append(masked_weighted_mean(per_sample_loss, mask, sample_weights))
    if not group_losses:
        return torch.tensor(0.0, device=per_sample_loss.device)
    stacked = torch.stack(group_losses)
    k = max(1, min(int(top_k), stacked.numel()))
    topk_vals, _ = torch.topk(stacked, k=k)
    return topk_vals.mean()


def _maybe_limit_samples(samples: list[dict], limit: int | None) -> list[dict]:
    if not limit or limit <= 0 or len(samples) <= limit:
        return samples
    return samples[:limit]


def _load_cfg_file(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_eval_target_cfg_path(target: str) -> Path:
    target = str(target).strip()
    if not target:
        raise ValueError("empty macro_dev_eval target")
    if target in EVAL_TARGET_REGISTRY:
        return PROJECT_ROOT / EVAL_TARGET_REGISTRY[target]
    path = Path(target)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"macro_dev_eval target config not found: {target}")
    return path


def _build_target_eval_cfg(source_cfg: dict, target_cfg: dict) -> dict:
    cfg = json.loads(json.dumps(source_cfg))
    for key in ['track', 'lang', 'domain', 'data_dir', 'train_file', 'dev_file', 'test_file', 'filter_unclear']:
        if key in target_cfg:
            cfg[key] = target_cfg[key]
    cfg['run_test'] = False
    cfg.setdefault('teacher', {})
    cfg['teacher']['enabled'] = False
    return cfg


def _build_macro_dev_targets(cfg: dict):
    macro_cfg = cfg.get("macro_dev_eval", {})
    if not macro_cfg.get("enabled", False):
        return []

    target_specs = macro_cfg.get("targets", [])
    if isinstance(target_specs, str):
        target_specs = [x.strip() for x in target_specs.split(",") if x.strip()]
    if not target_specs:
        raise ValueError("macro_dev_eval.enabled=true but no targets were provided")

    targets = []
    for spec in target_specs:
        cfg_path = _resolve_eval_target_cfg_path(spec)
        target_cfg = _load_cfg_file(cfg_path)
        eval_cfg = _build_target_eval_cfg(cfg, target_cfg)
        dev_data = expand_to_aspect_level(load_samples(eval_cfg, "dev"))
        targets.append({
            "name": spec if spec in EVAL_TARGET_REGISTRY else cfg_path.stem,
            "cfg_path": str(cfg_path),
            "eval_cfg": eval_cfg,
            "dev_data": dev_data,
        })
    logger.info("Macro-dev eval enabled on targets: %s", [t["name"] for t in targets])
    return targets


def _evaluate_macro_dev_rmse(model, macro_targets, tokenizer, device, va_norm_stats=None,
                             quantile_head=None, pivot_residual_head=None,
                             pivot_trust_head=None, va_residual_head=None,
                             group_routed_residual_head=None):
    if not macro_targets:
        return None, {}
    per_target = {}
    for target in macro_targets:
        dev_data = target["dev_data"]
        if not dev_data:
            continue
        preds = _predict(
            model,
            dev_data,
            tokenizer,
            device,
            target["eval_cfg"],
            va_norm_stats=va_norm_stats,
            quantile_head=quantile_head,
            pivot_residual_head=pivot_residual_head,
            pivot_trust_head=pivot_trust_head,
            va_residual_head=va_residual_head,
            group_routed_residual_head=group_routed_residual_head,
        )
        vp = np.array([p["v_pred"] for p in preds])
        ap = np.array([p["a_pred"] for p in preds])
        vt = np.array([p["v_true"] for p in preds])
        at = np.array([p["a_true"] for p in preds])
        ccc_v = _ccc_np(vp, vt)
        ccc_a = _ccc_np(ap, at)
        pccv = float(pcc_v(vp, vt))
        pcca = float(pcc_a(ap, at))
        per_target[target["name"]] = {
            "num_examples": len(preds),
            "rmse_va": float(rmse_va(vp, ap, vt, at)),
            "rmse_v": float(rmse_v(vp, vt)),
            "rmse_a": float(rmse_a(ap, at)),
            "pcc_v": pccv, "pcc_a": pcca,
            "ccc_v": ccc_v, "ccc_a": ccc_a,
            "ccc_macro": float((ccc_v + ccc_a) / 2.0),
            "pcc_macro": float((pccv + pcca) / 2.0),
            **_build_eval_diagnostics(vp, ap, vt, at),
        }
    if not per_target:
        return None, {}
    macro = float(np.mean([metrics["rmse_va"] for metrics in per_target.values()]))
    return macro, per_target


def teacher_weight_schedule(epoch: int, base_weight: float,
                            warmup_epochs: int = 0,
                            ramp_epochs: int = 0) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs > 0:
        elapsed = epoch - warmup_epochs
        if elapsed < ramp_epochs:
            return base_weight * (elapsed + 1) / ramp_epochs
    return base_weight


def resolve_teacher_loss_mode(teacher_cfg: dict) -> str:
    if "loss_mode" in teacher_cfg:
        return teacher_cfg["loss_mode"]
    return teacher_cfg.get("mode", "mse")


def _normalize_va(va: torch.Tensor, stats: dict | None) -> torch.Tensor:
    if stats is None:
        return va
    out = va.clone()
    out[:, 0] = (out[:, 0] - stats["v_mean"]) / stats["v_std"]
    out[:, 1] = (out[:, 1] - stats["a_mean"]) / stats["a_std"]
    return out


def _denormalize_va(va: torch.Tensor, stats: dict | None) -> torch.Tensor:
    if stats is None:
        return va
    out = va.clone()
    out[:, 0] = out[:, 0] * stats["v_std"] + stats["v_mean"]
    out[:, 1] = out[:, 1] * stats["a_std"] + stats["a_mean"]
    return out


def _denormalize_sigma(sigma: torch.Tensor, stats: dict | None) -> torch.Tensor:
    if stats is None:
        return sigma
    out = sigma.clone()
    out[:, 0] = out[:, 0] * stats["v_std"]
    out[:, 1] = out[:, 1] * stats["a_std"]
    return out


def _build_va_norm_stats(train_data: list[dict], enabled: bool) -> dict | None:
    if not enabled or not train_data:
        return None
    v = np.array([s["v"] for s in train_data if s.get("v") is not None], dtype=np.float32)
    a = np.array([s["a"] for s in train_data if s.get("a") is not None], dtype=np.float32)
    if len(v) == 0 or len(a) == 0:
        return None
    stats = {
        "v_mean": float(v.mean()),
        "v_std": max(float(v.std()), 1e-6),
        "a_mean": float(a.mean()),
        "a_std": max(float(a.std()), 1e-6),
    }
    logger.info(
        "VA target normalization: v=%.4f±%.4f a=%.4f±%.4f",
        stats["v_mean"], stats["v_std"], stats["a_mean"], stats["a_std"],
    )
    return stats


def _merge_auxiliary_data(cfg: dict, train_data: list[dict]) -> list[dict]:
    aux_cfg = cfg.get("auxiliary_data", {})
    merged = list(train_data)
    for sample in merged:
        sample.setdefault("_sample_weight", 1.0)
    if not aux_cfg.get("enabled", False):
        return merged

    n_before = len(merged)
    aux_counts: dict[str, int] = {}
    for src_cfg in aux_cfg.get("sources", []):
        aux_load_cfg = {
            "data_dir": src_cfg["data_dir"],
            "train_file": src_cfg.get("train_file", "train.jsonl"),
            "track": src_cfg.get("track", "A"),
            "lang": src_cfg.get("lang", "unknown"),
            "domain": src_cfg.get("domain", "unknown"),
        }
        aux_weight = float(src_cfg.get("weight", 1.0))
        aux_key = f"{aux_load_cfg['lang']}_{aux_load_cfg['domain']}"
        aux_samples = expand_to_aspect_level(load_samples(aux_load_cfg, "train"))
        aux_counts[aux_key] = len(aux_samples)
        for sample in aux_samples:
            sample["_is_auxiliary"] = True
            sample["_sample_weight"] = aux_weight
            sample["_aux_source"] = aux_key
        merged.extend(aux_samples)

    logger.info("Auxiliary data: %s -> %s aspect-level samples", n_before, len(merged))
    if aux_counts:
        logger.info("Auxiliary sources: %s", aux_counts)
    return merged


def _setup_teacher(cfg: dict, train_data: list[dict], run_dir: Path):
    teacher_cfg = cfg.get("teacher", {})
    if not teacher_cfg.get("enabled", False):
        return None, None

    source = teacher_cfg.get("artifact_paths") or teacher_cfg.get("artifact_path")
    if not source:
        raise ValueError("teacher.enabled=true but teacher.artifact_path/artifact_paths is missing")

    raw_map = load_teacher_map(source)
    sample_keys = [make_sample_key(s["id"], s["aspect_index"], s.get("lang")) for s in train_data]
    sample_langs = [s.get("lang", "unknown") for s in train_data]
    teacher_map, teacher_audit = join_teacher_to_samples(
        sample_keys=sample_keys,
        teacher_map=raw_map,
        lang_weights=teacher_cfg.get("lang_weights"),
        use_confidence=teacher_cfg.get("use_confidence", False),
        confidence_clip_min=teacher_cfg.get("confidence_clip_min", 0.0),
        confidence_clip_max=teacher_cfg.get("confidence_clip_max", 1.0),
        confidence_threshold=teacher_cfg.get("confidence_threshold", 0.0),
        sample_langs=sample_langs,
    )
    teacher_audit["teacher_sources"] = [str(p) for p in source] if isinstance(source, (list, tuple)) else [str(source)]
    teacher_audit["num_teacher_sources"] = len(teacher_audit["teacher_sources"])
    save_teacher_audit(teacher_audit, run_dir / "teacher_audit.json")
    logger.info(
        "Teacher join: matched=%s/%s ratio=%s",
        teacher_audit["num_teacher_matched"],
        teacher_audit["num_samples_total"],
        teacher_audit["teacher_match_ratio"],
    )
    return teacher_map, teacher_audit


def _normalize_checkpoint_canary_cfg(checkpoint_canary_cfg: dict | None) -> dict:
    raw = dict(checkpoint_canary_cfg or {})
    enabled = bool(raw.get("enabled", False))
    metric = str(raw.get("metric", "teacher_macro_rmse_va")).strip().lower() or "teacher_macro_rmse_va"
    aliases = {
        "teacher_macro": "teacher_macro_rmse_va",
        "teacher_macro_rmse": "teacher_macro_rmse_va",
        "teacher_macro_rmse_va": "teacher_macro_rmse_va",
        "teacher_mean": "teacher_rmse_va",
        "teacher_sample": "teacher_rmse_va",
        "teacher_rmse": "teacher_rmse_va",
        "teacher_rmse_va": "teacher_rmse_va",
        "label_macro": "label_macro_rmse_va",
        "label_macro_rmse": "label_macro_rmse_va",
        "label_macro_rmse_va": "label_macro_rmse_va",
        "label_mean": "label_rmse_va",
        "label_sample": "label_rmse_va",
        "label_rmse": "label_rmse_va",
        "label_rmse_va": "label_rmse_va",
    }
    metric = aliases.get(metric, metric)
    if metric not in {"teacher_macro_rmse_va", "teacher_rmse_va", "label_macro_rmse_va", "label_rmse_va"}:
        raise ValueError(
            "checkpoint_canary.metric must be one of "
            "{teacher_macro_rmse_va, teacher_rmse_va, label_macro_rmse_va, label_rmse_va}"
        )

    weight = float(raw.get("weight", raw.get("lambda", 0.0)))
    if weight < 0:
        raise ValueError("checkpoint_canary.weight must be >= 0")

    def _optional_float(name: str) -> float | None:
        value = raw.get(name)
        return None if value is None else float(value)

    max_score = _optional_float("max_score")
    max_increase = _optional_float("max_increase")
    if max_score is not None and max_score < 0:
        raise ValueError("checkpoint_canary.max_score must be >= 0")
    if max_increase is not None and max_increase < 0:
        raise ValueError("checkpoint_canary.max_increase must be >= 0")

    return {
        "enabled": enabled,
        "source": raw.get("artifact_paths") or raw.get("artifact_path"),
        "use_confidence": bool(raw.get("use_confidence", False)),
        "confidence_clip_min": float(raw.get("confidence_clip_min", 0.0)),
        "confidence_clip_max": float(raw.get("confidence_clip_max", 1.0)),
        "confidence_threshold": float(raw.get("confidence_threshold", 0.0)),
        "metric": metric,
        "weight": weight,
        "use_for_best": bool(raw.get("use_for_best", True)),
        "max_score": max_score,
        "max_increase": max_increase,
    }


def _setup_checkpoint_canary(checkpoint_canary_cfg: dict, train_data: list[dict], run_dir: Path):
    if not checkpoint_canary_cfg.get("enabled", False):
        return None, None, None

    source = checkpoint_canary_cfg.get("source")
    if not source:
        raise ValueError(
            "checkpoint_canary.enabled=true but checkpoint_canary.artifact_path/artifact_paths is missing"
        )

    sample_keys = [make_sample_key(s["id"], s["aspect_index"], s.get("lang")) for s in train_data]
    sample_langs = [s.get("lang", "unknown") for s in train_data]
    raw_map = load_teacher_map(source)
    canary_teacher_map, canary_audit = join_teacher_to_samples(
        sample_keys=sample_keys,
        teacher_map=raw_map,
        use_confidence=checkpoint_canary_cfg.get("use_confidence", False),
        confidence_clip_min=checkpoint_canary_cfg.get("confidence_clip_min", 0.0),
        confidence_clip_max=checkpoint_canary_cfg.get("confidence_clip_max", 1.0),
        confidence_threshold=checkpoint_canary_cfg.get("confidence_threshold", 0.0),
        sample_langs=sample_langs,
    )

    canary_samples = []
    group_counts: dict[str, int] = {}
    for sample, sample_key in zip(train_data, sample_keys):
        teacher_rec = canary_teacher_map.get(sample_key)
        if teacher_rec and teacher_rec.get("teacher_mask", 0):
            canary_samples.append(sample)
            group = _sample_group_name(sample)
            group_counts[group] = group_counts.get(group, 0) + 1

    canary_audit["canary_sources"] = [str(p) for p in source] if isinstance(source, (list, tuple)) else [str(source)]
    canary_audit["num_canary_sources"] = len(canary_audit["canary_sources"])
    canary_audit["num_canary_samples"] = len(canary_samples)
    canary_audit["groups"] = group_counts
    canary_audit["metric"] = checkpoint_canary_cfg["metric"]
    canary_audit["use_for_best"] = checkpoint_canary_cfg["use_for_best"]
    canary_audit["weight"] = checkpoint_canary_cfg["weight"]
    if checkpoint_canary_cfg.get("max_score") is not None:
        canary_audit["max_score"] = checkpoint_canary_cfg["max_score"]
    if checkpoint_canary_cfg.get("max_increase") is not None:
        canary_audit["max_increase"] = checkpoint_canary_cfg["max_increase"]

    (run_dir / "checkpoint_canary_audit.json").write_text(
        json.dumps(canary_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Checkpoint canary enabled: matched=%s/%s ratio=%s metric=%s use_for_best=%s weight=%.3f groups=%s",
        canary_audit["num_teacher_matched"],
        canary_audit["num_samples_total"],
        canary_audit["teacher_match_ratio"],
        checkpoint_canary_cfg["metric"],
        checkpoint_canary_cfg["use_for_best"],
        checkpoint_canary_cfg["weight"],
        group_counts,
    )
    return canary_samples, canary_teacher_map, canary_audit


def _resolve_artifact_path(path: str | Path) -> Path:
    artifact_path = Path(path)
    return artifact_path if artifact_path.is_absolute() else PROJECT_ROOT / artifact_path


def _setup_repair_pool(cfg: dict, train_data: list[dict], run_dir: Path):
    repair_cfg = cfg.get("repair_pool", {})
    if not repair_cfg.get("enabled", False):
        return None

    source = str(repair_cfg.get("source", "artifact")).strip().lower()
    if source in {"risk_gate", "risk"}:
        path = repair_cfg.get("artifact_path") or cfg.get("risk_gate", {}).get("weights_path")
    else:
        path = repair_cfg.get("artifact_path") or repair_cfg.get("weights_path")
    if not path:
        raise ValueError("repair_pool.enabled=true but repair_pool.artifact_path/weights_path is missing")

    weight_map = _load_weight_map(_resolve_artifact_path(path))
    mode = str(repair_cfg.get("mode", "hard_mask")).strip().lower()
    score_field = str(repair_cfg.get("score_field", "group_risk_rank")).strip()
    keep_quantile = repair_cfg.get("keep_quantile")
    min_score = repair_cfg.get("min_score")
    default_score = float(repair_cfg.get("default_score", 0.0))
    default_weight = float(repair_cfg.get("default_weight", 1.0))
    min_weight = float(repair_cfg.get("min_weight", 0.0))
    max_weight = float(repair_cfg.get("max_weight", 1.0))
    candidate_fields = tuple(
        str(v).strip()
        for v in repair_cfg.get("candidate_fields", ["repair_weight", "risk_weight", "weight"])
        if str(v).strip()
    ) or ("repair_weight", "risk_weight", "weight")

    hits = 0
    active = 0
    scores = []
    weights = []
    for sample in train_data:
        key = make_sample_key(sample["id"], sample["aspect_index"], sample.get("lang"))
        rec = weight_map.get(key)
        if rec is not None:
            hits += 1
        score = float(rec.get(score_field, default_score)) if rec is not None else default_score
        if keep_quantile is not None:
            is_active = score >= float(keep_quantile)
        elif min_score is not None:
            is_active = score >= float(min_score)
        else:
            is_active = rec is not None

        if mode == "soft_weight":
            repair_weight = _resolve_artifact_weight(rec, default_weight, candidate_fields)
            repair_weight = min(max(repair_weight, min_weight), max_weight)
            repair_weight = repair_weight if is_active else 0.0
        else:
            repair_weight = 1.0 if is_active else 0.0

        sample["_repair_mask"] = 1.0 if is_active else 0.0
        sample["_repair_weight"] = float(repair_weight)
        sample["_repair_score"] = score
        active += int(is_active)
        scores.append(score)
        weights.append(float(repair_weight))

    audit = {
        "source": source,
        "artifact_path": str(path),
        "score_field": score_field,
        "mode": mode,
        "num_samples": len(train_data),
        "matched": hits,
        "match_rate": round(hits / max(len(train_data), 1), 6),
        "active_samples": active,
        "active_rate": round(active / max(len(train_data), 1), 6),
        "keep_quantile": None if keep_quantile is None else float(keep_quantile),
        "min_score": None if min_score is None else float(min_score),
        "score_mean": round(float(np.mean(scores)), 6) if scores else 0.0,
        "score_std": round(float(np.std(scores)), 6) if scores else 0.0,
        "weight_mean": round(float(np.mean(weights)), 6) if weights else 0.0,
        "weight_std": round(float(np.std(weights)), 6) if weights else 0.0,
        "apply_to_base_loss": bool(repair_cfg.get("apply_to_base_loss", False)),
        "apply_to_residual_loss": bool(repair_cfg.get("apply_to_residual_loss", True)),
        "apply_to_group_routed_loss": bool(repair_cfg.get("apply_to_group_routed_loss", True)),
    }
    (run_dir / "repair_pool_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Repair pool enabled: matched=%s/%s rate=%.4f active=%s(%.4f) mode=%s score_field=%s",
        hits,
        len(train_data),
        audit["match_rate"],
        active,
        audit["active_rate"],
        mode,
        score_field,
    )
    return audit


def _setup_protected_pool(cfg: dict, train_data: list[dict], run_dir: Path):
    protected_cfg = cfg.get("protected_pool", {})
    if not protected_cfg.get("enabled", False):
        return None, None

    source = str(protected_cfg.get("source", "artifact")).strip().lower()
    artifact_source = (
        protected_cfg.get("artifact_paths")
        or protected_cfg.get("artifact_path")
        or protected_cfg.get("source_paths")
        or protected_cfg.get("source_path")
    )
    if artifact_source is None:
        if source in {"checkpoint_canary", "stable_teacher", "stable_canary_teacher"}:
            artifact_source = (
                cfg.get("checkpoint_canary", {}).get("artifact_paths")
                or cfg.get("checkpoint_canary", {}).get("artifact_path")
            )
        elif source == "teacher":
            artifact_source = cfg.get("teacher", {}).get("artifact_paths") or cfg.get("teacher", {}).get("artifact_path")
    if artifact_source is None:
        raise ValueError(
            "protected_pool.enabled=true but protected_pool artifact path is missing"
        )

    raw_map = load_teacher_map(artifact_source)
    sample_keys = [make_sample_key(s["id"], s["aspect_index"], s.get("lang")) for s in train_data]
    sample_langs = [s.get("lang", "unknown") for s in train_data]
    protected_map, protected_audit = join_teacher_to_samples(
        sample_keys=sample_keys,
        teacher_map=raw_map,
        use_confidence=protected_cfg.get("use_confidence", False),
        confidence_clip_min=protected_cfg.get("confidence_clip_min", 0.0),
        confidence_clip_max=protected_cfg.get("confidence_clip_max", 1.0),
        confidence_threshold=protected_cfg.get("confidence_threshold", 0.0),
        sample_langs=sample_langs,
    )

    active = 0
    group_counts: dict[str, int] = {}
    for sample, key in zip(train_data, sample_keys):
        rec = protected_map.get(key)
        is_active = bool(rec and rec.get("teacher_mask", 0))
        sample["_protected_mask"] = 1.0 if is_active else 0.0
        sample["_protected_weight"] = float(rec.get("teacher_weight", 1.0)) if is_active else 0.0
        sample["_protected_teacher_v"] = float(rec.get("teacher_v", 0.0)) if is_active else 0.0
        sample["_protected_teacher_a"] = float(rec.get("teacher_a", 0.0)) if is_active else 0.0
        if is_active:
            active += 1
            group = _sample_group_name(sample)
            group_counts[group] = group_counts.get(group, 0) + 1

    protected_audit["protected_sources"] = (
        [str(p) for p in artifact_source]
        if isinstance(artifact_source, (list, tuple))
        else [str(artifact_source)]
    )
    protected_audit["num_protected_sources"] = len(protected_audit["protected_sources"])
    protected_audit["num_protected_samples"] = active
    protected_audit["groups"] = group_counts
    (run_dir / "protected_pool_audit.json").write_text(
        json.dumps(protected_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Protected pool enabled: matched=%s/%s ratio=%s protected=%s groups=%s",
        protected_audit["num_teacher_matched"],
        protected_audit["num_samples_total"],
        protected_audit["teacher_match_ratio"],
        active,
        group_counts,
    )
    return protected_map, protected_audit


def _exclude_repair_pool_with_protected(
    cfg: dict,
    train_data: list[dict],
    run_dir: Path,
    repair_pool_audit: dict | None,
) -> dict | None:
    repair_cfg = cfg.get("repair_pool", {})
    if not repair_cfg.get("enabled", False):
        return repair_pool_audit
    if not repair_cfg.get("exclude_protected_pool", False):
        return repair_pool_audit

    removed = 0
    for sample in train_data:
        if float(sample.get("_repair_mask", 0.0)) <= 0.0:
            continue
        if float(sample.get("_protected_mask", 0.0)) <= 0.0:
            continue
        sample["_repair_mask"] = 0.0
        sample["_repair_weight"] = 0.0
        removed += 1

    if repair_pool_audit is None:
        repair_pool_audit = {}
    active = sum(1 for sample in train_data if float(sample.get("_repair_mask", 0.0)) > 0.0)
    weights = [float(sample.get("_repair_weight", 0.0)) for sample in train_data]
    repair_pool_audit["excluded_protected_overlap"] = int(removed)
    repair_pool_audit["active_samples"] = int(active)
    repair_pool_audit["active_rate"] = round(active / max(len(train_data), 1), 6)
    repair_pool_audit["weight_mean"] = round(float(np.mean(weights)), 6) if weights else 0.0
    repair_pool_audit["weight_std"] = round(float(np.std(weights)), 6) if weights else 0.0
    (run_dir / "repair_pool_audit.json").write_text(
        json.dumps(repair_pool_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Repair pool protected exclusion applied: removed_overlap=%s active=%s(%.4f)",
        removed,
        active,
        repair_pool_audit["active_rate"],
    )
    return repair_pool_audit


def _compute_post_ckpt_repair_adapter_scale(cfg: dict, global_update: int | None) -> float:
    adapter_cfg = cfg.get("post_ckpt_repair_adapter", {})
    if not adapter_cfg.get("enabled", False):
        return 1.0

    start_update = int(adapter_cfg.get("start_update", 0) or 0)
    warmup_updates = int(adapter_cfg.get("warmup_updates", 0) or 0)
    max_scale = max(float(adapter_cfg.get("max_scale", 1.0)), 0.0)
    current_update = int(global_update or 0)
    if current_update < start_update:
        return 0.0
    if warmup_updates <= 0:
        return max_scale
    progress = min(max((current_update - start_update + 1) / warmup_updates, 0.0), 1.0)
    return max_scale * progress


def _refresh_post_ckpt_repair_adapter_scale(model, cfg: dict, global_update: int | None) -> float:
    scale = _compute_post_ckpt_repair_adapter_scale(cfg, global_update)
    if hasattr(model, "set_post_ckpt_repair_adapter_scale"):
        model.set_post_ckpt_repair_adapter_scale(scale)
    return scale


def _resolve_translation_pivot_source(cfg: dict):
    translation_cfg = cfg.get("translation_pivot", {})
    return (
        translation_cfg.get("train_sidecar_paths")
        or translation_cfg.get("train_sidecar_path")
        or translation_cfg.get("sidecar_paths")
        or translation_cfg.get("sidecar_path")
    )


def _setup_translation_pivot(cfg: dict, train_data: list[dict], dev_data: list[dict], test_data: list[dict], run_dir: Path):
    translation_cfg = cfg.get("translation_pivot", {})
    if not translation_cfg.get("enabled", False):
        return None, None

    source = _resolve_translation_pivot_source(cfg)
    if not source:
        raise ValueError(
            "translation_pivot.enabled=true but translation_pivot.train_sidecar_path/"
            "train_sidecar_paths/sidecar_path is missing"
        )

    translation_map = load_translation_pivot_map(source)
    translation_audit = {
        "train": audit_translation_coverage(translation_map, train_data),
        "dev": audit_translation_coverage(translation_map, dev_data),
        "test": audit_translation_coverage(translation_map, test_data),
        "sidecar_sources": [str(p) for p in source] if isinstance(source, (list, tuple)) else [str(source)],
    }
    (run_dir / "translation_pivot_audit.json").write_text(
        json.dumps(translation_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Translation pivot coverage: train hit=%s non_eng_hit=%s sources=%s",
        translation_audit["train"]["hit_rate"],
        translation_audit["train"]["non_english_hit_rate"],
        translation_audit["sidecar_sources"],
    )
    return translation_map, translation_audit


def _merge_sample_weights(
    sample_weights: torch.Tensor | None,
    extra_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    if sample_weights is None:
        return extra_weights
    if extra_weights is None:
        return sample_weights
    return sample_weights * extra_weights


def _batch_pool_weights(batch: dict, prefix: str, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mask = batch.get(f"{prefix}_mask")
    weight = batch.get(f"{prefix}_weight")
    if mask is not None:
        mask = mask.to(device)
    if weight is not None:
        weight = weight.to(device)
    if weight is None:
        weight = mask
    elif mask is not None:
        weight = weight * mask
    return mask, weight


def _slice_va_focus(values: torch.Tensor, focus_dim: str) -> torch.Tensor:
    focus = str(focus_dim or "va").strip().lower()
    if focus == "v":
        return values[:, 0:1]
    if focus == "a":
        return values[:, 1:2]
    return values


def _prediction_consistency_loss(
    source_pred: torch.Tensor,
    pivot_pred: torch.Tensor,
    is_auxiliary: torch.Tensor | None,
    auxiliary_loss_weight: float,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    per_sample = F.smooth_l1_loss(source_pred, pivot_pred, reduction="none").mean(dim=-1)
    return combine_primary_aux_losses(
        per_sample,
        is_auxiliary=is_auxiliary,
        auxiliary_loss_weight=auxiliary_loss_weight,
        sample_weights=sample_weights,
    )


def _apply_metadata_output_residual_freeze(cfg: dict, model) -> dict:
    meta_out_cfg = cfg.get("metadata_output_residual", {})
    if not meta_out_cfg.get("enabled", False):
        return {}
    report = {
        "freeze_backbone": False,
        "freeze_head": False,
        "freeze_pool_gate": False,
        "backbone_params_frozen": 0,
        "head_params_frozen": 0,
        "pool_gate_params_frozen": 0,
    }
    if bool(meta_out_cfg.get("freeze_backbone", False)):
        for param in model.backbone.parameters():
            if param.requires_grad:
                report["backbone_params_frozen"] += param.numel()
                param.requires_grad = False
        report["freeze_backbone"] = True
    if bool(meta_out_cfg.get("freeze_head", meta_out_cfg.get("freeze_base_head", False))):
        base_head = getattr(getattr(model, "head", None), "base_head", None)
        if base_head is not None:
            for param in base_head.parameters():
                if param.requires_grad:
                    report["head_params_frozen"] += param.numel()
                    param.requires_grad = False
            report["freeze_head"] = True
    if bool(meta_out_cfg.get("freeze_pool_gate", False)):
        pool_gate = getattr(model, "pool_gate", None)
        if pool_gate is not None:
            for param in pool_gate.parameters():
                if param.requires_grad:
                    report["pool_gate_params_frozen"] += param.numel()
                    param.requires_grad = False
            report["freeze_pool_gate"] = True
    return report


def _apply_factorized_output_residual_freeze(cfg: dict, model) -> dict:
    residual_cfg = cfg.get("factorized_output_residual", {})
    if not residual_cfg.get("enabled", False):
        return {}
    report = {
        "freeze_backbone": False,
        "freeze_head": False,
        "freeze_pool_gate": False,
        "backbone_params_frozen": 0,
        "head_params_frozen": 0,
        "pool_gate_params_frozen": 0,
    }
    head = getattr(model, "head", None)

    def _freeze_module(module, report_key: str) -> None:
        if module is None:
            return
        frozen = 0
        for param in module.parameters():
            if param.requires_grad:
                frozen += param.numel()
                param.requires_grad = False
        if frozen > 0:
            report[report_key] = report.get(report_key, 0) + frozen

    if bool(residual_cfg.get("freeze_backbone", False)):
        for param in model.backbone.parameters():
            if param.requires_grad:
                report["backbone_params_frozen"] += param.numel()
                param.requires_grad = False
        report["freeze_backbone"] = True
    if bool(residual_cfg.get("freeze_head", residual_cfg.get("freeze_base_head", False))):
        base_head = getattr(getattr(model, "head", None), "base_head", None)
        if base_head is not None:
            for param in base_head.parameters():
                if param.requires_grad:
                    report["head_params_frozen"] += param.numel()
                    param.requires_grad = False
            report["freeze_head"] = True
    if bool(residual_cfg.get("freeze_pool_gate", False)):
        pool_gate = getattr(model, "pool_gate", None)
        if pool_gate is not None:
            for param in pool_gate.parameters():
                if param.requires_grad:
                    report["pool_gate_params_frozen"] += param.numel()
                    param.requires_grad = False
            report["freeze_pool_gate"] = True
    if isinstance(head, FactorizedOutputResidualHead):
        if bool(residual_cfg.get("freeze_shared_branch", False)):
            _freeze_module(head.shared_trunk, "shared_branch_params_frozen")
            _freeze_module(head.shared_out, "shared_branch_params_frozen")
            report["freeze_shared_branch"] = True
        if bool(residual_cfg.get("freeze_lang_branch", False)):
            _freeze_module(head.lang_trunk, "lang_branch_params_frozen")
            _freeze_module(head.lang_out, "lang_branch_params_frozen")
            report["freeze_lang_branch"] = True
        if bool(residual_cfg.get("freeze_domain_branch", False)):
            _freeze_module(head.domain_trunk, "domain_branch_params_frozen")
            _freeze_module(head.domain_out, "domain_branch_params_frozen")
            report["freeze_domain_branch"] = True
        if bool(residual_cfg.get("freeze_composition_branch", False)):
            _freeze_module(head.comp_trunk, "composition_branch_params_frozen")
            _freeze_module(head.comp_out, "composition_branch_params_frozen")
            report["freeze_composition_branch"] = True
        if bool(residual_cfg.get("freeze_gate_proj", False)):
            _freeze_module(head.gate_proj, "gate_proj_params_frozen")
            report["freeze_gate_proj"] = True
        if bool(residual_cfg.get("freeze_condition_embeddings", False)):
            _freeze_module(head.lang_embedding, "condition_embedding_params_frozen")
            _freeze_module(head.domain_embedding, "condition_embedding_params_frozen")
            report["freeze_condition_embeddings"] = True
        if bool(residual_cfg.get("freeze_soft_condition_proj", False)):
            _freeze_module(head.soft_condition_proj, "soft_condition_proj_params_frozen")
            report["freeze_soft_condition_proj"] = True
    return report


def _apply_text_routing_detector_freeze(cfg: dict, model) -> dict:
    detector_cfg = cfg.get("text_routing_detector", {})
    if not detector_cfg.get("enabled", False):
        return {}
    if not bool(detector_cfg.get("freeze", False)):
        return {}
    detector = getattr(model, "text_routing_detector", None)
    if detector is None:
        return {}
    frozen = 0
    for param in detector.parameters():
        if param.requires_grad:
            frozen += param.numel()
            param.requires_grad = False
    return {
        "freeze_text_routing_detector": True,
        "text_routing_detector_params_frozen": frozen,
    }


def _apply_backbone_conditional_adapter_freeze(cfg: dict, model) -> dict:
    adapter_cfg = cfg.get("backbone_conditional_adapter", {})
    if not adapter_cfg.get("enabled", False):
        return {}
    report = {
        "freeze_source": "backbone_conditional_adapter",
        "freeze_backbone": bool(adapter_cfg.get("freeze_backbone", False)),
        "freeze_head": bool(adapter_cfg.get("freeze_head", False)),
        "freeze_pool_gate": bool(adapter_cfg.get("freeze_pool_gate", False)),
        "keep_adapter_trainable": bool(adapter_cfg.get("keep_adapter_trainable", True)),
        "backbone_params_frozen": 0,
        "head_params_frozen": 0,
        "pool_gate_params_frozen": 0,
        "adapter_params_kept": 0,
    }
    keep_adapter_trainable = report["keep_adapter_trainable"]

    if report["freeze_backbone"]:
        for name, param in model.backbone.named_parameters():
            if keep_adapter_trainable and (".adapter." in name or ".repair_adapter." in name):
                report["adapter_params_kept"] += param.numel()
                continue
            if param.requires_grad:
                report["backbone_params_frozen"] += param.numel()
                param.requires_grad = False
    if report["freeze_head"]:
        for param in model.head.parameters():
            if param.requires_grad:
                report["head_params_frozen"] += param.numel()
                param.requires_grad = False
    if report["freeze_pool_gate"] and getattr(model, "pool_gate", None) is not None:
        for param in model.pool_gate.parameters():
            if param.requires_grad:
                report["pool_gate_params_frozen"] += param.numel()
                param.requires_grad = False
    return report


def _feature_consistency_loss(
    source_features: torch.Tensor,
    pivot_features: torch.Tensor,
    is_auxiliary: torch.Tensor | None,
    auxiliary_loss_weight: float,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    per_sample = 1.0 - F.cosine_similarity(source_features, pivot_features, dim=-1)
    return combine_primary_aux_losses(
        per_sample,
        is_auxiliary=is_auxiliary,
        auxiliary_loss_weight=auxiliary_loss_weight,
        sample_weights=sample_weights,
    )


def _build_pivot_pair_features(
    source_pred_raw: torch.Tensor,
    pivot_pred_raw: torch.Tensor | None,
    source_features: torch.Tensor | None,
    pivot_features: torch.Tensor | None,
    pivot_confidence: torch.Tensor | None,
    *,
    detach_inputs: bool = False,
) -> torch.Tensor | None:
    if (
        pivot_pred_raw is None
        or source_features is None
        or pivot_features is None
        or pivot_confidence is None
    ):
        return None

    src_a = source_pred_raw[:, 1:2]
    piv_a = pivot_pred_raw[:, 1:2]
    abs_delta_a = torch.abs(src_a - piv_a)
    cos_sim = F.cosine_similarity(source_features, pivot_features, dim=-1).unsqueeze(-1)
    conf = pivot_confidence.to(dtype=src_a.dtype).unsqueeze(-1)
    pair_features = torch.cat([src_a, piv_a, abs_delta_a, cos_sim, conf], dim=-1)
    return pair_features.detach() if detach_inputs else pair_features


def _build_pivot_va_pair_features(
    source_pred_raw: torch.Tensor,
    pivot_pred_raw: torch.Tensor | None,
    source_features: torch.Tensor | None,
    pivot_features: torch.Tensor | None,
    pivot_confidence: torch.Tensor | None,
    *,
    detach_inputs: bool = False,
) -> torch.Tensor | None:
    if (
        pivot_pred_raw is None
        or source_features is None
        or pivot_features is None
        or pivot_confidence is None
    ):
        return None

    src_v = source_pred_raw[:, 0:1]
    src_a = source_pred_raw[:, 1:2]
    piv_v = pivot_pred_raw[:, 0:1]
    piv_a = pivot_pred_raw[:, 1:2]
    abs_delta_v = torch.abs(src_v - piv_v)
    abs_delta_a = torch.abs(src_a - piv_a)
    cos_sim = F.cosine_similarity(source_features, pivot_features, dim=-1).unsqueeze(-1)
    conf = pivot_confidence.to(dtype=src_v.dtype).unsqueeze(-1)
    pair_features = torch.cat(
        [src_v, src_a, piv_v, piv_a, abs_delta_v, abs_delta_a, cos_sim, conf],
        dim=-1,
    )
    return pair_features.detach() if detach_inputs else pair_features


def _compute_learned_pivot_gate_multiplier(
    gate_cfg: dict,
    pair_features: torch.Tensor | None,
    pivot_gate_head,
    pivot_sample_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if pair_features is None or pivot_gate_head is None or pivot_sample_mask is None:
        return None
    gate_logits = pivot_gate_head(pair_features)
    gate_score = torch.sigmoid(gate_logits)
    min_multiplier = max(float(gate_cfg.get("min_multiplier", 1.0)), 1.0)
    max_multiplier = max(float(gate_cfg.get("max_multiplier", 2.0)), min_multiplier)
    multiplier = min_multiplier + (max_multiplier - min_multiplier) * gate_score
    active_mask = (pivot_sample_mask > 0.5).to(multiplier.dtype)
    return multiplier * active_mask + (1.0 - active_mask)


def _build_pivot_hardness_gate_weights(
    gate_cfg: dict,
    global_update: int,
    pivot_sample_mask: torch.Tensor | None,
    pivot_confidence: torch.Tensor | None,
    source_pred_raw: torch.Tensor,
    pivot_pred_raw: torch.Tensor | None,
) -> torch.Tensor | None:
    if not gate_cfg.get("enabled", False):
        return None
    start_update = max(int(gate_cfg.get("start_update", 0) or 0), 0)
    if global_update < start_update or pivot_sample_mask is None:
        return None

    active_mask = pivot_sample_mask > 0.5
    if float(active_mask.sum().item()) <= 0.0:
        return None

    dtype = source_pred_raw.dtype
    base = torch.ones_like(pivot_sample_mask, dtype=dtype)
    active = active_mask.to(dtype)

    use_confidence = bool(gate_cfg.get("use_confidence", True))
    use_prediction_disagreement = bool(gate_cfg.get("use_prediction_disagreement", True))
    combine_mode = str(gate_cfg.get("combine", "product")).strip().lower()

    if use_confidence and pivot_confidence is not None:
        confidence_norm = max(float(gate_cfg.get("confidence_norm", 1.0)), 1e-6)
        confidence_floor = max(float(gate_cfg.get("confidence_floor", 0.0)), 0.0)
        confidence_score = torch.clamp(pivot_confidence.to(dtype), min=confidence_floor) / confidence_norm
        confidence_score = torch.clamp(confidence_score, 0.0, 1.0)
    else:
        confidence_score = torch.ones_like(base)

    if use_prediction_disagreement and pivot_pred_raw is not None:
        focus_dim = str(gate_cfg.get("focus_dim", "a")).strip().lower()
        if focus_dim == "va":
            disagreement = torch.mean(torch.abs(source_pred_raw - pivot_pred_raw), dim=-1)
        else:
            disagreement = torch.abs(source_pred_raw[:, 1] - pivot_pred_raw[:, 1])
        disagreement_temperature = max(float(gate_cfg.get("disagreement_temperature", 0.5)), 1e-6)
        disagreement_score = torch.tanh(disagreement / disagreement_temperature)
    else:
        disagreement_score = torch.ones_like(base)

    if use_confidence and use_prediction_disagreement:
        if combine_mode == "mean":
            hardness = 0.5 * (confidence_score + disagreement_score)
        else:
            hardness = confidence_score * disagreement_score
    elif use_confidence:
        hardness = confidence_score
    elif use_prediction_disagreement:
        hardness = disagreement_score
    else:
        hardness = torch.zeros_like(base)

    min_multiplier = max(float(gate_cfg.get("min_multiplier", 1.0)), 1.0)
    max_multiplier = max(float(gate_cfg.get("max_multiplier", 2.0)), min_multiplier)
    multiplier = min_multiplier + (max_multiplier - min_multiplier) * torch.clamp(hardness, 0.0, 1.0)
    multiplier = torch.where(active_mask, multiplier, base)
    return multiplier * active + (1.0 - active)


def _compute_pivot_gate_multiplier(
    gate_cfg: dict,
    global_update: int,
    pivot_sample_mask: torch.Tensor | None,
    pivot_confidence: torch.Tensor | None,
    source_pred_raw: torch.Tensor,
    pivot_pred_raw: torch.Tensor | None,
    *,
    source_features: torch.Tensor | None = None,
    pivot_features: torch.Tensor | None = None,
    pivot_gate_head=None,
) -> torch.Tensor | None:
    mode = str(gate_cfg.get("mode", "heuristic")).strip().lower()
    if mode == "learned":
        pair_features = _build_pivot_pair_features(
            source_pred_raw,
            pivot_pred_raw,
            source_features,
            pivot_features,
            pivot_confidence,
            detach_inputs=bool(gate_cfg.get("detach_inputs", True)),
        )
        return _compute_learned_pivot_gate_multiplier(
            gate_cfg,
            pair_features,
            pivot_gate_head,
            pivot_sample_mask,
        )
    return _build_pivot_hardness_gate_weights(
        gate_cfg,
        global_update=global_update,
        pivot_sample_mask=pivot_sample_mask,
        pivot_confidence=pivot_confidence,
        source_pred_raw=source_pred_raw,
        pivot_pred_raw=pivot_pred_raw,
    )


def _build_pivot_residual_prediction(
    residual_cfg: dict,
    pivot_residual_head,
    pivot_sample_mask: torch.Tensor | None,
    source_pred_raw: torch.Tensor,
    pivot_pred_raw: torch.Tensor | None,
    source_features: torch.Tensor | None,
    pivot_features: torch.Tensor | None,
    pivot_confidence: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if pivot_residual_head is None or pivot_sample_mask is None:
        return source_pred_raw, None

    pair_features = _build_pivot_pair_features(
        source_pred_raw,
        pivot_pred_raw,
        source_features,
        pivot_features,
        pivot_confidence,
        detach_inputs=bool(residual_cfg.get("detach_inputs", True)),
    )
    if pair_features is None:
        return source_pred_raw, None

    delta_a = pivot_residual_head(pair_features)
    clamp_delta = float(residual_cfg.get("clamp_delta", 1.0))
    if clamp_delta > 0:
        delta_a = torch.clamp(delta_a, -clamp_delta, clamp_delta)

    corrected = source_pred_raw.clone()
    active_mask = (pivot_sample_mask > 0.5)
    corrected_a = corrected[:, 1] + delta_a
    corrected[:, 1] = torch.where(
        active_mask,
        corrected_a.clamp(1.0, 9.0),
        corrected[:, 1],
    )
    return corrected, delta_a


def _build_pivot_trust_residual_prediction(
    trust_cfg: dict,
    pivot_trust_head,
    pivot_sample_mask: torch.Tensor | None,
    source_pred_raw: torch.Tensor,
    pivot_pred_raw: torch.Tensor | None,
    source_features: torch.Tensor | None,
    pivot_features: torch.Tensor | None,
    pivot_confidence: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if pivot_trust_head is None or pivot_sample_mask is None:
        return source_pred_raw, None, None

    pair_features = _build_pivot_pair_features(
        source_pred_raw,
        pivot_pred_raw,
        source_features,
        pivot_features,
        pivot_confidence,
        detach_inputs=bool(trust_cfg.get("detach_inputs", True)),
    )
    if pair_features is None:
        return source_pred_raw, None, None

    gate_logits, delta_a = pivot_trust_head(pair_features)
    gate = torch.sigmoid(gate_logits)
    clamp_delta = float(trust_cfg.get("clamp_delta", 1.0))
    if clamp_delta > 0:
        delta_a = torch.clamp(delta_a, -clamp_delta, clamp_delta)

    correction = gate * delta_a
    corrected = source_pred_raw.clone()
    active_mask = (pivot_sample_mask > 0.5)
    corrected_a = corrected[:, 1] + correction
    corrected[:, 1] = torch.where(
        active_mask,
        corrected_a.clamp(1.0, 9.0),
        corrected[:, 1],
    )
    return corrected, gate, correction


def _build_va_residual_prediction(
    residual_cfg: dict,
    va_residual_head,
    source_pred_raw: torch.Tensor,
    source_features: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if va_residual_head is None:
        return source_pred_raw, None, None

    if source_features is None:
        return source_pred_raw, None, None

    detach_inputs = bool(residual_cfg.get("detach_inputs", False))
    source_inputs = source_features.detach() if detach_inputs else source_features
    pred_inputs = source_pred_raw.detach() if detach_inputs else source_pred_raw
    delta_raw, gate_logits = va_residual_head(source_features=source_inputs, source_pred_raw=pred_inputs)
    clamp_delta = float(residual_cfg.get("clamp_delta", 1.0))
    bound_mode = str(residual_cfg.get("bound_mode", "tanh")).strip().lower()
    if clamp_delta > 0:
        if bound_mode == "clip":
            delta_va = torch.clamp(delta_raw, -clamp_delta, clamp_delta)
        else:
            delta_va = torch.tanh(delta_raw) * clamp_delta
    else:
        delta_va = delta_raw

    gate = None
    if bool(residual_cfg.get("learned_gate", True)) and gate_logits is not None:
        gate_temperature = max(float(residual_cfg.get("gate_temperature", 1.0)), 1e-6)
        gate = torch.sigmoid(gate_logits / gate_temperature)
        if gate.shape[-1] == 1 and delta_va.shape[-1] > 1:
            gate = gate.expand_as(delta_va)
    else:
        gate = torch.ones_like(delta_va)

    focus_dim = str(residual_cfg.get("focus_dim", "va")).strip().lower()
    if focus_dim == "a":
        delta_va = torch.stack([torch.zeros_like(delta_va[:, 0]), delta_va[:, 1]], dim=-1)
        gate = torch.stack([torch.zeros_like(gate[:, 0]), gate[:, 1]], dim=-1)
    elif focus_dim == "v":
        delta_va = torch.stack([delta_va[:, 0], torch.zeros_like(delta_va[:, 1])], dim=-1)
        gate = torch.stack([gate[:, 0], torch.zeros_like(gate[:, 1])], dim=-1)

    correction = gate * delta_va

    corrected = (source_pred_raw + correction).clamp(1.0, 9.0)
    return corrected, correction, gate


def _build_group_routed_va_residual_prediction(
    group_cfg: dict,
    group_routed_residual_head,
    batch_lang,
    batch_domain,
    source_pred_raw: torch.Tensor,
    source_features: torch.Tensor | None,
    pivot_pair_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if group_routed_residual_head is None or source_features is None:
        return source_pred_raw, None

    lang_ids, domain_ids = group_routed_residual_head.encode_group_ids(
        batch_lang,
        batch_domain,
        device=source_pred_raw.device,
    )
    source_inputs = source_features.detach() if bool(group_cfg.get("detach_inputs", False)) else source_features
    source_pred_inputs = source_pred_raw.detach() if bool(group_cfg.get("detach_inputs", False)) else source_pred_raw
    pair_inputs = pivot_pair_features.detach() if (
        pivot_pair_features is not None and bool(group_cfg.get("detach_inputs", False))
    ) else pivot_pair_features

    delta_va = group_routed_residual_head(
        source_features=source_inputs,
        source_pred_raw=source_pred_inputs,
        lang_ids=lang_ids,
        domain_ids=domain_ids,
        pair_features=pair_inputs,
    )
    clamp_delta = float(group_cfg.get("clamp_delta", 1.0))
    if clamp_delta > 0:
        delta_va = torch.clamp(delta_va, -clamp_delta, clamp_delta)

    corrected = (source_pred_raw + delta_va).clamp(1.0, 9.0)
    return corrected, delta_va


def _explicit_metadata_allowed(cfg: dict) -> bool:
    return not bool(cfg.get("disable_explicit_metadata", False))


def _resolve_model_condition_kwargs(batch: dict, cfg: dict) -> dict:
    if cfg.get("text_only_inference", False):
        # Label supervision reads lang_metadata/domain_metadata separately.
        # Neither real labels nor placeholder metadata are model inputs here.
        return {}
    if not _explicit_metadata_allowed(cfg):
        # When explicit metadata is disabled, the collate path already scrubs
        # lang/domain into placeholder tokens. Keep passing those placeholders
        # through so metadata-conditioned legacy heads/adapters fall back to
        # __unk__ routing instead of crashing on missing batch dimensions.
        return {
            "batch_lang": batch.get("lang"),
            "batch_domain": batch.get("domain"),
        }
    return {
        "batch_lang": batch.get("lang"),
        "batch_domain": batch.get("domain"),
    }


def _unwrap_training_model(model):
    return model.module if hasattr(model, "module") else model


def _aggregate_backbone_conditional_forward_cache(model, detach: bool = False) -> dict[str, torch.Tensor]:
    wrapped_model = _unwrap_training_model(model)
    adapters = getattr(wrapped_model, "backbone_conditional_adapters", None) or []
    if not adapters:
        return {}

    buckets: dict[str, list[torch.Tensor]] = {}
    for adapter in adapters:
        if not hasattr(adapter, "forward_cache"):
            continue
        for key, value in adapter.forward_cache(detach=detach).items():
            if torch.is_tensor(value):
                buckets.setdefault(key, []).append(value)

    return {
        key: torch.stack(values, dim=0).mean(dim=0)
        for key, values in buckets.items()
        if values
    }


def _aggregate_factorized_output_forward_cache(model, detach: bool = False) -> dict[str, torch.Tensor]:
    wrapped_model = _unwrap_training_model(model)
    head = getattr(wrapped_model, "head", None)
    cache = dict(getattr(head, "_last_cache", {}) or {})
    if not detach:
        return cache
    detached: dict[str, torch.Tensor] = {}
    for key, value in cache.items():
        if torch.is_tensor(value):
            detached[key] = value.detach()
        else:
            detached[key] = value
    return detached


def _aggregate_post_ckpt_repair_forward_cache(model, detach: bool = False) -> dict[str, torch.Tensor]:
    wrapped_model = _unwrap_training_model(model)
    adapters = getattr(wrapped_model, "post_ckpt_repair_adapters", None) or []
    if not adapters:
        return {}

    buckets: dict[str, list[torch.Tensor]] = {}
    for adapter in adapters:
        if not hasattr(adapter, "forward_cache"):
            continue
        for key, value in adapter.forward_cache(detach=detach).items():
            if torch.is_tensor(value):
                buckets.setdefault(key, []).append(value)

    return {
        key: torch.stack(values, dim=0).mean(dim=0)
        for key, values in buckets.items()
        if values
    }


def _load_yaml_cfg(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw or {})


def _resolve_project_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_checkpoint_teacher_from_block(block_cfg: dict, *, block_name: str) -> tuple[dict, Path]:
    teacher_run = _resolve_project_path(block_cfg.get("teacher_checkpoint_run"))
    teacher_cfg_path = _resolve_project_path(block_cfg.get("teacher_config_path"))
    teacher_ckpt_path = _resolve_project_path(block_cfg.get("teacher_checkpoint_path"))

    if teacher_run is not None:
        if not teacher_run.is_dir():
            raise ValueError(f"{block_name}.teacher_checkpoint_run not found: {teacher_run}")
        cfg_path = teacher_run / "config.yaml"
        ckpt_path = teacher_ckpt_path or (teacher_run / "checkpoint.pt")
        if not cfg_path.exists():
            raise ValueError(f"{block_name} teacher config missing: {cfg_path}")
        if not ckpt_path.exists():
            raise ValueError(f"{block_name} teacher checkpoint missing: {ckpt_path}")
        return _load_yaml_cfg(cfg_path), ckpt_path

    if teacher_cfg_path is None or teacher_ckpt_path is None:
        raise ValueError(
            f"{block_name} requires either teacher_checkpoint_run or both "
            "teacher_config_path and teacher_checkpoint_path"
        )
    if not teacher_cfg_path.exists():
        raise ValueError(f"{block_name}.teacher_config_path not found: {teacher_cfg_path}")
    if not teacher_ckpt_path.exists():
        raise ValueError(f"{block_name}.teacher_checkpoint_path not found: {teacher_ckpt_path}")
    return _load_yaml_cfg(teacher_cfg_path), teacher_ckpt_path


def _resolve_meta_bootstrap_teacher(cfg: dict) -> tuple[dict, Path]:
    bootstrap_cfg = dict(cfg.get("meta_soft_bootstrap", {}) or {})
    return _resolve_checkpoint_teacher_from_block(
        bootstrap_cfg,
        block_name="meta_soft_bootstrap",
    )


def _setup_meta_soft_bootstrap(cfg: dict, device: torch.device):
    bootstrap_cfg = dict(cfg.get("meta_soft_bootstrap", {}) or {})
    if not bootstrap_cfg.get("enabled", False):
        return None

    teacher_cfg, teacher_ckpt_path = _resolve_meta_bootstrap_teacher(cfg)
    teacher_model, _, _ = build_small_llm_regressor(teacher_cfg)
    teacher_device_name = str(bootstrap_cfg.get("teacher_device", "same")).strip().lower()
    teacher_device = device if teacher_device_name in {"same", "auto", ""} else torch.device(teacher_device_name)
    teacher_model.to(teacher_device)
    _, load_report = _load_checkpoint(
        teacher_model,
        teacher_ckpt_path,
        cfg=teacher_cfg,
        load_backbone=True,
        load_head="always",
    )
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    logger.info(
        "Meta-soft bootstrap enabled: teacher_ckpt=%s teacher_device=%s gate_weight=%.4f condition_weight=%.4f delta_weight=%.4f",
        teacher_ckpt_path,
        teacher_device,
        float(bootstrap_cfg.get("gate_weight", 0.0)),
        float(bootstrap_cfg.get("condition_weight", 0.0)),
        float(bootstrap_cfg.get("delta_weight", 0.0)),
    )
    logger.info("Meta-soft bootstrap teacher load report: %s", load_report)
    return {
        "cfg": bootstrap_cfg,
        "teacher_cfg": teacher_cfg,
        "teacher_model": teacher_model,
        "teacher_device": teacher_device,
        "teacher_checkpoint_path": str(teacher_ckpt_path),
        "load_report": load_report,
    }


def _setup_stage1_pooled_feature_anchor(cfg: dict, device: torch.device):
    anchor_cfg = dict(cfg.get("stage1_pooled_feature_anchor", {}) or {})
    if not anchor_cfg.get("enabled", False):
        return None

    teacher_cfg, teacher_ckpt_path = _resolve_checkpoint_teacher_from_block(
        anchor_cfg,
        block_name="stage1_pooled_feature_anchor",
    )
    teacher_model, _, _ = build_small_llm_regressor(teacher_cfg)
    teacher_device_name = str(anchor_cfg.get("teacher_device", "same")).strip().lower()
    teacher_device = device if teacher_device_name in {"same", "auto", ""} else torch.device(teacher_device_name)
    teacher_model.to(teacher_device)
    _, load_report = _load_checkpoint(
        teacher_model,
        teacher_ckpt_path,
        cfg=teacher_cfg,
        load_backbone=True,
        load_head="always",
    )
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    logger.info(
        "Stage1 pooled-feature anchor enabled: teacher_ckpt=%s teacher_device=%s loss_weight=%.4f cosine_weight=%.4f mse_weight=%.4f normalize=%s start_update=%s",
        teacher_ckpt_path,
        teacher_device,
        float(anchor_cfg.get("loss_weight", 0.0)),
        float(anchor_cfg.get("cosine_weight", 1.0)),
        float(anchor_cfg.get("mse_weight", 0.0)),
        bool(anchor_cfg.get("normalize", True)),
        int(anchor_cfg.get("start_update", 0) or 0),
    )
    logger.info("Stage1 pooled-feature anchor teacher load report: %s", load_report)
    return {
        "cfg": anchor_cfg,
        "teacher_cfg": teacher_cfg,
        "teacher_model": teacher_model,
        "teacher_device": teacher_device,
        "teacher_checkpoint_path": str(teacher_ckpt_path),
        "load_report": load_report,
    }


def _compute_meta_soft_bootstrap_loss(
    meta_state,
    model,
    batch: dict,
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
    student_cache: dict[str, torch.Tensor] | None = None,
):
    zero = torch.tensor(0.0, device=device)
    if meta_state is None:
        return zero, {}

    bootstrap_cfg = meta_state["cfg"]
    student_cache = student_cache or _aggregate_backbone_conditional_forward_cache(model, detach=False)
    if not student_cache:
        return zero, {}

    teacher_model = meta_state["teacher_model"]
    teacher_device = meta_state["teacher_device"]
    teacher_ids = ids if ids.device == teacher_device else ids.to(teacher_device)
    teacher_mask = mask if mask.device == teacher_device else mask.to(teacher_device)
    teacher_condition_kwargs = {
        "batch_lang": batch.get("lang_metadata", batch.get("lang")),
        "batch_domain": batch.get("domain_metadata", batch.get("domain")),
    }
    with torch.no_grad():
        teacher_model.encode(
            teacher_ids,
            teacher_mask,
            **teacher_condition_kwargs,
        )
        teacher_cache = _aggregate_backbone_conditional_forward_cache(teacher_model, detach=True)
    if not teacher_cache:
        return zero, {}

    loss = torch.tensor(0.0, device=device)
    stats: dict[str, float] = {}

    def _loss_for_keys(keys: list[str], weight: float, prefix: str) -> None:
        nonlocal loss
        if weight <= 0:
            return
        terms = []
        for key in keys:
            student_value = student_cache.get(key)
            teacher_value = teacher_cache.get(key)
            if student_value is None or teacher_value is None:
                continue
            teacher_value = teacher_value.to(device=device, dtype=student_value.dtype)
            term = F.mse_loss(student_value, teacher_value, reduction="mean")
            terms.append(term)
            stats[f"{prefix}_{key}"] = float(term.detach().cpu())
        if terms:
            group_loss = torch.stack(terms).mean()
            loss = loss + float(weight) * group_loss
            stats[f"{prefix}_loss"] = float(group_loss.detach().cpu())

    _loss_for_keys(
        ["gate_lang", "gate_domain", "gate_composition"],
        float(bootstrap_cfg.get("gate_weight", 0.0)),
        "gate",
    )
    _loss_for_keys(
        ["lang_emb", "domain_emb"],
        float(bootstrap_cfg.get("condition_weight", 0.0)),
        "condition",
    )
    _loss_for_keys(
        [
            "shared_delta_pooled",
            "lang_delta_pooled",
            "domain_delta_pooled",
            "composition_delta_pooled",
            "total_delta_pooled",
        ],
        float(bootstrap_cfg.get("delta_weight", 0.0)),
        "delta",
    )

    return float(bootstrap_cfg.get("loss_weight", 1.0)) * loss, stats


def _compute_stage1_pooled_feature_anchor_loss(
    anchor_state,
    batch: dict,
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
    student_features: torch.Tensor | None,
    is_auxiliary: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    auxiliary_loss_weight: float = 1.0,
    global_update: int | None = None,
):
    zero = torch.tensor(0.0, device=device)
    if anchor_state is None:
        return zero, {}

    anchor_cfg = anchor_state["cfg"]
    if int(global_update or 0) < int(anchor_cfg.get("start_update", 0) or 0):
        return zero, {}
    if student_features is None:
        return zero, {}

    teacher_model = anchor_state["teacher_model"]
    teacher_cfg = anchor_state["teacher_cfg"]
    teacher_device = anchor_state["teacher_device"]
    teacher_ids = ids if ids.device == teacher_device else ids.to(teacher_device)
    teacher_mask = mask if mask.device == teacher_device else mask.to(teacher_device)
    teacher_condition_kwargs = _resolve_model_condition_kwargs(batch, teacher_cfg)

    with torch.no_grad():
        teacher_features = teacher_model.encode(
            teacher_ids,
            teacher_mask,
            **teacher_condition_kwargs,
        )

    teacher_features = teacher_features.to(device=device, dtype=student_features.dtype)
    student_view = student_features.float()
    teacher_view = teacher_features.float()
    if bool(anchor_cfg.get("normalize", True)):
        student_view = F.normalize(student_view, dim=-1)
        teacher_view = F.normalize(teacher_view, dim=-1)

    cosine_weight = float(anchor_cfg.get("cosine_weight", 1.0))
    mse_weight = float(anchor_cfg.get("mse_weight", 0.0))
    if cosine_weight <= 0 and mse_weight <= 0:
        return zero, {}

    per_sample = torch.zeros(student_view.size(0), device=device, dtype=student_view.dtype)
    stats: dict[str, float] = {}
    if cosine_weight > 0:
        cosine_loss = 1.0 - F.cosine_similarity(student_view, teacher_view, dim=-1)
        per_sample = per_sample + cosine_weight * cosine_loss
        stats["cosine_loss"] = float(cosine_loss.mean().detach().cpu())
    if mse_weight > 0:
        mse_loss = F.mse_loss(student_view, teacher_view, reduction="none").mean(dim=-1)
        per_sample = per_sample + mse_weight * mse_loss
        stats["mse_loss"] = float(mse_loss.mean().detach().cpu())

    raw_loss = combine_primary_aux_losses(
        per_sample,
        is_auxiliary=is_auxiliary,
        auxiliary_loss_weight=auxiliary_loss_weight,
        sample_weights=sample_weights,
    )
    weighted_loss = float(anchor_cfg.get("loss_weight", 1.0)) * raw_loss
    stats["raw_loss"] = float(raw_loss.detach().cpu())
    stats["weighted_loss"] = float(weighted_loss.detach().cpu())
    return weighted_loss, stats


def _ordered_group_values_from_idx(value_to_idx: dict[str, int]) -> list[str]:
    return [
        value
        for value, idx in sorted(value_to_idx.items(), key=lambda item: item[1])
        if int(idx) != 0
    ]


def _setup_soft_condition_label_supervision(cfg: dict, model, device: torch.device):
    sup_cfg = dict(cfg.get("soft_condition_label_supervision", {}) or {})
    if not sup_cfg.get("enabled", False):
        return None

    wrapped_model = _unwrap_training_model(model)
    adapters = [
        adapter
        for adapter in (getattr(wrapped_model, "backbone_conditional_adapters", None) or [])
        if getattr(adapter, "condition_source", "").strip().lower() == "soft_text"
    ]
    ref_adapter = adapters[0] if adapters else None
    if ref_adapter is None:
        ref_head = getattr(wrapped_model, "head", None)
        if not (
            isinstance(ref_head, FactorizedOutputResidualHead)
            and str(getattr(ref_head, "condition_source", "")).strip().lower() == "soft_text"
        ):
            raise ValueError(
                "soft_condition_label_supervision requires either a soft_text backbone conditional adapter "
                "or factorized_output_residual.enabled=true with condition_source=soft_text"
            )
        embed_dim = int(ref_head.lang_embedding.embedding_dim)
        langs = _ordered_group_values_from_idx(ref_head.lang_to_idx)
        domains = _ordered_group_values_from_idx(ref_head.domain_to_idx)
        lang_alias_map = getattr(ref_head, "lang_alias_map", {})
        domain_alias_map = getattr(ref_head, "domain_alias_map", {})
    else:
        embed_dim = int(ref_adapter.lang_embedding.embedding_dim)
        langs = _ordered_group_values_from_idx(ref_adapter.lang_to_idx)
        domains = _ordered_group_values_from_idx(ref_adapter.domain_to_idx)
        lang_alias_map = getattr(ref_adapter, "lang_alias_map", {})
        domain_alias_map = getattr(ref_adapter, "domain_alias_map", {})
    head = SoftConditionGroupClassifier(
        embed_dim=embed_dim,
        langs=langs,
        domains=domains,
        hidden_dim=int(sup_cfg.get("hidden_dim", 64)),
        dropout=float(sup_cfg.get("dropout", 0.1)),
        lang_alias_map=lang_alias_map,
        domain_alias_map=domain_alias_map,
    ).to(device)
    logger.info(
        "Soft-condition label supervision enabled: langs=%d domains=%d embed_dim=%d hidden=%d start_update=%d loss_weight=%.4f",
        head.num_lang_classes() - 1,
        head.num_domain_classes() - 1,
        embed_dim,
        int(sup_cfg.get("hidden_dim", 64)),
        int(sup_cfg.get("start_update", 0) or 0),
        float(sup_cfg.get("loss_weight", 1.0)),
    )
    return head


def _compute_soft_condition_label_supervision_loss(
    soft_condition_label_head,
    model,
    batch: dict,
    cfg: dict,
    device: torch.device,
    *,
    cache: dict[str, torch.Tensor] | None = None,
    is_auxiliary: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    auxiliary_loss_weight: float = 1.0,
):
    zero = torch.tensor(0.0, device=device)
    if soft_condition_label_head is None:
        return zero, {}

    cache = cache or _aggregate_backbone_conditional_forward_cache(model, detach=False)
    lang_emb = cache.get("lang_emb") if cache else None
    domain_emb = cache.get("domain_emb") if cache else None
    if lang_emb is None or domain_emb is None:
        factorized_cache = _aggregate_factorized_output_forward_cache(model, detach=False)
        lang_emb = factorized_cache.get("lang_emb")
        domain_emb = factorized_cache.get("domain_emb")
    if lang_emb is None or domain_emb is None:
        return zero, {"skip_reason": "missing_condition_embeddings"}

    label_cfg = dict(cfg.get("soft_condition_label_supervision", {}) or {})
    label_smoothing = float(label_cfg.get("label_smoothing", 0.0))
    lang_logits, domain_logits = soft_condition_label_head(lang_emb, domain_emb)
    lang_targets, domain_targets = soft_condition_label_head.encode_group_ids(
        batch.get("lang_metadata"),
        batch.get("domain_metadata"),
        device=lang_logits.device,
    )

    base_weights = sample_weights
    if base_weights is None:
        base_weights = torch.ones(lang_logits.size(0), device=lang_logits.device, dtype=lang_logits.dtype)
    else:
        base_weights = base_weights.to(device=lang_logits.device, dtype=lang_logits.dtype)
    aux_mask = is_auxiliary.to(lang_logits.device) if is_auxiliary is not None else None

    loss = torch.tensor(0.0, device=device)
    stats: dict[str, float] = {}
    for prefix, logits, targets, weight in [
        ("lang", lang_logits, lang_targets, float(label_cfg.get("lang_weight", 1.0))),
        ("domain", domain_logits, domain_targets, float(label_cfg.get("domain_weight", 1.0))),
    ]:
        if weight <= 0:
            continue
        valid_mask = (targets > 0).to(logits.dtype)
        valid_count = int(valid_mask.sum().item())
        stats[f"{prefix}_valid"] = valid_count
        if valid_count <= 0:
            continue
        per_sample = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        masked_weights = base_weights * valid_mask
        branch_loss = combine_primary_aux_losses(
            per_sample,
            is_auxiliary=aux_mask,
            auxiliary_loss_weight=auxiliary_loss_weight,
            sample_weights=masked_weights,
        )
        loss = loss + weight * branch_loss
        stats[f"{prefix}_loss"] = float(branch_loss.detach().cpu())
        preds = logits.argmax(dim=-1)
        correct = ((preds == targets) & (targets > 0)).sum().item()
        stats[f"{prefix}_acc"] = float(correct / max(valid_count, 1))

    if not stats:
        return zero, {}
    return float(label_cfg.get("loss_weight", 1.0)) * loss, stats


def _collect_backbone_conditional_layer_caches(model, detach: bool = False) -> list[dict[str, torch.Tensor | None]]:
    wrapped_model = _unwrap_training_model(model)
    adapters = getattr(wrapped_model, "backbone_conditional_adapters", None) or []
    return [adapter.forward_cache(detach=detach) for adapter in adapters if hasattr(adapter, "forward_cache")]


def _schedule_scalar(
    start_value: float,
    end_value: float,
    *,
    current_update: int,
    start_update: int,
    end_update: int,
) -> float:
    if end_update <= start_update:
        return float(end_value)
    if current_update <= start_update:
        return float(start_value)
    if current_update >= end_update:
        return float(end_value)
    ratio = (current_update - start_update) / max(end_update - start_update, 1)
    return float(start_value + ratio * (end_value - start_value))


def _refresh_backbone_runtime_controls(model, cfg: dict, global_update: int | None) -> dict[str, float]:
    wrapped_model = _unwrap_training_model(model)
    if not hasattr(wrapped_model, "set_backbone_runtime_controls"):
        return {}

    current_update = int(global_update or 0)
    schedule_cfg = dict(cfg.get("gate_trajectory_schedule", {}) or {})
    shared_cfg = dict(cfg.get("shared_branch_regularization", {}) or {})

    shared_residual_scale = None
    shared_branch_dropout = None
    composition_gate_max = None
    composition_residual_scale = None
    stats: dict[str, float] = {}

    if schedule_cfg.get("enabled", False):
        start_update = int(schedule_cfg.get("start_update", 0) or 0)
        end_update = int(schedule_cfg.get("end_update", start_update) or start_update)
        if "shared_residual_scale_start" in schedule_cfg or "shared_residual_scale_end" in schedule_cfg:
            shared_residual_scale = _schedule_scalar(
                float(schedule_cfg.get("shared_residual_scale_start", 1.0)),
                float(schedule_cfg.get("shared_residual_scale_end", 1.0)),
                current_update=current_update,
                start_update=start_update,
                end_update=end_update,
            )
            stats["shared_residual_scale"] = shared_residual_scale
        if "shared_branch_dropout_start" in schedule_cfg or "shared_branch_dropout_end" in schedule_cfg:
            shared_branch_dropout = _schedule_scalar(
                float(schedule_cfg.get("shared_branch_dropout_start", 0.0)),
                float(schedule_cfg.get("shared_branch_dropout_end", 0.0)),
                current_update=current_update,
                start_update=start_update,
                end_update=end_update,
            )
            stats["shared_branch_dropout"] = shared_branch_dropout
        if "composition_gate_max_start" in schedule_cfg or "composition_gate_max_end" in schedule_cfg:
            composition_gate_max = _schedule_scalar(
                float(schedule_cfg.get("composition_gate_max_start", 1.0)),
                float(schedule_cfg.get("composition_gate_max_end", 1.0)),
                current_update=current_update,
                start_update=start_update,
                end_update=end_update,
            )
            stats["composition_gate_max"] = composition_gate_max
        if "composition_residual_scale_start" in schedule_cfg or "composition_residual_scale_end" in schedule_cfg:
            composition_residual_scale = _schedule_scalar(
                float(schedule_cfg.get("composition_residual_scale_start", 1.0)),
                float(schedule_cfg.get("composition_residual_scale_end", 1.0)),
                current_update=current_update,
                start_update=start_update,
                end_update=end_update,
            )
            stats["composition_residual_scale"] = composition_residual_scale

    if shared_cfg.get("enabled", False) and shared_branch_dropout is None:
        shared_branch_dropout = float(shared_cfg.get("shared_branch_dropout", 0.0))
        if shared_branch_dropout > 0:
            stats["shared_branch_dropout"] = shared_branch_dropout
    if shared_cfg.get("enabled", False) and shared_residual_scale is None:
        shared_residual_scale = float(shared_cfg.get("shared_residual_scale", 1.0))
        if shared_residual_scale != 1.0:
            stats["shared_residual_scale"] = shared_residual_scale

    wrapped_model.set_backbone_runtime_controls(
        shared_residual_scale=shared_residual_scale,
        shared_branch_dropout=shared_branch_dropout,
        composition_gate_max=composition_gate_max,
        composition_residual_scale=composition_residual_scale,
    )
    return stats


def _refresh_effective_post_ckpt_repair_adapter_scale(
    model,
    cfg: dict,
    global_update: int | None,
) -> tuple[float, float]:
    scale = _refresh_post_ckpt_repair_adapter_scale(model, cfg, global_update)
    bundle_scale = _protected_repair_bundle_scale(cfg, global_update)
    effective_scale = scale * bundle_scale
    wrapped_model = _unwrap_training_model(model)
    if hasattr(wrapped_model, "set_post_ckpt_repair_adapter_scale"):
        wrapped_model.set_post_ckpt_repair_adapter_scale(effective_scale)
    return effective_scale, bundle_scale


def _compute_branch_decorrelation_loss(
    model,
    cfg: dict,
    device: torch.device,
    *,
    cache: dict[str, torch.Tensor] | None = None,
):
    decorr_cfg = dict(cfg.get("branch_decorrelation", {}) or {})
    if not decorr_cfg.get("enabled", False):
        return torch.tensor(0.0, device=device), {}

    cache = cache or _aggregate_backbone_conditional_forward_cache(model, detach=False)
    if not cache:
        return torch.tensor(0.0, device=device), {}

    keys = [
        ("shared", "shared_delta_pooled"),
        ("lang", "lang_delta_pooled"),
        ("domain", "domain_delta_pooled"),
        ("composition", "composition_delta_pooled"),
    ]
    tensors = {
        name: cache[key]
        for name, key in keys
        if cache.get(key) is not None
    }
    if len(tensors) < 2:
        return torch.tensor(0.0, device=device), {}

    center = bool(decorr_cfg.get("center", True))
    pair_weight = float(decorr_cfg.get("pair_weight", 1.0))
    shared_weight = float(decorr_cfg.get("shared_weight", 1.0))
    loss = torch.tensor(0.0, device=device)
    stats: dict[str, float] = {}
    names = list(tensors.keys())
    n_terms = 0
    for i, left_name in enumerate(names):
        left = tensors[left_name]
        if center:
            left = left - left.mean(dim=0, keepdim=True)
        left = F.normalize(left, dim=-1, eps=1e-6)
        for right_name in names[i + 1:]:
            right = tensors[right_name]
            if center:
                right = right - right.mean(dim=0, keepdim=True)
            right = F.normalize(right, dim=-1, eps=1e-6)
            cosine = (left * right).sum(dim=-1).mean()
            term = cosine * cosine
            weight = shared_weight if "shared" in {left_name, right_name} else pair_weight
            loss = loss + weight * term
            stats[f"{left_name}_{right_name}_cos2"] = float(term.detach().cpu())
            n_terms += 1
    if n_terms <= 0:
        return torch.tensor(0.0, device=device), {}
    return float(decorr_cfg.get("loss_weight", 1.0)) * (loss / n_terms), stats


def _compute_layer_consistency_routing_loss(
    model,
    cfg: dict,
    device: torch.device,
):
    consistency_cfg = dict(cfg.get("layer_consistency_routing", {}) or {})
    if not consistency_cfg.get("enabled", False):
        return torch.tensor(0.0, device=device), {}

    layer_caches = _collect_backbone_conditional_layer_caches(model, detach=False)
    if len(layer_caches) < 2:
        return torch.tensor(0.0, device=device), {}

    embedding_weight = float(consistency_cfg.get("embedding_weight", 1.0))
    gate_weight = float(consistency_cfg.get("gate_weight", 1.0))
    composition_gate_weight = float(consistency_cfg.get("composition_gate_weight", gate_weight))
    loss = torch.tensor(0.0, device=device)
    stats: dict[str, float] = {}
    n_terms = 0

    for idx in range(len(layer_caches) - 1):
        left = layer_caches[idx]
        right = layer_caches[idx + 1]
        for key, weight in [
            ("lang_emb", embedding_weight),
            ("domain_emb", embedding_weight),
            ("gate_lang", gate_weight),
            ("gate_domain", gate_weight),
            ("gate_composition", composition_gate_weight),
        ]:
            if weight <= 0:
                continue
            left_tensor = left.get(key)
            right_tensor = right.get(key)
            if left_tensor is None or right_tensor is None:
                continue
            right_tensor = right_tensor.to(device=left_tensor.device, dtype=left_tensor.dtype)
            term = F.mse_loss(left_tensor, right_tensor, reduction="mean")
            loss = loss + weight * term
            stats[f"{key}_layer{idx}_layer{idx+1}"] = float(term.detach().cpu())
            n_terms += 1
    if n_terms <= 0:
        return torch.tensor(0.0, device=device), {}
    return float(consistency_cfg.get("loss_weight", 1.0)) * (loss / n_terms), stats


def _setup_backbone_conditional_ema_teacher(cfg: dict, model, device: torch.device):
    ema_cfg = dict(cfg.get("ema_route_teacher", {}) or {})
    if not ema_cfg.get("enabled", False):
        wrapped_model = _unwrap_training_model(model)
        if hasattr(wrapped_model, "set_ema_teacher_adapters"):
            wrapped_model.set_ema_teacher_adapters([])
        return None

    wrapped_model = _unwrap_training_model(model)
    adapters = getattr(wrapped_model, "backbone_conditional_adapters", None) or []
    if not adapters:
        raise ValueError("ema_route_teacher requires backbone_conditional_adapter.enabled=true")

    ema_adapters = []
    for adapter in adapters:
        teacher_adapter = copy.deepcopy(adapter)
        for param in teacher_adapter.parameters():
            param.requires_grad = False
        teacher_adapter.eval()
        ema_adapters.append(teacher_adapter.to(device))
    wrapped_model.set_ema_teacher_adapters(ema_adapters)
    logger.info(
        "EMA route teacher enabled: adapters=%d momentum=%.4f momentum_end=%.4f start_update=%d loss_weight=%.4f",
        len(ema_adapters),
        float(ema_cfg.get("momentum", 0.995)),
        float(ema_cfg.get("momentum_end", ema_cfg.get("momentum", 0.995))),
        int(ema_cfg.get("start_update", 0) or 0),
        float(ema_cfg.get("loss_weight", 1.0)),
    )
    return {"cfg": ema_cfg, "adapters": ema_adapters}


def _collect_ema_teacher_cache(ema_state, detach: bool = True) -> dict[str, torch.Tensor | None]:
    if ema_state is None:
        return {}
    buckets: dict[str, list[torch.Tensor]] = {}
    for adapter in ema_state.get("adapters", []):
        if not hasattr(adapter, "forward_cache"):
            continue
        cache = adapter.forward_cache(detach=detach)
        for key, value in cache.items():
            if torch.is_tensor(value):
                buckets.setdefault(key, []).append(value)
    return {
        key: torch.stack(values, dim=0).mean(dim=0)
        for key, values in buckets.items()
        if values
    }


def _ema_momentum_value(ema_cfg: dict, global_update: int | None) -> float:
    momentum = float(ema_cfg.get("momentum", 0.995))
    momentum_end = float(ema_cfg.get("momentum_end", momentum))
    start_update = int(ema_cfg.get("start_update", 0) or 0)
    end_update = int(ema_cfg.get("momentum_end_update", start_update) or start_update)
    return _schedule_scalar(
        momentum,
        momentum_end,
        current_update=int(global_update or 0),
        start_update=start_update,
        end_update=end_update,
    )


def _update_backbone_conditional_ema_teacher(ema_state, model, global_update: int | None) -> float | None:
    if ema_state is None:
        return None
    wrapped_model = _unwrap_training_model(model)
    student_adapters = getattr(wrapped_model, "backbone_conditional_adapters", None) or []
    teacher_adapters = ema_state.get("adapters", [])
    if len(student_adapters) != len(teacher_adapters):
        return None
    momentum = _ema_momentum_value(ema_state["cfg"], global_update)
    with torch.no_grad():
        for student, teacher in zip(student_adapters, teacher_adapters):
            teacher_sd = teacher.state_dict()
            student_sd = student.state_dict()
            for key, teacher_value in teacher_sd.items():
                student_value = student_sd[key].to(device=teacher_value.device, dtype=teacher_value.dtype)
                teacher_value.mul_(momentum).add_(student_value, alpha=1.0 - momentum)
    return momentum


def _compute_ema_route_teacher_loss(
    ema_state,
    model,
    cfg: dict,
    device: torch.device,
    *,
    student_cache: dict[str, torch.Tensor] | None = None,
    global_update: int | None = None,
):
    zero = torch.tensor(0.0, device=device)
    if ema_state is None:
        return zero, {}

    teacher_cfg = ema_state["cfg"]
    if int(global_update or 0) < int(teacher_cfg.get("start_update", 0) or 0):
        return zero, {}

    student_cache = student_cache or _aggregate_backbone_conditional_forward_cache(model, detach=False)
    teacher_cache = _collect_ema_teacher_cache(ema_state, detach=True)
    if not student_cache or not teacher_cache:
        return zero, {}

    loss = torch.tensor(0.0, device=device)
    stats: dict[str, float] = {}

    def _loss_for_keys(keys: list[str], weight: float, prefix: str) -> None:
        nonlocal loss
        if weight <= 0:
            return
        terms = []
        for key in keys:
            student_value = student_cache.get(key)
            teacher_value = teacher_cache.get(key)
            if student_value is None or teacher_value is None:
                continue
            teacher_value = teacher_value.to(device=device, dtype=student_value.dtype)
            term = F.mse_loss(student_value, teacher_value, reduction="mean")
            terms.append(term)
            stats[f"{prefix}_{key}"] = float(term.detach().cpu())
        if terms:
            group_loss = torch.stack(terms).mean()
            loss = loss + float(weight) * group_loss
            stats[f"{prefix}_loss"] = float(group_loss.detach().cpu())

    _loss_for_keys(
        ["gate_lang", "gate_domain", "gate_composition"],
        float(teacher_cfg.get("gate_weight", 0.0)),
        "gate",
    )
    _loss_for_keys(
        ["lang_emb", "domain_emb"],
        float(teacher_cfg.get("condition_weight", 0.0)),
        "condition",
    )
    _loss_for_keys(
        ["shared_delta_pooled", "lang_delta_pooled", "domain_delta_pooled", "composition_delta_pooled", "total_delta_pooled"],
        float(teacher_cfg.get("delta_weight", 0.0)),
        "delta",
    )
    return float(teacher_cfg.get("loss_weight", 1.0)) * loss, stats


def _resolve_repair_gate_values(
    model,
    *,
    va_gate: torch.Tensor | None = None,
) -> torch.Tensor | None:
    gate_values = va_gate
    if gate_values is None:
        repair_cache = _aggregate_post_ckpt_repair_forward_cache(model, detach=False)
        gate_values = repair_cache.get("gate")
    if gate_values is None:
        return None
    if gate_values.dim() > 1:
        gate_values = gate_values.mean(dim=-1)
    return gate_values.view(-1)


def _phase_schedule_items(cfg: dict) -> list[dict]:
    alt_cfg = dict(cfg.get("alternating_optimization", {}) or {})
    phases = alt_cfg.get("phases", []) or []
    out = []
    for phase in phases:
        updates = int(phase.get("updates", 0) or 0)
        if updates <= 0:
            continue
        out.append(dict(phase))
    return out


def _resolve_alternating_phase(cfg: dict, global_update: int | None) -> dict | None:
    alt_cfg = dict(cfg.get("alternating_optimization", {}) or {})
    if not alt_cfg.get("enabled", False):
        return None
    phases = _phase_schedule_items(cfg)
    if not phases:
        return None
    cycle = sum(int(p.get("updates", 0) or 0) for p in phases)
    if cycle <= 0:
        return None
    step = int(global_update or 0) % cycle
    offset = 0
    for phase in phases:
        width = int(phase.get("updates", 0) or 0)
        if offset <= step < offset + width:
            return phase
        offset += width
    return phases[-1]


def _apply_alternating_optimization_phase(
    cfg: dict,
    model,
    base_requires_grad_map: dict[str, bool],
    global_update: int | None,
) -> str | None:
    phase = _resolve_alternating_phase(cfg, global_update)
    if phase is None:
        return None
    wrapped_model = _unwrap_training_model(model)
    train_shared = bool(phase.get("train_shared", True))
    train_lang = bool(phase.get("train_lang", True))
    train_domain = bool(phase.get("train_domain", True))
    train_composition = bool(phase.get("train_composition", True))
    train_backbone_subset = bool(phase.get("train_backbone_subset", True))
    train_head = bool(phase.get("train_head", True))
    train_pool_gate = bool(phase.get("train_pool_gate", True))

    for name, param in wrapped_model.named_parameters():
        base_enabled = bool(base_requires_grad_map.get(name, False))
        enabled = base_enabled
        if ".shared_" in name and not train_shared:
            enabled = False
        if ".lang_" in name and not train_lang:
            enabled = False
        if ".domain_" in name and not train_domain:
            enabled = False
        if ".comp_" in name and not train_composition:
            enabled = False
        if name.startswith("backbone.") and ".adapter." not in name and ".repair_adapter." not in name and not train_backbone_subset:
            enabled = False
        if name.startswith("head.") and not train_head:
            enabled = False
        if name.startswith("pool_gate.") and not train_pool_gate:
            enabled = False
        param.requires_grad = enabled
    return str(phase.get("name", "unnamed"))


def _materialize_protected_repair_bundle(cfg: dict) -> dict:
    bundle_cfg = dict(cfg.get("protected_repair_bundle", {}) or {})
    if not bundle_cfg.get("enabled", False):
        return {}

    protected_anchor = dict(cfg.get("protected_prediction_anchor", {}) or {})
    protected_anchor.setdefault("enabled", True)
    protected_anchor.setdefault("loss_weight", float(bundle_cfg.get("protected_anchor_weight", 0.2)))
    protected_anchor.setdefault("start_update", int(bundle_cfg.get("start_update", 0) or 0))
    cfg["protected_prediction_anchor"] = protected_anchor

    repair_gate = dict(cfg.get("repair_gate", {}) or {})
    repair_gate.setdefault("enabled", True)
    repair_gate.setdefault("repair_open_weight", float(bundle_cfg.get("repair_open_weight", 0.2)))
    repair_gate.setdefault("protected_closed_weight", float(bundle_cfg.get("protected_closed_weight", 0.2)))
    repair_gate.setdefault("global_sparse_weight", float(bundle_cfg.get("global_sparse_weight", 0.0)))
    cfg["repair_gate"] = repair_gate

    repair_adapter = dict(cfg.get("post_ckpt_repair_adapter", {}) or {})
    if bundle_cfg.get("enable_repair_adapter", False):
        repair_adapter.setdefault("enabled", True)
        repair_adapter.setdefault("residual_scale", float(bundle_cfg.get("repair_adapter_residual_scale", 1.0)))
        cfg["post_ckpt_repair_adapter"] = repair_adapter
    return bundle_cfg


def _protected_repair_bundle_scale(cfg: dict, global_update: int | None) -> float:
    bundle_cfg = dict(cfg.get("protected_repair_bundle", {}) or {})
    if not bundle_cfg.get("enabled", False):
        return 1.0
    return _schedule_scalar(
        float(bundle_cfg.get("scale_start", 1.0)),
        float(bundle_cfg.get("scale_end", 1.0)),
        current_update=int(global_update or 0),
        start_update=int(bundle_cfg.get("start_update", 0) or 0),
        end_update=int(bundle_cfg.get("end_update", bundle_cfg.get("start_update", 0) or 0) or (bundle_cfg.get("start_update", 0) or 0)),
    )


def _compute_backbone_conditional_adapter_regularization(
    model,
    cfg: dict,
    device: torch.device,
    cache: dict[str, torch.Tensor] | None = None,
    current_update: int | None = None,
):
    reg_cfg = dict(cfg.get("backbone_conditional_adapter_regularization", {}) or {})
    if not reg_cfg.get("enabled", False):
        return torch.tensor(0.0, device=device), {}

    cache = cache or _aggregate_backbone_conditional_forward_cache(model, detach=False)
    if not cache:
        return torch.tensor(0.0, device=device), {}

    loss = torch.tensor(0.0, device=device)
    stats: dict[str, float] = {}
    for branch in ("lang", "domain", "composition"):
        gate = cache.get(f"gate_{branch}")
        if gate is None:
            continue
        gate_mean = gate.mean()
        stats[f"{branch}_gate_mean"] = float(gate_mean.detach().cpu())
        floor = float(reg_cfg.get(f"{branch}_gate_floor", 0.0))
        weight = float(reg_cfg.get(f"{branch}_gate_weight", 0.0))
        schedule_cfg = dict(cfg.get("gate_trajectory_schedule", {}) or {})
        if schedule_cfg.get("enabled", False):
            start_update = int(schedule_cfg.get("start_update", 0) or 0)
            end_update = int(schedule_cfg.get("end_update", start_update) or start_update)
            if f"{branch}_gate_floor_start" in schedule_cfg or f"{branch}_gate_floor_end" in schedule_cfg:
                floor = _schedule_scalar(
                    float(schedule_cfg.get(f"{branch}_gate_floor_start", floor)),
                    float(schedule_cfg.get(f"{branch}_gate_floor_end", floor)),
                    current_update=int(current_update or 0),
                    start_update=start_update,
                    end_update=end_update,
                )
            if f"{branch}_gate_weight_start" in schedule_cfg or f"{branch}_gate_weight_end" in schedule_cfg:
                weight = _schedule_scalar(
                    float(schedule_cfg.get(f"{branch}_gate_weight_start", weight)),
                    float(schedule_cfg.get(f"{branch}_gate_weight_end", weight)),
                    current_update=int(current_update or 0),
                    start_update=start_update,
                    end_update=end_update,
                )
            stats[f"{branch}_gate_floor_runtime"] = float(floor)
            stats[f"{branch}_gate_weight_runtime"] = float(weight)
        if weight > 0 and floor > 0:
            gap = torch.relu(gate.new_tensor(floor) - gate_mean)
            term = gap * gap
            loss = loss + weight * term
            stats[f"{branch}_gate_floor_penalty"] = float(term.detach().cpu())
    return loss, stats


def _compute_text_routing_detector_loss(
    model,
    batch: dict,
    cfg: dict,
    device: torch.device,
    *,
    cache: dict[str, torch.Tensor] | None = None,
    is_auxiliary: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    auxiliary_loss_weight: float = 1.0,
):
    zero = torch.tensor(0.0, device=device)
    detector_cfg = dict(cfg.get("text_routing_detector", {}) or {})
    if not detector_cfg.get("enabled", False):
        return zero, {}

    cache = cache or _aggregate_factorized_output_forward_cache(model, detach=False)
    lang_logits = cache.get("routing_lang_logits") if cache else None
    domain_logits = cache.get("routing_domain_logits") if cache else None
    if lang_logits is None or domain_logits is None:
        return zero, {"skip_reason": "missing_detector_logits"}

    wrapped_model = _unwrap_training_model(model)
    head = getattr(wrapped_model, "head", None)
    if not isinstance(head, FactorizedOutputResidualHead):
        return zero, {"skip_reason": "unsupported_head"}

    lang_targets, domain_targets = head.encode_group_ids(
        batch.get("lang_metadata"),
        batch.get("domain_metadata"),
        device=lang_logits.device,
    )
    label_smoothing = float(detector_cfg.get("label_smoothing", 0.0))
    loss_weight = float(detector_cfg.get("loss_weight", 1.0))
    lang_weight = float(detector_cfg.get("lang_weight", 1.0))
    domain_weight = float(detector_cfg.get("domain_weight", 1.0))

    total = torch.tensor(0.0, device=device)
    stats: dict[str, float] = {}
    for prefix, logits, targets, branch_weight in [
        ("lang", lang_logits, lang_targets, lang_weight),
        ("domain", domain_logits, domain_targets, domain_weight),
    ]:
        if branch_weight <= 0:
            continue
        per_sample = F.cross_entropy(
            logits.float(),
            targets,
            reduction="none",
            label_smoothing=label_smoothing,
        ).to(device)
        branch_loss = combine_primary_aux_losses(
            per_sample,
            is_auxiliary=is_auxiliary,
            auxiliary_loss_weight=auxiliary_loss_weight,
            sample_weights=sample_weights,
        )
        total = total + branch_weight * branch_loss
        preds = torch.argmax(logits, dim=-1)
        stats[f"{prefix}_acc"] = float((preds == targets).float().mean().detach().cpu())
        stats[f"{prefix}_valid"] = int(targets.numel())

    if total.detach().abs().item() == 0.0:
        return zero, stats
    weighted = loss_weight * total
    stats["weighted_loss"] = float(weighted.detach().cpu())
    return weighted, stats


def _build_optimizer(cfg: dict, model):
    lr = cfg.get("lr", 2e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    opt_cfg = cfg.get("optimizer", {})
    ds_cfg = _deepspeed_cfg(cfg)
    use_ds_cpu_adam = (
        bool(ds_cfg.get("enabled", False))
        and str(ds_cfg.get("offload_optimizer_device", "")).strip().lower() == "cpu"
    )
    llrd_cfg = opt_cfg.get("llrd", {})
    repair_adapter_cfg = cfg.get("post_ckpt_repair_adapter", {})
    repair_named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and ".repair_adapter." in name
    ]
    repair_param_ids = {id(param) for _, param in repair_named_params}
    if llrd_cfg.get("enabled", False):
        raw_param_groups = build_llrd_param_groups(
            model.backbone,
            [model.head],
            lr=lr,
            weight_decay=weight_decay,
            llrd_factor=llrd_cfg.get("factor", 0.95),
        )
        param_groups = []
        for group in raw_param_groups:
            filtered = [param for param in group["params"] if id(param) not in repair_param_ids]
            if not filtered:
                continue
            next_group = dict(group)
            next_group["params"] = filtered
            param_groups.append(next_group)
    else:
        params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) not in repair_param_ids
        ]
        param_groups = []
        if params:
            param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
    if repair_named_params:
        repair_params = [param for _, param in repair_named_params]
        param_groups.append(
            {
                "params": repair_params,
                "lr": float(repair_adapter_cfg.get("lr", lr)),
                "weight_decay": float(repair_adapter_cfg.get("weight_decay", weight_decay)),
            }
        )
        logger.info(
            "Post-ckpt repair adapter optimizer group: params=%s lr=%.6g wd=%.6g",
            sum(param.numel() for param in repair_params),
            float(repair_adapter_cfg.get("lr", lr)),
            float(repair_adapter_cfg.get("weight_decay", weight_decay)),
        )
    if not param_groups:
        raise ValueError("no trainable parameters found for optimizer")
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    if use_ds_cpu_adam:
        if deepspeed is None:
            raise RuntimeError("DeepSpeedCPUAdam requested but deepspeed is unavailable")
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return DeepSpeedCPUAdam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=float(opt_cfg.get("eps", 1e-8)),
            weight_decay=weight_decay,
            adamw_mode=True,
            fp32_optimizer_states=True,
        )
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)


def _apply_post_optimizer_freeze_for_va_residual(cfg: dict, model) -> dict:
    residual_cfg = cfg.get("va_residual_corrector", {})
    group_cfg = cfg.get("group_routed_va_residual", {})
    shared_group_cfg = cfg.get("shared_group_routed_va_residual", {})
    use_va_residual = bool(residual_cfg.get("enabled", False))
    use_group_routed = bool(group_cfg.get("enabled", False))
    use_shared_group_routed = bool(shared_group_cfg.get("enabled", False))
    if not use_va_residual and not use_group_routed and not use_shared_group_routed:
        return {}

    active_cfg = residual_cfg if use_va_residual else (group_cfg if use_group_routed else shared_group_cfg)
    freeze_report = {
        "freeze_source": (
            "va_residual_corrector"
            if use_va_residual
            else ("group_routed_va_residual" if use_group_routed else "shared_group_routed_va_residual")
        ),
        "freeze_backbone": bool(active_cfg.get("freeze_backbone", False)),
        "freeze_head": bool(active_cfg.get("freeze_head", False)),
        "freeze_pool_gate": bool(active_cfg.get("freeze_pool_gate", False)),
        "backbone_params_frozen": 0,
        "head_params_frozen": 0,
        "pool_gate_params_frozen": 0,
    }
    keep_post_ckpt_repair = bool(cfg.get("post_ckpt_repair_adapter", {}).get("enabled", False))

    if freeze_report["freeze_backbone"]:
        for name, param in model.backbone.named_parameters():
            if keep_post_ckpt_repair and ".repair_adapter." in name:
                continue
            if param.requires_grad:
                freeze_report["backbone_params_frozen"] += param.numel()
                param.requires_grad = False

    if freeze_report["freeze_head"]:
        for param in model.head.parameters():
            if param.requires_grad:
                freeze_report["head_params_frozen"] += param.numel()
                param.requires_grad = False

    if freeze_report["freeze_pool_gate"] and getattr(model, "pool_gate", None) is not None:
        for param in model.pool_gate.parameters():
            if param.requires_grad:
                freeze_report["pool_gate_params_frozen"] += param.numel()
                param.requires_grad = False

    return freeze_report


def _build_scheduler(cfg: dict, optimizer, total_steps: int):
    if total_steps <= 0:
        return None
    warmup_ratio = float(cfg.get("warmup_ratio", 0.1))
    warmup_steps = int(total_steps * warmup_ratio)
    name = cfg.get("scheduler", "cosine")
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if name in {"cosine", "cos"}:
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return None


def _module_optimizer_group(module, cfg_section: dict | None, *, default_lr: float, default_weight_decay: float):
    if module is None:
        return None
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return None
    cfg_section = cfg_section or {}
    return {
        "params": params,
        "lr": float(cfg_section.get("lr", default_lr)),
        "weight_decay": float(cfg_section.get("weight_decay", default_weight_decay)),
    }


def _validate_arousal_anti_collapse_cfg(cfg: dict) -> None:
    arousal_cfg = cfg.get("arousal_anti_collapse", {})
    if not arousal_cfg.get("enabled", False):
        return

    mode = str(arousal_cfg.get("mode", "ordinal"))
    batch_size = int(cfg.get("batch_size", 1))
    grad_accum = int(cfg.get("grad_accum", 1))
    min_batch_by_mode = {
        "ordinal": 1,
        "distribution": 4,
        "ccc": 4,
        "variance_penalty": 4,
    }
    min_batch = min_batch_by_mode.get(mode)
    if min_batch is None:
        return
    if batch_size >= min_batch:
        return

    raise ValueError(
        "arousal_anti_collapse.mode="
        f"{mode!r} requires real in-batch sample count >= {min_batch}, "
        f"but batch_size={batch_size} and grad_accum={grad_accum}. "
        "Gradient accumulation does not increase the sample count seen by the "
        "anti-collapse loss inside a single forward pass. "
        "Use mode='ordinal' for small-batch training, or increase batch_size."
    )


def _validate_metadata_conditioning_cfg(cfg: dict) -> None:
    meta_cfg = cfg.get("metadata_conditioning", {})
    if not meta_cfg.get("enabled", False):
        return
    langs = _resolve_group_values_from_cfg(meta_cfg, cfg, "lang")
    domains = _resolve_group_values_from_cfg(meta_cfg, cfg, "domain")
    if not langs:
        raise ValueError("metadata_conditioning requires at least one language")
    if not domains:
        raise ValueError("metadata_conditioning requires at least one domain")
    for alias_key in ["lang_alias_map", "domain_alias_map"]:
        alias_map = meta_cfg.get(alias_key, {})
        if alias_map is None:
            continue
        if not isinstance(alias_map, dict):
            raise ValueError(f"metadata_conditioning.{alias_key} must be a mapping if provided")


def _validate_metadata_output_residual_cfg(cfg: dict) -> None:
    out_cfg = cfg.get("metadata_output_residual", {})
    if not out_cfg.get("enabled", False):
        return
    langs = _resolve_group_values_from_cfg(out_cfg, cfg, "lang")
    domains = _resolve_group_values_from_cfg(out_cfg, cfg, "domain")
    if not langs:
        raise ValueError("metadata_output_residual requires at least one language")
    if not domains:
        raise ValueError("metadata_output_residual requires at least one domain")
    for alias_key in ["lang_alias_map", "domain_alias_map"]:
        alias_map = out_cfg.get(alias_key, {})
        if alias_map is None:
            continue
        if not isinstance(alias_map, dict):
            raise ValueError(f"metadata_output_residual.{alias_key} must be a mapping if provided")


def _validate_factorized_output_residual_cfg(cfg: dict) -> None:
    residual_cfg = cfg.get("factorized_output_residual", {})
    if not residual_cfg.get("enabled", False):
        return
    if cfg.get("metadata_output_residual", {}).get("enabled", False):
        raise ValueError("factorized_output_residual is incompatible with metadata_output_residual.enabled=true")
    langs = _resolve_group_values_from_cfg(residual_cfg, cfg, "lang")
    domains = _resolve_group_values_from_cfg(residual_cfg, cfg, "domain")
    if not langs:
        raise ValueError("factorized_output_residual requires at least one language")
    if not domains:
        raise ValueError("factorized_output_residual requires at least one domain")
    condition_source = str(residual_cfg.get("condition_source", "metadata")).strip().lower()
    if condition_source not in {"metadata", "soft_text"}:
        raise ValueError("factorized_output_residual.condition_source must be 'metadata' or 'soft_text'")
    if condition_source == "soft_text" and not bool(cfg.get("disable_explicit_metadata", False)):
        raise ValueError(
            "factorized_output_residual.condition_source=soft_text expects disable_explicit_metadata=true "
            "for no-metadata evaluation parity"
        )


def _validate_soft_condition_label_supervision_cfg(cfg: dict) -> None:
    sup_cfg = cfg.get("soft_condition_label_supervision", {})
    if not sup_cfg.get("enabled", False):
        return

    adapter_cfg = cfg.get("backbone_conditional_adapter", {})
    factorized_cfg = cfg.get("factorized_output_residual", {})
    adapter_ok = bool(adapter_cfg.get("enabled", False)) and (
        str(adapter_cfg.get("condition_source", "metadata")).strip().lower() == "soft_text"
    )
    factorized_ok = bool(factorized_cfg.get("enabled", False)) and (
        str(factorized_cfg.get("condition_source", "metadata")).strip().lower() == "soft_text"
    )
    if not (adapter_ok or factorized_ok):
        raise ValueError(
            "soft_condition_label_supervision requires either "
            "backbone_conditional_adapter.condition_source=soft_text or "
            "factorized_output_residual.condition_source=soft_text"
        )
    if not bool(cfg.get("disable_explicit_metadata", False)):
        raise ValueError(
            "soft_condition_label_supervision requires disable_explicit_metadata=true "
            "so lang/domain labels are supervision only and never model inputs"
        )
    if cfg.get("metadata_conditioning", {}).get("enabled", False):
        raise ValueError(
            "soft_condition_label_supervision is incompatible with metadata_conditioning.enabled=true"
        )
    if cfg.get("metadata_output_residual", {}).get("enabled", False):
        raise ValueError(
            "soft_condition_label_supervision is incompatible with metadata_output_residual.enabled=true"
        )


def _validate_text_routing_detector_cfg(cfg: dict) -> None:
    detector_cfg = cfg.get("text_routing_detector", {})
    if not detector_cfg.get("enabled", False):
        return

    factorized_cfg = cfg.get("factorized_output_residual", {})
    if not factorized_cfg.get("enabled", False):
        raise ValueError("text_routing_detector requires factorized_output_residual.enabled=true")
    if cfg.get("metadata_conditioning", {}).get("enabled", False):
        raise ValueError(
            "text_routing_detector is incompatible with metadata_conditioning.enabled=true"
        )
    if cfg.get("metadata_output_residual", {}).get("enabled", False):
        raise ValueError(
            "text_routing_detector is incompatible with metadata_output_residual.enabled=true"
        )
    if not bool(cfg.get("disable_explicit_metadata", False)):
        raise ValueError(
            "text_routing_detector requires disable_explicit_metadata=true so detector supervision "
            "uses labels without feeding metadata to the model"
        )


def _validate_gate_trajectory_schedule_cfg(cfg: dict) -> None:
    schedule_cfg = cfg.get("gate_trajectory_schedule", {})
    if not schedule_cfg.get("enabled", False):
        return
    if not cfg.get("backbone_conditional_adapter", {}).get("enabled", False):
        raise ValueError("gate_trajectory_schedule requires backbone_conditional_adapter.enabled=true")


def _validate_confidence_gated_composition_cfg(cfg: dict) -> None:
    comp_cfg = cfg.get("confidence_gated_composition", {})
    if not comp_cfg.get("enabled", False):
        return
    adapter_cfg = cfg.get("backbone_conditional_adapter", {})
    if not adapter_cfg.get("enabled", False) or not adapter_cfg.get("use_composition", False):
        raise ValueError(
            "confidence_gated_composition requires backbone_conditional_adapter.enabled=true and use_composition=true"
        )


def _validate_branch_decorrelation_cfg(cfg: dict) -> None:
    decorr_cfg = cfg.get("branch_decorrelation", {})
    if not decorr_cfg.get("enabled", False):
        return
    if not cfg.get("backbone_conditional_adapter", {}).get("enabled", False):
        raise ValueError("branch_decorrelation requires backbone_conditional_adapter.enabled=true")


def _validate_layer_consistency_routing_cfg(cfg: dict) -> None:
    consistency_cfg = cfg.get("layer_consistency_routing", {})
    if not consistency_cfg.get("enabled", False):
        return
    if not cfg.get("backbone_conditional_adapter", {}).get("enabled", False):
        raise ValueError("layer_consistency_routing requires backbone_conditional_adapter.enabled=true")


def _validate_ema_route_teacher_cfg(cfg: dict) -> None:
    ema_cfg = cfg.get("ema_route_teacher", {})
    if not ema_cfg.get("enabled", False):
        return
    if not cfg.get("backbone_conditional_adapter", {}).get("enabled", False):
        raise ValueError("ema_route_teacher requires backbone_conditional_adapter.enabled=true")


def _validate_stage1_pooled_feature_anchor_cfg(cfg: dict) -> None:
    anchor_cfg = cfg.get("stage1_pooled_feature_anchor", {})
    if not anchor_cfg.get("enabled", False):
        return
    has_run = bool(anchor_cfg.get("teacher_checkpoint_run"))
    has_pair = bool(anchor_cfg.get("teacher_config_path")) and bool(anchor_cfg.get("teacher_checkpoint_path"))
    if not has_run and not has_pair:
        raise ValueError(
            "stage1_pooled_feature_anchor.enabled=true requires either "
            "teacher_checkpoint_run or both teacher_config_path and teacher_checkpoint_path"
        )


def _validate_alternating_optimization_cfg(cfg: dict) -> None:
    alt_cfg = cfg.get("alternating_optimization", {})
    if not alt_cfg.get("enabled", False):
        return
    if len(_phase_schedule_items(cfg)) == 0:
        raise ValueError("alternating_optimization.enabled=true requires at least one phase with updates>0")


def _validate_protected_repair_bundle_cfg(cfg: dict) -> None:
    bundle_cfg = cfg.get("protected_repair_bundle", {})
    if not bundle_cfg.get("enabled", False):
        return
    if not cfg.get("backbone_conditional_adapter", {}).get("enabled", False):
        raise ValueError("protected_repair_bundle currently expects backbone_conditional_adapter.enabled=true")


def _validate_pivot_hardness_gate_cfg(cfg: dict) -> None:
    gate_cfg = cfg.get("pivot_hardness_gate", {})
    if not gate_cfg.get("enabled", False):
        return

    translation_cfg = cfg.get("translation_pivot", {})
    if not translation_cfg.get("enabled", False):
        raise ValueError(
            "pivot_hardness_gate.enabled=true requires translation_pivot.enabled=true"
        )

    focus_dim = str(gate_cfg.get("focus_dim", "a")).strip().lower()
    if focus_dim not in {"a", "va"}:
        raise ValueError(
            f"pivot_hardness_gate.focus_dim={focus_dim!r} must be 'a' or 'va'"
        )
    mode = str(gate_cfg.get("mode", "heuristic")).strip().lower()
    if mode not in {"heuristic", "learned"}:
        raise ValueError(
            f"pivot_hardness_gate.mode={mode!r} must be 'heuristic' or 'learned'"
        )

    max_multiplier = float(gate_cfg.get("max_multiplier", 2.0))
    min_multiplier = float(gate_cfg.get("min_multiplier", 1.0))
    if min_multiplier < 1.0 or max_multiplier < min_multiplier:
        raise ValueError(
            "pivot_hardness_gate requires 1.0 <= min_multiplier <= max_multiplier"
        )


def _validate_pivot_a_residual_cfg(cfg: dict) -> None:
    residual_cfg = cfg.get("pivot_a_residual", {})
    if not residual_cfg.get("enabled", False):
        return

    translation_cfg = cfg.get("translation_pivot", {})
    if not translation_cfg.get("enabled", False):
        raise ValueError(
            "pivot_a_residual.enabled=true requires translation_pivot.enabled=true"
        )

    start_update = int(residual_cfg.get("start_update", 0) or 0)
    if start_update < 0:
        raise ValueError("pivot_a_residual.start_update must be >= 0")

    clamp_delta = float(residual_cfg.get("clamp_delta", 1.0))
    if clamp_delta <= 0:
        raise ValueError("pivot_a_residual.clamp_delta must be > 0")


def _validate_pivot_trust_residual_cfg(cfg: dict) -> None:
    trust_cfg = cfg.get("pivot_trust_residual", {})
    if not trust_cfg.get("enabled", False):
        return

    translation_cfg = cfg.get("translation_pivot", {})
    if not translation_cfg.get("enabled", False):
        raise ValueError(
            "pivot_trust_residual.enabled=true requires translation_pivot.enabled=true"
        )

    if cfg.get("pivot_hardness_gate", {}).get("enabled", False):
        raise ValueError(
            "pivot_trust_residual should not be enabled together with pivot_hardness_gate"
        )
    if cfg.get("pivot_a_residual", {}).get("enabled", False):
        raise ValueError(
            "pivot_trust_residual should not be enabled together with pivot_a_residual"
        )

    start_update = int(trust_cfg.get("start_update", 0) or 0)
    if start_update < 0:
        raise ValueError("pivot_trust_residual.start_update must be >= 0")

    clamp_delta = float(trust_cfg.get("clamp_delta", 1.0))
    if clamp_delta <= 0:
        raise ValueError("pivot_trust_residual.clamp_delta must be > 0")


def _validate_va_residual_corrector_cfg(cfg: dict) -> None:
    residual_cfg = cfg.get("va_residual_corrector", {})
    if not residual_cfg.get("enabled", False):
        return

    start_update = int(residual_cfg.get("start_update", 0) or 0)
    if start_update < 0:
        raise ValueError("va_residual_corrector.start_update must be >= 0")

    clamp_delta = float(residual_cfg.get("clamp_delta", 1.0))
    if clamp_delta <= 0:
        raise ValueError("va_residual_corrector.clamp_delta must be > 0")

    focus_dim = str(residual_cfg.get("focus_dim", "va")).strip().lower()
    if focus_dim not in {"va", "a", "v"}:
        raise ValueError("va_residual_corrector.focus_dim must be one of {'va','a','v'}")

    bound_mode = str(residual_cfg.get("bound_mode", "tanh")).strip().lower()
    if bound_mode not in {"tanh", "clip"}:
        raise ValueError("va_residual_corrector.bound_mode must be 'tanh' or 'clip'")

    if cfg.get("group_routed_va_residual", {}).get("enabled", False) or cfg.get("shared_group_routed_va_residual", {}).get("enabled", False):
        raise ValueError(
            "va_residual_corrector should not be enabled together with group-routed residual modules"
        )


def _validate_post_ckpt_repair_adapter_cfg(cfg: dict) -> None:
    adapter_cfg = cfg.get("post_ckpt_repair_adapter", {})
    if not adapter_cfg.get("enabled", False):
        return
    if not cfg.get("backbone_conditional_adapter", {}).get("enabled", False):
        raise ValueError(
            "post_ckpt_repair_adapter currently requires backbone_conditional_adapter.enabled=true"
        )
    if int(adapter_cfg.get("bottleneck_dim", 128)) <= 0:
        raise ValueError("post_ckpt_repair_adapter.bottleneck_dim must be > 0")
    if int(adapter_cfg.get("gate_hidden", 64)) <= 0:
        raise ValueError("post_ckpt_repair_adapter.gate_hidden must be > 0")
    if int(adapter_cfg.get("start_update", 0) or 0) < 0:
        raise ValueError("post_ckpt_repair_adapter.start_update must be >= 0")
    if int(adapter_cfg.get("warmup_updates", 0) or 0) < 0:
        raise ValueError("post_ckpt_repair_adapter.warmup_updates must be >= 0")
    if float(adapter_cfg.get("max_scale", 1.0)) < 0.0:
        raise ValueError("post_ckpt_repair_adapter.max_scale must be >= 0")
    start_layer = int(adapter_cfg.get("start_layer", 0))
    end_layer = int(adapter_cfg.get("end_layer", start_layer + 1))
    if end_layer <= start_layer:
        raise ValueError("post_ckpt_repair_adapter.end_layer must be > start_layer")


def _validate_group_routed_va_residual_cfg(cfg: dict) -> None:
    group_cfg = cfg.get("group_routed_va_residual", {})
    if not group_cfg.get("enabled", False):
        return

    if cfg.get("va_residual_corrector", {}).get("enabled", False) or cfg.get("shared_group_routed_va_residual", {}).get("enabled", False):
        raise ValueError(
            "group_routed_va_residual should not be enabled together with other residual modules"
        )
    if cfg.get("pivot_hardness_gate", {}).get("enabled", False):
        raise ValueError(
            "group_routed_va_residual should not be enabled together with pivot_hardness_gate"
        )
    if cfg.get("pivot_a_residual", {}).get("enabled", False):
        raise ValueError(
            "group_routed_va_residual should not be enabled together with pivot_a_residual"
        )
    if cfg.get("pivot_trust_residual", {}).get("enabled", False):
        raise ValueError(
            "group_routed_va_residual should not be enabled together with pivot_trust_residual"
        )

    start_update = int(group_cfg.get("start_update", 0) or 0)
    if start_update < 0:
        raise ValueError("group_routed_va_residual.start_update must be >= 0")

    clamp_delta = float(group_cfg.get("clamp_delta", 1.0))
    if clamp_delta <= 0:
        raise ValueError("group_routed_va_residual.clamp_delta must be > 0")

    langs = [str(v).strip() for v in group_cfg.get("langs", []) if str(v).strip()]
    domains = [str(v).strip() for v in group_cfg.get("domains", []) if str(v).strip()]
    if not langs:
        raise ValueError("group_routed_va_residual.langs must provide at least one language")
    if not domains:
        raise ValueError("group_routed_va_residual.domains must provide at least one domain")

    for alias_key in ["lang_alias_map", "domain_alias_map"]:
        alias_map = group_cfg.get(alias_key, {})
        if alias_map is None:
            continue
        if not isinstance(alias_map, dict):
            raise ValueError(f"group_routed_va_residual.{alias_key} must be a mapping if provided")

    if bool(group_cfg.get("use_pivot_features", False)):
        translation_cfg = cfg.get("translation_pivot", {})
        if not translation_cfg.get("enabled", False):
            raise ValueError(
                "group_routed_va_residual.use_pivot_features=true requires translation_pivot.enabled=true"
            )


def _validate_shared_group_routed_va_residual_cfg(cfg: dict) -> None:
    shared_group_cfg = cfg.get("shared_group_routed_va_residual", {})
    if not shared_group_cfg.get("enabled", False):
        return

    for other_key in ["va_residual_corrector", "group_routed_va_residual", "pivot_hardness_gate", "pivot_a_residual", "pivot_trust_residual"]:
        if cfg.get(other_key, {}).get("enabled", False):
            raise ValueError(
                f"shared_group_routed_va_residual should not be enabled together with {other_key}"
            )

    start_update = int(shared_group_cfg.get("start_update", 0) or 0)
    if start_update < 0:
        raise ValueError("shared_group_routed_va_residual.start_update must be >= 0")

    clamp_delta = float(shared_group_cfg.get("clamp_delta", 1.0))
    if clamp_delta <= 0:
        raise ValueError("shared_group_routed_va_residual.clamp_delta must be > 0")

    shared_clamp_delta = float(shared_group_cfg.get("shared_clamp_delta", clamp_delta))
    if shared_clamp_delta <= 0:
        raise ValueError("shared_group_routed_va_residual.shared_clamp_delta must be > 0")

    langs = _resolve_group_values_from_cfg(shared_group_cfg, cfg, "lang")
    domains = _resolve_group_values_from_cfg(shared_group_cfg, cfg, "domain")
    if not langs:
        raise ValueError("shared_group_routed_va_residual requires at least one language")
    if not domains:
        raise ValueError("shared_group_routed_va_residual requires at least one domain")

    for alias_key in ["lang_alias_map", "domain_alias_map"]:
        alias_map = shared_group_cfg.get(alias_key, {})
        if alias_map is None:
            continue
        if not isinstance(alias_map, dict):
            raise ValueError(f"shared_group_routed_va_residual.{alias_key} must be a mapping if provided")

    if bool(shared_group_cfg.get("use_pivot_features", False)):
        translation_cfg = cfg.get("translation_pivot", {})
        if not translation_cfg.get("enabled", False):
            raise ValueError(
                "shared_group_routed_va_residual.use_pivot_features=true requires translation_pivot.enabled=true"
            )


def _validate_group_dro_cfg(cfg: dict) -> None:
    dro_cfg = cfg.get("group_dro", {})
    if not dro_cfg.get("enabled", False):
        return
    if float(dro_cfg.get("weight", 0.0)) <= 0:
        raise ValueError("group_dro.weight must be > 0 when enabled")
    top_k = int(dro_cfg.get("top_k_groups", 1) or 1)
    if top_k < 1:
        raise ValueError("group_dro.top_k_groups must be >= 1")


def _validate_group_balanced_sampling_cfg(cfg: dict) -> None:
    sampler_cfg = cfg.get("group_balanced_sampling", {})
    if not sampler_cfg.get("enabled", False):
        return
    power = float(sampler_cfg.get("power", 1.0))
    if power < 0:
        raise ValueError("group_balanced_sampling.power must be >= 0")


def _build_sample_weights(batch: dict, target_raw: torch.Tensor, cfg: dict, device: torch.device):
    sample_weights = batch.get("sample_weight")
    if sample_weights is not None:
        sample_weights = sample_weights.to(device)

    region_cfg = cfg.get("va_region_reweight", {})
    if region_cfg.get("enabled", False):
        region_weights = compute_va_region_weights(
            target_raw,
            region_weights=region_cfg.get("weights"),
            normalize=region_cfg.get("normalize", True),
        ).to(device)
        sample_weights = region_weights if sample_weights is None else sample_weights * region_weights

    return sample_weights


def _load_weight_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"defect_focus weights file not found: {path}")
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"unsupported JSON payload in {path}")

    mapping: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("sample_key")
            if key:
                mapping[key] = rec
    return mapping


def _resolve_artifact_weight(rec: dict | None, default_weight: float, candidate_fields: tuple[str, ...]) -> float:
    if rec is None:
        return float(default_weight)
    for field in candidate_fields:
        value = rec.get(field)
        if value is not None:
            return float(value)
    return float(default_weight)


def _apply_sample_weight_artifact(
    cfg: dict,
    cfg_key: str,
    train_data: list[dict],
    run_dir: Path,
    *,
    candidate_fields: tuple[str, ...],
    audit_filename: str,
    log_prefix: str,
):
    section_cfg = cfg.get(cfg_key, {})
    if not section_cfg.get("enabled", False):
        return None

    path = section_cfg.get("weights_path")
    if not path:
        raise ValueError(f"{cfg_key}.enabled=true but {cfg_key}.weights_path is missing")
    weight_map = _load_weight_map(PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path))

    default_weight = float(section_cfg.get("default_weight", 1.0))
    min_weight = float(section_cfg.get("min_weight", 0.0))
    max_weight = float(section_cfg.get("max_weight", 10.0))

    hit = 0
    weights = []
    for sample in train_data:
        key = make_sample_key(sample["id"], sample["aspect_index"], sample.get("lang"))
        rec = weight_map.get(key)
        if rec is not None:
            hit += 1
        w = _resolve_artifact_weight(rec, default_weight, candidate_fields)
        w = min(max(w, min_weight), max_weight)
        sample["_sample_weight"] = float(sample.get("_sample_weight", 1.0)) * w
        weights.append(w)

    audit = {
        "section": cfg_key,
        "weights_path": str(path),
        "num_samples": len(train_data),
        "matched": hit,
        "match_rate": round(hit / max(len(train_data), 1), 6),
        "default_weight": default_weight,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "candidate_fields": list(candidate_fields),
        "applied_mean": round(float(np.mean(weights)), 6) if weights else 0.0,
        "applied_std": round(float(np.std(weights)), 6) if weights else 0.0,
        "applied_min": round(float(np.min(weights)), 6) if weights else 0.0,
        "applied_max": round(float(np.max(weights)), 6) if weights else 0.0,
    }
    (run_dir / audit_filename).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "%s enabled: matched=%s/%s rate=%.4f mean=%.4f std=%.4f",
        log_prefix,
        hit,
        len(train_data),
        audit["match_rate"],
        audit["applied_mean"],
        audit["applied_std"],
    )
    return audit


def _apply_risk_gate_weights(cfg: dict, train_data: list[dict], run_dir: Path):
    return _apply_sample_weight_artifact(
        cfg,
        "risk_gate",
        train_data,
        run_dir,
        candidate_fields=("risk_weight", "gate_weight", "weight", "defect_weight"),
        audit_filename="risk_gate_audit.json",
        log_prefix="Risk gate",
    )


def _apply_defect_focus_weights(cfg: dict, train_data: list[dict], run_dir: Path):
    return _apply_sample_weight_artifact(
        cfg,
        "defect_focus",
        train_data,
        run_dir,
        candidate_fields=("defect_weight", "weight"),
        audit_filename="defect_focus_audit.json",
        log_prefix="Defect focus",
    )


def _apply_error_aware_training_fields(cfg: dict, train_data: list[dict], run_dir: Path):
    section_cfg = cfg.get("error_aware_sampling", {})
    if not section_cfg.get("enabled", False):
        return None

    path = section_cfg.get("weights_path")
    if not path:
        raise ValueError("error_aware_sampling.enabled=true but error_aware_sampling.weights_path is missing")
    weight_map = _load_weight_map(PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path))

    candidate_fields = tuple(
        str(v).strip()
        for v in section_cfg.get("candidate_fields", ["teacher_weight", "error_weight", "weight", "stage1_error_va"])
        if str(v).strip()
    )
    hardness_field = str(section_cfg.get("hardness_field", "stage1_error_va")).strip()
    default_weight = float(section_cfg.get("default_weight", 1.0))
    min_weight = float(section_cfg.get("min_weight", 0.0))
    max_weight = float(section_cfg.get("max_weight", 10.0))
    default_hardness = float(section_cfg.get("default_hardness", 0.0))
    use_loss_weight = bool(section_cfg.get("use_loss_weight", True))

    matched = 0
    weights = []
    hardness_values = []
    recs = []
    for sample in train_data:
        key = make_sample_key(sample["id"], sample["aspect_index"], sample.get("lang"))
        rec = weight_map.get(key)
        if rec is not None:
            matched += 1
        recs.append(rec)
        hardness = float(rec.get(hardness_field, default_hardness) if rec is not None else default_hardness)
        hardness_values.append(hardness)
        w = _resolve_artifact_weight(rec, default_weight, candidate_fields)
        w = min(max(w, min_weight), max_weight)
        weights.append(w)
        sample["_error_sampler_weight"] = float(w)
        if use_loss_weight:
            sample["_sample_weight"] = float(sample.get("_sample_weight", 1.0)) * float(w)

    hardness_np = np.asarray(hardness_values, dtype=np.float64) if hardness_values else np.zeros(1, dtype=np.float64)
    h_min = float(hardness_np.min()) if len(hardness_np) else 0.0
    h_max = float(hardness_np.max()) if len(hardness_np) else 0.0
    denom = max(h_max - h_min, 1e-9)
    easy_q = float(section_cfg.get("easy_quantile", 0.3))
    hard_q = float(section_cfg.get("hard_quantile", 0.7))
    easy_threshold = float(np.quantile(hardness_np, easy_q)) if len(hardness_np) else 0.0
    hard_threshold = float(np.quantile(hardness_np, hard_q)) if len(hardness_np) else 0.0

    for sample, hardness in zip(train_data, hardness_values):
        hardness_norm = (float(hardness) - h_min) / denom
        sample["_error_hardness"] = float(hardness_norm)
        sample["_error_easy_mask"] = 1.0 if float(hardness) <= easy_threshold else 0.0
        sample["_error_hard_mask"] = 1.0 if float(hardness) >= hard_threshold else 0.0

    audit = {
        "weights_path": str(path),
        "candidate_fields": list(candidate_fields),
        "hardness_field": hardness_field,
        "matched": matched,
        "num_samples": len(train_data),
        "match_rate": round(matched / max(len(train_data), 1), 6),
        "use_loss_weight": use_loss_weight,
        "use_sampler": bool(section_cfg.get("use_sampler", True)),
        "easy_quantile": easy_q,
        "hard_quantile": hard_q,
        "easy_threshold": easy_threshold,
        "hard_threshold": hard_threshold,
        "raw_hardness_min": h_min,
        "raw_hardness_max": h_max,
        "weight_mean": round(float(np.mean(weights)), 6) if weights else 0.0,
        "weight_std": round(float(np.std(weights)), 6) if weights else 0.0,
    }
    (run_dir / "error_aware_sampling_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Error-aware training fields enabled: matched=%s/%s rate=%.4f hard_q=%.2f easy_q=%.2f weight_mean=%.4f",
        matched,
        len(train_data),
        audit["match_rate"],
        hard_q,
        easy_q,
        audit["weight_mean"],
    )
    return audit


def _build_error_aware_sampler(cfg: dict, train_data: list[dict], run_dir: Path):
    section_cfg = cfg.get("error_aware_sampling", {})
    if not section_cfg.get("enabled", False) or not bool(section_cfg.get("use_sampler", True)):
        return None, None

    weights = np.asarray(
        [float(sample.get("_error_sampler_weight", 1.0)) for sample in train_data],
        dtype=np.float64,
    )
    power = float(section_cfg.get("sampler_power", 1.0))
    if power != 1.0:
        weights = np.power(np.clip(weights, a_min=1e-9, a_max=None), power)
    min_weight = float(section_cfg.get("sampler_min_weight", 0.1))
    max_weight = float(section_cfg.get("sampler_max_weight", 10.0))
    weights = np.clip(weights, min_weight, max_weight)
    if weights.mean() > 0:
        weights = weights / weights.mean()
    replacement = bool(section_cfg.get("replacement", True))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_data),
        replacement=replacement,
    )
    audit = {
        "replacement": replacement,
        "sampler_power": power,
        "mean_weight": float(weights.mean()) if len(weights) else 0.0,
        "min_weight": float(weights.min()) if len(weights) else 0.0,
        "max_weight": float(weights.max()) if len(weights) else 0.0,
    }
    (run_dir / "error_aware_sampler_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Error-aware sampler enabled: mean=%.4f range=[%.4f, %.4f] power=%.2f",
        audit["mean_weight"],
        audit["min_weight"],
        audit["max_weight"],
        power,
    )
    return sampler, audit


def _capture_anchor_state(model, cfg: dict):
    anchor_cfg = cfg.get("w2_anchor", {})
    if not anchor_cfg.get("enabled", False):
        return None
    scope = str(anchor_cfg.get("scope", "backbone_trainable"))
    include_adapter = bool(anchor_cfg.get("include_backbone_conditional_adapter", True))
    state = {}
    if scope == "backbone_trainable":
        for name, param in model.backbone.named_parameters():
            if param.requires_grad:
                if not include_adapter and (".adapter." in name or ".repair_adapter." in name):
                    continue
                state[name] = param.detach().clone()
    elif scope == "all_trainable":
        for name, param in model.named_parameters():
            if param.requires_grad:
                if not include_adapter and (".adapter." in name or ".repair_adapter." in name):
                    continue
                state[name] = param.detach().clone()
    else:
        raise ValueError(f"unsupported w2_anchor.scope={scope!r}")
    logger.info(
        "W2 anchor enabled: scope=%s include_backbone_conditional_adapter=%s tensors=%d",
        scope,
        include_adapter,
        len(state),
    )
    return state


def _apply_backbone_trainable_subset(cfg: dict, model) -> dict:
    residual_cfg = cfg.get("va_residual_corrector", {})
    group_cfg = cfg.get("group_routed_va_residual", {})
    shared_group_cfg = cfg.get("shared_group_routed_va_residual", {})
    adapter_cfg = cfg.get("backbone_conditional_adapter", {})
    active_cfg = None
    if residual_cfg.get("enabled", False):
        active_cfg = residual_cfg
    elif group_cfg.get("enabled", False):
        active_cfg = group_cfg
    elif shared_group_cfg.get("enabled", False):
        active_cfg = shared_group_cfg
    elif adapter_cfg.get("enabled", False):
        active_cfg = adapter_cfg
    if not active_cfg or bool(active_cfg.get("freeze_backbone", False)):
        return {}

    patterns = [str(p).strip() for p in active_cfg.get("backbone_trainable_patterns", []) or [] if str(p).strip()]
    if not patterns:
        return {}

    report = {
        "patterns": patterns,
        "kept_tensors": 0,
        "kept_params": 0,
        "frozen_tensors": 0,
        "frozen_params": 0,
    }
    keep_post_ckpt_repair = bool(cfg.get("post_ckpt_repair_adapter", {}).get("enabled", False))
    keep_adapter_trainable = bool(adapter_cfg.get("enabled", False) and active_cfg is adapter_cfg and active_cfg.get("keep_adapter_trainable", True))
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if keep_post_ckpt_repair and ".repair_adapter." in name:
            report["kept_tensors"] += 1
            report["kept_params"] += param.numel()
            continue
        if keep_adapter_trainable and ".adapter." in name:
            report["kept_tensors"] += 1
            report["kept_params"] += param.numel()
            continue
        if any(pattern in name for pattern in patterns):
            report["kept_tensors"] += 1
            report["kept_params"] += param.numel()
        else:
            param.requires_grad = False
            report["frozen_tensors"] += 1
            report["frozen_params"] += param.numel()
    logger.info("Backbone trainable subset applied: %s", report)
    return report


def _anchor_regularization_loss(model, anchor_state, cfg: dict, device: torch.device):
    if not anchor_state:
        return torch.tensor(0.0, device=device)
    anchor_cfg = cfg.get("w2_anchor", {})
    scope = str(anchor_cfg.get("scope", "backbone_trainable"))
    include_adapter = bool(anchor_cfg.get("include_backbone_conditional_adapter", True))
    terms = []
    if scope == "backbone_trainable":
        iterator = model.backbone.named_parameters()
    else:
        iterator = model.named_parameters()
    for name, param in iterator:
        if not param.requires_grad:
            continue
        if not include_adapter and (".adapter." in name or ".repair_adapter." in name):
            continue
        ref = anchor_state.get(name)
        if ref is None:
            continue
        ref = ref.to(device=param.device, dtype=param.dtype)
        terms.append(F.mse_loss(param, ref, reduction="mean"))
    if not terms:
        return torch.tensor(0.0, device=device)
    return float(anchor_cfg.get("weight", 0.0)) * torch.stack(terms).mean()


def _save_checkpoint(model, run_dir: Path, epoch: int, metrics: dict, cfg: dict,
                     filename: str = "checkpoint.pt", global_update: int | None = None,
                     extra_modules: dict[str, object] | None = None,
                     dist_ctx: DistributedContext | None = None):
    ds_engine = model if hasattr(model, "_consolidated_16bit_state_dict") and hasattr(model, "module") else None
    wrapped_model = ds_engine.module if ds_engine is not None else model
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "checkpoint_type": "head_only",
        "head": wrapped_model.head.state_dict(),
        "cfg_tag": cfg.get("tag", "small_llm_va"),
        "head_type": cfg.get("head_type", "unknown"),
        "pooling": cfg.get("small_llm", {}).get("pooling", getattr(wrapped_model, "pooling", "last_token")),
        "freeze_until": cfg.get("freeze_until", 0),
    }
    if cfg.get("matched_protocol", {}).get("enabled", False):
        stats = getattr(wrapped_model, "matched_va_norm_stats", None)
        validate_normalization(stats, bool(cfg.get("va_target_normalization", False)))
        payload["matched_protocol"] = copy.deepcopy(cfg["matched_protocol"])
        payload["effective_inference_config"] = copy.deepcopy(cfg)
        payload["va_norm_stats"] = stats
        payload["loading_provenance"] = copy.deepcopy(getattr(wrapped_model, "loading_provenance", {}))
        payload["optimizer_policy"] = "reset"
        payload["optimizer_scheduler_or_rng_state"] = "intentionally absent; weights-only continuation"
    if global_update is not None:
        payload["global_update"] = int(global_update)
    if cfg.get("text_only_inference", False):
        validate_text_only_cfg(cfg, required=True)
        stats = getattr(wrapped_model, "text_only_va_norm_stats", None)
        validate_normalization(stats, bool(cfg.get("va_target_normalization", False)))
        payload["inference_protocol"] = PROTOCOL
        payload["va_norm_stats"] = stats
    # Save pool_gate if it exists (gated_last_mean pooling)
    if hasattr(wrapped_model, "pool_gate") and wrapped_model.pool_gate is not None:
        payload["pool_gate"] = wrapped_model.pool_gate.state_dict()
    if hasattr(wrapped_model.backbone, "peft_config"):
        try:
            from peft import get_peft_model_state_dict
            payload["backbone_peft"] = get_peft_model_state_dict(wrapped_model.backbone)
            payload["checkpoint_type"] = "peft_head"
        except Exception as exc:
            logger.warning("Failed to export PEFT weights cleanly: %s", exc)
    elif uses_base_overlay(cfg):
        if ds_engine is not None or (dist_ctx is not None and dist_ctx.enabled):
            raise ValueError("pinned-base overlay saving is accepted only for the single-process matched runtime")
        payload["backbone"], payload["checkpoint_storage"] = export_base_overlay(wrapped_model, cfg)
        payload["checkpoint_type"] = "pinned_base_overlay"
    elif cfg.get("save_full_backbone_checkpoint", False):
        if ds_engine is not None and dist_ctx is not None and dist_ctx.enabled:
            consolidated = ds_engine._consolidated_16bit_state_dict(exclude_frozen_parameters=False)
            if dist_ctx.is_main:
                payload["backbone"] = _extract_submodule_state_dict(consolidated, "backbone")
                head_sd = _extract_submodule_state_dict(consolidated, "head")
                if head_sd:
                    payload["head"] = head_sd
                pool_gate_sd = _extract_submodule_state_dict(consolidated, "pool_gate")
                if pool_gate_sd:
                    payload["pool_gate"] = pool_gate_sd
        elif dist_ctx is not None and dist_ctx.enabled:
            payload["backbone"] = gather_full_state_dict(wrapped_model.backbone)
        else:
            payload["backbone"] = wrapped_model.backbone.state_dict()
        payload["checkpoint_type"] = "full_backbone"
    else:
        logger.warning("Saving head-only checkpoint; set save_full_backbone_checkpoint=true for full restoration")
    for module_name, module in (extra_modules or {}).items():
        if module is not None:
            payload[module_name] = module.state_dict()
            if module_name == "soft_condition_label_head":
                payload["soft_condition_label_head_config"] = export_soft_condition_classifier_config(module)
    if dist_ctx is not None and dist_ctx.enabled:
        if dist_ctx.is_main:
            torch.save(payload, run_dir / filename)
        distributed_barrier(dist_ctx)
    else:
        torch.save(payload, run_dir / filename)


def _save_best_checkpoint(model, run_dir, epoch, metrics, cfg, *, event, global_update,
                          extra_modules=None, dist_ctx=None):
    from src.train.checkpoint_retention import policy, publish_best_alias
    instruction = policy()
    if (cfg.get("matched_protocol", {}).get("protocol_version") == "method_validation_v2"
            and instruction is not None and instruction.get("future_best_as_archive_hardlink")):
        if dist_ctx is not None and dist_ctx.enabled:
            raise ValueError("V2 archive best aliases require single-process training")
        publish_best_alias(run_dir, event=event, epoch=epoch, update=global_update, cfg=cfg)
        return
    _save_checkpoint(model, run_dir, epoch, metrics, cfg, global_update=global_update,
                     extra_modules=extra_modules, dist_ctx=dist_ctx)


def _prune_step_checkpoints(run_dir: Path, keep: int) -> list[str]:
    if keep <= 0:
        return []

    step_ckpts = []
    for path in run_dir.glob("checkpoint_step*.pt"):
        stem = path.stem
        try:
            step = int(stem.replace("checkpoint_step", ""))
        except ValueError:
            continue
        step_ckpts.append((step, path))

    removed = []
    if len(step_ckpts) <= keep:
        return removed

    step_ckpts.sort(key=lambda item: item[0])
    for _, path in step_ckpts[:-keep]:
        try:
            path.unlink()
            removed.append(path.name)
        except FileNotFoundError:
            continue
    return removed


def _filter_compatible_state_dict(module, incoming_state: dict | None):
    if not incoming_state:
        return {}, {}
    target_state = module.state_dict()
    compatible = {}
    skipped = {}
    for key, value in incoming_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            skipped[key] = "missing_target_key"
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped[key] = f"shape_mismatch:{tuple(value.shape)}!={tuple(target_value.shape)}"
            continue
        compatible[key] = value
    return compatible, skipped


def _remap_backbone_checkpoint_keys_for_wrapped_layers(module, incoming_state: dict | None):
    if not incoming_state:
        return incoming_state
    target_state = module.state_dict()
    if not target_state:
        return incoming_state

    remapped = {}
    for key, value in incoming_state.items():
        target_key = key
        if key not in target_state and ".adapter." not in key and ".repair_adapter." not in key:
            parts = key.split(".")
            if len(parts) >= 3 and parts[0] == "layers":
                candidate = ".".join(parts[:2] + ["layer"] + parts[2:])
                if candidate in target_state:
                    target_key = candidate
        remapped[target_key] = value
    return remapped


def _load_state_subset(module, incoming_state: dict | None, module_name: str):
    if module_name == "backbone":
        incoming_state = _remap_backbone_checkpoint_keys_for_wrapped_layers(module, incoming_state)
    compatible, skipped = _filter_compatible_state_dict(module, incoming_state)
    missing = []
    unexpected = []
    if compatible:
        status = module.load_state_dict(compatible, strict=False)
        missing = list(getattr(status, "missing_keys", []))
        unexpected = list(getattr(status, "unexpected_keys", []))
    return {
        "module": module_name,
        "loaded_keys": len(compatible),
        "skipped_keys": len(skipped),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_examples": dict(list(skipped.items())[:5]),
    }


def _load_head_state_subset(model_head, incoming_state: dict | None):
    direct_compatible, direct_skipped = _filter_compatible_state_dict(model_head, incoming_state)
    if direct_compatible:
        return _load_state_subset(model_head, incoming_state, "head")

    base_head = getattr(model_head, "base_head", None)
    if base_head is not None:
        base_compatible, base_skipped = _filter_compatible_state_dict(base_head, incoming_state)
        if base_compatible:
            report = _load_state_subset(base_head, incoming_state, "head.base_head")
            report["loaded_via_wrapper"] = True
            return report
        direct_skipped = base_skipped

    return {
        "module": "head",
        "loaded_keys": 0,
        "skipped_keys": len(direct_skipped),
        "skip_reason": f"incompatible_head_keys:{len(direct_skipped)}",
        "skipped_examples": dict(list(direct_skipped.items())[:5]),
    }


def _load_peft_state_subset(module, incoming_state: dict | None, module_name: str):
    if not incoming_state:
        return {
            "module": module_name,
            "loaded_keys": 0,
            "skipped_keys": 0,
            "missing_keys": 0,
            "unexpected_keys": 0,
            "skipped_examples": {},
        }
    try:
        from peft import set_peft_model_state_dict

        status = set_peft_model_state_dict(module, incoming_state, adapter_name="default")
        missing = []
        unexpected = []
        if isinstance(status, tuple):
            if len(status) >= 1 and status[0] is not None:
                missing = list(getattr(status[0], "missing_keys", status[0] if isinstance(status[0], list) else []))
            if len(status) >= 2 and status[1] is not None:
                unexpected = list(getattr(status[1], "unexpected_keys", status[1] if isinstance(status[1], list) else []))
        else:
            missing = list(getattr(status, "missing_keys", []))
            unexpected = list(getattr(status, "unexpected_keys", []))
        return {
            "module": module_name,
            "loaded_keys": len(incoming_state),
            "skipped_keys": 0,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "skipped_examples": {},
        }
    except Exception as exc:
        return {
            "module": module_name,
            "loaded_keys": 0,
            "skipped_keys": len(incoming_state),
            "missing_keys": 0,
            "unexpected_keys": 0,
            "skip_reason": f"peft_load_failed:{exc}",
            "skipped_examples": dict(list((incoming_state or {}).items())[:1]),
        }


def _load_checkpoint(model, ckpt_path: Path, cfg: dict | None = None,
                     load_backbone: bool = True,
                     load_head: str = "if_compatible",
                     extra_modules: dict[str, object] | None = None,
                     matched_stage1: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if (cfg or {}).get("matched_protocol", {}).get("enabled", False) and load_backbone:
        if load_head == "never":
            raise ValueError("matched full restoration cannot skip the common readout")
        report = restore_matched_checkpoint(
            model, ckpt, cfg, from_stage1=matched_stage1, extra_modules=extra_modules,
        )
        if matched_stage1:
            model.matched_stage1_va_norm_stats = ckpt.get("va_norm_stats")
        return ckpt, {"ckpt_path": str(ckpt_path), **report}
    load_report = {
        "ckpt_path": str(ckpt_path),
        "checkpoint_type": ckpt.get("checkpoint_type", "unknown"),
        "load_backbone": bool(load_backbone),
        "load_head": load_head,
    }

    if load_backbone and "backbone" in ckpt:
        load_report["backbone"] = _load_state_subset(model.backbone, ckpt["backbone"], "backbone")
    if load_backbone and "backbone_peft" in ckpt:
        load_report["backbone_peft"] = _load_peft_state_subset(model.backbone, ckpt["backbone_peft"], "backbone_peft")

    incoming_head = ckpt.get("head") or ckpt.get("va_head")
    current_head_type = (cfg or {}).get("head_type")
    current_pooling = (cfg or {}).get("small_llm", {}).get("pooling", getattr(model, "pooling", None))
    ckpt_head_type = ckpt.get("head_type")
    ckpt_pooling = ckpt.get("pooling")

    head_meta_match = True
    head_skip_reason = None
    if load_head == "never":
        head_meta_match = False
        head_skip_reason = "disabled_by_policy"
    elif incoming_head is None:
        head_meta_match = False
        head_skip_reason = "missing_head_in_checkpoint"
    elif current_head_type and ckpt_head_type and current_head_type != ckpt_head_type:
        head_meta_match = False
        head_skip_reason = f"head_type_mismatch:{ckpt_head_type}->{current_head_type}"
    elif current_pooling and ckpt_pooling and current_pooling != ckpt_pooling:
        head_meta_match = False
        head_skip_reason = f"pooling_mismatch:{ckpt_pooling}->{current_pooling}"

    if incoming_head is not None and load_head == "always":
        load_report["head"] = _load_head_state_subset(model.head, incoming_head)
    elif incoming_head is not None and head_meta_match:
        compatible, skipped = _filter_compatible_state_dict(model.head, incoming_head)
        if len(compatible) == len(incoming_head):
            load_report["head"] = _load_head_state_subset(model.head, incoming_head)
        else:
            load_report["head"] = _load_head_state_subset(model.head, incoming_head)
    else:
        load_report["head"] = {
            "module": "head",
            "loaded_keys": 0,
            "skipped_keys": len(incoming_head or {}),
            "skip_reason": head_skip_reason or "not_loaded",
        }

    # Load pool_gate if present in checkpoint and model
    if "pool_gate" in ckpt and hasattr(model, "pool_gate") and model.pool_gate is not None:
        model.pool_gate.load_state_dict(ckpt["pool_gate"])
        load_report["pool_gate"] = {"loaded": True}

    for module_name, module in (extra_modules or {}).items():
        incoming_state = ckpt.get(module_name)
        if module is None:
            continue
        if incoming_state is not None:
            load_report[module_name] = _load_state_subset(module, incoming_state, module_name)
        else:
            load_report[module_name] = {
                "module": module_name,
                "loaded_keys": 0,
                "skipped_keys": 0,
                "skip_reason": "missing_in_checkpoint",
            }

    return ckpt, load_report


def _archive_matched_dev(model, run_dir, event, epoch, update, per_target, cfg,
                         extra_modules, dist_ctx):
    return archive_evaluation(
        run_dir, event=event, epoch=epoch, update=update, per_target=per_target, cfg=cfg,
        checkpoint_writer=lambda directory, filename: _save_checkpoint(
            model, directory, epoch, {"per_target_dev": per_target}, cfg,
            filename=filename, global_update=update, extra_modules=extra_modules,
            dist_ctx=dist_ctx,
        ),
        is_main=dist_ctx.is_main, barrier=lambda: distributed_barrier(dist_ctx),
    )


def _predict(model, data, tokenizer, device, cfg, va_norm_stats=None,
             quantile_head=None, pivot_residual_head=None,
             pivot_trust_head=None, va_residual_head=None,
             group_routed_residual_head=None):
    validate_text_only_cfg(cfg)
    if cfg.get("text_only_inference", False):
        validate_text_only_model(_unwrap_training_model(model))
        if any(head is not None for head in (quantile_head, pivot_residual_head,
                                            pivot_trust_head, va_residual_head, group_routed_residual_head)):
            raise ValueError("extra prediction heads are outside the text-only protocol")
    model.eval()
    small_llm_cfg = cfg.get("small_llm", {})
    teacher_cfg = cfg.get("teacher", {})
    translation_cfg = cfg.get("translation_pivot", {})
    translation_source = _resolve_translation_pivot_source(cfg)
    translation_map = (
        load_translation_pivot_map(translation_source)
        if translation_cfg.get("enabled", False) and translation_source
        else None
    )

    # TTA configuration
    tta_cfg = cfg.get("tta", {})
    tta_enabled = tta_cfg.get("enabled", False)
    tta_n_samples = tta_cfg.get("n_samples", 10)
    tta_aggregation = tta_cfg.get("aggregation", "mean")
    tta_mc_dropout = tta_cfg.get("mc_dropout", True)

    if tta_enabled:
        logger.info(f"TTA enabled: n_samples={tta_n_samples}, aggregation={tta_aggregation}, mc_dropout={tta_mc_dropout}")

    eval_num_workers = max(int(cfg.get("eval_dataloader_num_workers", cfg.get("dataloader_num_workers", 4))), 0)
    eval_pin_memory = bool(cfg.get("eval_dataloader_pin_memory", cfg.get("dataloader_pin_memory", device.type == "cuda")))
    eval_persistent_workers = bool(
        cfg.get("eval_dataloader_persistent_workers", cfg.get("dataloader_persistent_workers", eval_num_workers > 0))
    )
    loader = build_dimasr_dataloader(
        data,
        tokenizer,
        batch_size=cfg.get("eval_batch_size", cfg.get("batch_size", 1)),
        shuffle=False,
        max_len=cfg.get("max_len", 256),
        input_format=cfg.get("input_format", "instruction_va_v1"),
        prompt_template=small_llm_cfg.get("prompt_template"),
        teacher_explanation_as_input=teacher_cfg.get("explanation_as_input", False),
        teacher_explanation_prefix=teacher_cfg.get("explanation_prefix", "teacher_explanation"),
        teacher_explanation_min_quality=teacher_cfg.get("teacher_explanation_min_quality", 0.0),
        translation_map=translation_map,
        translation_pivot=translation_cfg,
        disable_explicit_metadata=bool(cfg.get("disable_explicit_metadata", False)),
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory,
        persistent_workers=eval_persistent_workers,
    )

    preds = []
    use_uncertainty = _uses_uncertainty_head(model)

    for batch in loader:
        def _forward_single_pass():
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            condition_kwargs = _resolve_model_condition_kwargs(batch, cfg)
            head_param = next(model.head.parameters(), None)
            group_routed_cfg = cfg.get("group_routed_va_residual", {})
            shared_group_routed_cfg = cfg.get("shared_group_routed_va_residual", {})
            active_group_cfg = shared_group_routed_cfg if shared_group_routed_cfg.get("enabled", False) else group_routed_cfg
            need_group_routed_pivot = (
                group_routed_residual_head is not None
                and bool(active_group_cfg.get("use_pivot_features", False))
            )
            need_source_features = (
                quantile_head is not None
                or pivot_residual_head is not None
                or pivot_trust_head is not None
                or va_residual_head is not None
                or group_routed_residual_head is not None
            )
            if need_source_features:
                features = model.encode(
                    ids,
                    mask,
                    **condition_kwargs,
                )
                if head_param is not None and features.dtype != head_param.dtype:
                    features = features.to(head_param.dtype)
                outputs = model.apply_head(
                    features,
                    **condition_kwargs,
                )
            else:
                features = None
                outputs = model(
                    ids,
                    mask,
                    **condition_kwargs,
                )
            if use_uncertainty:
                mu, log_sigma = outputs
                sigma_local = _denormalize_sigma(log_sigma.exp().cpu(), va_norm_stats)
                va_device_local = _denormalize_va(mu, va_norm_stats).clamp(1.0, 9.0)
            else:
                sigma_local = None
                va_device_local = _denormalize_va(outputs, va_norm_stats).clamp(1.0, 9.0)

            if quantile_head is not None:
                if features is None:
                    features = model.encode(
                        ids,
                        mask,
                        **condition_kwargs,
                    )
                    if head_param is not None and features.dtype != head_param.dtype:
                        features = features.to(head_param.dtype)
                quantile_head.eval()
                a_median = quantile_head.predict_median(features)
                va_device_local[:, 1] = a_median.clamp(1.0, 9.0)

            pivot_features = None
            pivot_va_raw = None
            pivot_sample_mask = None
            pivot_confidence = None
            if (
                translation_map is not None
                and "pivot_input_ids" in batch
                and (
                    pivot_trust_head is not None
                    or pivot_residual_head is not None
                    or need_group_routed_pivot
                )
            ):
                pivot_ids = batch["pivot_input_ids"].to(device)
                pivot_mask = batch["pivot_attention_mask"].to(device)
                pivot_sample_mask = batch["pivot_mask"].to(device)
                pivot_confidence = batch["pivot_weight"].to(device)
                if float((pivot_sample_mask > 0.5).sum().item()) > 0.0:
                    pivot_features = model.encode(
                        pivot_ids,
                        pivot_mask,
                        **condition_kwargs,
                    )
                    if head_param is not None and pivot_features.dtype != head_param.dtype:
                        pivot_features = pivot_features.to(head_param.dtype)
                    pivot_outputs = model.apply_head(
                        pivot_features,
                        **condition_kwargs,
                    )
                    pivot_mu = pivot_outputs[0] if use_uncertainty else pivot_outputs
                    pivot_va_raw = _denormalize_va(pivot_mu, va_norm_stats).clamp(1.0, 9.0)

            if pivot_trust_head is not None and pivot_va_raw is not None:
                va_device_local, _, _ = _build_pivot_trust_residual_prediction(
                    cfg.get("pivot_trust_residual", {}),
                    pivot_trust_head,
                    pivot_sample_mask,
                    va_device_local,
                    pivot_va_raw,
                    features,
                    pivot_features,
                    pivot_confidence,
                )
                va_device_local = va_device_local.clamp(1.0, 9.0)
            elif pivot_residual_head is not None and pivot_va_raw is not None:
                va_device_local, _ = _build_pivot_residual_prediction(
                    cfg.get("pivot_a_residual", {}),
                    pivot_residual_head,
                    pivot_sample_mask,
                    va_device_local,
                    pivot_va_raw,
                    features,
                    pivot_features,
                    pivot_confidence,
                )
                va_device_local = va_device_local.clamp(1.0, 9.0)

            if va_residual_head is not None:
                va_device_local, _, _ = _build_va_residual_prediction(
                    cfg.get("va_residual_corrector", {}),
                    va_residual_head,
                    va_device_local,
                    features,
                )
                va_device_local = va_device_local.clamp(1.0, 9.0)

            if group_routed_residual_head is not None:
                pivot_pair_features = None
                if need_group_routed_pivot:
                    pivot_pair_features = _build_pivot_va_pair_features(
                        va_device_local,
                        pivot_va_raw,
                        features,
                        pivot_features,
                        pivot_confidence,
                        detach_inputs=bool(active_group_cfg.get("detach_inputs", False)),
                    )
                va_device_local, _ = _build_group_routed_va_residual_prediction(
                    active_group_cfg,
                    group_routed_residual_head,
                    batch["lang"],
                    batch["domain"],
                    va_device_local,
                    features,
                    pivot_pair_features=pivot_pair_features,
                )
                va_device_local = va_device_local.clamp(1.0, 9.0)
            return va_device_local, sigma_local

        if tta_enabled:
            original_training_state = model.training
            model.eval()
            if tta_mc_dropout:
                enable_dropout_for_inference(model)
            va_samples = []
            with torch.no_grad():
                for _ in range(max(int(tta_n_samples), 1)):
                    va_device_sample, _ = _forward_single_pass()
                    va_samples.append(va_device_sample.detach())
            if tta_mc_dropout:
                disable_dropout_for_inference(model)
            model.train(original_training_state)

            va_stack = torch.stack(va_samples, dim=0)
            if tta_aggregation == "median":
                va = va_stack.median(dim=0).values.cpu()
            else:
                va = va_stack.mean(dim=0).cpu()
            tta_uncertainty = va_stack.std(dim=0).cpu()
            sigma = None
        else:
            with torch.no_grad():
                va_device, sigma = _forward_single_pass()
                va = va_device.cpu()
            tta_uncertainty = None

        # Matched experiments retain continuous predictions for all arm/DP metrics.
        if cfg.get("matched_protocol", {}).get("protocol_version") in {"original_source_stage2_textfix_20261104", "recovered_original_stage1_source_20261104"}:
            if cfg.get("legacy_prediction_rounding") != "python_two_decimals_before_metrics":
                raise ValueError("Recovered source protocol requires original Python two-decimal prediction rounding")
            if cfg.get("matched_protocol", {}).get("protocol_version") == "recovered_original_stage1_source_20261104":
                va = (va * 100).round() / 100
            else:
                va = torch.tensor([[round(float(x), 2) for x in row] for row in va], dtype=torch.float64)
        elif not cfg.get("matched_protocol", {}).get("enabled", False):
            va = (va * 100).round() / 100

        for i, idx in enumerate(batch["idx"]):
            sample = data[idx]
            pred = {
                "id": sample["id"],
                "aspect": sample["aspect"],
                "aspect_index": sample["aspect_index"],
                "sample_aspects": sample["sample_aspects"],
                "v_pred": va[i, 0].item(),
                "a_pred": va[i, 1].item(),
            }
            if sigma is not None:
                pred["sigma_v"] = sigma[i, 0].item()
                pred["sigma_a"] = sigma[i, 1].item()
            if tta_uncertainty is not None:
                pred["tta_uncertainty_v"] = tta_uncertainty[i, 0].item()
                pred["tta_uncertainty_a"] = tta_uncertainty[i, 1].item()
            if sample.get("v") is not None:
                pred["v_true"] = sample["v"]
                pred["a_true"] = sample["a"]
            preds.append(pred)
    return preds


def _evaluate_dev_rmse(model, dev_data, tokenizer, device, cfg, va_norm_stats=None,
                       quantile_head=None, pivot_residual_head=None,
                       pivot_trust_head=None, va_residual_head=None,
                       group_routed_residual_head=None) -> float | None:
    if not dev_data:
        return None
    dev_preds = _predict(
        model,
        dev_data,
        tokenizer,
        device,
        cfg,
        va_norm_stats=va_norm_stats,
        quantile_head=quantile_head,
        pivot_residual_head=pivot_residual_head,
        pivot_trust_head=pivot_trust_head,
        va_residual_head=va_residual_head,
        group_routed_residual_head=group_routed_residual_head,
    )
    vp = np.array([p["v_pred"] for p in dev_preds])
    ap = np.array([p["a_pred"] for p in dev_preds])
    vt = np.array([p["v_true"] for p in dev_preds])
    at = np.array([p["a_true"] for p in dev_preds])
    return rmse_va(vp, ap, vt, at)


def _evaluate_checkpoint_canary(
    model,
    canary_samples,
    canary_teacher_map,
    tokenizer,
    device,
    cfg,
    va_norm_stats=None,
    quantile_head=None,
    pivot_residual_head=None,
    pivot_trust_head=None,
    va_residual_head=None,
    group_routed_residual_head=None,
) -> dict | None:
    if not canary_samples or not canary_teacher_map:
        return None

    canary_preds = _predict(
        model,
        canary_samples,
        tokenizer,
        device,
        cfg,
        va_norm_stats=va_norm_stats,
        quantile_head=quantile_head,
        pivot_residual_head=pivot_residual_head,
        pivot_trust_head=pivot_trust_head,
        va_residual_head=va_residual_head,
        group_routed_residual_head=group_routed_residual_head,
    )

    teacher_sq_errors = []
    label_sq_errors = []
    per_group_teacher_sq: dict[str, list[float]] = {}
    per_group_label_sq: dict[str, list[float]] = {}

    for sample, pred in zip(canary_samples, canary_preds):
        sample_key = make_sample_key(sample["id"], sample["aspect_index"], sample.get("lang"))
        teacher_rec = canary_teacher_map.get(sample_key)
        if teacher_rec is None or not teacher_rec.get("teacher_mask", 0):
            continue

        group = _sample_group_name(sample)
        teacher_sq = 0.5 * (
            (float(pred["v_pred"]) - float(teacher_rec["teacher_v"])) ** 2
            + (float(pred["a_pred"]) - float(teacher_rec["teacher_a"])) ** 2
        )
        teacher_sq_errors.append(teacher_sq)
        per_group_teacher_sq.setdefault(group, []).append(teacher_sq)

        if sample.get("v") is not None and sample.get("a") is not None:
            label_sq = 0.5 * (
                (float(pred["v_pred"]) - float(sample["v"])) ** 2
                + (float(pred["a_pred"]) - float(sample["a"])) ** 2
            )
            label_sq_errors.append(label_sq)
            per_group_label_sq.setdefault(group, []).append(label_sq)

    if not teacher_sq_errors:
        return None

    per_group_teacher_rmse = {
        group: float(np.sqrt(np.mean(values)))
        for group, values in sorted(per_group_teacher_sq.items())
        if values
    }
    per_group_label_rmse = {
        group: float(np.sqrt(np.mean(values)))
        for group, values in sorted(per_group_label_sq.items())
        if values
    }
    teacher_rmse = float(np.sqrt(np.mean(teacher_sq_errors)))
    label_rmse = float(np.sqrt(np.mean(label_sq_errors))) if label_sq_errors else None
    teacher_macro_rmse = (
        float(np.mean(list(per_group_teacher_rmse.values())))
        if per_group_teacher_rmse
        else None
    )
    label_macro_rmse = (
        float(np.mean(list(per_group_label_rmse.values())))
        if per_group_label_rmse
        else None
    )
    return {
        "num_samples": len(teacher_sq_errors),
        "num_groups": len(per_group_teacher_rmse),
        "teacher_rmse_va": teacher_rmse,
        "teacher_macro_rmse_va": teacher_macro_rmse,
        "label_rmse_va": label_rmse,
        "label_macro_rmse_va": label_macro_rmse,
        "per_group_teacher_rmse_va": per_group_teacher_rmse,
        "per_group_label_rmse_va": per_group_label_rmse,
    }


def _checkpoint_canary_metric_value(canary_metrics: dict | None, metric_name: str) -> float | None:
    if not canary_metrics:
        return None
    value = canary_metrics.get(metric_name)
    return None if value is None else float(value)


def _checkpoint_canary_guard_reasons(
    checkpoint_canary_cfg: dict,
    canary_score: float | None,
    baseline_score: float | None,
) -> list[str]:
    if canary_score is None:
        return []

    reasons = []
    max_score = checkpoint_canary_cfg.get("max_score")
    if max_score is not None and canary_score > max_score:
        reasons.append(
            f"{checkpoint_canary_cfg['metric']}={canary_score:.6f} exceeds max_score={max_score:.6f}"
        )

    max_increase = checkpoint_canary_cfg.get("max_increase")
    if max_increase is not None and baseline_score is not None and canary_score > baseline_score + max_increase:
        reasons.append(
            f"{checkpoint_canary_cfg['metric']}={canary_score:.6f} exceeds baseline+max_increase="
            f"{baseline_score + max_increase:.6f}"
        )
    return reasons


def _set_training_mode(model, polarity_head=None, region_cls_head=None, quantile_head=None,
                       pivot_gate_head=None, pivot_residual_head=None,
                       pivot_trust_head=None, va_residual_head=None,
                       group_routed_residual_head=None, soft_condition_label_head=None):
    model.train()
    if polarity_head is not None:
        polarity_head.train()
    if region_cls_head is not None:
        region_cls_head.train()
    if quantile_head is not None:
        quantile_head.train()
    if pivot_gate_head is not None:
        pivot_gate_head.train()
    if pivot_residual_head is not None:
        pivot_residual_head.train()
    if pivot_trust_head is not None:
        pivot_trust_head.train()
    if va_residual_head is not None:
        va_residual_head.train()
    if group_routed_residual_head is not None:
        group_routed_residual_head.train()
    if soft_condition_label_head is not None:
        soft_condition_label_head.train()


def train_small_llm_va(cfg: dict) -> dict:
    from src.train.execution import require_training_release
    require_training_release(cfg)
    # Validate before initializing distributed/CUDA state or loading base weights.
    matched_protocol = validate_matched_protocol(cfg)
    # A missing continuation is fatal before model construction/CUDA setup.
    required_pretrain_checkpoint = (
        resolve_pretrain_checkpoint(cfg["pretrain_from"], PROJECT_ROOT)
        if cfg.get("pretrain_from") else None
    )
    if matched_protocol:
        from src.data.matched_data import validate_matched_data_config
        validate_matched_data_config(cfg)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    dist_cfg = cfg.get("distributed", {}) or {}
    dist_ctx = init_distributed_context(
        cfg.get("device", "auto"),
        backend=str(dist_cfg.get("backend", "nccl")),
    )
    device = dist_ctx.device
    if device.type == "cuda":
        set_cuda_seed(seed)
    base_run_dir = make_run_dir(base=cfg.get("output_dir", "outputs/runs"), tag=cfg.get("tag", "small_llm_va"), seed=seed) if dist_ctx.is_main else None
    shared_run_dir = broadcast_path(base_run_dir, dist_ctx) if dist_ctx.enabled else base_run_dir
    run_dir = shared_run_dir if dist_ctx.is_main else (shared_run_dir / f"rank{dist_ctx.rank}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if dist_ctx.is_main:
        save_config(cfg, shared_run_dir)
        add_file_handler(logger, shared_run_dir / "stdout.log")
    else:
        add_file_handler(logger, run_dir / f"stdout.rank{dist_ctx.rank}.log")
    _materialize_protected_repair_bundle(cfg)
    _validate_metadata_conditioning_cfg(cfg)
    _validate_metadata_output_residual_cfg(cfg)
    _validate_factorized_output_residual_cfg(cfg)
    _validate_soft_condition_label_supervision_cfg(cfg)
    _validate_text_routing_detector_cfg(cfg)
    _validate_gate_trajectory_schedule_cfg(cfg)
    _validate_confidence_gated_composition_cfg(cfg)
    _validate_branch_decorrelation_cfg(cfg)
    _validate_layer_consistency_routing_cfg(cfg)
    _validate_ema_route_teacher_cfg(cfg)
    _validate_stage1_pooled_feature_anchor_cfg(cfg)
    _validate_alternating_optimization_cfg(cfg)
    _validate_protected_repair_bundle_cfg(cfg)
    _validate_arousal_anti_collapse_cfg(cfg)
    _validate_pivot_hardness_gate_cfg(cfg)
    _validate_pivot_a_residual_cfg(cfg)
    _validate_pivot_trust_residual_cfg(cfg)
    _validate_va_residual_corrector_cfg(cfg)
    _validate_post_ckpt_repair_adapter_cfg(cfg)
    _validate_group_routed_va_residual_cfg(cfg)
    _validate_shared_group_routed_va_residual_cfg(cfg)
    _validate_group_dro_cfg(cfg)
    _validate_group_balanced_sampling_cfg(cfg)

    if matched_protocol and not cfg.get("run_test", False):
        # Do not even open formal test files during development-only runs.
        splits = {name: expand_to_aspect_level(load_samples(cfg, name)) for name in ("train", "dev")}
        splits["test"] = []
    else:
        splits = load_splits(cfg)
    train_data = _merge_auxiliary_data(cfg, splits["train"])
    dev_data = splits["dev"]
    test_data = splits["test"]
    if matched_protocol:
        from src.data.matched_data import apply_training_vocabulary
        data_audit = apply_training_vocabulary(cfg, train_data)
        write_json(run_dir / "matched_training_data_audit.json", data_audit)
        if dist_ctx.is_main:
            save_config(cfg, shared_run_dir)  # Includes vocabulary fixed from actual training rows.
    if not train_data:
        logger.error("No training data found")
        return {"run_dir": str(run_dir)}

    debug_cfg = cfg.get("debug_smoke", {})
    debug_enabled = bool(debug_cfg.get("enabled", False))
    if debug_enabled:
        train_before, dev_before, test_before = len(train_data), len(dev_data), len(test_data)
        train_data = _maybe_limit_samples(train_data, int(debug_cfg.get("train_limit", 0)))
        # Keep dev/test complete by default; truncating aspect-level eval data can break
        # the grouped prediction writer when a sample's aspects are only partially kept.
        if bool(debug_cfg.get("truncate_eval", False)):
            dev_data = _maybe_limit_samples(dev_data, int(debug_cfg.get("dev_limit", 0)))
            test_data = _maybe_limit_samples(test_data, int(debug_cfg.get("test_limit", 0)))
        logger.info(
            "Debug smoke enabled: train %d -> %d, dev %d -> %d, test %d -> %d, "
            "truncate_eval=%s, max_train_batches=%s, log_first_n_steps=%s, log_timing=%s",
            train_before, len(train_data),
            dev_before, len(dev_data),
            test_before, len(test_data),
            debug_cfg.get("truncate_eval", False),
            debug_cfg.get("max_train_batches", 0),
            debug_cfg.get("log_first_n_steps", 3),
            debug_cfg.get("log_timing", True),
        )

    macro_dev_targets = _build_macro_dev_targets(cfg)
    if matched_protocol:
        names = [target["name"] for target in macro_dev_targets]
        if len(names) != 12 or set(names) != set(matched_protocol["training_targets"]):
            raise ValueError("loaded macro-dev target set does not equal the fixed 12 targets")
        empty_targets = [target["name"] for target in macro_dev_targets if not target["dev_data"]]
        if empty_targets or not dev_data:
            raise ValueError(f"matched development data are empty: {empty_targets}")

    # LDS: Label Distribution Smoothing — inject per-sample weights
    lds_cfg = cfg.get("lds", {})
    if lds_cfg.get("enabled", False):
        from src.models.lds_fds import compute_lds_weights_va
        import numpy as _np_lds
        _tv = _np_lds.array([float(s.get("v", 5.0)) for s in train_data])
        _ta = _np_lds.array([float(s.get("a", 5.0)) for s in train_data])
        lds_weights = compute_lds_weights_va(
            _tv, _ta,
            focus_dim=lds_cfg.get("focus_dim", "a"),
            num_bins=lds_cfg.get("num_bins", 50),
            kernel_sigma=lds_cfg.get("kernel_sigma", 2.0),
            reweight=lds_cfg.get("reweight", "sqrt_inv"),
            clip_max=lds_cfg.get("clip_max", 10.0),
        )
        for i, s in enumerate(train_data):
            s["_sample_weight"] = float(s.get("_sample_weight", 1.0)) * float(lds_weights[i])
        logger.info("LDS enabled: focus=%s bins=%d sigma=%.1f reweight=%s weight_range=[%.2f, %.2f]",
                     lds_cfg.get("focus_dim", "a"), lds_cfg.get("num_bins", 50),
                     lds_cfg.get("kernel_sigma", 2.0), lds_cfg.get("reweight", "sqrt_inv"),
                     float(lds_weights.min()), float(lds_weights.max()))

    risk_gate_audit = _apply_risk_gate_weights(cfg, train_data, run_dir)
    defect_focus_audit = _apply_defect_focus_weights(cfg, train_data, run_dir)
    error_aware_sampling_audit = _apply_error_aware_training_fields(cfg, train_data, run_dir)
    repair_pool_audit = _setup_repair_pool(cfg, train_data, run_dir)

    model, tokenizer, hidden_dim = build_small_llm_regressor(cfg)
    metadata_output_freeze_report = _apply_metadata_output_residual_freeze(cfg, model)
    if metadata_output_freeze_report:
        logger.info("Metadata output residual freeze: %s", metadata_output_freeze_report)
    factorized_output_freeze_report = _apply_factorized_output_residual_freeze(cfg, model)
    if factorized_output_freeze_report:
        logger.info("Factorized output residual freeze: %s", factorized_output_freeze_report)
    text_routing_detector_freeze_report = _apply_text_routing_detector_freeze(cfg, model)
    if text_routing_detector_freeze_report:
        logger.info("Text routing detector freeze: %s", text_routing_detector_freeze_report)
    backbone_conditional_freeze_report = _apply_backbone_conditional_adapter_freeze(cfg, model)
    if backbone_conditional_freeze_report:
        logger.info("Backbone conditional adapter freeze: %s", backbone_conditional_freeze_report)
    soft_condition_label_head = _setup_soft_condition_label_supervision(cfg, model, device)

    pretrain_from = cfg.get("pretrain_from")
    ckpt_path = None
    if pretrain_from:
        ckpt_path = required_pretrain_checkpoint
        pretrain_load_cfg = cfg.get("pretrain_load", {}) or {}
        if matched_protocol and (not pretrain_load_cfg.get("load_backbone", True)
                                 or pretrain_load_cfg.get("load_head") == "never"):
            raise ValueError("matched continuation must restore all common backbone/readout weights")
        _, load_report = _load_checkpoint(
            model, ckpt_path, cfg=cfg,
            load_backbone=pretrain_load_cfg.get("load_backbone", True),
            load_head=pretrain_load_cfg.get("load_head", "if_compatible"),
            extra_modules={
                "soft_condition_label_head": soft_condition_label_head,
                "pre_pool_repair": getattr(model, "pre_pool_repair", None),
                "top_block_repair": getattr(model, "top_block_repair", None),
                "text_routing_detector": getattr(model, "text_routing_detector", None),
            },
            matched_stage1=bool(matched_protocol),
        )
        write_json(run_dir / "pretrain_load_report.json", load_report)
        logger.info("Loaded small LLM checkpoint from %s: %s", ckpt_path, load_report)
    freeze_small_llm_layers(model, cfg.get("freeze_until", 0))
    if matched_protocol:
        write_json(run_dir / "checkpoint_storage.json", initialize_checkpoint_storage(model, cfg))
    deepspeed_enabled = _deepspeed_enabled(cfg, dist_ctx)
    if deepspeed_enabled and deepspeed is None:
        raise RuntimeError("cfg.deepspeed.enabled=true but deepspeed is not installed in this Python environment")
    if deepspeed_enabled:
        _patch_deepspeed_count_used_parameters_in_backward()
    fsdp2_enabled = bool(dist_cfg.get("enabled", False)) and dist_ctx.enabled and not deepspeed_enabled
    if dist_ctx.enabled and not fsdp2_enabled:
        if not deepspeed_enabled:
            raise ValueError("WORLD_SIZE>1 detected but cfg.distributed.enabled is false")
    if fsdp2_enabled:
        apply_fsdp2_to_backbone(
            model,
            param_dtype=torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16,
            root_reshard_after_forward=bool(dist_cfg.get("root_reshard_after_forward", False)),
        )
    if not deepspeed_enabled:
        model.to(device)
    backbone_trainable_subset_report = _apply_backbone_trainable_subset(cfg, model)
    anchor_state = _capture_anchor_state(model, cfg)

    teacher_map, teacher_audit = _setup_teacher(cfg, train_data, run_dir)
    checkpoint_canary_cfg = _normalize_checkpoint_canary_cfg(cfg.get("checkpoint_canary"))
    checkpoint_canary_samples, checkpoint_canary_teacher_map, checkpoint_canary_audit = _setup_checkpoint_canary(
        checkpoint_canary_cfg,
        train_data,
        run_dir,
    )
    protected_pool_map, protected_pool_audit = _setup_protected_pool(cfg, train_data, run_dir)
    repair_pool_audit = _exclude_repair_pool_with_protected(cfg, train_data, run_dir, repair_pool_audit)
    translation_map, translation_audit = _setup_translation_pivot(cfg, train_data, dev_data, test_data, run_dir)
    teacher_cfg = cfg.get("teacher", {})
    translation_cfg = cfg.get("translation_pivot", {})
    translation_enabled = bool(translation_cfg.get("enabled", False)) and translation_map is not None
    translation_supervised_weight = float(translation_cfg.get("supervised_weight", 0.5))
    translation_pred_consistency_weight = float(translation_cfg.get("pred_consistency_weight", 0.1))
    translation_feature_consistency_weight = float(translation_cfg.get("feature_consistency_weight", 0.05))
    translation_warmup_updates = max(int(translation_cfg.get("warmup_updates", 0) or 0), 0)
    translation_consistency_start_update = max(
        int(translation_cfg.get("consistency_start_update", translation_warmup_updates) or 0),
        0,
    )
    meta_soft_bootstrap_state = _setup_meta_soft_bootstrap(cfg, device)
    stage1_pooled_feature_anchor_state = _setup_stage1_pooled_feature_anchor(cfg, device)
    repair_pool_cfg = cfg.get("repair_pool", {})
    repair_pool_enabled = bool(repair_pool_cfg.get("enabled", False))
    repair_apply_to_base_loss = bool(repair_pool_cfg.get("apply_to_base_loss", False)) and repair_pool_enabled
    repair_apply_to_residual_loss = bool(repair_pool_cfg.get("apply_to_residual_loss", True)) and repair_pool_enabled
    repair_apply_to_group_routed_loss = bool(repair_pool_cfg.get("apply_to_group_routed_loss", True)) and repair_pool_enabled
    protected_pool_cfg = cfg.get("protected_pool", {})
    protected_pool_enabled = bool(protected_pool_cfg.get("enabled", False))
    protected_anchor_cfg = cfg.get("protected_prediction_anchor", {})
    protected_anchor_enabled = bool(protected_anchor_cfg.get("enabled", False)) and protected_pool_enabled
    repair_gate_cfg = cfg.get("repair_gate", {})
    repair_gate_enabled = bool(repair_gate_cfg.get("enabled", False))
    pivot_gate_cfg = cfg.get("pivot_hardness_gate", {})
    pivot_gate_enabled = bool(pivot_gate_cfg.get("enabled", False)) and translation_enabled
    pivot_gate_mode = str(pivot_gate_cfg.get("mode", "heuristic")).strip().lower()
    pivot_gate_head = None
    if pivot_gate_enabled and pivot_gate_mode == "learned":
        from src.models.heads import PivotHardnessGateHead
        pivot_gate_head = PivotHardnessGateHead(
            input_dim=int(pivot_gate_cfg.get("input_dim", 5)),
            hidden_dim=int(pivot_gate_cfg.get("hidden_dim", 16)),
            dropout=float(pivot_gate_cfg.get("dropout", 0.0)),
        ).to(device)
    pivot_residual_cfg = cfg.get("pivot_a_residual", {})
    pivot_residual_enabled = bool(pivot_residual_cfg.get("enabled", False)) and translation_enabled
    pivot_residual_head = None
    if pivot_residual_enabled:
        from src.models.heads import PivotAResidualHead
        pivot_residual_head = PivotAResidualHead(
            input_dim=int(pivot_residual_cfg.get("input_dim", 5)),
            hidden_dim=int(pivot_residual_cfg.get("hidden_dim", 16)),
            dropout=float(pivot_residual_cfg.get("dropout", 0.0)),
        ).to(device)
    pivot_trust_cfg = cfg.get("pivot_trust_residual", {})
    pivot_trust_enabled = bool(pivot_trust_cfg.get("enabled", False)) and translation_enabled
    pivot_trust_head = None
    if pivot_trust_enabled:
        from src.models.heads import PivotTrustResidualHead
        pivot_trust_head = PivotTrustResidualHead(
            input_dim=int(pivot_trust_cfg.get("input_dim", 5)),
            pair_hidden=int(pivot_trust_cfg.get("pair_hidden", 16)),
            dropout=float(pivot_trust_cfg.get("dropout", 0.0)),
        ).to(device)
    va_residual_cfg = cfg.get("va_residual_corrector", {})
    va_residual_enabled = bool(va_residual_cfg.get("enabled", False))
    va_residual_head = None
    if va_residual_enabled:
        from src.models.heads import VAResidualCorrectorHead
        va_residual_head = VAResidualCorrectorHead(
            hidden_dim=hidden_dim,
            corrector_hidden=int(va_residual_cfg.get("hidden_dim", 128)),
            dropout=float(va_residual_cfg.get("dropout", 0.1)),
            output_dim=2,
            learned_gate=bool(va_residual_cfg.get("learned_gate", True)),
            per_dim_gate=bool(va_residual_cfg.get("per_dim_gate", False)),
            zero_init=bool(va_residual_cfg.get("zero_init", True)),
            gate_bias_init=float(va_residual_cfg.get("gate_bias_init", 0.0)),
        ).to(device)
    shared_group_routed_cfg = cfg.get("shared_group_routed_va_residual", {})
    shared_group_routed_enabled = bool(shared_group_routed_cfg.get("enabled", False))
    group_routed_cfg = cfg.get("group_routed_va_residual", {})
    group_routed_enabled = bool(group_routed_cfg.get("enabled", False)) or shared_group_routed_enabled
    active_group_routed_cfg = shared_group_routed_cfg if shared_group_routed_enabled else group_routed_cfg
    group_routed_residual_head = None
    if group_routed_enabled:
        langs = _resolve_group_values_from_cfg(active_group_routed_cfg, cfg, "lang")
        domains = _resolve_group_values_from_cfg(active_group_routed_cfg, cfg, "domain")
        pair_input_dim = int(active_group_routed_cfg.get("pair_input_dim", 8)) if bool(active_group_routed_cfg.get("use_pivot_features", False)) else 0
        if shared_group_routed_enabled:
            from src.models.heads import SharedPlusRoutedVAResidualHead
            group_routed_residual_head = SharedPlusRoutedVAResidualHead(
                hidden_dim=hidden_dim,
                langs=langs,
                domains=domains,
                shared_hidden=int(active_group_routed_cfg.get("shared_hidden", 128)),
                feature_hidden=int(active_group_routed_cfg.get("feature_hidden", 128)),
                group_embed_dim=int(active_group_routed_cfg.get("group_embed_dim", 16)),
                router_hidden=int(active_group_routed_cfg.get("router_hidden", 32)),
                expert_hidden=int(active_group_routed_cfg.get("expert_hidden", 128)),
                num_experts=int(active_group_routed_cfg.get("num_experts", 4)),
                top_k=int(active_group_routed_cfg.get("top_k", 2)),
                pair_input_dim=pair_input_dim,
                dropout=float(active_group_routed_cfg.get("dropout", 0.1)),
                router_aux_weight=float(active_group_routed_cfg.get("router_aux_weight", 0.01)),
                router_z_weight=float(active_group_routed_cfg.get("router_z_weight", 0.0)),
                shared_learned_gate=bool(active_group_routed_cfg.get("shared_learned_gate", True)),
                shared_per_dim_gate=bool(active_group_routed_cfg.get("shared_per_dim_gate", False)),
                shared_gate_bias_init=float(active_group_routed_cfg.get("shared_gate_bias_init", 0.0)),
                shared_bound_mode=str(active_group_routed_cfg.get("shared_bound_mode", "tanh")),
                shared_clamp_delta=float(active_group_routed_cfg.get("shared_clamp_delta", active_group_routed_cfg.get("clamp_delta", 1.0))),
                total_clamp_delta=float(active_group_routed_cfg.get("clamp_delta", 1.0)),
                lang_alias_map=active_group_routed_cfg.get("lang_alias_map", {}),
                domain_alias_map=active_group_routed_cfg.get("domain_alias_map", {}),
            ).to(device)
        else:
            from src.models.heads import GroupRoutedVAResidualHead
            group_routed_residual_head = GroupRoutedVAResidualHead(
                hidden_dim=hidden_dim,
                langs=langs,
                domains=domains,
                feature_hidden=int(active_group_routed_cfg.get("feature_hidden", 128)),
                group_embed_dim=int(active_group_routed_cfg.get("group_embed_dim", 16)),
                router_hidden=int(active_group_routed_cfg.get("router_hidden", 32)),
                expert_hidden=int(active_group_routed_cfg.get("expert_hidden", 128)),
                num_experts=int(active_group_routed_cfg.get("num_experts", 4)),
                top_k=int(active_group_routed_cfg.get("top_k", 2)),
                pair_input_dim=pair_input_dim,
                dropout=float(active_group_routed_cfg.get("dropout", 0.1)),
                router_aux_weight=float(active_group_routed_cfg.get("router_aux_weight", 0.01)),
                router_z_weight=float(active_group_routed_cfg.get("router_z_weight", 0.0)),
                lang_alias_map=active_group_routed_cfg.get("lang_alias_map", {}),
                domain_alias_map=active_group_routed_cfg.get("domain_alias_map", {}),
            ).to(device)
    small_llm_cfg = cfg.get("small_llm", {})
    aux_cfg = cfg.get("auxiliary_data", {})
    macro_dev_cfg = cfg.get("macro_dev_eval", {})
    macro_dev_enabled = bool(macro_dev_cfg.get("enabled", False)) and len(macro_dev_targets) > 0
    macro_dev_use_for_best = bool(macro_dev_cfg.get("use_for_best", True)) if macro_dev_enabled else False
    macro_dev_selection_cfg = macro_dev_cfg.get("selection", {})
    macro_dev_selection_metric = (
        macro_dev_selection_metric_name(macro_dev_selection_cfg)
        if macro_dev_enabled and macro_dev_use_for_best
        else "dev_rmse_va"
    )
    if translation_enabled:
        logger.info(
            "Translation pivot enabled: supervised_weight=%.3f pred_consistency=%.3f feat_consistency=%.3f warmup_updates=%d consistency_start_update=%d",
            translation_supervised_weight,
            translation_pred_consistency_weight,
            translation_feature_consistency_weight,
            translation_warmup_updates,
            translation_consistency_start_update,
        )
    if pivot_gate_enabled:
        logger.info(
            "Pivot hardness gate enabled: mode=%s focus_dim=%s start_update=%d min_multiplier=%.3f max_multiplier=%.3f combine=%s",
            pivot_gate_mode,
            pivot_gate_cfg.get("focus_dim", "a"),
            int(pivot_gate_cfg.get("start_update", 0) or 0),
            float(pivot_gate_cfg.get("min_multiplier", 1.0)),
            float(pivot_gate_cfg.get("max_multiplier", 2.0)),
            pivot_gate_cfg.get("combine", "product"),
        )
    if pivot_residual_enabled:
        logger.info(
            "Pivot A residual enabled: start_update=%d loss_weight=%.3f clamp_delta=%.3f",
            int(pivot_residual_cfg.get("start_update", 0) or 0),
            float(pivot_residual_cfg.get("loss_weight", 1.0)),
            float(pivot_residual_cfg.get("clamp_delta", 1.0)),
        )
    if pivot_trust_enabled:
        logger.info(
            "Pivot trust residual enabled: start_update=%d loss_weight=%.3f clamp_delta=%.3f pair_hidden=%d",
            int(pivot_trust_cfg.get("start_update", 0) or 0),
            float(pivot_trust_cfg.get("loss_weight", 1.0)),
            float(pivot_trust_cfg.get("clamp_delta", 1.0)),
            int(pivot_trust_cfg.get("pair_hidden", 16)),
        )
    if va_residual_enabled:
        logger.info(
            "VA residual corrector enabled: start_update=%d loss_weight=%.3f clamp_delta=%.3f hidden=%d gate=%s focus=%s bound=%s zero_init=%s",
            int(va_residual_cfg.get("start_update", 0) or 0),
            float(va_residual_cfg.get("loss_weight", 1.0)),
            float(va_residual_cfg.get("clamp_delta", 1.0)),
            int(va_residual_cfg.get("hidden_dim", 128)),
            bool(va_residual_cfg.get("learned_gate", True)),
            va_residual_cfg.get("focus_dim", "va"),
            va_residual_cfg.get("bound_mode", "tanh"),
            bool(va_residual_cfg.get("zero_init", True)),
        )
    factorized_output_residual_cfg = cfg.get("factorized_output_residual", {})
    factorized_output_residual_enabled = bool(factorized_output_residual_cfg.get("enabled", False))
    if factorized_output_residual_enabled:
        logger.info(
            "Factorized output residual enabled: condition_source=%s shared=%s lang_branch=%s domain_branch=%s composition=%s clamp=%.3f delta_sup=%.3f easy_zero=%.3f hard_threshold=%.3f comp_threshold=%.3f feature_repair=%s feature_repair_scale=%.3f feature_repair_hard_only=%s repair_probe=%s",
            factorized_output_residual_cfg.get("condition_source", "metadata"),
            bool(factorized_output_residual_cfg.get("use_shared", True)),
            bool(factorized_output_residual_cfg.get("use_lang_branch", True)),
            bool(factorized_output_residual_cfg.get("use_domain_branch", True)),
            bool(factorized_output_residual_cfg.get("use_composition", True)),
            float(factorized_output_residual_cfg.get("clamp_delta", 1.0)),
            float(factorized_output_residual_cfg.get("delta_supervision_weight", 0.0)),
            float(factorized_output_residual_cfg.get("easy_zero_weight", 0.0)),
            float(factorized_output_residual_cfg.get("hard_routing_threshold", 0.0)),
            float(factorized_output_residual_cfg.get("composition_activation_threshold", 0.0)),
            bool(factorized_output_residual_cfg.get("feature_repair_enabled", False)),
            float(factorized_output_residual_cfg.get("feature_repair_scale", 0.5)),
            bool(factorized_output_residual_cfg.get("feature_repair_hard_only", False)),
            bool(factorized_output_residual_cfg.get("repair_probe_enabled", False)),
        )
    text_routing_detector_cfg = cfg.get("text_routing_detector", {})
    if text_routing_detector_cfg.get("enabled", False):
        logger.info(
            "Text routing detector enabled: pooling=%s hidden=%d loss_weight=%.3f lang_weight=%.3f domain_weight=%.3f temp=%.3f",
            text_routing_detector_cfg.get("pooling", "masked_mean"),
            int(text_routing_detector_cfg.get("detector_hidden", 128)),
            float(text_routing_detector_cfg.get("loss_weight", 1.0)),
            float(text_routing_detector_cfg.get("lang_weight", 1.0)),
            float(text_routing_detector_cfg.get("domain_weight", 1.0)),
            float(text_routing_detector_cfg.get("temperature", 1.0)),
        )
    meta_cfg = cfg.get("metadata_conditioning", {})
    if meta_cfg.get("enabled", False):
        logger.info(
            "Metadata conditioning enabled: langs=%d domains=%d embed_dim=%d hidden=%d gate=%s",
            len(_resolve_group_values_from_cfg(meta_cfg, cfg, "lang")),
            len(_resolve_group_values_from_cfg(meta_cfg, cfg, "domain")),
            int(meta_cfg.get("embed_dim", 16)),
            int(meta_cfg.get("conditioner_hidden", 128)),
            bool(meta_cfg.get("learned_gate", True)),
        )
    meta_out_cfg = cfg.get("metadata_output_residual", {})
    if meta_out_cfg.get("enabled", False):
        logger.info(
            "Metadata output residual enabled: langs=%d domains=%d embed_dim=%d hidden=%d per_dim_gate=%s detach_base_pred=%s clamp=%.3f",
            len(_resolve_group_values_from_cfg(meta_out_cfg, cfg, "lang")),
            len(_resolve_group_values_from_cfg(meta_out_cfg, cfg, "domain")),
            int(meta_out_cfg.get("embed_dim", 16)),
            int(meta_out_cfg.get("conditioner_hidden", 128)),
            bool(meta_out_cfg.get("per_dim_gate", True)),
            bool(meta_out_cfg.get("detach_base_prediction", True)),
            float(meta_out_cfg.get("clamp_delta", 0.5)),
        )
    repair_pool_cfg = cfg.get("repair_pool", {})
    if repair_pool_cfg.get("enabled", False):
        logger.info(
            "Repair pool enabled: source=%s mode=%s keep_quantile=%s min_score=%s apply_to_base=%s apply_to_residual=%s apply_to_group_routed=%s",
            repair_pool_cfg.get("source", "artifact"),
            repair_pool_cfg.get("mode", "hard_mask"),
            repair_pool_cfg.get("keep_quantile"),
            repair_pool_cfg.get("min_score"),
            bool(repair_pool_cfg.get("apply_to_base_loss", False)),
            bool(repair_pool_cfg.get("apply_to_residual_loss", True)),
            bool(repair_pool_cfg.get("apply_to_group_routed_loss", True)),
        )
    protected_pool_cfg = cfg.get("protected_pool", {})
    if protected_pool_cfg.get("enabled", False):
        logger.info(
            "Protected pool enabled: source=%s use_confidence=%s confidence_threshold=%.3f",
            protected_pool_cfg.get("source", "artifact"),
            bool(protected_pool_cfg.get("use_confidence", False)),
            float(protected_pool_cfg.get("confidence_threshold", 0.0)),
        )
    protected_anchor_cfg = cfg.get("protected_prediction_anchor", {})
    if protected_anchor_cfg.get("enabled", False):
        logger.info(
            "Protected prediction anchor enabled: start_update=%d loss_weight=%.3f focus=%s",
            int(protected_anchor_cfg.get("start_update", 0) or 0),
            float(protected_anchor_cfg.get("loss_weight", 0.0)),
            protected_anchor_cfg.get("focus_dim", "va"),
        )
    repair_gate_cfg = cfg.get("repair_gate", {})
    if repair_gate_cfg.get("enabled", False):
        logger.info(
            "Repair gate regularization enabled: repair_open=%.3f protected_closed=%.3f sparse=%.3f",
            float(repair_gate_cfg.get("repair_open_weight", 0.0)),
            float(repair_gate_cfg.get("protected_closed_weight", 0.0)),
            float(repair_gate_cfg.get("global_sparse_weight", 0.0)),
        )
    if group_routed_enabled:
        active_group_cfg = shared_group_routed_cfg if shared_group_routed_enabled else group_routed_cfg
        logger.info(
            "%s enabled: start_update=%d loss_weight=%.3f clamp_delta=%.3f experts=%d top_k=%d pivot=%s lang_alias=%s domain_alias=%s",
            "Shared+group-routed residual" if shared_group_routed_enabled else "Group-routed VA residual",
            int(active_group_cfg.get("start_update", 0) or 0),
            float(active_group_cfg.get("loss_weight", 1.0)),
            float(active_group_cfg.get("clamp_delta", 1.0)),
            int(active_group_cfg.get("num_experts", 4)),
            int(active_group_cfg.get("top_k", 2)),
            bool(active_group_cfg.get("use_pivot_features", False)),
            bool(active_group_cfg.get("lang_alias_map")),
            bool(active_group_cfg.get("domain_alias_map")),
        )
    if cfg.get("group_dro", {}).get("enabled", False):
        logger.info(
            "GroupDRO enabled: group_by=%s weight=%.3f top_k=%d",
            cfg["group_dro"].get("group_by", "domain"),
            float(cfg["group_dro"].get("weight", 0.0)),
            int(cfg["group_dro"].get("top_k_groups", 1)),
        )
    checkpoint_extra_modules = {
        "pivot_gate_head": pivot_gate_head,
        "pivot_residual_head": pivot_residual_head,
        "pivot_trust_head": pivot_trust_head,
        "va_residual_head": va_residual_head,
        "group_routed_residual_head": group_routed_residual_head,
        "soft_condition_label_head": soft_condition_label_head,
        "pre_pool_repair": getattr(_unwrap_training_model(model), "pre_pool_repair", None),
        "top_block_repair": getattr(_unwrap_training_model(model), "top_block_repair", None),
        "text_routing_detector": getattr(_unwrap_training_model(model), "text_routing_detector", None),
    }
    if not matched_protocol and ckpt_path is not None and ckpt_path.exists() and (
        pivot_gate_head is not None
        or pivot_residual_head is not None
        or pivot_trust_head is not None
        or va_residual_head is not None
        or group_routed_residual_head is not None
        or soft_condition_label_head is not None
    ):
        _, aux_load_report = _load_checkpoint(
            model,
            ckpt_path,
            cfg=cfg,
            load_backbone=False,
            load_head="never",
            extra_modules=checkpoint_extra_modules,
        )
        logger.info("Auxiliary module load report: %s", aux_load_report)
    if macro_dev_enabled:
        logger.info(
            "Macro-dev eval enabled: targets=%s use_for_best=%s selection=%s",
            [t["name"] for t in macro_dev_targets],
            macro_dev_use_for_best,
            macro_dev_selection_cfg or {"mode": "macro_mean"},
        )

    # FDS (Feature Distribution Smoothing) - smooths feature statistics across A bins
    fds_cfg = cfg.get("fds", {})
    fds_enabled = bool(fds_cfg.get("enabled", False))
    fds_module = None
    if fds_enabled:
        from src.models.lds_fds import FDS
        fds_module = FDS(
            feature_dim=hidden_dim,
            num_bins=int(fds_cfg.get("num_bins", 50)),
            start_update=int(fds_cfg.get("start_update", 0)),
            start_smooth=int(fds_cfg.get("start_smooth", 1)),
            kernel_sigma=float(fds_cfg.get("kernel_sigma", 2.0)),
        ).to(device)
        logger.info(f"FDS enabled: feature_dim={hidden_dim}, num_bins={fds_cfg.get('num_bins', 50)}, "
                     f"sigma={fds_cfg.get('kernel_sigma', 2.0)}, start_smooth={fds_cfg.get('start_smooth', 1)}")

    dataloader_num_workers = max(int(cfg.get("dataloader_num_workers", 4)), 0)
    dataloader_pin_memory = bool(cfg.get("dataloader_pin_memory", device.type == "cuda"))
    dataloader_persistent_workers = bool(
        cfg.get("dataloader_persistent_workers", dataloader_num_workers > 0)
    )
    train_sampler, error_aware_sampler_audit = _build_error_aware_sampler(cfg, train_data, run_dir)
    group_sampling_audit = None
    if train_sampler is None:
        train_sampler, group_sampling_audit = _build_group_balanced_sampler(cfg, train_data, run_dir)

    train_loader = build_dimasr_dataloader(
        train_data,
        tokenizer,
        batch_size=cfg.get("batch_size", 1),
        shuffle=train_sampler is None,
        max_len=cfg.get("max_len", 256),
        input_format=cfg.get("input_format", "instruction_va_v1"),
        teacher_map=teacher_map,
        prompt_template=small_llm_cfg.get("prompt_template"),
        translation_map=translation_map,
        translation_pivot=translation_cfg,
        teacher_explanation_as_input=teacher_cfg.get("explanation_as_input", False),
        teacher_explanation_prefix=teacher_cfg.get("explanation_prefix", "teacher_explanation"),
        teacher_explanation_min_quality=teacher_cfg.get("teacher_explanation_min_quality", 0.0),
        sampler=train_sampler,
        sampler_seed=(None if cfg.get("matched_protocol", {}).get("protocol_version") == "original_source_stage2_textfix_20261104"
                      else cfg.get("sampler_seed", cfg.get("seed", 42))),
        distributed_num_replicas=dist_ctx.world_size if dist_ctx.enabled else None,
        distributed_rank=dist_ctx.rank if dist_ctx.enabled else None,
        disable_explicit_metadata=bool(cfg.get("disable_explicit_metadata", False)),
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        persistent_workers=dataloader_persistent_workers,
    )

    use_uncertainty = _uses_uncertainty_head(model)
    va_norm_stats = _build_va_norm_stats(train_data, cfg.get("va_target_normalization", False))
    if matched_protocol:
        common_model = _unwrap_training_model(model)
        common_model.matched_va_norm_stats = va_norm_stats
        if ckpt_path is not None and common_model.matched_stage1_va_norm_stats != va_norm_stats:
            raise ValueError("continuation training normalization differs from shared Stage-1 checkpoint")
        if dist_ctx.is_main:
            assets = shared_run_dir / "inference_assets"
            tokenizer.save_pretrained(assets / "tokenizer")
            common_model.backbone.config.save_pretrained(assets / "backbone_config")
        distributed_barrier(dist_ctx)
        write_json(run_dir / "inference_dependencies.json", {
            "effective_config": cfg, "va_norm_stats": va_norm_stats,
            "loading_provenance": copy.deepcopy(getattr(common_model, "loading_provenance", {})),
            "base_revision": getattr(getattr(common_model.backbone, "config", None), "_commit_hash", None),
            "tokenizer_assets": "inference_assets/tokenizer (relative to run root)",
            "backbone_config_assets": "inference_assets/backbone_config (relative to run root)",
            "prediction_policy": "continuous predictions, official [1,9] clamp, no rounding; raw invariance diagnostic is separate",
        })
    if cfg.get("text_only_inference", False):
        # Export preprocessing state so deployment never needs evaluation labels
        # or the original training dataset to denormalize its predictions.
        _unwrap_training_model(model).text_only_va_norm_stats = va_norm_stats

    # A-tail reweighting: compute quantile thresholds from train data (in normalized space)
    _a_tail_lo = None
    _a_tail_hi = None
    if float(cfg.get("a_tail_weight", 1.0)) > 1.0:
        a_vals = np.array([s["a"] for s in train_data if s.get("a") is not None], dtype=np.float32)
        q_lo = float(cfg.get("a_tail_q_lo", 0.2))
        q_hi = float(cfg.get("a_tail_q_hi", 0.8))
        raw_lo = float(np.quantile(a_vals, q_lo))
        raw_hi = float(np.quantile(a_vals, q_hi))
        if va_norm_stats:
            _a_tail_lo = (raw_lo - va_norm_stats["a_mean"]) / va_norm_stats["a_std"]
            _a_tail_hi = (raw_hi - va_norm_stats["a_mean"]) / va_norm_stats["a_std"]
        else:
            _a_tail_lo = raw_lo
            _a_tail_hi = raw_hi
        logger.info(f"A-tail reweighting: q_lo={q_lo}, q_hi={q_hi}, "
                     f"raw=[{raw_lo:.2f}, {raw_hi:.2f}], "
                     f"norm=[{_a_tail_lo:.4f}, {_a_tail_hi:.4f}], "
                     f"weight={cfg.get('a_tail_weight')}")

    quantile_cfg = cfg.get("arousal_quantile", {})
    quantile_enabled = bool(quantile_cfg.get("enabled", False))
    quantile_start_epoch = int(quantile_cfg.get("start_epoch", 0))
    quantile_tail_only = bool(quantile_cfg.get("tail_only", False))
    quantile_tail_lo = None
    quantile_tail_hi = None
    if quantile_enabled and quantile_tail_only:
        a_vals_raw = np.array([s["a"] for s in train_data if s.get("a") is not None], dtype=np.float32)
        tail_q_lo = float(quantile_cfg.get("tail_q_lo", 0.2))
        tail_q_hi = float(quantile_cfg.get("tail_q_hi", 0.8))
        quantile_tail_lo = float(np.quantile(a_vals_raw, tail_q_lo))
        quantile_tail_hi = float(np.quantile(a_vals_raw, tail_q_hi))
        logger.info(
            "W3 quantile tail-only enabled: q_lo=%.2f q_hi=%.2f raw_thresholds=[%.3f, %.3f]",
            tail_q_lo, tail_q_hi, quantile_tail_lo, quantile_tail_hi,
        )

    arousal_cfg = cfg.get("arousal_anti_collapse", {})
    arousal_enabled = bool(arousal_cfg.get("enabled", False))

    # W3: Valence sharpening
    valence_sharp_cfg = cfg.get("valence_sharpening", {})
    valence_sharp_enabled = bool(valence_sharp_cfg.get("enabled", False))

    # D39: Polarity classification auxiliary task
    polarity_cfg = cfg.get("polarity_aux", {})
    polarity_enabled = bool(polarity_cfg.get("enabled", False))
    polarity_head = None
    if polarity_enabled:
        from src.models.heads import PolarityHead
        from src.models.losses import polarity_classification_loss
        polarity_boundaries = tuple(polarity_cfg.get("boundaries", [4.0, 6.0]))
        polarity_head = PolarityHead(
            hidden_dim, n_classes=3,
            dropout=polarity_cfg.get("dropout", 0.1)
        ).to(device)
        logger.info(f"D39: polarity auxiliary enabled, "
                    f"weight={polarity_cfg.get('weight', 0.1)}, "
                    f"boundaries={polarity_boundaries}")

    # D40: VA region classification auxiliary task (Two-Stage)
    region_cls_cfg = cfg.get("va_region_cls", {})
    region_cls_enabled = bool(region_cls_cfg.get("enabled", False))
    region_cls_head = None
    if region_cls_enabled:
        from src.models.heads import VARegionHead
        from src.models.losses import va_region_classification_loss
        region_v_boundaries = region_cls_cfg.get("v_boundaries", [5.0, 7.0])
        region_a_boundaries = region_cls_cfg.get("a_boundaries", [6.0, 6.6])
        n_v_bins = len(region_v_boundaries) + 1
        n_a_bins = len(region_a_boundaries) + 1
        region_cls_head = VARegionHead(
            hidden_dim, n_v_bins=n_v_bins, n_a_bins=n_a_bins,
            dropout=region_cls_cfg.get("dropout", 0.1)
        ).to(device)
        logger.info(f"D40: VA region classification enabled, "
                    f"weight={region_cls_cfg.get('weight', 0.1)}, "
                    f"v_boundaries={region_v_boundaries}, a_boundaries={region_a_boundaries}, "
                    f"grid={n_v_bins}x{n_a_bins}={n_v_bins * n_a_bins} regions")

    # W3: Arousal quantile regression auxiliary task
    quantile_head = None
    if quantile_enabled:
        from src.models.heads import ArousalQuantileHead
        from src.models.losses import quantile_pinball_loss
        quantile_quantiles = quantile_cfg.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
        quantile_head = ArousalQuantileHead(
            hidden_dim,
            quantiles=quantile_quantiles,
            dropout=quantile_cfg.get("dropout", 0.1),
        ).to(device)
        logger.info(f"W3: arousal quantile head enabled, "
                    f"weight={quantile_cfg.get('weight', 0.1)}, "
                    f"quantiles={quantile_quantiles}, "
                    f"start_epoch={quantile_start_epoch}, tail_only={quantile_tail_only}")

    decorr_cfg = cfg.get("va_decorrelation", {})
    decorr_enabled = bool(decorr_cfg.get("enabled", False))
    if decorr_enabled:
        logger.info(f"W3: V-A decorrelation loss enabled, weight={decorr_cfg.get('weight', 0.1)}")

    var_reg_cfg = cfg.get("variance_reg", {})
    var_reg_enabled = bool(var_reg_cfg.get("enabled", False))
    if var_reg_enabled:
        logger.info(f"W3: variance regularization enabled, weight={var_reg_cfg.get('weight', 0.05)}, mode={var_reg_cfg.get('mode', 'ratio')}")

    # EMA variance tracking: batch_size=1 compatible anti-collapse
    ema_var_cfg = cfg.get("ema_variance_tracking", {})
    ema_var_enabled = bool(ema_var_cfg.get("enabled", False))
    ema_var_alpha = float(ema_var_cfg.get("alpha", 0.99))
    ema_var_weight = float(ema_var_cfg.get("weight", 0.1))
    # Initialize EMA state (rough initial estimates)
    ema_pred_a_mean = 5.0
    ema_pred_a_sq = 26.0
    ema_true_a_mean = 5.0
    ema_true_a_sq = 26.0
    if ema_var_enabled:
        logger.info(f"W3: EMA variance tracking enabled, alpha={ema_var_alpha}, weight={ema_var_weight}")

    # Feature Memory Bank + VICReg variance regularization
    mem_bank_cfg = cfg.get("feature_memory_bank", {})
    mem_bank_enabled = bool(mem_bank_cfg.get("enabled", False))
    mem_bank = None
    vicreg_var_weight = float(mem_bank_cfg.get("vicreg_var_weight", 0.05))
    mem_bank_min_samples = int(mem_bank_cfg.get("min_samples", 8))
    if mem_bank_enabled:
        from src.train.feature_memory_bank import FeatureMemoryBank
        mem_bank = FeatureMemoryBank(
            capacity=int(mem_bank_cfg.get("capacity", 32)),
            feature_dim=hidden_dim,
            device=device,
        )
        logger.info(f"W3: Feature memory bank enabled, capacity={mem_bank_cfg.get('capacity', 32)}, "
                     f"vicreg_var_weight={vicreg_var_weight}, min_samples={mem_bank_min_samples}")

    debug_max_train_batches = int(debug_cfg.get("max_train_batches", 0)) if debug_enabled else 0
    debug_log_first_n_steps = int(debug_cfg.get("log_first_n_steps", 3)) if debug_enabled else 0
    debug_log_timing = bool(debug_cfg.get("log_timing", True)) if debug_enabled else False

    grad_accum = max(int(cfg.get("grad_accum", 1)), 1)
    n_epochs = int(cfg.get("n_epochs", 3))
    base_total_updates = math.ceil(len(train_loader) / grad_accum) * max(n_epochs, 1)
    max_updates = max(int(cfg.get("max_updates", 0) or 0), 0)
    total_updates = min(base_total_updates, max_updates) if max_updates > 0 else base_total_updates
    eval_every_updates = max(int(cfg.get("eval_every_updates", 0) or 0), 0)
    save_every_updates = max(int(cfg.get("save_every_updates", 0) or 0), 0)
    max_step_checkpoints_to_keep = max(int(cfg.get("max_step_checkpoints_to_keep", 0) or 0), 0)
    if max_updates > 0:
        logger.info(
            "Max updates override enabled: requested=%s base_total_updates=%s effective_total_updates=%s",
            max_updates,
            base_total_updates,
            total_updates,
        )
    optimizer = _build_optimizer(cfg, model)
    default_lr = float(cfg.get("lr", 2e-4))
    default_weight_decay = float(cfg.get("weight_decay", 0.01))
    # Add polarity head params to optimizer
    if polarity_enabled and polarity_head is not None:
        group = _module_optimizer_group(
            polarity_head,
            polarity_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    # Add region classification head params to optimizer
    if region_cls_enabled and region_cls_head is not None:
        group = _module_optimizer_group(
            region_cls_head,
            region_cls_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    # Add quantile head params to optimizer
    if quantile_enabled and quantile_head is not None:
        group = _module_optimizer_group(
            quantile_head,
            quantile_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    if pivot_gate_head is not None:
        group = _module_optimizer_group(
            pivot_gate_head,
            pivot_gate_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    if pivot_residual_head is not None:
        group = _module_optimizer_group(
            pivot_residual_head,
            pivot_residual_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    if pivot_trust_head is not None:
        group = _module_optimizer_group(
            pivot_trust_head,
            pivot_trust_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    if va_residual_head is not None:
        group = _module_optimizer_group(
            va_residual_head,
            va_residual_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    if group_routed_residual_head is not None:
        group = _module_optimizer_group(
            group_routed_residual_head,
            active_group_routed_cfg,
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    if soft_condition_label_head is not None:
        group = _module_optimizer_group(
            soft_condition_label_head,
            cfg.get("soft_condition_label_supervision", {}),
            default_lr=default_lr,
            default_weight_decay=default_weight_decay,
        )
        if group is not None:
            optimizer.add_param_group(group)
    va_residual_freeze_report = _apply_post_optimizer_freeze_for_va_residual(cfg, model)
    if va_residual_freeze_report:
        logger.info("VA residual post-optimizer freeze: %s", va_residual_freeze_report)
    scheduler = _build_scheduler(cfg, optimizer, total_updates)
    if deepspeed_enabled:
        ds_runtime_cfg = _build_deepspeed_runtime_config(cfg, dist_ctx)
        logger.info("DeepSpeed enabled: %s", ds_runtime_cfg)
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_runtime_cfg,
            dist_init_required=False,
        )

    amp_enabled = bool(cfg.get("amp", device.type == "cuda")) and device.type == "cuda" and not deepspeed_enabled
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(not deepspeed_enabled) and amp_enabled and amp_dtype == torch.float16)
    ema_route_teacher_state = _setup_backbone_conditional_ema_teacher(cfg, model, device)
    wrapped_training_model = _unwrap_training_model(model)
    if matched_protocol:
        write_json(run_dir / "effective_parameters.json", parameter_manifest(
            wrapped_training_model, checkpoint_extra_modules, cfg, optimizer=optimizer,
        ))
    update_observer = None
    if matched_protocol.get("protocol_version") == "method_validation_v2":
        from src.train.validation_v2 import UpdateObserver, assert_effective_freeze
        write_json(run_dir / "effective_freeze.json", assert_effective_freeze(
            wrapped_training_model, checkpoint_extra_modules, cfg))
        update_observer = UpdateObserver(wrapped_training_model, checkpoint_extra_modules, cfg,
                                         run_dir / "observed_parameter_updates.jsonl")
    elif matched_protocol.get("protocol_version") == "original_stage2_reference_20261104":
        # Observation only: retain the recovered training mathematics and use
        # the existing first-32-scalars observer to record actual updates.
        from src.train.validation_v2 import UpdateObserver
        update_observer = UpdateObserver(wrapped_training_model, checkpoint_extra_modules, cfg,
                                         run_dir / "observed_parameter_updates.jsonl")
    base_requires_grad_map = {
        name: bool(param.requires_grad)
        for name, param in wrapped_training_model.named_parameters()
    }
    _refresh_backbone_runtime_controls(model, cfg, 0)
    current_post_ckpt_repair_scale, current_protected_repair_bundle_scale = (
        _refresh_effective_post_ckpt_repair_adapter_scale(model, cfg, 0)
    )

    best_dev_rmse = None
    best_macro_dev_rmse = None
    best_selection_score = None
    best_epoch = -1
    best_update = -1
    global_update = 0
    last_eval_update = -1
    last_eval_rmse = None
    metrics = {
        "run_dir": str(run_dir),
        "hidden_dim": hidden_dim,
        "num_train_samples": len(train_data),
        "num_primary_samples": sum(1 for s in train_data if not s.get("_is_auxiliary")),
        "num_auxiliary_samples": sum(1 for s in train_data if s.get("_is_auxiliary")),
    }
    if eval_every_updates > 0:
        metrics["eval_every_updates"] = eval_every_updates
    if save_every_updates > 0:
        metrics["save_every_updates"] = save_every_updates
    if max_step_checkpoints_to_keep > 0:
        metrics["max_step_checkpoints_to_keep"] = max_step_checkpoints_to_keep
    if eval_every_updates > 0 or save_every_updates > 0:
        logger.info(
            "Step checkpoint/eval enabled: eval_every_updates=%s save_every_updates=%s keep_recent_step_ckpts=%s",
            eval_every_updates or "off",
            save_every_updates or "off",
            max_step_checkpoints_to_keep or "unlimited",
        )
    if teacher_audit:
        metrics["teacher_match_ratio"] = teacher_audit["teacher_match_ratio"]
        metrics["teacher_num_matched"] = teacher_audit["num_teacher_matched"]
        metrics["teacher_num_missing"] = teacher_audit["num_teacher_missing"]
    if checkpoint_canary_audit:
        metrics["checkpoint_canary_enabled"] = True
        metrics["checkpoint_canary_num_samples"] = checkpoint_canary_audit["num_canary_samples"]
        metrics["checkpoint_canary_match_ratio"] = checkpoint_canary_audit["teacher_match_ratio"]
        metrics["checkpoint_canary_metric"] = checkpoint_canary_cfg["metric"]
        metrics["checkpoint_canary_weight"] = checkpoint_canary_cfg["weight"]
        metrics["checkpoint_canary_use_for_best"] = checkpoint_canary_cfg["use_for_best"]
        if checkpoint_canary_cfg.get("max_score") is not None:
            metrics["checkpoint_canary_max_score"] = checkpoint_canary_cfg["max_score"]
        if checkpoint_canary_cfg.get("max_increase") is not None:
            metrics["checkpoint_canary_max_increase"] = checkpoint_canary_cfg["max_increase"]
    if translation_audit:
        metrics["translation_pivot_train_hit_rate"] = translation_audit["train"]["hit_rate"]
        metrics["translation_pivot_train_non_english_hit_rate"] = translation_audit["train"]["non_english_hit_rate"]
        metrics["translation_pivot_enabled"] = True
    if pivot_gate_enabled:
        metrics["pivot_hardness_gate_enabled"] = True
        metrics["pivot_hardness_gate_mode"] = pivot_gate_mode
    if pivot_residual_enabled:
        metrics["pivot_a_residual_enabled"] = True
    if pivot_trust_enabled:
        metrics["pivot_trust_residual_enabled"] = True
    if va_residual_enabled:
        metrics["va_residual_corrector_enabled"] = True
    if cfg.get("metadata_conditioning", {}).get("enabled", False):
        metrics["metadata_conditioning_enabled"] = True
    if cfg.get("metadata_output_residual", {}).get("enabled", False):
        metrics["metadata_output_residual_enabled"] = True
    if cfg.get("factorized_output_residual", {}).get("enabled", False):
        metrics["factorized_output_residual_enabled"] = True
    if cfg.get("backbone_conditional_adapter", {}).get("composition_state_aware", False):
        metrics["backbone_conditional_composition_state_aware"] = True
    if cfg.get("backbone_conditional_adapter_regularization", {}).get("enabled", False):
        metrics["backbone_conditional_adapter_regularization_enabled"] = True
    if cfg.get("gate_trajectory_schedule", {}).get("enabled", False):
        metrics["gate_trajectory_schedule_enabled"] = True
    if cfg.get("confidence_gated_composition", {}).get("enabled", False):
        metrics["confidence_gated_composition_enabled"] = True
    if cfg.get("shared_branch_regularization", {}).get("enabled", False):
        metrics["shared_branch_regularization_enabled"] = True
    if cfg.get("branch_decorrelation", {}).get("enabled", False):
        metrics["branch_decorrelation_enabled"] = True
    if cfg.get("layer_consistency_routing", {}).get("enabled", False):
        metrics["layer_consistency_routing_enabled"] = True
    if ema_route_teacher_state is not None:
        metrics["ema_route_teacher_enabled"] = True
    if cfg.get("alternating_optimization", {}).get("enabled", False):
        metrics["alternating_optimization_enabled"] = True
    if cfg.get("protected_repair_bundle", {}).get("enabled", False):
        metrics["protected_repair_bundle_enabled"] = True
    if soft_condition_label_head is not None:
        metrics["soft_condition_label_supervision_enabled"] = True
    if cfg.get("text_routing_detector", {}).get("enabled", False):
        metrics["text_routing_detector_enabled"] = True
    if meta_soft_bootstrap_state is not None:
        metrics["meta_soft_bootstrap_enabled"] = True
        metrics["meta_soft_bootstrap_teacher_checkpoint"] = meta_soft_bootstrap_state["teacher_checkpoint_path"]
    if stage1_pooled_feature_anchor_state is not None:
        metrics["stage1_pooled_feature_anchor_enabled"] = True
        metrics["stage1_pooled_feature_anchor_teacher_checkpoint"] = (
            stage1_pooled_feature_anchor_state["teacher_checkpoint_path"]
        )
    if repair_pool_audit:
        metrics["repair_pool_enabled"] = True
        metrics["repair_pool_match_rate"] = repair_pool_audit["match_rate"]
        metrics["repair_pool_active_rate"] = repair_pool_audit["active_rate"]
    if protected_pool_audit:
        metrics["protected_pool_enabled"] = True
        metrics["protected_pool_match_ratio"] = protected_pool_audit["teacher_match_ratio"]
        metrics["protected_pool_num_samples"] = protected_pool_audit["num_protected_samples"]
    if cfg.get("protected_prediction_anchor", {}).get("enabled", False):
        metrics["protected_prediction_anchor_enabled"] = True
    if cfg.get("repair_gate", {}).get("enabled", False):
        metrics["repair_gate_enabled"] = True
    if group_routed_enabled:
        metrics["group_routed_va_residual_enabled"] = True
        active_group_cfg = shared_group_routed_cfg if shared_group_routed_enabled else group_routed_cfg
        metrics["group_routed_va_residual_use_pivot_features"] = bool(active_group_cfg.get("use_pivot_features", False))
        if shared_group_routed_enabled:
            metrics["shared_group_routed_va_residual_enabled"] = True
    if risk_gate_audit:
        metrics["risk_gate_enabled"] = True
        metrics["risk_gate_match_rate"] = risk_gate_audit["match_rate"]
        metrics["risk_gate_mean_weight"] = risk_gate_audit["applied_mean"]
    if defect_focus_audit:
        metrics["defect_focus_enabled"] = True
        metrics["defect_focus_match_rate"] = defect_focus_audit["match_rate"]
        metrics["defect_focus_mean_weight"] = defect_focus_audit["applied_mean"]
    if error_aware_sampling_audit:
        metrics["error_aware_sampling_enabled"] = True
        metrics["error_aware_sampling_match_rate"] = error_aware_sampling_audit["match_rate"]
        metrics["error_aware_sampling_weight_mean"] = error_aware_sampling_audit["weight_mean"]
        metrics["error_aware_sampling_hard_threshold"] = error_aware_sampling_audit["hard_threshold"]
        metrics["error_aware_sampling_easy_threshold"] = error_aware_sampling_audit["easy_threshold"]
    if error_aware_sampler_audit:
        metrics["error_aware_sampler_enabled"] = True
        metrics["error_aware_sampler_mean_weight"] = error_aware_sampler_audit["mean_weight"]
    if group_sampling_audit:
        metrics["group_balanced_sampling_enabled"] = True
        metrics["group_balanced_sampling_group_by"] = group_sampling_audit["group_by"]
        metrics["group_balanced_sampling_num_groups"] = group_sampling_audit["num_groups"]
    if va_residual_freeze_report:
        metrics["va_residual_freeze_backbone"] = va_residual_freeze_report["freeze_backbone"]
        metrics["va_residual_freeze_head"] = va_residual_freeze_report["freeze_head"]
        metrics["va_residual_freeze_pool_gate"] = va_residual_freeze_report["freeze_pool_gate"]
    if backbone_conditional_freeze_report:
        metrics["backbone_conditional_freeze_backbone"] = backbone_conditional_freeze_report["freeze_backbone"]
        metrics["backbone_conditional_freeze_head"] = backbone_conditional_freeze_report["freeze_head"]
        metrics["backbone_conditional_freeze_pool_gate"] = backbone_conditional_freeze_report["freeze_pool_gate"]
        metrics["backbone_conditional_keep_adapter_trainable"] = backbone_conditional_freeze_report["keep_adapter_trainable"]
    if cfg.get("post_ckpt_repair_adapter", {}).get("enabled", False):
        metrics["post_ckpt_repair_adapter_enabled"] = True
        metrics["post_ckpt_repair_adapter_start_update"] = int(
            cfg["post_ckpt_repair_adapter"].get("start_update", 0) or 0
        )
        metrics["post_ckpt_repair_adapter_warmup_updates"] = int(
            cfg["post_ckpt_repair_adapter"].get("warmup_updates", 0) or 0
        )
    if cfg.get("group_dro", {}).get("enabled", False):
        metrics["group_dro_enabled"] = True
        metrics["group_dro_group_by"] = cfg["group_dro"].get("group_by", "domain")
        metrics["group_dro_weight"] = float(cfg["group_dro"].get("weight", 0.0))

    checkpoint_canary_baseline_score = None
    if checkpoint_canary_samples and checkpoint_canary_teacher_map:
        _refresh_backbone_runtime_controls(model, cfg, 0)
        current_post_ckpt_repair_scale, current_protected_repair_bundle_scale = (
            _refresh_effective_post_ckpt_repair_adapter_scale(model, cfg, 0)
        )
        checkpoint_canary_initial_metrics = _evaluate_checkpoint_canary(
            model,
            checkpoint_canary_samples,
            checkpoint_canary_teacher_map,
            tokenizer,
            device,
            cfg,
            va_norm_stats=va_norm_stats,
            quantile_head=quantile_head if quantile_enabled else None,
            pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
            pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
            va_residual_head=va_residual_head if va_residual_enabled else None,
            group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
        )
        if checkpoint_canary_initial_metrics is not None:
            checkpoint_canary_baseline_score = _checkpoint_canary_metric_value(
                checkpoint_canary_initial_metrics,
                checkpoint_canary_cfg["metric"],
            )
            metrics["checkpoint_canary_initial_teacher_rmse_va"] = checkpoint_canary_initial_metrics["teacher_rmse_va"]
            metrics["checkpoint_canary_initial_teacher_macro_rmse_va"] = checkpoint_canary_initial_metrics["teacher_macro_rmse_va"]
            if checkpoint_canary_initial_metrics.get("label_rmse_va") is not None:
                metrics["checkpoint_canary_initial_label_rmse_va"] = checkpoint_canary_initial_metrics["label_rmse_va"]
            if checkpoint_canary_initial_metrics.get("label_macro_rmse_va") is not None:
                metrics["checkpoint_canary_initial_label_macro_rmse_va"] = checkpoint_canary_initial_metrics["label_macro_rmse_va"]
            if checkpoint_canary_baseline_score is not None:
                metrics["checkpoint_canary_initial_selection_score"] = checkpoint_canary_baseline_score
            logger.info(
                "Checkpoint canary baseline: %s=%.6f teacher_macro=%.6f label_macro=%s samples=%s groups=%s",
                checkpoint_canary_cfg["metric"],
                checkpoint_canary_baseline_score if checkpoint_canary_baseline_score is not None else float("nan"),
                checkpoint_canary_initial_metrics["teacher_macro_rmse_va"],
                (
                    f"{checkpoint_canary_initial_metrics['label_macro_rmse_va']:.6f}"
                    if checkpoint_canary_initial_metrics.get("label_macro_rmse_va") is not None
                    else "n/a"
                ),
                checkpoint_canary_initial_metrics["num_samples"],
                checkpoint_canary_initial_metrics["num_groups"],
            )

    last_alternating_phase = None
    last_alternating_phase_update = None
    stop_after_max_updates = False
    for epoch in range(n_epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        _set_training_mode(
            model,
            polarity_head=polarity_head if polarity_enabled else None,
            region_cls_head=region_cls_head if region_cls_enabled else None,
            quantile_head=quantile_head if quantile_enabled else None,
            pivot_gate_head=pivot_gate_head,
            pivot_residual_head=pivot_residual_head,
            pivot_trust_head=pivot_trust_head,
            va_residual_head=va_residual_head,
            group_routed_residual_head=group_routed_residual_head,
            soft_condition_label_head=soft_condition_label_head,
        )
        if deepspeed_enabled:
            model.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_teacher_loss = 0.0
        total_arousal_aux_loss = 0.0
        total_polarity_loss = 0.0
        total_region_cls_loss = 0.0
        total_quantile_loss = 0.0
        total_decorr_loss = 0.0
        total_var_reg_loss = 0.0
        total_ema_var_loss = 0.0
        total_vicreg_bank_loss = 0.0
        total_translation_pivot_loss = 0.0
        total_translation_pred_consistency_loss = 0.0
        total_translation_feat_consistency_loss = 0.0
        total_pivot_gate_multiplier = 0.0
        n_pivot_gate_batches = 0
        total_pivot_residual_loss = 0.0
        total_pivot_trust_loss = 0.0
        total_va_residual_loss = 0.0
        total_va_residual_gate = 0.0
        total_va_residual_correction = 0.0
        n_va_residual_batches = 0
        total_group_routed_residual_loss = 0.0
        total_group_routed_router_reg_loss = 0.0
        total_group_dro_loss = 0.0
        total_protected_anchor_loss = 0.0
        total_repair_gate_loss = 0.0
        total_repair_gate_repair_mean = 0.0
        total_repair_gate_protected_mean = 0.0
        n_repair_gate_batches = 0
        total_pivot_trust_gate = 0.0
        total_pivot_trust_correction = 0.0
        n_pivot_trust_batches = 0
        total_group_routed_correction = 0.0
        n_group_routed_batches = 0
        total_anchor_loss = 0.0
        total_backbone_conditional_adapter_reg_loss = 0.0
        total_branch_decorrelation_loss = 0.0
        total_layer_consistency_loss = 0.0
        total_ema_route_teacher_loss = 0.0
        total_ema_route_teacher_momentum = 0.0
        n_ema_route_teacher_updates = 0
        total_factorized_output_delta_loss = 0.0
        total_factorized_output_easy_zero_loss = 0.0
        total_factorized_output_feature_anchor_loss = 0.0
        total_factorized_output_repair_l2_loss = 0.0
        total_factorized_output_easy_feature_zero_loss = 0.0
        total_factorized_output_repair_probe_loss = 0.0
        total_soft_condition_label_supervision_loss = 0.0
        total_soft_condition_lang_correct = 0.0
        total_soft_condition_lang_count = 0
        total_soft_condition_domain_correct = 0.0
        total_soft_condition_domain_count = 0
        total_text_routing_detector_loss = 0.0
        total_text_routing_lang_correct = 0.0
        total_text_routing_lang_count = 0
        total_text_routing_domain_correct = 0.0
        total_text_routing_domain_count = 0
        total_meta_soft_bootstrap_loss = 0.0
        total_stage1_pooled_feature_anchor_loss = 0.0
        total_runtime_shared_residual_scale = 0.0
        total_runtime_shared_branch_dropout = 0.0
        total_runtime_composition_gate_max = 0.0
        total_runtime_composition_residual_scale = 0.0
        n_runtime_control_batches = 0
        total_composition_confidence = 0.0
        n_composition_confidence_batches = 0
        total_protected_repair_bundle_scale = 0.0
        n_protected_repair_bundle_batches = 0
        alternating_phase_counts: Counter[str] = Counter()
        n_batches = 0
        teacher_eff_weight = teacher_weight_schedule(
            epoch,
            float(teacher_cfg.get("loss_weight", 0.0)),
            int(teacher_cfg.get("warmup_epochs", 0)),
            int(teacher_cfg.get("ramp_epochs", 0)),
        )
        _refresh_backbone_runtime_controls(model, cfg, global_update)
        current_post_ckpt_repair_scale, current_protected_repair_bundle_scale = (
            _refresh_effective_post_ckpt_repair_adapter_scale(
                model,
                cfg,
                global_update,
            )
        )

        if debug_enabled:
            logger.info(
                "Debug smoke epoch %d/%d: train_loader_batches=%d max_train_batches=%d",
                epoch + 1, n_epochs, len(train_loader), debug_max_train_batches,
            )

        train_iter = iter(train_loader)
        step = 0
        while True:
            if max_updates > 0 and global_update >= total_updates:
                logger.info(
                    "Reached max_updates=%s at epoch=%s step=%s; stopping training loop",
                    total_updates,
                    epoch + 1,
                    step,
                )
                stop_after_max_updates = True
                break
            if debug_max_train_batches and step >= debug_max_train_batches:
                logger.info(
                    "Debug smoke: stopping epoch %d after %d batches",
                    epoch + 1, step,
                )
                break

            fetch_t0 = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            fetch_dt = time.perf_counter() - fetch_t0
            step += 1
            if matched_protocol:
                order_record = batch_order_record(
                    batch, train_data, epoch=epoch, update=global_update + 1,
                    microbatch=step, rank=dist_ctx.rank,
                )
                with (run_dir / f"training_order.rank{dist_ctx.rank}.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(order_record, ensure_ascii=False, allow_nan=False) + "\n")
            debug_step = debug_enabled and step <= debug_log_first_n_steps

            if debug_step and debug_log_timing:
                logger.info(
                    "Debug step %d: fetched batch in %.3fs",
                    step, fetch_dt,
                )

            device_t0 = time.perf_counter()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target_raw = batch["va"].to(device)
            target = _normalize_va(target_raw, va_norm_stats)
            is_auxiliary = batch.get("is_auxiliary")
            if is_auxiliary is not None:
                is_auxiliary = is_auxiliary.to(device)
            sample_weights = _build_sample_weights(batch, target_raw, cfg, device)
            repair_mask, repair_weights = _batch_pool_weights(batch, "repair", device)
            protected_mask, protected_weights = _batch_pool_weights(batch, "protected", device)
            protected_va_raw = batch.get("protected_va")
            if protected_va_raw is not None:
                protected_va_raw = protected_va_raw.to(device)
            base_sample_weights = sample_weights
            if repair_apply_to_base_loss:
                base_sample_weights = _merge_sample_weights(base_sample_weights, repair_weights)
            pivot_ids = None
            pivot_mask = None
            pivot_sample_mask = None
            pivot_sample_weights = None
            pivot_confidence = None
            if translation_enabled and "pivot_input_ids" in batch:
                pivot_ids = batch["pivot_input_ids"].to(device)
                pivot_mask = batch["pivot_attention_mask"].to(device)
                pivot_sample_mask = batch["pivot_mask"].to(device)
                pivot_confidence = batch["pivot_weight"].to(device)
                pivot_sample_weights = _merge_sample_weights(
                    sample_weights,
                    pivot_confidence,
                )
            device_dt = time.perf_counter() - device_t0
            runtime_control_stats = _refresh_backbone_runtime_controls(
                model,
                cfg,
                global_update,
            )
            current_post_ckpt_repair_scale, current_protected_repair_bundle_scale = (
                _refresh_effective_post_ckpt_repair_adapter_scale(
                    model,
                    cfg,
                    global_update,
                )
            )
            current_alternating_phase = _apply_alternating_optimization_phase(
                cfg,
                model,
                base_requires_grad_map,
                global_update,
            )
            if current_alternating_phase is not None and last_alternating_phase_update != global_update:
                alternating_phase_counts[current_alternating_phase] += 1
                last_alternating_phase_update = global_update
                if current_alternating_phase != last_alternating_phase:
                    logger.info(
                        "Alternating optimization phase update=%s phase=%s",
                        global_update,
                        current_alternating_phase,
                    )
                    last_alternating_phase = current_alternating_phase
            condition_kwargs = _resolve_model_condition_kwargs(batch, cfg)

            main_forward_dt = 0.0
            quantile_reencode_dt = 0.0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                # Extract features for auxiliary heads or FDS when needed
                features = None
                pivot_features = None
                pivot_student_va = None
                pivot_student_va_raw = None
                pivot_residual_loss = torch.tensor(0.0, device=device)
                pivot_trust_loss = torch.tensor(0.0, device=device)
                va_residual_loss = torch.tensor(0.0, device=device)
                group_routed_residual_loss = torch.tensor(0.0, device=device)
                group_routed_router_reg_loss = torch.tensor(0.0, device=device)
                pivot_supervised_loss = torch.tensor(0.0, device=device)
                pivot_pred_consistency_loss = torch.tensor(0.0, device=device)
                pivot_feat_consistency_loss = torch.tensor(0.0, device=device)
                head_regularization_loss = torch.tensor(0.0, device=device)
                pivot_head_regularization_loss = torch.tensor(0.0, device=device)
                anchor_loss = torch.tensor(0.0, device=device)
                protected_anchor_loss = torch.tensor(0.0, device=device)
                repair_gate_loss = torch.tensor(0.0, device=device)
                branch_decorrelation_loss = torch.tensor(0.0, device=device)
                layer_consistency_loss = torch.tensor(0.0, device=device)
                ema_route_teacher_loss = torch.tensor(0.0, device=device)
                factorized_output_delta_loss = torch.tensor(0.0, device=device)
                factorized_output_easy_zero_loss = torch.tensor(0.0, device=device)
                factorized_output_feature_anchor_loss = torch.tensor(0.0, device=device)
                factorized_output_repair_l2_loss = torch.tensor(0.0, device=device)
                factorized_output_easy_feature_zero_loss = torch.tensor(0.0, device=device)
                factorized_output_repair_probe_loss = torch.tensor(0.0, device=device)
                main_factorized_cache = {}
                va_delta = None
                va_gate = None
                main_forward_t0 = time.perf_counter()
                if (
                    translation_enabled
                    or fds_enabled
                    or polarity_enabled
                    or va_residual_enabled
                    or group_routed_enabled
                    or quantile_enabled
                    or stage1_pooled_feature_anchor_state is not None
                    or factorized_output_residual_enabled
                ):
                    features = model.encode(
                        ids,
                        mask,
                        **condition_kwargs,
                    )
                    head_param = next(model.head.parameters(), None)
                    if head_param is not None and features.dtype != head_param.dtype:
                        features = features.to(head_param.dtype)
                    head_runtime_kwargs = {}
                    if factorized_output_residual_enabled:
                        if batch.get("error_hard_mask") is not None:
                            head_runtime_kwargs["sample_hard_mask"] = batch["error_hard_mask"].to(device)
                        if batch.get("error_hardness") is not None:
                            head_runtime_kwargs["sample_hardness"] = batch["error_hardness"].to(device)
                    if fds_enabled:
                        features = fds_module(features, target_raw[:, 1], epoch)
                    outputs = model.apply_head(
                        features,
                        **condition_kwargs,
                        **head_runtime_kwargs,
                    )
                    if factorized_output_residual_enabled:
                        runtime_head = getattr(_unwrap_training_model(model), "head", None)
                        main_factorized_cache = dict(getattr(runtime_head, "_last_cache", {}) or {})
                else:
                    outputs = model(
                        ids,
                        mask,
                        **condition_kwargs,
                    )
                    if factorized_output_residual_enabled:
                        runtime_head = getattr(_unwrap_training_model(model), "head", None)
                        main_factorized_cache = dict(getattr(runtime_head, "_last_cache", {}) or {})
                if hasattr(model.head, "regularization_loss"):
                    reg_loss = model.head.regularization_loss()
                    if reg_loss is not None:
                        head_regularization_loss = reg_loss
                main_forward_dt = time.perf_counter() - main_forward_t0
                main_backbone_conditional_cache = _aggregate_backbone_conditional_forward_cache(
                    model,
                    detach=False,
                )
                if use_uncertainty:
                    mu, log_sigma = outputs
                    base_loss = gaussian_nll_va(
                        mu,
                        log_sigma,
                        target,
                        sample_weights=base_sample_weights,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                    )
                    student_va = mu
                else:
                    pred = outputs
                    base_loss = va_loss(
                        pred,
                        target,
                        mode=cfg.get("loss", "mse"),
                        weight_v=float(cfg.get("weight_v", 1.0)),
                        weight_a=float(cfg.get("weight_a", 1.0)),
                        sample_weights=base_sample_weights,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        loss_a_mode=cfg.get("loss_a_mode"),
                        a_tail_weight=float(cfg.get("a_tail_weight", 1.0)),
                        a_tail_lo=_a_tail_lo,
                        a_tail_hi=_a_tail_hi,
                    )
                    student_va = pred

                if (
                    translation_enabled
                    and pivot_ids is not None
                    and pivot_sample_mask is not None
                    and float(pivot_sample_mask.sum().item()) > 0.0
                    and global_update >= translation_warmup_updates
                ):
                    pivot_features = model.encode(
                        pivot_ids,
                        pivot_mask,
                        **condition_kwargs,
                    )
                    head_param = next(model.head.parameters(), None)
                    if head_param is not None and pivot_features.dtype != head_param.dtype:
                        pivot_features = pivot_features.to(head_param.dtype)
                    pivot_outputs = model.apply_head(
                        pivot_features,
                        **condition_kwargs,
                    )
                    if hasattr(model.head, "regularization_loss"):
                        pivot_reg_loss = model.head.regularization_loss()
                        if pivot_reg_loss is not None:
                            pivot_head_regularization_loss = pivot_reg_loss
                    if use_uncertainty:
                        pivot_mu, pivot_log_sigma = pivot_outputs
                        pivot_student_va = pivot_mu
                        pivot_supervised_loss = gaussian_nll_va(
                            pivot_mu,
                            pivot_log_sigma,
                            target,
                            sample_weights=pivot_sample_weights,
                            is_auxiliary=is_auxiliary,
                            auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        )
                    else:
                        pivot_student_va = pivot_outputs
                        pivot_supervised_loss = va_loss(
                            pivot_student_va,
                            target,
                            mode=cfg.get("loss", "mse"),
                            weight_v=float(cfg.get("weight_v", 1.0)),
                            weight_a=float(cfg.get("weight_a", 1.0)),
                            sample_weights=pivot_sample_weights,
                            is_auxiliary=is_auxiliary,
                            auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                            loss_a_mode=cfg.get("loss_a_mode"),
                            a_tail_weight=float(cfg.get("a_tail_weight", 1.0)),
                            a_tail_lo=_a_tail_lo,
                            a_tail_hi=_a_tail_hi,
                        )

                    if global_update >= translation_consistency_start_update and pivot_student_va is not None:
                        if translation_pred_consistency_weight > 0:
                            pivot_pred_consistency_loss = _prediction_consistency_loss(
                                student_va,
                                pivot_student_va,
                                is_auxiliary=is_auxiliary,
                                auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                                sample_weights=pivot_sample_weights,
                            )
                        if (
                            translation_feature_consistency_weight > 0
                            and features is not None
                            and pivot_features is not None
                        ):
                            pivot_feat_consistency_loss = _feature_consistency_loss(
                                features,
                                pivot_features,
                                is_auxiliary=is_auxiliary,
                                auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                                sample_weights=pivot_sample_weights,
                            )

                teacher_loss = torch.tensor(0.0, device=device)
                if teacher_eff_weight > 0 and "teacher_va" in batch:
                    teacher_target = _normalize_va(batch["teacher_va"].to(device), va_norm_stats)
                    teacher_loss = teacher_va_distill_loss(
                        student_va,
                        teacher_target,
                        mask=batch["teacher_mask"].to(device),
                        weight=batch["teacher_weight"].to(device),
                        mode=resolve_teacher_loss_mode(teacher_cfg),
                    )
                backbone_conditional_adapter_reg_loss, backbone_conditional_adapter_reg_stats = (
                    _compute_backbone_conditional_adapter_regularization(
                        model,
                        cfg,
                        device,
                        cache=main_backbone_conditional_cache,
                        current_update=global_update,
                    )
                )
                branch_decorrelation_loss, branch_decorrelation_stats = _compute_branch_decorrelation_loss(
                    model,
                    cfg,
                    device,
                    cache=main_backbone_conditional_cache,
                )
                layer_consistency_loss, layer_consistency_stats = _compute_layer_consistency_routing_loss(
                    model,
                    cfg,
                    device,
                )
                soft_condition_label_supervision_loss = torch.tensor(0.0, device=device)
                soft_condition_label_supervision_stats = {}
                soft_condition_label_cfg = dict(cfg.get("soft_condition_label_supervision", {}) or {})
                if (
                    soft_condition_label_head is not None
                    and global_update >= int(soft_condition_label_cfg.get("start_update", 0) or 0)
                ):
                    soft_condition_label_supervision_loss, soft_condition_label_supervision_stats = (
                        _compute_soft_condition_label_supervision_loss(
                            soft_condition_label_head,
                            model,
                            batch,
                            cfg,
                            device,
                            cache=main_backbone_conditional_cache,
                            is_auxiliary=is_auxiliary,
                            sample_weights=sample_weights,
                            auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        )
                    )
                text_routing_detector_loss = torch.tensor(0.0, device=device)
                text_routing_detector_stats = {}
                text_routing_detector_cfg = dict(cfg.get("text_routing_detector", {}) or {})
                if (
                    text_routing_detector_cfg.get("enabled", False)
                    and global_update >= int(text_routing_detector_cfg.get("start_update", 0) or 0)
                ):
                    text_routing_detector_loss, text_routing_detector_stats = (
                        _compute_text_routing_detector_loss(
                            model,
                            batch,
                            cfg,
                            device,
                            cache=main_factorized_cache,
                            is_auxiliary=is_auxiliary,
                            sample_weights=sample_weights,
                            auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        )
                    )
                meta_soft_bootstrap_loss = torch.tensor(0.0, device=device)
                meta_soft_bootstrap_stats = {}
                meta_soft_bootstrap_cfg = dict(cfg.get("meta_soft_bootstrap", {}) or {})
                if (
                    meta_soft_bootstrap_state is not None
                    and global_update >= int(meta_soft_bootstrap_cfg.get("start_update", 0) or 0)
                ):
                    meta_soft_bootstrap_loss, meta_soft_bootstrap_stats = _compute_meta_soft_bootstrap_loss(
                        meta_soft_bootstrap_state,
                        model,
                        batch,
                        ids,
                        mask,
                        device=device,
                        student_cache=main_backbone_conditional_cache,
                    )
                stage1_pooled_feature_anchor_loss = torch.tensor(0.0, device=device)
                stage1_pooled_feature_anchor_stats = {}
                if stage1_pooled_feature_anchor_state is not None:
                    stage1_pooled_feature_anchor_loss, stage1_pooled_feature_anchor_stats = (
                        _compute_stage1_pooled_feature_anchor_loss(
                            stage1_pooled_feature_anchor_state,
                            batch,
                            ids,
                            mask,
                            device=device,
                            student_features=features,
                            is_auxiliary=is_auxiliary,
                            sample_weights=sample_weights,
                            auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                            global_update=global_update,
                        )
                    )
                ema_route_teacher_loss, ema_route_teacher_stats = _compute_ema_route_teacher_loss(
                    ema_route_teacher_state,
                    model,
                    cfg,
                    device,
                    student_cache=main_backbone_conditional_cache,
                    global_update=global_update,
                )
                if factorized_output_residual_enabled:
                    factorized_cache = main_factorized_cache
                    base_pred = factorized_cache.get("base_pred")
                    total_delta = factorized_cache.get("total_delta")
                    base_features = factorized_cache.get("base_features")
                    repaired_features = factorized_cache.get("repaired_features")
                    feature_repair_delta = factorized_cache.get("feature_repair_delta")
                    repair_probe_pred = factorized_cache.get("repair_probe_pred")
                    if base_pred is not None and total_delta is not None:
                        if float(factorized_output_residual_cfg.get("delta_supervision_weight", 0.0)) > 0:
                            delta_target = target - base_pred.detach()
                            delta_per_sample = F.smooth_l1_loss(
                                total_delta,
                                delta_target,
                                reduction="none",
                            ).mean(dim=-1)
                            factorized_output_delta_loss = float(
                                factorized_output_residual_cfg.get("delta_supervision_weight", 0.0)
                            ) * combine_primary_aux_losses(
                                delta_per_sample,
                                is_auxiliary=is_auxiliary,
                                auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                                sample_weights=sample_weights,
                            )
                        if float(factorized_output_residual_cfg.get("easy_zero_weight", 0.0)) > 0:
                            easy_mask = batch.get("error_easy_mask")
                            if easy_mask is not None:
                                easy_mask = easy_mask.to(device)
                            else:
                                easy_threshold = float(
                                    factorized_output_residual_cfg.get("easy_zero_threshold", 0.0)
                                )
                                if easy_threshold > 0:
                                    base_error = torch.mean(torch.abs(target - base_pred.detach()), dim=-1)
                                    easy_mask = (base_error <= easy_threshold).to(target.dtype)
                            if easy_mask is not None and float(easy_mask.sum().item()) > 0.0:
                                easy_zero_per_sample = torch.mean(torch.abs(total_delta), dim=-1)
                                easy_zero_weights = _merge_sample_weights(
                                    sample_weights,
                                    easy_mask.to(target.dtype),
                                )
                                factorized_output_easy_zero_loss = float(
                                    factorized_output_residual_cfg.get("easy_zero_weight", 0.0)
                                ) * combine_primary_aux_losses(
                                    easy_zero_per_sample,
                                    is_auxiliary=is_auxiliary,
                                    auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                                    sample_weights=easy_zero_weights,
                                )
                    repair_probe_weight = float(
                        factorized_output_residual_cfg.get("repair_probe_weight", 0.0)
                    )
                    repair_probe_end_update = int(
                        factorized_output_residual_cfg.get("repair_probe_end_update", 0) or 0
                    )
                    if repair_probe_end_update > 0 and global_update >= repair_probe_end_update:
                        repair_probe_weight = 0.0
                    if repair_probe_weight > 0 and repair_probe_pred is not None:
                        repair_probe_per_sample = F.smooth_l1_loss(
                            repair_probe_pred,
                            target,
                            reduction="none",
                        ).mean(dim=-1)
                        factorized_output_repair_probe_loss = repair_probe_weight * combine_primary_aux_losses(
                            repair_probe_per_sample,
                            is_auxiliary=is_auxiliary,
                            auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                            sample_weights=sample_weights,
                        )
                    if (
                        base_features is not None
                        and repaired_features is not None
                        and feature_repair_delta is not None
                    ):
                        feature_anchor_weight = float(
                            factorized_output_residual_cfg.get("feature_anchor_weight", 0.0)
                        )
                        feature_anchor_end_update = int(
                            factorized_output_residual_cfg.get("feature_anchor_end_update", 0) or 0
                        )
                        if feature_anchor_end_update > 0 and global_update >= feature_anchor_end_update:
                            feature_anchor_weight = 0.0
                        if feature_anchor_weight > 0:
                            feature_anchor_per_sample = F.smooth_l1_loss(
                                repaired_features,
                                base_features.detach(),
                                reduction="none",
                            ).mean(dim=-1)
                            factorized_output_feature_anchor_loss = feature_anchor_weight * combine_primary_aux_losses(
                                feature_anchor_per_sample,
                                is_auxiliary=is_auxiliary,
                                auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                                sample_weights=sample_weights,
                            )
                        repair_l2_weight = float(
                            factorized_output_residual_cfg.get("repair_l2_weight", 0.0)
                        )
                        if repair_l2_weight > 0:
                            repair_l2_per_sample = torch.mean(feature_repair_delta.pow(2), dim=-1)
                            factorized_output_repair_l2_loss = repair_l2_weight * combine_primary_aux_losses(
                                repair_l2_per_sample,
                                is_auxiliary=is_auxiliary,
                                auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                                sample_weights=sample_weights,
                            )
                        easy_feature_zero_weight = float(
                            factorized_output_residual_cfg.get("easy_feature_zero_weight", 0.0)
                        )
                        if easy_feature_zero_weight > 0:
                            easy_mask = batch.get("error_easy_mask")
                            if easy_mask is not None:
                                easy_mask = easy_mask.to(device)
                            if easy_mask is not None and float(easy_mask.sum().item()) > 0.0:
                                easy_feature_zero_per_sample = torch.mean(torch.abs(feature_repair_delta), dim=-1)
                                easy_feature_zero_weights = _merge_sample_weights(
                                    sample_weights,
                                    easy_mask.to(target.dtype),
                                )
                                factorized_output_easy_feature_zero_loss = (
                                    easy_feature_zero_weight
                                    * combine_primary_aux_losses(
                                        easy_feature_zero_per_sample,
                                        is_auxiliary=is_auxiliary,
                                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                                        sample_weights=easy_feature_zero_weights,
                                    )
                                )
                student_va_raw = _denormalize_va(student_va, va_norm_stats)
                if pivot_student_va is not None:
                    pivot_student_va_raw = _denormalize_va(pivot_student_va, va_norm_stats)
                corrected_student_va_raw = student_va_raw
                if (
                    pivot_trust_enabled
                    and pivot_trust_head is not None
                    and pivot_student_va_raw is not None
                    and global_update >= int(pivot_trust_cfg.get("start_update", 0) or 0)
                ):
                    corrected_student_va_raw, trust_gate, trust_correction = _build_pivot_trust_residual_prediction(
                        pivot_trust_cfg,
                        pivot_trust_head,
                        pivot_sample_mask,
                        student_va_raw,
                        pivot_student_va_raw,
                        features,
                        pivot_features,
                        pivot_confidence,
                    )
                    trust_sample_weights = sample_weights
                    if pivot_sample_mask is not None:
                        trust_sample_weights = _merge_sample_weights(
                            trust_sample_weights,
                            (pivot_sample_mask > 0.5).to(student_va_raw.dtype),
                        )
                    trust_per_sample = F.smooth_l1_loss(
                        corrected_student_va_raw[:, 1],
                        target_raw[:, 1],
                        reduction="none",
                    )
                    pivot_trust_loss = float(pivot_trust_cfg.get("loss_weight", 1.0)) * combine_primary_aux_losses(
                        trust_per_sample,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        sample_weights=trust_sample_weights,
                    )
                    if trust_gate is not None and trust_correction is not None:
                        total_pivot_trust_gate += float(trust_gate.detach().mean().cpu())
                        total_pivot_trust_correction += float(trust_correction.detach().abs().mean().cpu())
                        n_pivot_trust_batches += 1
                elif (
                    pivot_residual_enabled
                    and pivot_residual_head is not None
                    and pivot_student_va_raw is not None
                    and global_update >= int(pivot_residual_cfg.get("start_update", 0) or 0)
                ):
                    corrected_student_va_raw, _ = _build_pivot_residual_prediction(
                        pivot_residual_cfg,
                        pivot_residual_head,
                        pivot_sample_mask,
                        student_va_raw,
                        pivot_student_va_raw,
                        features,
                        pivot_features,
                        pivot_confidence,
                    )
                    residual_sample_weights = sample_weights
                    if pivot_sample_mask is not None:
                        residual_sample_weights = _merge_sample_weights(
                            residual_sample_weights,
                            (pivot_sample_mask > 0.5).to(student_va_raw.dtype),
                        )
                    residual_per_sample = F.smooth_l1_loss(
                        corrected_student_va_raw[:, 1],
                        target_raw[:, 1],
                        reduction="none",
                    )
                    pivot_residual_loss = float(pivot_residual_cfg.get("loss_weight", 1.0)) * combine_primary_aux_losses(
                        residual_per_sample,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        sample_weights=residual_sample_weights,
                    )
                if (
                    va_residual_enabled
                    and va_residual_head is not None
                    and global_update >= int(va_residual_cfg.get("start_update", 0) or 0)
                ):
                    corrected_student_va_raw, va_delta, va_gate = _build_va_residual_prediction(
                        va_residual_cfg,
                        va_residual_head,
                        corrected_student_va_raw,
                        features,
                    )
                    va_residual_per_sample = F.smooth_l1_loss(
                        corrected_student_va_raw,
                        target_raw,
                        reduction="none",
                    ).mean(dim=-1)
                    residual_sample_weights = sample_weights
                    if repair_apply_to_residual_loss:
                        residual_sample_weights = _merge_sample_weights(
                            residual_sample_weights,
                            repair_weights,
                        )
                    va_residual_loss = float(va_residual_cfg.get("loss_weight", 1.0)) * combine_primary_aux_losses(
                        va_residual_per_sample,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        sample_weights=residual_sample_weights,
                    )
                    if va_delta is not None and va_gate is not None:
                        total_va_residual_gate += float(va_gate.detach().mean().cpu())
                        total_va_residual_correction += float(va_delta.detach().abs().mean().cpu())
                        n_va_residual_batches += 1
                if (
                    group_routed_enabled
                    and group_routed_residual_head is not None
                    and global_update >= int(active_group_routed_cfg.get("start_update", 0) or 0)
                ):
                    group_pair_features = None
                    if bool(active_group_routed_cfg.get("use_pivot_features", False)):
                        group_pair_features = _build_pivot_va_pair_features(
                            corrected_student_va_raw,
                            pivot_student_va_raw,
                            features,
                            pivot_features,
                            pivot_confidence,
                            detach_inputs=bool(active_group_routed_cfg.get("detach_inputs", False)),
                        )
                    corrected_student_va_raw, group_delta = _build_group_routed_va_residual_prediction(
                        active_group_routed_cfg,
                        group_routed_residual_head,
                        batch["lang"],
                        batch["domain"],
                        corrected_student_va_raw,
                        features,
                        pivot_pair_features=group_pair_features,
                    )
                    group_residual_per_sample = F.smooth_l1_loss(
                        corrected_student_va_raw,
                        target_raw,
                        reduction="none",
                    ).mean(dim=-1)
                    group_residual_sample_weights = sample_weights
                    if repair_apply_to_group_routed_loss:
                        group_residual_sample_weights = _merge_sample_weights(
                            group_residual_sample_weights,
                            repair_weights,
                        )
                    group_routed_residual_loss = float(active_group_routed_cfg.get("loss_weight", 1.0)) * combine_primary_aux_losses(
                        group_residual_per_sample,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        sample_weights=group_residual_sample_weights,
                    )
                    if hasattr(group_routed_residual_head, "regularization_loss"):
                        router_reg = group_routed_residual_head.regularization_loss()
                        if router_reg is not None:
                            group_routed_router_reg_loss = router_reg
                    if group_delta is not None:
                        total_group_routed_correction += float(group_delta.detach().abs().mean().cpu())
                        n_group_routed_batches += 1
                if (
                    protected_anchor_enabled
                    and protected_va_raw is not None
                    and protected_weights is not None
                    and float(protected_weights.sum().item()) > 0.0
                    and global_update >= int(protected_anchor_cfg.get("start_update", 0) or 0)
                ):
                    protected_pred = _slice_va_focus(
                        corrected_student_va_raw,
                        protected_anchor_cfg.get("focus_dim", "va"),
                    )
                    protected_target = _slice_va_focus(
                        protected_va_raw,
                        protected_anchor_cfg.get("focus_dim", "va"),
                    )
                    protected_anchor_loss = float(protected_anchor_cfg.get("loss_weight", 0.0)) * _prediction_consistency_loss(
                        protected_pred,
                        protected_target,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        sample_weights=protected_weights,
                    )
                    if current_protected_repair_bundle_scale != 1.0:
                        protected_anchor_loss = protected_anchor_loss * current_protected_repair_bundle_scale
                if (
                    repair_gate_enabled
                ):
                    gate_values = _resolve_repair_gate_values(
                        model,
                        va_gate=va_gate,
                    )
                    if gate_values is None:
                        gate_values = None
                    else:
                        repair_gate_loss = torch.tensor(0.0, device=device)
                    if gate_values is None:
                        pass
                    else:
                        repair_open_weight = float(repair_gate_cfg.get("repair_open_weight", 0.0))
                        protected_closed_weight = float(repair_gate_cfg.get("protected_closed_weight", 0.0))
                        global_sparse_weight = float(repair_gate_cfg.get("global_sparse_weight", 0.0))
                        if (
                            repair_open_weight > 0
                            and repair_mask is not None
                            and float(repair_mask.sum().item()) > 0.0
                        ):
                            repair_gate_loss = repair_gate_loss + repair_open_weight * masked_weighted_mean(
                                1.0 - gate_values,
                                repair_mask > 0.5,
                                repair_weights,
                            )
                        if (
                            protected_closed_weight > 0
                            and protected_mask is not None
                            and float(protected_mask.sum().item()) > 0.0
                        ):
                            repair_gate_loss = repair_gate_loss + protected_closed_weight * masked_weighted_mean(
                                gate_values,
                                protected_mask > 0.5,
                                protected_weights,
                            )
                        if global_sparse_weight > 0:
                            repair_gate_loss = repair_gate_loss + global_sparse_weight * gate_values.mean()
                        if current_protected_repair_bundle_scale != 1.0:
                            repair_gate_loss = repair_gate_loss * current_protected_repair_bundle_scale
                        if repair_mask is not None and float(repair_mask.sum().item()) > 0.0:
                            total_repair_gate_repair_mean += float(
                                masked_weighted_mean(
                                    gate_values.detach(),
                                    repair_mask > 0.5,
                                    repair_weights,
                                ).cpu()
                            )
                        if protected_mask is not None and float(protected_mask.sum().item()) > 0.0:
                            total_repair_gate_protected_mean += float(
                                masked_weighted_mean(
                                    gate_values.detach(),
                                    protected_mask > 0.5,
                                    protected_weights,
                                ).cpu()
                            )
                        n_repair_gate_batches += 1
                group_dro_loss = torch.tensor(0.0, device=device)
                group_dro_cfg = cfg.get("group_dro", {})
                if group_dro_cfg.get("enabled", False):
                    group_by = str(group_dro_cfg.get("group_by", "domain")).strip().lower()
                    alias_map = group_dro_cfg.get("alias_map")
                    if group_by in {"langfamily", "lang_family", "family"} and alias_map is None:
                        alias_map = DEFAULT_LANGFAMILY_ALIAS
                    group_values = _batch_group_values(batch, group_by, alias_map)
                    group_per_sample = F.smooth_l1_loss(
                        corrected_student_va_raw,
                        target_raw,
                        reduction="none",
                    ).mean(dim=-1)
                    group_dro_loss = float(group_dro_cfg.get("weight", 0.0)) * _group_dro_loss(
                        group_per_sample,
                        group_values,
                        sample_weights=sample_weights,
                        top_k=int(group_dro_cfg.get("top_k_groups", 1)),
                    )
                arousal_aux_loss = torch.tensor(0.0, device=device)
                if arousal_enabled:
                    arousal_sample_weights = sample_weights
                    pivot_gate_multiplier = _compute_pivot_gate_multiplier(
                        pivot_gate_cfg,
                        global_update=global_update,
                        pivot_sample_mask=pivot_sample_mask,
                        pivot_confidence=pivot_confidence,
                        source_pred_raw=student_va_raw,
                        pivot_pred_raw=pivot_student_va_raw,
                        source_features=features,
                        pivot_features=pivot_features,
                        pivot_gate_head=pivot_gate_head,
                    ) if pivot_gate_enabled else None
                    if pivot_gate_multiplier is not None:
                        arousal_sample_weights = _merge_sample_weights(
                            arousal_sample_weights,
                            pivot_gate_multiplier,
                        )
                    arousal_aux_loss = arousal_anti_collapse_loss(
                        corrected_student_va_raw,
                        target_raw,
                        mode=arousal_cfg.get("mode", "ordinal"),
                        loss_weight=float(arousal_cfg.get("loss_weight", 0.0)),
                        boundaries=arousal_cfg.get("boundaries"),
                        ordinal_temperature=float(arousal_cfg.get("ordinal_temperature", 0.5)),
                        distribution_mode=arousal_cfg.get("distribution_mode", "wasserstein"),
                        distribution_num_bins=int(arousal_cfg.get("distribution_num_bins", 20)),
                        sample_weights=arousal_sample_weights,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                        primary_only=bool(arousal_cfg.get("primary_only", False)),
                    )
                    if pivot_gate_multiplier is not None:
                        total_pivot_gate_multiplier += float(pivot_gate_multiplier.detach().mean().cpu())
                        n_pivot_gate_batches += 1
                polarity_loss = torch.tensor(0.0, device=device)
                if polarity_enabled and polarity_head is not None:
                    polarity_logits = polarity_head(features)
                    polarity_loss = polarity_classification_loss(
                        polarity_logits, target_raw[:, 0], polarity_boundaries)
                    polarity_loss = float(polarity_cfg.get("weight", 0.1)) * polarity_loss
                region_cls_loss = torch.tensor(0.0, device=device)
                if region_cls_enabled and region_cls_head is not None:
                    # Use features if available (polarity path), otherwise encode
                    if polarity_enabled:
                        region_features = features
                    else:
                        region_features = model.encode(
                            ids,
                            mask,
                            **condition_kwargs,
                        )
                        head_param = next(model.head.parameters(), None)
                        if head_param is not None and region_features.dtype != head_param.dtype:
                            region_features = region_features.to(head_param.dtype)
                    region_logits = region_cls_head(region_features)
                    region_cls_loss = va_region_classification_loss(
                        region_logits, target_raw, region_v_boundaries, region_a_boundaries)
                    region_cls_loss = float(region_cls_cfg.get("weight", 0.1)) * region_cls_loss
                valence_sharp_loss = torch.tensor(0.0, device=device)
                if valence_sharp_enabled:
                    valence_sharp_loss = valence_sharpening_loss(
                        pred_v=student_va_raw[:, 0],
                        true_v=target_raw[:, 0],
                        pred_a=student_va_raw[:, 1],
                        true_a=target_raw[:, 1],
                        v_extreme_threshold=float(valence_sharp_cfg.get("v_extreme_threshold", 1.5)),
                        weight=float(valence_sharp_cfg.get("weight", 0.3)),
                    )
                quantile_loss = torch.tensor(0.0, device=device)
                quantile_active = quantile_enabled and quantile_head is not None and epoch >= quantile_start_epoch
                if quantile_active:
                    # Reuse features if already extracted, otherwise encode
                    quantile_reencode_t0 = time.perf_counter()
                    if polarity_enabled or region_cls_enabled:
                        q_features = features if polarity_enabled else region_features
                    else:
                        q_features = model.encode(
                            ids,
                            mask,
                            **condition_kwargs,
                        )
                        head_param = next(model.head.parameters(), None)
                        if head_param is not None and q_features.dtype != head_param.dtype:
                            q_features = q_features.to(head_param.dtype)
                    quantile_reencode_dt = time.perf_counter() - quantile_reencode_t0
                    q_preds = quantile_head(q_features)
                    quantile_sample_weights = sample_weights
                    if quantile_tail_only and quantile_tail_lo is not None and quantile_tail_hi is not None:
                        tail_mask = (
                            (target_raw[:, 1] <= quantile_tail_lo)
                            | (target_raw[:, 1] >= quantile_tail_hi)
                        ).float()
                        quantile_sample_weights = tail_mask if quantile_sample_weights is None else quantile_sample_weights * tail_mask
                    quantile_loss = float(quantile_cfg.get("weight", 0.1)) * quantile_pinball_loss(
                        q_preds, target_raw[:, 1], quantile_head.tau,
                        sample_weights=quantile_sample_weights,
                        is_auxiliary=is_auxiliary,
                        auxiliary_loss_weight=float(aux_cfg.get("loss_weight", 1.0)),
                    )
                loss = (
                    base_loss
                    + teacher_eff_weight * teacher_loss
                    + arousal_aux_loss
                    + polarity_loss
                    + region_cls_loss
                    + valence_sharp_loss
                    + quantile_loss
                    + pivot_residual_loss
                    + pivot_trust_loss
                    + va_residual_loss
                    + group_routed_residual_loss
                    + group_routed_router_reg_loss
                    + group_dro_loss
                    + protected_anchor_loss
                    + repair_gate_loss
                    + head_regularization_loss
                    + pivot_head_regularization_loss
                    + backbone_conditional_adapter_reg_loss
                    + branch_decorrelation_loss
                    + layer_consistency_loss
                    + ema_route_teacher_loss
                    + factorized_output_delta_loss
                    + factorized_output_easy_zero_loss
                    + factorized_output_feature_anchor_loss
                    + factorized_output_repair_l2_loss
                    + factorized_output_easy_feature_zero_loss
                    + factorized_output_repair_probe_loss
                    + soft_condition_label_supervision_loss
                    + text_routing_detector_loss
                    + meta_soft_bootstrap_loss
                    + stage1_pooled_feature_anchor_loss
                )
                anchor_loss = _anchor_regularization_loss(model, anchor_state, cfg, device)
                loss = loss + anchor_loss
                if translation_supervised_weight > 0:
                    loss = loss + translation_supervised_weight * pivot_supervised_loss
                if translation_pred_consistency_weight > 0:
                    loss = loss + translation_pred_consistency_weight * pivot_pred_consistency_loss
                if translation_feature_consistency_weight > 0:
                    loss = loss + translation_feature_consistency_weight * pivot_feat_consistency_loss
                decorr_loss = torch.tensor(0.0, device=device)
                if decorr_enabled and student_va.size(0) > 4:
                    decorr_loss = float(decorr_cfg.get("weight", 0.1)) * va_decorrelation_loss(
                        student_va, target)
                var_reg_loss = torch.tensor(0.0, device=device)
                if var_reg_enabled and student_va.size(0) > 4:
                    var_reg_loss = float(var_reg_cfg.get("weight", 0.05)) * variance_regularization_loss(
                        student_va, target, focus_dim=1, mode=var_reg_cfg.get("mode", "ratio"))
                loss = loss + decorr_loss + var_reg_loss

                # EMA variance tracking loss (works with batch_size=1)
                ema_var_loss = torch.tensor(0.0, device=device)
                if ema_var_enabled:
                    with torch.no_grad():
                        pred_a_val = float(student_va_raw[:, 1].mean())
                        true_a_val = float(target_raw[:, 1].mean())
                        ema_pred_a_mean = ema_var_alpha * ema_pred_a_mean + (1 - ema_var_alpha) * pred_a_val
                        ema_pred_a_sq = ema_var_alpha * ema_pred_a_sq + (1 - ema_var_alpha) * pred_a_val ** 2
                        ema_true_a_mean = ema_var_alpha * ema_true_a_mean + (1 - ema_var_alpha) * true_a_val
                        ema_true_a_sq = ema_var_alpha * ema_true_a_sq + (1 - ema_var_alpha) * true_a_val ** 2
                        ema_pred_var = max(ema_pred_a_sq - ema_pred_a_mean ** 2, 1e-8)
                        ema_true_var = max(ema_true_a_sq - ema_true_a_mean ** 2, 1e-8)
                        ema_pred_std = ema_pred_var ** 0.5
                        ema_true_std = ema_true_var ** 0.5
                        # Only penalize when predicted std < true std (collapse)
                        std_gap = max(0.0, ema_true_std - ema_pred_std)
                    if std_gap > 0:
                        # Encourage current prediction to move away from mean
                        pred_a_dev = (student_va_raw[:, 1] - ema_pred_a_mean) ** 2
                        # Penalize small deviations (push predictions outward)
                        ema_var_loss = ema_var_weight * std_gap * torch.exp(-pred_a_dev).mean()
                    loss = loss + ema_var_loss

                # Feature Memory Bank + VICReg variance regularization
                vicreg_bank_loss = torch.tensor(0.0, device=device)
                if mem_bank_enabled:
                    # Need features; extract if not already done
                    if features is None:
                        features = model.encode(
                            ids,
                            mask,
                            **condition_kwargs,
                        )
                        head_param = next(model.head.parameters(), None)
                        if head_param is not None and features.dtype != head_param.dtype:
                            features = features.to(head_param.dtype)
                    bank_data = mem_bank.get(min_samples=mem_bank_min_samples)
                    if bank_data is not None:
                        bank_feats, _ = bank_data
                        # Current features have gradient, bank features are detached
                        combined = torch.cat([features, bank_feats.to(features.dtype)], dim=0)
                        std_per_dim = combined.std(dim=0).clamp(min=1e-4)
                        vicreg_bank_loss = vicreg_var_weight * torch.mean(torch.relu(1.0 - std_per_dim))
                    # Push current features (detached) into bank
                    mem_bank.push(features.detach(), target_raw.detach())
                    loss = loss + vicreg_bank_loss

            should_step = (
                step % grad_accum == 0
                or step == len(train_loader)
                or (debug_max_train_batches and step == debug_max_train_batches)
            )
            backward_t0 = time.perf_counter()
            loss_for_backward = loss if deepspeed_enabled else (loss / grad_accum)
            if deepspeed_enabled:
                model.set_gradient_accumulation_boundary(should_step)
                model.backward(loss_for_backward)
            elif scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            backward_dt = time.perf_counter() - backward_t0

            optimizer_dt = 0.0
            if should_step:
                optimizer_t0 = time.perf_counter()
                if update_observer is not None:
                    update_observer.before(global_update + 1)
                if not deepspeed_enabled:
                    if cfg.get("grad_clip", 0.0):
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                else:
                    torch.cuda.empty_cache()
                    model.step()
                optimizer_dt = time.perf_counter() - optimizer_t0
                global_update += 1
                if update_observer is not None:
                    update_observer.after(global_update)
                ema_route_teacher_momentum = _update_backbone_conditional_ema_teacher(
                    ema_route_teacher_state,
                    model,
                    global_update,
                )
                if ema_route_teacher_momentum is not None:
                    total_ema_route_teacher_momentum += float(ema_route_teacher_momentum)
                    n_ema_route_teacher_updates += 1

                if dev_data and eval_every_updates > 0 and global_update % eval_every_updates == 0:
                    step_runtime_control_stats = _refresh_backbone_runtime_controls(model, cfg, global_update)
                    current_post_ckpt_repair_scale, current_protected_repair_bundle_scale = (
                        _refresh_effective_post_ckpt_repair_adapter_scale(
                            model,
                            cfg,
                            global_update,
                        )
                    )
                    step_dev_rmse = _evaluate_dev_rmse(
                        model,
                        dev_data,
                        tokenizer,
                        device,
                        cfg,
                        va_norm_stats=va_norm_stats,
                        quantile_head=quantile_head if quantile_enabled else None,
                        pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
                        pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
                        va_residual_head=va_residual_head if va_residual_enabled else None,
                        group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
                    )
                    last_eval_update = global_update
                    last_eval_rmse = step_dev_rmse
                    metrics[f"update_{global_update}_dev_rmse_va"] = step_dev_rmse
                    step_macro_rmse = None
                    step_macro_breakdown = {}
                    step_macro_selection_score = None
                    step_macro_selection = {}
                    step_canary_metrics = None
                    step_canary_score = None
                    step_canary_guard_reasons = []
                    if macro_dev_enabled:
                        step_macro_rmse, step_macro_breakdown = _evaluate_macro_dev_rmse(
                            model,
                            macro_dev_targets,
                            tokenizer,
                            device,
                            va_norm_stats=va_norm_stats,
                            quantile_head=quantile_head if quantile_enabled else None,
                            pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
                            pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
                            va_residual_head=va_residual_head if va_residual_enabled else None,
                            group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
                        )
                        if matched_protocol:
                            _archive_matched_dev(
                                model, shared_run_dir, "step", epoch, global_update,
                                step_macro_breakdown, cfg, checkpoint_extra_modules, dist_ctx,
                            )
                        if step_macro_rmse is not None:
                            metrics[f"update_{global_update}_macro_dev_rmse_va"] = step_macro_rmse
                            step_macro_breakdown_rmse = {
                                name: float(values["rmse_va"]) if isinstance(values, dict) else float(values)
                                for name, values in step_macro_breakdown.items()
                            }
                            step_macro_selection_score, step_macro_selection = compute_macro_dev_selection(
                                step_macro_breakdown,
                                macro_dev_selection_cfg,
                            )
                            metrics[f"update_{global_update}_macro_dev_selection_score"] = step_macro_selection_score
                        else:
                            step_macro_breakdown_rmse = {}
                    else:
                        step_macro_breakdown_rmse = {}
                    if checkpoint_canary_samples and checkpoint_canary_teacher_map:
                        step_canary_metrics = _evaluate_checkpoint_canary(
                            model,
                            checkpoint_canary_samples,
                            checkpoint_canary_teacher_map,
                            tokenizer,
                            device,
                            cfg,
                            va_norm_stats=va_norm_stats,
                            quantile_head=quantile_head if quantile_enabled else None,
                            pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
                            pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
                            va_residual_head=va_residual_head if va_residual_enabled else None,
                            group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
                        )
                        if step_canary_metrics is not None:
                            step_canary_score = _checkpoint_canary_metric_value(
                                step_canary_metrics,
                                checkpoint_canary_cfg["metric"],
                            )
                            metrics[f"update_{global_update}_checkpoint_canary_teacher_rmse_va"] = step_canary_metrics["teacher_rmse_va"]
                            metrics[f"update_{global_update}_checkpoint_canary_teacher_macro_rmse_va"] = step_canary_metrics["teacher_macro_rmse_va"]
                            if step_canary_metrics.get("label_rmse_va") is not None:
                                metrics[f"update_{global_update}_checkpoint_canary_label_rmse_va"] = step_canary_metrics["label_rmse_va"]
                            if step_canary_metrics.get("label_macro_rmse_va") is not None:
                                metrics[f"update_{global_update}_checkpoint_canary_label_macro_rmse_va"] = step_canary_metrics["label_macro_rmse_va"]
                            if step_canary_score is not None:
                                metrics[f"update_{global_update}_checkpoint_canary_selection_score"] = step_canary_score
                            step_canary_guard_reasons = _checkpoint_canary_guard_reasons(
                                checkpoint_canary_cfg,
                                step_canary_score,
                                checkpoint_canary_baseline_score,
                            )
                    logger.info(
                        "Update %s/%s dev RMSE_VA=%.6f (epoch=%s step=%s)",
                        global_update, total_updates, step_dev_rmse, epoch + 1, step,
                    )
                    if current_alternating_phase is not None:
                        metrics[f"update_{global_update}_alternating_phase"] = current_alternating_phase
                    if step_runtime_control_stats:
                        for key, value in step_runtime_control_stats.items():
                            metrics[f"update_{global_update}_runtime_{key}"] = float(value)
                    metrics[f"update_{global_update}_protected_repair_bundle_scale"] = (
                        current_protected_repair_bundle_scale
                    )
                    if ema_route_teacher_momentum is not None:
                        metrics[f"update_{global_update}_ema_route_teacher_momentum"] = (
                            ema_route_teacher_momentum
                        )
                    if step_macro_rmse is not None:
                        logger.info(
                            "Update %s/%s macro-dev RMSE_VA=%.6f selection=%.6f targets=%s",
                            global_update,
                            total_updates,
                            step_macro_rmse,
                            step_macro_selection_score if step_macro_selection_score is not None else step_macro_rmse,
                            step_macro_breakdown_rmse,
                        )
                        adapter_usage = model.backbone_adapter_usage() if hasattr(model, "backbone_adapter_usage") else {}
                        if adapter_usage:
                            for key, value in adapter_usage.items():
                                metrics[f"update_{global_update}_{key}"] = value
                            logger.info(
                                "Update %s/%s backbone-conditional usage=%s",
                                global_update,
                                total_updates,
                                {k: round(v, 6) for k, v in adapter_usage.items()},
                            )
                    if va_residual_enabled and n_va_residual_batches > 0:
                        residual_gate_mean = total_va_residual_gate / n_va_residual_batches
                        residual_corr_mean = total_va_residual_correction / n_va_residual_batches
                        metrics[f"update_{global_update}_va_residual_gate"] = residual_gate_mean
                        metrics[f"update_{global_update}_va_residual_correction"] = residual_corr_mean
                        logger.info(
                            "Update %s/%s va-residual health gate=%.6f correction=%.6f",
                            global_update,
                            total_updates,
                            residual_gate_mean,
                            residual_corr_mean,
                        )
                    if repair_gate_enabled and n_repair_gate_batches > 0:
                        metrics[f"update_{global_update}_repair_gate_loss"] = total_repair_gate_loss / n_repair_gate_batches
                        metrics[f"update_{global_update}_repair_gate_repair_mean"] = (
                            total_repair_gate_repair_mean / n_repair_gate_batches
                        )
                        metrics[f"update_{global_update}_repair_gate_protected_mean"] = (
                            total_repair_gate_protected_mean / n_repair_gate_batches
                        )
                        logger.info(
                            "Update %s/%s repair-gate loss=%.6f repair_mean=%.6f protected_mean=%.6f",
                            global_update,
                            total_updates,
                            total_repair_gate_loss / n_repair_gate_batches,
                            total_repair_gate_repair_mean / n_repair_gate_batches,
                            total_repair_gate_protected_mean / n_repair_gate_batches,
                        )
                    if cfg.get("branch_decorrelation", {}).get("enabled", False):
                        metrics[f"update_{global_update}_branch_decorrelation_loss"] = (
                            total_branch_decorrelation_loss / max(n_batches, 1)
                        )
                    if cfg.get("layer_consistency_routing", {}).get("enabled", False):
                        metrics[f"update_{global_update}_layer_consistency_loss"] = (
                            total_layer_consistency_loss / max(n_batches, 1)
                        )
                    if ema_route_teacher_state is not None:
                        metrics[f"update_{global_update}_ema_route_teacher_loss"] = (
                            total_ema_route_teacher_loss / max(n_batches, 1)
                        )
                    if step_canary_metrics is not None:
                        logger.info(
                            "Update %s/%s checkpoint-canary %s=%.6f teacher_macro=%.6f label_macro=%s guard=%s",
                            global_update,
                            total_updates,
                            checkpoint_canary_cfg["metric"],
                            step_canary_score if step_canary_score is not None else float("nan"),
                            step_canary_metrics["teacher_macro_rmse_va"],
                            (
                                f"{step_canary_metrics['label_macro_rmse_va']:.6f}"
                                if step_canary_metrics.get("label_macro_rmse_va") is not None
                                else "n/a"
                            ),
                            "pass" if not step_canary_guard_reasons else "; ".join(step_canary_guard_reasons),
                        )
                    selection_score = (
                        step_macro_selection_score
                        if macro_dev_use_for_best and step_macro_selection_score is not None
                        else step_dev_rmse
                    )
                    if checkpoint_canary_cfg.get("use_for_best", False) and step_canary_score is not None:
                        selection_score += checkpoint_canary_cfg["weight"] * step_canary_score
                    selection_blocked = (
                        checkpoint_canary_cfg.get("use_for_best", False)
                        and len(step_canary_guard_reasons) > 0
                    )
                    if selection_blocked:
                        logger.info(
                            "Skipped checkpoint update at epoch=%s update=%s due to checkpoint-canary guard: %s",
                            epoch,
                            global_update,
                            "; ".join(step_canary_guard_reasons),
                        )
                    elif best_selection_score is None or selection_score < best_selection_score:
                        best_selection_score = selection_score
                        best_dev_rmse = step_dev_rmse
                        if step_macro_rmse is not None:
                            best_macro_dev_rmse = step_macro_rmse
                        best_epoch = epoch
                        best_update = global_update
                        _save_best_checkpoint(
                            model,
                            run_dir,
                            epoch,
                            {"dev_rmse_va": step_dev_rmse},
                            cfg,
                            event="step",
                            global_update=global_update,
                            extra_modules=checkpoint_extra_modules,
                            dist_ctx=dist_ctx,
                        )
                        logger.info(
                            "New best selection score %.6f at epoch=%s update=%s (dev=%.6f macro=%s selection=%s)",
                            selection_score,
                            epoch,
                            global_update,
                            step_dev_rmse,
                            f"{step_macro_rmse:.6f}" if step_macro_rmse is not None else "n/a",
                            f"{step_macro_selection:.6f}" if isinstance(step_macro_selection, float) else step_macro_selection,
                        )
                    _set_training_mode(
                        model,
                        polarity_head=polarity_head if polarity_enabled else None,
                        region_cls_head=region_cls_head if region_cls_enabled else None,
                        quantile_head=quantile_head if quantile_enabled else None,
                        pivot_gate_head=pivot_gate_head,
                        pivot_residual_head=pivot_residual_head,
                        pivot_trust_head=pivot_trust_head,
                        va_residual_head=va_residual_head,
                        group_routed_residual_head=group_routed_residual_head,
                        soft_condition_label_head=soft_condition_label_head,
                    )

                if save_every_updates > 0 and global_update % save_every_updates == 0:
                    step_ckpt_metrics = {"update": global_update}
                    if last_eval_update == global_update and last_eval_rmse is not None:
                        step_ckpt_metrics["dev_rmse_va"] = last_eval_rmse
                    step_ckpt_name = f"checkpoint_step{global_update:06d}.pt"
                    _save_checkpoint(
                        model,
                        run_dir,
                        epoch,
                        step_ckpt_metrics,
                        cfg,
                        filename=step_ckpt_name,
                        global_update=global_update,
                        extra_modules=checkpoint_extra_modules,
                        dist_ctx=dist_ctx,
                    )
                    removed = _prune_step_checkpoints(run_dir, max_step_checkpoints_to_keep)
                    if removed:
                        logger.info(
                            "Pruned %d old step checkpoints after update %s: %s",
                            len(removed), global_update, ", ".join(removed),
                        )

            if debug_step and debug_log_timing:
                logger.info(
                    "Debug step %d timings: to_device=%.3fs main_forward=%.3fs quantile_extra_encode=%.3fs backward=%.3fs optimizer=%.3fs loss=%.6f",
                    step,
                    device_dt,
                    main_forward_dt,
                    quantile_reencode_dt,
                    backward_dt,
                    optimizer_dt,
                    float(loss.detach().cpu()),
                )

            total_loss += float(loss.detach().cpu())
            total_teacher_loss += float(teacher_loss.detach().cpu())
            total_arousal_aux_loss += float(arousal_aux_loss.detach().cpu())
            total_polarity_loss += float(polarity_loss.detach().cpu())
            total_region_cls_loss += float(region_cls_loss.detach().cpu())
            total_quantile_loss += float(quantile_loss.detach().cpu())
            total_decorr_loss += float(decorr_loss.detach().cpu())
            total_var_reg_loss += float(var_reg_loss.detach().cpu())
            total_ema_var_loss += float(ema_var_loss.detach().cpu())
            total_vicreg_bank_loss += float(vicreg_bank_loss.detach().cpu())
            total_translation_pivot_loss += float(pivot_supervised_loss.detach().cpu())
            total_translation_pred_consistency_loss += float(pivot_pred_consistency_loss.detach().cpu())
            total_translation_feat_consistency_loss += float(pivot_feat_consistency_loss.detach().cpu())
            total_pivot_residual_loss += float(pivot_residual_loss.detach().cpu())
            total_pivot_trust_loss += float(pivot_trust_loss.detach().cpu())
            total_va_residual_loss += float(va_residual_loss.detach().cpu())
            total_group_routed_residual_loss += float(group_routed_residual_loss.detach().cpu())
            total_group_routed_router_reg_loss += float(group_routed_router_reg_loss.detach().cpu())
            total_group_dro_loss += float(group_dro_loss.detach().cpu())
            total_protected_anchor_loss += float(protected_anchor_loss.detach().cpu())
            total_repair_gate_loss += float(repair_gate_loss.detach().cpu())
            total_anchor_loss += float(anchor_loss.detach().cpu())
            total_backbone_conditional_adapter_reg_loss += float(
                backbone_conditional_adapter_reg_loss.detach().cpu()
            )
            total_branch_decorrelation_loss += float(branch_decorrelation_loss.detach().cpu())
            total_layer_consistency_loss += float(layer_consistency_loss.detach().cpu())
            total_ema_route_teacher_loss += float(ema_route_teacher_loss.detach().cpu())
            total_factorized_output_delta_loss += float(factorized_output_delta_loss.detach().cpu())
            total_factorized_output_easy_zero_loss += float(factorized_output_easy_zero_loss.detach().cpu())
            total_factorized_output_feature_anchor_loss += float(
                factorized_output_feature_anchor_loss.detach().cpu()
            )
            total_factorized_output_repair_l2_loss += float(
                factorized_output_repair_l2_loss.detach().cpu()
            )
            total_factorized_output_easy_feature_zero_loss += float(
                factorized_output_easy_feature_zero_loss.detach().cpu()
            )
            total_factorized_output_repair_probe_loss += float(
                factorized_output_repair_probe_loss.detach().cpu()
            )
            total_soft_condition_label_supervision_loss += float(
                soft_condition_label_supervision_loss.detach().cpu()
            )
            if "lang_acc" in soft_condition_label_supervision_stats:
                lang_valid = int(soft_condition_label_supervision_stats.get("lang_valid", 0))
                total_soft_condition_lang_count += lang_valid
                total_soft_condition_lang_correct += (
                    float(soft_condition_label_supervision_stats["lang_acc"]) * lang_valid
                )
            if "domain_acc" in soft_condition_label_supervision_stats:
                domain_valid = int(soft_condition_label_supervision_stats.get("domain_valid", 0))
                total_soft_condition_domain_count += domain_valid
                total_soft_condition_domain_correct += (
                    float(soft_condition_label_supervision_stats["domain_acc"]) * domain_valid
                )
            total_text_routing_detector_loss += float(
                text_routing_detector_loss.detach().cpu()
            )
            if "lang_acc" in text_routing_detector_stats:
                lang_valid = int(text_routing_detector_stats.get("lang_valid", 0))
                total_text_routing_lang_count += lang_valid
                total_text_routing_lang_correct += (
                    float(text_routing_detector_stats["lang_acc"]) * lang_valid
                )
            if "domain_acc" in text_routing_detector_stats:
                domain_valid = int(text_routing_detector_stats.get("domain_valid", 0))
                total_text_routing_domain_count += domain_valid
                total_text_routing_domain_correct += (
                    float(text_routing_detector_stats["domain_acc"]) * domain_valid
                )
            total_meta_soft_bootstrap_loss += float(meta_soft_bootstrap_loss.detach().cpu())
            total_stage1_pooled_feature_anchor_loss += float(
                stage1_pooled_feature_anchor_loss.detach().cpu()
            )
            if runtime_control_stats:
                total_runtime_shared_residual_scale += float(runtime_control_stats.get("shared_residual_scale", 0.0))
                total_runtime_shared_branch_dropout += float(runtime_control_stats.get("shared_branch_dropout", 0.0))
                total_runtime_composition_gate_max += float(runtime_control_stats.get("composition_gate_max", 0.0))
                total_runtime_composition_residual_scale += float(
                    runtime_control_stats.get("composition_residual_scale", 0.0)
                )
                n_runtime_control_batches += 1
            composition_confidence = main_backbone_conditional_cache.get("composition_confidence")
            if composition_confidence is not None:
                total_composition_confidence += float(composition_confidence.detach().mean().cpu())
                n_composition_confidence_batches += 1
            total_protected_repair_bundle_scale += float(current_protected_repair_bundle_scale)
            n_protected_repair_bundle_batches += 1
            # FDS: update per-bin feature statistics
            if fds_enabled and features is not None:
                fds_module.update_stats(features.detach(), target_raw[:, 1])
            n_batches += 1

        train_loss_epoch = round(total_loss / max(n_batches, 1), 6)
        metrics[f"epoch_{epoch}_train_loss"] = train_loss_epoch
        if teacher_cfg.get("enabled", False):
            metrics[f"epoch_{epoch}_teacher_loss"] = round(total_teacher_loss / max(n_batches, 1), 6)
            metrics[f"epoch_{epoch}_teacher_w_eff"] = round(teacher_eff_weight, 6)
        if cfg.get("backbone_conditional_adapter_regularization", {}).get("enabled", False):
            metrics[f"epoch_{epoch}_backbone_conditional_adapter_reg_loss"] = round(
                total_backbone_conditional_adapter_reg_loss / max(n_batches, 1), 6
            )
        if cfg.get("branch_decorrelation", {}).get("enabled", False):
            metrics[f"epoch_{epoch}_branch_decorrelation_loss"] = round(
                total_branch_decorrelation_loss / max(n_batches, 1), 6
            )
        if cfg.get("layer_consistency_routing", {}).get("enabled", False):
            metrics[f"epoch_{epoch}_layer_consistency_loss"] = round(
                total_layer_consistency_loss / max(n_batches, 1), 6
            )
        if ema_route_teacher_state is not None:
            metrics[f"epoch_{epoch}_ema_route_teacher_loss"] = round(
                total_ema_route_teacher_loss / max(n_batches, 1), 6
            )
            if n_ema_route_teacher_updates > 0:
                metrics[f"epoch_{epoch}_ema_route_teacher_momentum"] = round(
                    total_ema_route_teacher_momentum / n_ema_route_teacher_updates,
                    6,
                )
        if n_runtime_control_batches > 0:
            metrics[f"epoch_{epoch}_runtime_shared_residual_scale"] = round(
                total_runtime_shared_residual_scale / n_runtime_control_batches,
                6,
            )
            metrics[f"epoch_{epoch}_runtime_shared_branch_dropout"] = round(
                total_runtime_shared_branch_dropout / n_runtime_control_batches,
                6,
            )
            metrics[f"epoch_{epoch}_runtime_composition_gate_max"] = round(
                total_runtime_composition_gate_max / n_runtime_control_batches,
                6,
            )
            metrics[f"epoch_{epoch}_runtime_composition_residual_scale"] = round(
                total_runtime_composition_residual_scale / n_runtime_control_batches,
                6,
            )
        if n_composition_confidence_batches > 0:
            metrics[f"epoch_{epoch}_composition_confidence"] = round(
                total_composition_confidence / n_composition_confidence_batches,
                6,
            )
        if n_protected_repair_bundle_batches > 0:
            metrics[f"epoch_{epoch}_protected_repair_bundle_scale"] = round(
                total_protected_repair_bundle_scale / n_protected_repair_bundle_batches,
                6,
            )
        if alternating_phase_counts:
            metrics[f"epoch_{epoch}_alternating_phase_counts"] = dict(alternating_phase_counts)
        if soft_condition_label_head is not None:
            metrics[f"epoch_{epoch}_soft_condition_label_supervision_loss"] = round(
                total_soft_condition_label_supervision_loss / max(n_batches, 1), 6
            )
            if total_soft_condition_lang_count > 0:
                metrics[f"epoch_{epoch}_soft_condition_lang_acc"] = round(
                    total_soft_condition_lang_correct / total_soft_condition_lang_count,
                    6,
                )
            if total_soft_condition_domain_count > 0:
                metrics[f"epoch_{epoch}_soft_condition_domain_acc"] = round(
                    total_soft_condition_domain_correct / total_soft_condition_domain_count,
                    6,
                )
        if cfg.get("text_routing_detector", {}).get("enabled", False):
            metrics[f"epoch_{epoch}_text_routing_detector_loss"] = round(
                total_text_routing_detector_loss / max(n_batches, 1),
                6,
            )
            if total_text_routing_lang_count > 0:
                metrics[f"epoch_{epoch}_text_routing_lang_acc"] = round(
                    total_text_routing_lang_correct / total_text_routing_lang_count,
                    6,
                )
            if total_text_routing_domain_count > 0:
                metrics[f"epoch_{epoch}_text_routing_domain_acc"] = round(
                    total_text_routing_domain_correct / total_text_routing_domain_count,
                    6,
                )
        if factorized_output_residual_enabled:
            metrics[f"epoch_{epoch}_factorized_output_repair_probe_loss"] = round(
                total_factorized_output_repair_probe_loss / max(n_batches, 1),
                6,
            )
        if meta_soft_bootstrap_state is not None:
            metrics[f"epoch_{epoch}_meta_soft_bootstrap_loss"] = round(
                total_meta_soft_bootstrap_loss / max(n_batches, 1), 6
            )
        if stage1_pooled_feature_anchor_state is not None:
            metrics[f"epoch_{epoch}_stage1_pooled_feature_anchor_loss"] = round(
                total_stage1_pooled_feature_anchor_loss / max(n_batches, 1), 6
            )
        if arousal_enabled:
            metrics[f"epoch_{epoch}_arousal_aux_loss"] = round(total_arousal_aux_loss / max(n_batches, 1), 6)
        if polarity_enabled:
            metrics[f"epoch_{epoch}_polarity_loss"] = round(total_polarity_loss / max(n_batches, 1), 6)
        if region_cls_enabled:
            metrics[f"epoch_{epoch}_region_cls_loss"] = round(total_region_cls_loss / max(n_batches, 1), 6)
        if quantile_enabled:
            metrics[f"epoch_{epoch}_quantile_loss"] = round(total_quantile_loss / max(n_batches, 1), 6)
        if translation_enabled:
            metrics[f"epoch_{epoch}_translation_pivot_loss"] = round(total_translation_pivot_loss / max(n_batches, 1), 6)
            metrics[f"epoch_{epoch}_translation_pred_consistency_loss"] = round(
                total_translation_pred_consistency_loss / max(n_batches, 1), 6
            )
            metrics[f"epoch_{epoch}_translation_feat_consistency_loss"] = round(
                total_translation_feat_consistency_loss / max(n_batches, 1), 6
            )
        if pivot_residual_enabled:
            metrics[f"epoch_{epoch}_pivot_residual_loss"] = round(
                total_pivot_residual_loss / max(n_batches, 1), 6
            )
        if pivot_trust_enabled:
            metrics[f"epoch_{epoch}_pivot_trust_loss"] = round(
                total_pivot_trust_loss / max(n_batches, 1), 6
            )
            if n_pivot_trust_batches > 0:
                metrics[f"epoch_{epoch}_pivot_trust_gate"] = round(
                    total_pivot_trust_gate / n_pivot_trust_batches, 6
                )
                metrics[f"epoch_{epoch}_pivot_trust_correction"] = round(
                    total_pivot_trust_correction / n_pivot_trust_batches, 6
                )
        if va_residual_enabled:
            metrics[f"epoch_{epoch}_va_residual_loss"] = round(
                total_va_residual_loss / max(n_batches, 1), 6
            )
            if n_va_residual_batches > 0:
                metrics[f"epoch_{epoch}_va_residual_gate"] = round(
                    total_va_residual_gate / n_va_residual_batches, 6
                )
                metrics[f"epoch_{epoch}_va_residual_correction"] = round(
                    total_va_residual_correction / n_va_residual_batches, 6
                )
        if factorized_output_residual_enabled:
            metrics[f"epoch_{epoch}_factorized_output_delta_loss"] = round(
                total_factorized_output_delta_loss / max(n_batches, 1), 6
            )
            metrics[f"epoch_{epoch}_factorized_output_easy_zero_loss"] = round(
                total_factorized_output_easy_zero_loss / max(n_batches, 1), 6
            )
            metrics[f"epoch_{epoch}_factorized_output_feature_anchor_loss"] = round(
                total_factorized_output_feature_anchor_loss / max(n_batches, 1), 6
            )
            metrics[f"epoch_{epoch}_factorized_output_repair_l2_loss"] = round(
                total_factorized_output_repair_l2_loss / max(n_batches, 1), 6
            )
            metrics[f"epoch_{epoch}_factorized_output_easy_feature_zero_loss"] = round(
                total_factorized_output_easy_feature_zero_loss / max(n_batches, 1), 6
            )
        if protected_anchor_enabled:
            metrics[f"epoch_{epoch}_protected_anchor_loss"] = round(
                total_protected_anchor_loss / max(n_batches, 1), 6
            )
        if repair_gate_enabled:
            metrics[f"epoch_{epoch}_repair_gate_loss"] = round(
                total_repair_gate_loss / max(n_batches, 1), 6
            )
            if n_repair_gate_batches > 0:
                metrics[f"epoch_{epoch}_repair_gate_repair_mean"] = round(
                    total_repair_gate_repair_mean / n_repair_gate_batches, 6
                )
                metrics[f"epoch_{epoch}_repair_gate_protected_mean"] = round(
                    total_repair_gate_protected_mean / n_repair_gate_batches, 6
                )
        if group_routed_enabled:
            metrics[f"epoch_{epoch}_group_routed_residual_loss"] = round(
                total_group_routed_residual_loss / max(n_batches, 1), 6
            )
            metrics[f"epoch_{epoch}_group_routed_router_reg_loss"] = round(
                total_group_routed_router_reg_loss / max(n_batches, 1), 6
            )
            if n_group_routed_batches > 0:
                metrics[f"epoch_{epoch}_group_routed_correction"] = round(
                    total_group_routed_correction / n_group_routed_batches, 6
                )
        if cfg.get("group_dro", {}).get("enabled", False):
            metrics[f"epoch_{epoch}_group_dro_loss"] = round(
                total_group_dro_loss / max(n_batches, 1), 6
            )
        if pivot_gate_enabled and n_pivot_gate_batches > 0:
            metrics[f"epoch_{epoch}_pivot_gate_multiplier"] = round(
                total_pivot_gate_multiplier / n_pivot_gate_batches, 6
            )
        if anchor_state is not None:
            metrics[f"epoch_{epoch}_anchor_loss"] = round(total_anchor_loss / max(n_batches, 1), 6)
        if cfg.get("post_ckpt_repair_adapter", {}).get("enabled", False):
            metrics[f"epoch_{epoch}_post_ckpt_repair_adapter_scale"] = round(
                current_post_ckpt_repair_scale,
                6,
            )

        log_parts = [f"train_loss={train_loss_epoch:.6f}", f"teacher_w={teacher_eff_weight:.4f}"]
        if cfg.get("backbone_conditional_adapter_regularization", {}).get("enabled", False):
            log_parts.append(
                f"bca_reg={total_backbone_conditional_adapter_reg_loss / max(n_batches, 1):.4f}"
            )
        if cfg.get("branch_decorrelation", {}).get("enabled", False):
            log_parts.append(
                f"branch_decorr={total_branch_decorrelation_loss / max(n_batches, 1):.4f}"
            )
        if cfg.get("layer_consistency_routing", {}).get("enabled", False):
            log_parts.append(
                f"layer_cons={total_layer_consistency_loss / max(n_batches, 1):.4f}"
            )
        if ema_route_teacher_state is not None:
            log_parts.append(f"ema_route={total_ema_route_teacher_loss / max(n_batches, 1):.4f}")
            if n_ema_route_teacher_updates > 0:
                log_parts.append(
                    f"ema_m={total_ema_route_teacher_momentum / n_ema_route_teacher_updates:.4f}"
                )
        if stage1_pooled_feature_anchor_state is not None:
            log_parts.append(
                f"feat_anchor={total_stage1_pooled_feature_anchor_loss / max(n_batches, 1):.4f}"
            )
        if n_runtime_control_batches > 0:
            log_parts.append(
                f"comp_gate_max={total_runtime_composition_gate_max / n_runtime_control_batches:.4f}"
            )
            log_parts.append(
                f"comp_scale={total_runtime_composition_residual_scale / n_runtime_control_batches:.4f}"
            )
        if n_composition_confidence_batches > 0:
            log_parts.append(
                f"comp_conf={total_composition_confidence / n_composition_confidence_batches:.4f}"
            )
        if soft_condition_label_head is not None:
            log_parts.append(
                f"soft_label={total_soft_condition_label_supervision_loss / max(n_batches, 1):.4f}"
            )
            if total_soft_condition_lang_count > 0:
                log_parts.append(
                    f"lang_acc={total_soft_condition_lang_correct / total_soft_condition_lang_count:.4f}"
                )
            if total_soft_condition_domain_count > 0:
                log_parts.append(
                    f"domain_acc={total_soft_condition_domain_correct / total_soft_condition_domain_count:.4f}"
                )
        if meta_soft_bootstrap_state is not None:
            log_parts.append(f"meta_boot={total_meta_soft_bootstrap_loss / max(n_batches, 1):.4f}")
        if cfg.get("post_ckpt_repair_adapter", {}).get("enabled", False):
            log_parts.append(f"repair_scale={current_post_ckpt_repair_scale:.4f}")
        if n_protected_repair_bundle_batches > 0:
            log_parts.append(
                f"bundle={total_protected_repair_bundle_scale / n_protected_repair_bundle_batches:.4f}"
            )
        if polarity_enabled:
            log_parts.append(f"polarity={total_polarity_loss / max(n_batches, 1):.4f}")
        if region_cls_enabled:
            log_parts.append(f"region_cls={total_region_cls_loss / max(n_batches, 1):.4f}")
        if quantile_enabled:
            log_parts.append(f"quantile={total_quantile_loss / max(n_batches, 1):.4f}")
        if translation_enabled:
            log_parts.append(f"tp_sup={total_translation_pivot_loss / max(n_batches, 1):.4f}")
            log_parts.append(f"tp_pred={total_translation_pred_consistency_loss / max(n_batches, 1):.4f}")
            log_parts.append(f"tp_feat={total_translation_feat_consistency_loss / max(n_batches, 1):.4f}")
        if pivot_residual_enabled:
            log_parts.append(f"piv_res={total_pivot_residual_loss / max(n_batches, 1):.4f}")
        if pivot_trust_enabled:
            log_parts.append(f"piv_trust={total_pivot_trust_loss / max(n_batches, 1):.4f}")
            if n_pivot_trust_batches > 0:
                log_parts.append(f"piv_trust_gate={total_pivot_trust_gate / n_pivot_trust_batches:.4f}")
                log_parts.append(f"piv_trust_corr={total_pivot_trust_correction / n_pivot_trust_batches:.4f}")
        if va_residual_enabled:
            log_parts.append(f"va_res={total_va_residual_loss / max(n_batches, 1):.4f}")
            if n_va_residual_batches > 0:
                log_parts.append(f"va_gate={total_va_residual_gate / n_va_residual_batches:.4f}")
                log_parts.append(f"va_corr={total_va_residual_correction / n_va_residual_batches:.4f}")
        if protected_anchor_enabled:
            log_parts.append(f"prot_anchor={total_protected_anchor_loss / max(n_batches, 1):.4f}")
        if repair_gate_enabled:
            log_parts.append(f"repair_gate={total_repair_gate_loss / max(n_batches, 1):.4f}")
            if n_repair_gate_batches > 0:
                log_parts.append(f"gate_repair={total_repair_gate_repair_mean / n_repair_gate_batches:.4f}")
                log_parts.append(f"gate_prot={total_repair_gate_protected_mean / n_repair_gate_batches:.4f}")
        if group_routed_enabled:
            log_parts.append(f"group_res={total_group_routed_residual_loss / max(n_batches, 1):.4f}")
            log_parts.append(f"group_reg={total_group_routed_router_reg_loss / max(n_batches, 1):.4f}")
            if n_group_routed_batches > 0:
                log_parts.append(f"group_corr={total_group_routed_correction / n_group_routed_batches:.4f}")
        if cfg.get("group_dro", {}).get("enabled", False):
            log_parts.append(f"group_dro={total_group_dro_loss / max(n_batches, 1):.4f}")
        if pivot_gate_enabled and n_pivot_gate_batches > 0:
            log_parts.append(f"piv_gate={total_pivot_gate_multiplier / n_pivot_gate_batches:.4f}")
        if ema_var_enabled:
            log_parts.append(f"ema_var={total_ema_var_loss / max(n_batches, 1):.6f}")
            log_parts.append(f"ema_std_pred={ema_pred_std:.4f}")
            log_parts.append(f"ema_std_true={ema_true_std:.4f}")
            metrics[f"epoch_{epoch}_ema_var_loss"] = round(total_ema_var_loss / max(n_batches, 1), 6)
            metrics[f"epoch_{epoch}_ema_pred_std_a"] = round(ema_pred_std, 4)
            metrics[f"epoch_{epoch}_ema_true_std_a"] = round(ema_true_std, 4)
        if mem_bank_enabled:
            log_parts.append(f"vicreg_bank={total_vicreg_bank_loss / max(n_batches, 1):.6f}")
            metrics[f"epoch_{epoch}_vicreg_bank_loss"] = round(total_vicreg_bank_loss / max(n_batches, 1), 6)
        if alternating_phase_counts:
            log_parts.append(
                "alt="
                + ",".join(
                    f"{name}:{count}"
                    for name, count in sorted(alternating_phase_counts.items())
                )
            )
        logger.info(
            "Epoch %s/%s finished: %s",
            epoch + 1, n_epochs, " ".join(log_parts),
        )

        # FDS: smooth per-bin statistics at end of each epoch
        if fds_enabled:
            fds_module.smooth_stats()
            logger.info("FDS: smoothed statistics at end of epoch %d", epoch + 1)

        if dev_data:
            _refresh_backbone_runtime_controls(model, cfg, global_update)
            current_post_ckpt_repair_scale, current_protected_repair_bundle_scale = (
                _refresh_effective_post_ckpt_repair_adapter_scale(
                    model,
                    cfg,
                    global_update,
                )
            )
            if last_eval_update == global_update and last_eval_rmse is not None:
                dev_rmse = last_eval_rmse
                logger.info(
                    "Epoch %s/%s dev RMSE_VA reused from update %s: %.6f",
                    epoch + 1, n_epochs, global_update, dev_rmse,
                )
            else:
                dev_rmse = _evaluate_dev_rmse(
                    model,
                    dev_data,
                    tokenizer,
                    device,
                    cfg,
                    va_norm_stats=va_norm_stats,
                    quantile_head=quantile_head if quantile_enabled else None,
                    pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
                    pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
                    va_residual_head=va_residual_head if va_residual_enabled else None,
                    group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
                )
            metrics[f"epoch_{epoch}_dev_rmse_va"] = dev_rmse
            logger.info("Epoch %s/%s dev RMSE_VA=%.6f", epoch + 1, n_epochs, dev_rmse)
            epoch_macro_rmse = None
            epoch_macro_breakdown = {}
            epoch_macro_selection_score = None
            epoch_macro_selection = {}
            epoch_canary_metrics = None
            epoch_canary_score = None
            epoch_canary_guard_reasons = []
            if macro_dev_enabled:
                epoch_macro_rmse, epoch_macro_breakdown = _evaluate_macro_dev_rmse(
                    model,
                    macro_dev_targets,
                    tokenizer,
                    device,
                    va_norm_stats=va_norm_stats,
                    quantile_head=quantile_head if quantile_enabled else None,
                    pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
                    pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
                    va_residual_head=va_residual_head if va_residual_enabled else None,
                    group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
                )
                if matched_protocol:
                    _archive_matched_dev(
                        model, shared_run_dir, "epoch", epoch, global_update,
                        epoch_macro_breakdown, cfg, checkpoint_extra_modules, dist_ctx,
                    )
                if epoch_macro_rmse is not None:
                    metrics[f"epoch_{epoch}_macro_dev_rmse_va"] = epoch_macro_rmse
                    epoch_macro_breakdown_rmse = {
                        name: float(values["rmse_va"]) if isinstance(values, dict) else float(values)
                        for name, values in epoch_macro_breakdown.items()
                    }
                    epoch_macro_selection_score, epoch_macro_selection = compute_macro_dev_selection(
                        epoch_macro_breakdown,
                        macro_dev_selection_cfg,
                    )
                    metrics[f"epoch_{epoch}_macro_dev_selection_score"] = epoch_macro_selection_score
                else:
                    epoch_macro_breakdown_rmse = {}
                logger.info(
                    "Epoch %s/%s macro-dev RMSE_VA=%.6f selection=%.6f targets=%s",
                    epoch + 1,
                    n_epochs,
                    epoch_macro_rmse,
                    epoch_macro_selection_score if epoch_macro_selection_score is not None else epoch_macro_rmse,
                    epoch_macro_breakdown_rmse,
                )
            else:
                epoch_macro_breakdown_rmse = {}
            adapter_usage = model.backbone_adapter_usage() if hasattr(model, "backbone_adapter_usage") else {}
            if adapter_usage:
                for key, value in adapter_usage.items():
                    metrics[f"epoch_{epoch}_{key}"] = value
                logger.info(
                    "Epoch %s/%s backbone-conditional usage=%s",
                    epoch + 1,
                    n_epochs,
                    {k: round(v, 6) for k, v in adapter_usage.items()},
                )
            if checkpoint_canary_samples and checkpoint_canary_teacher_map:
                epoch_canary_metrics = _evaluate_checkpoint_canary(
                    model,
                    checkpoint_canary_samples,
                    checkpoint_canary_teacher_map,
                    tokenizer,
                    device,
                    cfg,
                    va_norm_stats=va_norm_stats,
                    quantile_head=quantile_head if quantile_enabled else None,
                    pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
                    pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
                    va_residual_head=va_residual_head if va_residual_enabled else None,
                    group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
                )
                if epoch_canary_metrics is not None:
                    epoch_canary_score = _checkpoint_canary_metric_value(
                        epoch_canary_metrics,
                        checkpoint_canary_cfg["metric"],
                    )
                    metrics[f"epoch_{epoch}_checkpoint_canary_teacher_rmse_va"] = epoch_canary_metrics["teacher_rmse_va"]
                    metrics[f"epoch_{epoch}_checkpoint_canary_teacher_macro_rmse_va"] = epoch_canary_metrics["teacher_macro_rmse_va"]
                    if epoch_canary_metrics.get("label_rmse_va") is not None:
                        metrics[f"epoch_{epoch}_checkpoint_canary_label_rmse_va"] = epoch_canary_metrics["label_rmse_va"]
                    if epoch_canary_metrics.get("label_macro_rmse_va") is not None:
                        metrics[f"epoch_{epoch}_checkpoint_canary_label_macro_rmse_va"] = epoch_canary_metrics["label_macro_rmse_va"]
                    if epoch_canary_score is not None:
                        metrics[f"epoch_{epoch}_checkpoint_canary_selection_score"] = epoch_canary_score
                    epoch_canary_guard_reasons = _checkpoint_canary_guard_reasons(
                        checkpoint_canary_cfg,
                        epoch_canary_score,
                        checkpoint_canary_baseline_score,
                    )
                    logger.info(
                        "Epoch %s/%s checkpoint-canary %s=%.6f teacher_macro=%.6f label_macro=%s guard=%s",
                        epoch + 1,
                        n_epochs,
                        checkpoint_canary_cfg["metric"],
                        epoch_canary_score if epoch_canary_score is not None else float("nan"),
                        epoch_canary_metrics["teacher_macro_rmse_va"],
                        (
                            f"{epoch_canary_metrics['label_macro_rmse_va']:.6f}"
                            if epoch_canary_metrics.get("label_macro_rmse_va") is not None
                            else "n/a"
                        ),
                        "pass" if not epoch_canary_guard_reasons else "; ".join(epoch_canary_guard_reasons),
                    )
            if cfg.get("save_all_checkpoints", False) and not matched_protocol:
                _save_checkpoint(
                    model,
                    run_dir,
                    epoch,
                    {"dev_rmse_va": dev_rmse},
                    cfg,
                    filename=f"checkpoint_epoch{epoch}.pt",
                    global_update=global_update,
                    extra_modules=checkpoint_extra_modules,
                    dist_ctx=dist_ctx,
                )
            selection_score = (
                epoch_macro_selection_score
                if macro_dev_use_for_best and epoch_macro_selection_score is not None
                else dev_rmse
            )
            if checkpoint_canary_cfg.get("use_for_best", False) and epoch_canary_score is not None:
                selection_score += checkpoint_canary_cfg["weight"] * epoch_canary_score
            selection_blocked = (
                checkpoint_canary_cfg.get("use_for_best", False)
                and len(epoch_canary_guard_reasons) > 0
            )
            if selection_blocked:
                logger.info(
                    "Skipped epoch checkpoint update at epoch=%s due to checkpoint-canary guard: %s",
                    epoch,
                    "; ".join(epoch_canary_guard_reasons),
                )
            elif best_selection_score is None or selection_score < best_selection_score:
                best_selection_score = selection_score
                best_dev_rmse = dev_rmse
                if epoch_macro_rmse is not None:
                    best_macro_dev_rmse = epoch_macro_rmse
                best_epoch = epoch
                best_update = global_update
                _save_best_checkpoint(
                    model,
                    run_dir,
                    epoch,
                    {"dev_rmse_va": dev_rmse},
                    cfg,
                    event="epoch",
                    global_update=global_update,
                    extra_modules=checkpoint_extra_modules,
                    dist_ctx=dist_ctx,
                )
                logger.info(
                    "New best selection score %.6f at epoch %s (dev=%.6f macro=%s selection=%s)",
                    selection_score,
                    epoch,
                    dev_rmse,
                    f"{epoch_macro_rmse:.6f}" if epoch_macro_rmse is not None else "n/a",
                    epoch_macro_selection,
                )
        if stop_after_max_updates:
            break

    checkpoint_dir = shared_run_dir if dist_ctx.enabled else run_dir
    if (checkpoint_dir / "checkpoint.pt").exists():
        _refresh_backbone_runtime_controls(model, cfg, global_update)
        _refresh_effective_post_ckpt_repair_adapter_scale(model, cfg, global_update)
        _, reload_report = _load_checkpoint(
            model,
            checkpoint_dir / "checkpoint.pt",
            cfg=cfg,
            load_backbone=True,
            load_head="always",
            extra_modules=checkpoint_extra_modules,
        )
        logger.info("Reloaded best checkpoint for final eval: %s", reload_report)

    metrics["best_epoch"] = best_epoch
    metrics["best_update"] = best_update
    metrics["num_optimizer_updates"] = global_update
    if best_dev_rmse is not None:
        metrics["best_dev_rmse_va"] = best_dev_rmse
    if best_macro_dev_rmse is not None:
        metrics["best_macro_dev_rmse_va"] = best_macro_dev_rmse
    if best_selection_score is not None:
        selection_metric_name = macro_dev_selection_metric
        if checkpoint_canary_cfg.get("enabled", False) and checkpoint_canary_cfg.get("use_for_best", False):
            selection_metric_name = (
                f"{macro_dev_selection_metric}+{checkpoint_canary_cfg['weight']:.3f}*checkpoint_canary_{checkpoint_canary_cfg['metric']}"
            )
        metrics["best_selection_score"] = best_selection_score
        if selection_metric_name in {"macro_dev_rmse_va", "dev_rmse_va"}:
            metrics["best_selection_rmse_va"] = best_selection_score
        metrics["best_selection_metric"] = selection_metric_name

    if macro_dev_enabled:
        final_macro_rmse, final_macro_breakdown = _evaluate_macro_dev_rmse(
            model,
            macro_dev_targets,
            tokenizer,
            device,
            va_norm_stats=va_norm_stats,
            quantile_head=quantile_head if quantile_enabled else None,
            pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
            pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
            va_residual_head=va_residual_head if va_residual_enabled else None,
            group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
        )
        if matched_protocol:
            validate_complete_dev(final_macro_breakdown, matched_protocol["training_targets"])
        if final_macro_rmse is not None:
            metrics["final_macro_dev_rmse_va"] = final_macro_rmse
            final_macro_selection_score, final_macro_selection = compute_macro_dev_selection(
                final_macro_breakdown,
                macro_dev_selection_cfg,
            )
            metrics["final_macro_dev_selection_score"] = final_macro_selection_score
            (run_dir / "macro_dev_eval.json").write_text(
                json.dumps(
                    {
                        "targets": final_macro_breakdown,
                        "macro_dev_rmse_va": final_macro_rmse,
                        "selection_metric": macro_dev_selection_metric,
                        "selection_score": final_macro_selection_score,
                        "selection": final_macro_selection,
                        "use_for_best": macro_dev_use_for_best,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    if checkpoint_canary_samples and checkpoint_canary_teacher_map:
        final_canary_metrics = _evaluate_checkpoint_canary(
            model,
            checkpoint_canary_samples,
            checkpoint_canary_teacher_map,
            tokenizer,
            device,
            cfg,
            va_norm_stats=va_norm_stats,
            quantile_head=quantile_head if quantile_enabled else None,
            pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
            pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
            va_residual_head=va_residual_head if va_residual_enabled else None,
            group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
        )
        if final_canary_metrics is not None:
            final_canary_score = _checkpoint_canary_metric_value(
                final_canary_metrics,
                checkpoint_canary_cfg["metric"],
            )
            metrics["final_checkpoint_canary_teacher_rmse_va"] = final_canary_metrics["teacher_rmse_va"]
            metrics["final_checkpoint_canary_teacher_macro_rmse_va"] = final_canary_metrics["teacher_macro_rmse_va"]
            if final_canary_metrics.get("label_rmse_va") is not None:
                metrics["final_checkpoint_canary_label_rmse_va"] = final_canary_metrics["label_rmse_va"]
            if final_canary_metrics.get("label_macro_rmse_va") is not None:
                metrics["final_checkpoint_canary_label_macro_rmse_va"] = final_canary_metrics["label_macro_rmse_va"]
            if final_canary_score is not None:
                metrics["final_checkpoint_canary_selection_score"] = final_canary_score
            (run_dir / "checkpoint_canary_eval.json").write_text(
                json.dumps(
                    {
                        "metric": checkpoint_canary_cfg["metric"],
                        "selection_score": final_canary_score,
                        "baseline_score": checkpoint_canary_baseline_score,
                        "weight": checkpoint_canary_cfg["weight"],
                        "max_score": checkpoint_canary_cfg.get("max_score"),
                        "max_increase": checkpoint_canary_cfg.get("max_increase"),
                        **final_canary_metrics,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    run_test = bool(cfg.get("run_test", True))
    eval_splits = [("dev", dev_data)]
    if run_test:
        eval_splits.append(("test", test_data))

    quantile_inference_head = None
    if quantile_enabled and quantile_head is not None and quantile_cfg.get("use_median_inference", False):
        quantile_inference_head = quantile_head
        logger.info("W3: using quantile median for arousal inference")

    va_calib_enabled = cfg.get("va_calibration", False)
    va_calib_v = None
    va_calib_a = None
    for split_name, data in eval_splits:
        if not data:
            continue
        preds = _predict(
            model,
            data,
            tokenizer,
            device,
            cfg,
            va_norm_stats=va_norm_stats,
            quantile_head=quantile_inference_head,
            pivot_residual_head=pivot_residual_head if pivot_residual_enabled else None,
            pivot_trust_head=pivot_trust_head if pivot_trust_enabled else None,
            va_residual_head=va_residual_head if va_residual_enabled else None,
            group_routed_residual_head=group_routed_residual_head if group_routed_enabled else None,
        )

        if va_calib_enabled and va_calib_v is not None and split_name == "test":
            vp_raw = np.array([p["v_pred"] for p in preds])
            ap_raw = np.array([p["a_pred"] for p in preds])
            vp_cal = np.clip(va_calib_v.transform(vp_raw), 1.0, 9.0)
            ap_cal = np.clip(va_calib_a.transform(ap_raw), 1.0, 9.0)
            for i, pred in enumerate(preds):
                pred["v_pred"] = round(float(vp_cal[i]), 2)
                pred["a_pred"] = round(float(ap_cal[i]), 2)

        write_predictions(preds, run_dir / f"predictions_{split_name}.jsonl")
        has_gold = all(pred.get("v_true") is not None for pred in preds)
        if not has_gold:
            continue

        vp = np.array([p["v_pred"] for p in preds])
        ap = np.array([p["a_pred"] for p in preds])
        vt = np.array([p["v_true"] for p in preds])
        at = np.array([p["a_true"] for p in preds])
        prefix = "" if split_name == "dev" else f"{split_name}_"
        metrics[f"{prefix}rmse_v"] = rmse_v(vp, vt)
        metrics[f"{prefix}rmse_a"] = rmse_a(ap, at)
        metrics[f"{prefix}rmse_va"] = rmse_va(vp, ap, vt, at)
        metrics[f"{prefix}pcc_v"] = pcc_v(vp, vt)
        metrics[f"{prefix}pcc_a"] = pcc_a(ap, at)
        for diag_key, diag_val in _build_eval_diagnostics(vp, ap, vt, at).items():
            metrics[f"{prefix}{diag_key}"] = diag_val

        if va_calib_enabled and split_name == "dev":
            calib_cfg = cfg.get("va_calibration_method", {})
            method = calib_cfg.get("method", "ols") if isinstance(calib_cfg, dict) else "ols"
            kwargs = {k: v for k, v in calib_cfg.items() if k != "method"} if isinstance(calib_cfg, dict) else {}
            va_calib_v = make_calibrator(method, **kwargs).fit(vp, vt)
            va_calib_a = make_calibrator(method, **kwargs).fit(ap, at)
            metrics["va_calibration_method"] = method

    test_pred_path = run_dir / "predictions_test.jsonl"
    dev_pred_path = run_dir / "predictions_dev.jsonl"
    main_pred = run_dir / "predictions.jsonl"
    if test_pred_path.exists():
        shutil.copy(test_pred_path, main_pred)
    elif dev_pred_path.exists():
        shutil.copy(dev_pred_path, main_pred)

    metrics["run_test"] = run_test
    save_metrics(metrics, run_dir)
    logger.info("Metrics: %s", metrics)
    return metrics
