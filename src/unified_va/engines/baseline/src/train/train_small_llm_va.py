"""Training entry for decoder-only small LLM VA regression."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from src.data.datamodule import build_dimasr_dataloader
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
from src.models.heads import VAUncertaintyHead
from src.models.losses import (
    arousal_anti_collapse_loss,
    combine_primary_aux_losses,
    gaussian_nll_va,
    teacher_va_distill_loss,
    va_decorrelation_loss,
    va_loss,
)
from src.models.valence_sharpening import valence_sharpening_loss
from src.methods.D37_variance.variance_regularization import variance_regularization_loss
from src.models.va_region_weights import compute_va_region_weights
from src.models.small_llm_regressor import build_small_llm_regressor, freeze_small_llm_layers
from src.train.calibration import make_calibrator
from src.train.macro_selection import compute_macro_dev_selection, macro_dev_selection_metric_name
from src.train.recipe import build_llrd_param_groups
from src.train.tta import tta_predict_batch
from src.utils.device import setup_device
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


def _evaluate_macro_dev_rmse(model, macro_targets, tokenizer, device, va_norm_stats=None):
    if not macro_targets:
        return None, {}
    per_target = {}
    for target in macro_targets:
        dev_data = target["dev_data"]
        if not dev_data:
            continue
        preds = _predict(model, dev_data, tokenizer, device, target["eval_cfg"], va_norm_stats=va_norm_stats)
        vp = np.array([p["v_pred"] for p in preds])
        ap = np.array([p["a_pred"] for p in preds])
        vt = np.array([p["v_true"] for p in preds])
        at = np.array([p["a_true"] for p in preds])
        per_target[target["name"]] = float(rmse_va(vp, ap, vt, at))
    if not per_target:
        return None, {}
    macro = float(np.mean(list(per_target.values())))
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


def _setup_translation_pivot(cfg: dict, train_data: list[dict], dev_data: list[dict], test_data: list[dict], run_dir: Path):
    translation_cfg = cfg.get("translation_pivot", {})
    if not translation_cfg.get("enabled", False):
        return None, None

    source = (
        translation_cfg.get("train_sidecar_paths")
        or translation_cfg.get("train_sidecar_path")
        or translation_cfg.get("sidecar_paths")
        or translation_cfg.get("sidecar_path")
    )
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


def _build_optimizer(cfg: dict, model):
    lr = cfg.get("lr", 2e-4)
    weight_decay = cfg.get("weight_decay", 0.01)
    opt_cfg = cfg.get("optimizer", {})
    llrd_cfg = opt_cfg.get("llrd", {})
    if llrd_cfg.get("enabled", False):
        param_groups = build_llrd_param_groups(
            model.backbone,
            [model.head],
            lr=lr,
            weight_decay=weight_decay,
            llrd_factor=llrd_cfg.get("factor", 0.95),
        )
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": params, "lr": lr, "weight_decay": weight_decay}]
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas)


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

    max_multiplier = float(gate_cfg.get("max_multiplier", 2.0))
    min_multiplier = float(gate_cfg.get("min_multiplier", 1.0))
    if min_multiplier < 1.0 or max_multiplier < min_multiplier:
        raise ValueError(
            "pivot_hardness_gate requires 1.0 <= min_multiplier <= max_multiplier"
        )


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


def _capture_anchor_state(model, cfg: dict):
    anchor_cfg = cfg.get("w2_anchor", {})
    if not anchor_cfg.get("enabled", False):
        return None
    scope = str(anchor_cfg.get("scope", "backbone_trainable"))
    state = {}
    if scope == "backbone_trainable":
        for name, param in model.backbone.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().clone()
    elif scope == "all_trainable":
        for name, param in model.named_parameters():
            if param.requires_grad:
                state[name] = param.detach().clone()
    else:
        raise ValueError(f"unsupported w2_anchor.scope={scope!r}")
    logger.info("W2 anchor enabled: scope=%s tensors=%d", scope, len(state))
    return state


def _anchor_regularization_loss(model, anchor_state, cfg: dict, device: torch.device):
    if not anchor_state:
        return torch.tensor(0.0, device=device)
    anchor_cfg = cfg.get("w2_anchor", {})
    scope = str(anchor_cfg.get("scope", "backbone_trainable"))
    terms = []
    if scope == "backbone_trainable":
        iterator = model.backbone.named_parameters()
    else:
        iterator = model.named_parameters()
    for name, param in iterator:
        if not param.requires_grad:
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
                     filename: str = "checkpoint.pt", global_update: int | None = None):
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "checkpoint_type": "head_only",
        "head": model.head.state_dict(),
        "cfg_tag": cfg.get("tag", "small_llm_va"),
        "head_type": cfg.get("head_type", "unknown"),
        "pooling": cfg.get("small_llm", {}).get("pooling", getattr(model, "pooling", "last_token")),
        "freeze_until": cfg.get("freeze_until", 0),
    }
    if global_update is not None:
        payload["global_update"] = int(global_update)
    # Save pool_gate if it exists (gated_last_mean pooling)
    if hasattr(model, "pool_gate") and model.pool_gate is not None:
        payload["pool_gate"] = model.pool_gate.state_dict()
    if hasattr(model.backbone, "peft_config"):
        try:
            from peft import get_peft_model_state_dict
            payload["backbone_peft"] = get_peft_model_state_dict(model.backbone)
            payload["checkpoint_type"] = "peft_head"
        except Exception as exc:
            logger.warning("Failed to export PEFT weights cleanly: %s", exc)
    elif cfg.get("save_full_backbone_checkpoint", False):
        payload["backbone"] = model.backbone.state_dict()
        payload["checkpoint_type"] = "full_backbone"
    else:
        logger.warning("Saving head-only checkpoint; set save_full_backbone_checkpoint=true for full restoration")
    torch.save(payload, run_dir / filename)


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


def _load_state_subset(module, incoming_state: dict | None, module_name: str):
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
                     load_head: str = "if_compatible"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
        load_report["head"] = _load_state_subset(model.head, incoming_head, "head")
    elif incoming_head is not None and head_meta_match:
        compatible, skipped = _filter_compatible_state_dict(model.head, incoming_head)
        if len(compatible) == len(incoming_head):
            load_report["head"] = _load_state_subset(model.head, incoming_head, "head")
        else:
            head_skip_reason = f"incompatible_head_keys:{len(skipped)}"
            load_report["head"] = {
                "module": "head",
                "loaded_keys": 0,
                "skipped_keys": len(skipped),
                "skip_reason": head_skip_reason,
                "skipped_examples": dict(list(skipped.items())[:5]),
            }
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

    return ckpt, load_report


def _predict(model, data, tokenizer, device, cfg, va_norm_stats=None, quantile_head=None):
    model.eval()
    small_llm_cfg = cfg.get("small_llm", {})
    teacher_cfg = cfg.get("teacher", {})

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
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory,
        persistent_workers=eval_persistent_workers,
    )

    preds = []
    use_uncertainty = isinstance(model.head, VAUncertaintyHead)

    for batch in loader:
        if tta_enabled:
            # Use TTA prediction
            va, tta_uncertainty = tta_predict_batch(
                model=model,
                batch=batch,
                device=device,
                n_samples=tta_n_samples,
                aggregation=tta_aggregation,
                mc_dropout=tta_mc_dropout,
            )
            va = va.cpu()
            if tta_uncertainty is not None:
                tta_uncertainty = tta_uncertainty.cpu()
            sigma = None  # TTA doesn't use uncertainty head
        else:
            # Standard prediction
            with torch.no_grad():
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                outputs = model(ids, mask)
                if use_uncertainty:
                    mu, log_sigma = outputs
                    sigma = _denormalize_sigma(log_sigma.exp().cpu(), va_norm_stats)
                    va = mu.cpu()
                else:
                    sigma = None
                    va = outputs.cpu()
            tta_uncertainty = None

        va = _denormalize_va(va, va_norm_stats).clamp(1.0, 9.0)

        # Quantile median override for arousal (in raw space, after denormalization)
        if quantile_head is not None:
            with torch.no_grad():
                ids_q = batch["input_ids"].to(device)
                mask_q = batch["attention_mask"].to(device)
                features = model.encode(ids_q, mask_q)
                head_param = next(model.head.parameters(), None)
                if head_param is not None and features.dtype != head_param.dtype:
                    features = features.to(head_param.dtype)
                quantile_head.eval()
                a_median = quantile_head.predict_median(features).cpu()
                # a_median is in raw space (trained on target_raw), clamp to valid range
                va[:, 1] = a_median.clamp(1.0, 9.0)

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


def _evaluate_dev_rmse(model, dev_data, tokenizer, device, cfg, va_norm_stats=None) -> float | None:
    if not dev_data:
        return None
    dev_preds = _predict(model, dev_data, tokenizer, device, cfg, va_norm_stats=va_norm_stats)
    vp = np.array([p["v_pred"] for p in dev_preds])
    ap = np.array([p["a_pred"] for p in dev_preds])
    vt = np.array([p["v_true"] for p in dev_preds])
    at = np.array([p["a_true"] for p in dev_preds])
    return rmse_va(vp, ap, vt, at)


def _set_training_mode(model, polarity_head=None, region_cls_head=None, quantile_head=None):
    model.train()
    if polarity_head is not None:
        polarity_head.train()
    if region_cls_head is not None:
        region_cls_head.train()
    if quantile_head is not None:
        quantile_head.train()


def train_small_llm_va(cfg: dict) -> dict:
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = setup_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        set_cuda_seed(seed)
    run_dir = make_run_dir(tag=cfg.get("tag", "small_llm_va"), seed=seed)
    save_config(cfg, run_dir)
    add_file_handler(logger, run_dir / "stdout.log")
    _validate_arousal_anti_collapse_cfg(cfg)
    _validate_pivot_hardness_gate_cfg(cfg)

    splits = load_splits(cfg)
    train_data = _merge_auxiliary_data(cfg, splits["train"])
    dev_data = splits["dev"]
    test_data = splits["test"]
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

    model, tokenizer, hidden_dim = build_small_llm_regressor(cfg)
    model.to(device)
    freeze_small_llm_layers(model, cfg.get("freeze_until", 0))

    pretrain_from = cfg.get("pretrain_from")
    if pretrain_from:
        ckpt_path = Path(pretrain_from) / "checkpoint.pt"
        if ckpt_path.exists():
            pretrain_load_cfg = cfg.get("pretrain_load", {}) or {}
            _, load_report = _load_checkpoint(
                model,
                ckpt_path,
                cfg=cfg,
                load_backbone=pretrain_load_cfg.get("load_backbone", True),
                load_head=pretrain_load_cfg.get("load_head", "if_compatible"),
            )
            logger.info("Loaded small LLM checkpoint from %s", ckpt_path)
            logger.info("Checkpoint load report: %s", load_report)
    anchor_state = _capture_anchor_state(model, cfg)

    teacher_map, teacher_audit = _setup_teacher(cfg, train_data, run_dir)
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
    pivot_gate_cfg = cfg.get("pivot_hardness_gate", {})
    pivot_gate_enabled = bool(pivot_gate_cfg.get("enabled", False)) and translation_enabled
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
            "Pivot hardness gate enabled: focus_dim=%s start_update=%d min_multiplier=%.3f max_multiplier=%.3f combine=%s",
            pivot_gate_cfg.get("focus_dim", "a"),
            int(pivot_gate_cfg.get("start_update", 0) or 0),
            float(pivot_gate_cfg.get("min_multiplier", 1.0)),
            float(pivot_gate_cfg.get("max_multiplier", 2.0)),
            pivot_gate_cfg.get("combine", "product"),
        )
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

    train_loader = build_dimasr_dataloader(
        train_data,
        tokenizer,
        batch_size=cfg.get("batch_size", 1),
        shuffle=True,
        max_len=cfg.get("max_len", 256),
        input_format=cfg.get("input_format", "instruction_va_v1"),
        teacher_map=teacher_map,
        prompt_template=small_llm_cfg.get("prompt_template"),
        translation_map=translation_map,
        translation_pivot=translation_cfg,
        teacher_explanation_as_input=teacher_cfg.get("explanation_as_input", False),
        teacher_explanation_prefix=teacher_cfg.get("explanation_prefix", "teacher_explanation"),
        teacher_explanation_min_quality=teacher_cfg.get("teacher_explanation_min_quality", 0.0),
        num_workers=dataloader_num_workers,
        pin_memory=dataloader_pin_memory,
        persistent_workers=dataloader_persistent_workers,
    )

    use_uncertainty = isinstance(model.head, VAUncertaintyHead)
    va_norm_stats = _build_va_norm_stats(train_data, cfg.get("va_target_normalization", False))

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
    total_updates = math.ceil(len(train_loader) / grad_accum) * max(n_epochs, 1)
    eval_every_updates = max(int(cfg.get("eval_every_updates", 0) or 0), 0)
    save_every_updates = max(int(cfg.get("save_every_updates", 0) or 0), 0)
    max_step_checkpoints_to_keep = max(int(cfg.get("max_step_checkpoints_to_keep", 0) or 0), 0)
    optimizer = _build_optimizer(cfg, model)
    # Add polarity head params to optimizer
    if polarity_enabled and polarity_head is not None:
        optimizer.add_param_group({
            "params": list(polarity_head.parameters()),
            "lr": cfg.get("lr", 2e-4),
            "weight_decay": cfg.get("weight_decay", 0.01),
        })
    # Add region classification head params to optimizer
    if region_cls_enabled and region_cls_head is not None:
        optimizer.add_param_group({
            "params": list(region_cls_head.parameters()),
            "lr": cfg.get("lr", 2e-4),
            "weight_decay": cfg.get("weight_decay", 0.01),
        })
    # Add quantile head params to optimizer
    if quantile_enabled and quantile_head is not None:
        optimizer.add_param_group({
            "params": list(quantile_head.parameters()),
            "lr": cfg.get("lr", 2e-4),
            "weight_decay": cfg.get("weight_decay", 0.01),
        })
    scheduler = _build_scheduler(cfg, optimizer, total_updates)

    amp_enabled = bool(cfg.get("amp", device.type == "cuda")) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

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
    if translation_audit:
        metrics["translation_pivot_train_hit_rate"] = translation_audit["train"]["hit_rate"]
        metrics["translation_pivot_train_non_english_hit_rate"] = translation_audit["train"]["non_english_hit_rate"]
        metrics["translation_pivot_enabled"] = True
    if pivot_gate_enabled:
        metrics["pivot_hardness_gate_enabled"] = True
    if risk_gate_audit:
        metrics["risk_gate_enabled"] = True
        metrics["risk_gate_match_rate"] = risk_gate_audit["match_rate"]
        metrics["risk_gate_mean_weight"] = risk_gate_audit["applied_mean"]
    if defect_focus_audit:
        metrics["defect_focus_enabled"] = True
        metrics["defect_focus_match_rate"] = defect_focus_audit["match_rate"]
        metrics["defect_focus_mean_weight"] = defect_focus_audit["applied_mean"]

    for epoch in range(n_epochs):
        _set_training_mode(
            model,
            polarity_head=polarity_head if polarity_enabled else None,
            region_cls_head=region_cls_head if region_cls_enabled else None,
            quantile_head=quantile_head if quantile_enabled else None,
        )
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
        total_anchor_loss = 0.0
        n_batches = 0
        teacher_eff_weight = teacher_weight_schedule(
            epoch,
            float(teacher_cfg.get("loss_weight", 0.0)),
            int(teacher_cfg.get("warmup_epochs", 0)),
            int(teacher_cfg.get("ramp_epochs", 0)),
        )

        if debug_enabled:
            logger.info(
                "Debug smoke epoch %d/%d: train_loader_batches=%d max_train_batches=%d",
                epoch + 1, n_epochs, len(train_loader), debug_max_train_batches,
            )

        train_iter = iter(train_loader)
        step = 0
        while True:
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

            main_forward_dt = 0.0
            quantile_reencode_dt = 0.0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                # Extract features for auxiliary heads or FDS when needed
                features = None
                pivot_features = None
                pivot_student_va = None
                pivot_student_va_raw = None
                pivot_supervised_loss = torch.tensor(0.0, device=device)
                pivot_pred_consistency_loss = torch.tensor(0.0, device=device)
                pivot_feat_consistency_loss = torch.tensor(0.0, device=device)
                head_regularization_loss = torch.tensor(0.0, device=device)
                pivot_head_regularization_loss = torch.tensor(0.0, device=device)
                anchor_loss = torch.tensor(0.0, device=device)
                main_forward_t0 = time.perf_counter()
                if translation_enabled or fds_enabled or polarity_enabled:
                    features = model.encode(ids, mask)
                    head_param = next(model.head.parameters(), None)
                    if head_param is not None and features.dtype != head_param.dtype:
                        features = features.to(head_param.dtype)
                    if fds_enabled:
                        features = fds_module(features, target_raw[:, 1], epoch)
                    outputs = model.head(features)
                else:
                    outputs = model(ids, mask)
                if hasattr(model.head, "regularization_loss"):
                    reg_loss = model.head.regularization_loss()
                    if reg_loss is not None:
                        head_regularization_loss = reg_loss
                main_forward_dt = time.perf_counter() - main_forward_t0
                if use_uncertainty:
                    mu, log_sigma = outputs
                    base_loss = gaussian_nll_va(
                        mu,
                        log_sigma,
                        target,
                        sample_weights=sample_weights,
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
                        sample_weights=sample_weights,
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
                    pivot_features = model.encode(pivot_ids, pivot_mask)
                    head_param = next(model.head.parameters(), None)
                    if head_param is not None and pivot_features.dtype != head_param.dtype:
                        pivot_features = pivot_features.to(head_param.dtype)
                    pivot_outputs = model.head(pivot_features)
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
                student_va_raw = _denormalize_va(student_va, va_norm_stats)
                if pivot_student_va is not None:
                    pivot_student_va_raw = _denormalize_va(pivot_student_va, va_norm_stats)
                arousal_aux_loss = torch.tensor(0.0, device=device)
                if arousal_enabled:
                    arousal_sample_weights = sample_weights
                    pivot_gate_multiplier = _build_pivot_hardness_gate_weights(
                        pivot_gate_cfg,
                        global_update=global_update,
                        pivot_sample_mask=pivot_sample_mask,
                        pivot_confidence=pivot_confidence,
                        source_pred_raw=student_va_raw,
                        pivot_pred_raw=pivot_student_va_raw,
                    ) if pivot_gate_enabled else None
                    if pivot_gate_multiplier is not None:
                        arousal_sample_weights = _merge_sample_weights(
                            arousal_sample_weights,
                            pivot_gate_multiplier,
                        )
                    arousal_aux_loss = arousal_anti_collapse_loss(
                        student_va_raw,
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
                        region_features = model.encode(ids, mask)
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
                        q_features = model.encode(ids, mask)
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
                    + head_regularization_loss
                    + pivot_head_regularization_loss
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
                if decorr_enabled and pred.size(0) > 4:
                    decorr_loss = float(decorr_cfg.get("weight", 0.1)) * va_decorrelation_loss(
                        student_va, target)
                var_reg_loss = torch.tensor(0.0, device=device)
                if var_reg_enabled and pred.size(0) > 4:
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
                        features = model.encode(ids, mask)
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

            backward_t0 = time.perf_counter()
            loss_for_backward = loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            backward_dt = time.perf_counter() - backward_t0

            optimizer_dt = 0.0
            should_step = (
                step % grad_accum == 0
                or step == len(train_loader)
                or (debug_max_train_batches and step == debug_max_train_batches)
            )
            if should_step:
                optimizer_t0 = time.perf_counter()
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
                optimizer_dt = time.perf_counter() - optimizer_t0
                global_update += 1

                if dev_data and eval_every_updates > 0 and global_update % eval_every_updates == 0:
                    step_dev_rmse = _evaluate_dev_rmse(
                        model, dev_data, tokenizer, device, cfg, va_norm_stats=va_norm_stats
                    )
                    last_eval_update = global_update
                    last_eval_rmse = step_dev_rmse
                    metrics[f"update_{global_update}_dev_rmse_va"] = step_dev_rmse
                    step_macro_rmse = None
                    step_macro_breakdown = {}
                    step_macro_selection_score = None
                    step_macro_selection = {}
                    if macro_dev_enabled:
                        step_macro_rmse, step_macro_breakdown = _evaluate_macro_dev_rmse(
                            model, macro_dev_targets, tokenizer, device, va_norm_stats=va_norm_stats
                        )
                        if step_macro_rmse is not None:
                            metrics[f"update_{global_update}_macro_dev_rmse_va"] = step_macro_rmse
                            step_macro_selection_score, step_macro_selection = compute_macro_dev_selection(
                                step_macro_breakdown,
                                macro_dev_selection_cfg,
                            )
                            metrics[f"update_{global_update}_macro_dev_selection_score"] = step_macro_selection_score
                    logger.info(
                        "Update %s/%s dev RMSE_VA=%.6f (epoch=%s step=%s)",
                        global_update, total_updates, step_dev_rmse, epoch + 1, step,
                    )
                    if step_macro_rmse is not None:
                        logger.info(
                            "Update %s/%s macro-dev RMSE_VA=%.6f selection=%.6f targets=%s",
                            global_update,
                            total_updates,
                            step_macro_rmse,
                            step_macro_selection_score if step_macro_selection_score is not None else step_macro_rmse,
                            step_macro_breakdown,
                        )
                    selection_score = (
                        step_macro_selection_score
                        if macro_dev_use_for_best and step_macro_selection_score is not None
                        else step_dev_rmse
                    )
                    if best_selection_score is None or selection_score < best_selection_score:
                        best_selection_score = selection_score
                        best_dev_rmse = step_dev_rmse
                        if step_macro_rmse is not None:
                            best_macro_dev_rmse = step_macro_rmse
                        best_epoch = epoch
                        best_update = global_update
                        _save_checkpoint(
                            model,
                            run_dir,
                            epoch,
                            {"dev_rmse_va": step_dev_rmse},
                            cfg,
                            global_update=global_update,
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
            total_anchor_loss += float(anchor_loss.detach().cpu())
            # FDS: update per-bin feature statistics
            if fds_enabled and features is not None:
                fds_module.update_stats(features.detach(), target_raw[:, 1])
            n_batches += 1

        train_loss_epoch = round(total_loss / max(n_batches, 1), 6)
        metrics[f"epoch_{epoch}_train_loss"] = train_loss_epoch
        if teacher_cfg.get("enabled", False):
            metrics[f"epoch_{epoch}_teacher_loss"] = round(total_teacher_loss / max(n_batches, 1), 6)
            metrics[f"epoch_{epoch}_teacher_w_eff"] = round(teacher_eff_weight, 6)
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
        if pivot_gate_enabled and n_pivot_gate_batches > 0:
            metrics[f"epoch_{epoch}_pivot_gate_multiplier"] = round(
                total_pivot_gate_multiplier / n_pivot_gate_batches, 6
            )
        if anchor_state is not None:
            metrics[f"epoch_{epoch}_anchor_loss"] = round(total_anchor_loss / max(n_batches, 1), 6)

        log_parts = [f"train_loss={train_loss_epoch:.6f}", f"teacher_w={teacher_eff_weight:.4f}"]
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
        logger.info(
            "Epoch %s/%s finished: %s",
            epoch + 1, n_epochs, " ".join(log_parts),
        )

        # FDS: smooth per-bin statistics at end of each epoch
        if fds_enabled:
            fds_module.smooth_stats()
            logger.info("FDS: smoothed statistics at end of epoch %d", epoch + 1)

        if dev_data:
            if last_eval_update == global_update and last_eval_rmse is not None:
                dev_rmse = last_eval_rmse
                logger.info(
                    "Epoch %s/%s dev RMSE_VA reused from update %s: %.6f",
                    epoch + 1, n_epochs, global_update, dev_rmse,
                )
            else:
                dev_rmse = _evaluate_dev_rmse(
                    model, dev_data, tokenizer, device, cfg, va_norm_stats=va_norm_stats
                )
            metrics[f"epoch_{epoch}_dev_rmse_va"] = dev_rmse
            logger.info("Epoch %s/%s dev RMSE_VA=%.6f", epoch + 1, n_epochs, dev_rmse)
            epoch_macro_rmse = None
            epoch_macro_breakdown = {}
            epoch_macro_selection_score = None
            epoch_macro_selection = {}
            if macro_dev_enabled:
                epoch_macro_rmse, epoch_macro_breakdown = _evaluate_macro_dev_rmse(
                    model, macro_dev_targets, tokenizer, device, va_norm_stats=va_norm_stats
                )
                if epoch_macro_rmse is not None:
                    metrics[f"epoch_{epoch}_macro_dev_rmse_va"] = epoch_macro_rmse
                    epoch_macro_selection_score, epoch_macro_selection = compute_macro_dev_selection(
                        epoch_macro_breakdown,
                        macro_dev_selection_cfg,
                    )
                    metrics[f"epoch_{epoch}_macro_dev_selection_score"] = epoch_macro_selection_score
                    logger.info(
                        "Epoch %s/%s macro-dev RMSE_VA=%.6f selection=%.6f targets=%s",
                        epoch + 1,
                        n_epochs,
                        epoch_macro_rmse,
                        epoch_macro_selection_score if epoch_macro_selection_score is not None else epoch_macro_rmse,
                        epoch_macro_breakdown,
                    )
            # Save per-epoch checkpoint if requested (for macro-dev selection)
            if cfg.get("save_all_checkpoints", False):
                _save_checkpoint(
                    model,
                    run_dir,
                    epoch,
                    {"dev_rmse_va": dev_rmse},
                    cfg,
                    filename=f"checkpoint_epoch{epoch}.pt",
                    global_update=global_update,
                )
            selection_score = (
                epoch_macro_selection_score
                if macro_dev_use_for_best and epoch_macro_selection_score is not None
                else dev_rmse
            )
            if best_selection_score is None or selection_score < best_selection_score:
                best_selection_score = selection_score
                best_dev_rmse = dev_rmse
                if epoch_macro_rmse is not None:
                    best_macro_dev_rmse = epoch_macro_rmse
                best_epoch = epoch
                best_update = global_update
                _save_checkpoint(
                    model,
                    run_dir,
                    epoch,
                    {"dev_rmse_va": dev_rmse},
                    cfg,
                    global_update=global_update,
                )
                logger.info(
                    "New best selection score %.6f at epoch %s (dev=%.6f macro=%s selection=%s)",
                    selection_score,
                    epoch,
                    dev_rmse,
                    f"{epoch_macro_rmse:.6f}" if epoch_macro_rmse is not None else "n/a",
                    epoch_macro_selection,
                )

    if (run_dir / "checkpoint.pt").exists():
        _, reload_report = _load_checkpoint(
            model,
            run_dir / "checkpoint.pt",
            cfg=cfg,
            load_backbone=True,
            load_head="always",
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
        metrics["best_selection_score"] = best_selection_score
        if macro_dev_selection_metric in {"macro_dev_rmse_va", "dev_rmse_va"}:
            metrics["best_selection_rmse_va"] = best_selection_score
        metrics["best_selection_metric"] = macro_dev_selection_metric

    if macro_dev_enabled:
        final_macro_rmse, final_macro_breakdown = _evaluate_macro_dev_rmse(
            model, macro_dev_targets, tokenizer, device, va_norm_stats=va_norm_stats
        )
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

    run_test = bool(cfg.get("run_test", True))
    eval_splits = [("dev", dev_data)]
    if run_test:
        eval_splits.append(("test", test_data))

    # Quantile median inference: use quantile head's median for A prediction
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
        preds = _predict(model, data, tokenizer, device, cfg, va_norm_stats=va_norm_stats,
                         quantile_head=quantile_inference_head)

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
