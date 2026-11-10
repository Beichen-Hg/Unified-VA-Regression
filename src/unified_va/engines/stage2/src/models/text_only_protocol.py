"""Strict text/aspect inference contract for the soft-text backbone adapters.

Condition labels may supervise an auxiliary TRAINING loss. They never enter
this prediction interface. Existing metadata experiments remain opt-in legacy
paths; this contract does not relabel their checkpoints or results.
"""
from __future__ import annotations

import math
from string import Formatter

import torch

PROTOCOL = "text_only_soft_adapter_v1"
DEFAULT_PROMPT = (
    "Predict valence and arousal for the aspect from the text.\n\n"
    "Text: {text}\nAspect: {aspect}"
)


def validate_text_only_cfg(cfg: dict, *, required: bool = False) -> None:
    if not cfg.get("text_only_inference", False):
        if required:
            raise ValueError("text_only_inference=true is required; legacy results are not text-only results")
        return
    if cfg.get("disable_explicit_metadata") is not True:
        raise ValueError("text-only inference requires disable_explicit_metadata=true")
    adapter = cfg.get("backbone_conditional_adapter", {}) or {}
    if not adapter.get("enabled") or adapter.get("condition_source") != "soft_text":
        raise ValueError("text-only inference requires an enabled soft_text backbone adapter")
    if adapter.get("soft_condition_pooling") != "masked_mean":
        raise ValueError("text-only inference requires masked_mean soft conditioning")
    if cfg.get("head_type") != "dual_mlp" or cfg.get("uncertainty", False):
        raise ValueError("this text-only protocol supports the plain dual_mlp VA head")
    # Keep the protocol narrow and auditable. Additional inference branches need
    # their own tests before they can become part of this contract.
    forbidden = (
        "metadata_conditioning", "metadata_output_residual", "factorized_output_residual",
        "text_routing_detector", "group_routed_va_residual", "shared_group_routed_va_residual",
        "translation_pivot", "pivot_a_residual", "pivot_trust_residual", "pivot_hardness_gate",
        "va_residual_corrector", "arousal_quantile", "tta", "lora", "decoupled_adapter", "pre_pool_repair",
        "top_block_repair", "post_ckpt_repair_adapter",
    )
    for key in forbidden:
        if (cfg.get(key, {}) or {}).get("enabled", False):
            raise ValueError(f"{key} is outside the text-only soft-adapter protocol")
    if (cfg.get("teacher", {}) or {}).get("explanation_as_input", False):
        raise ValueError("teacher explanations cannot be inference inputs in this protocol")
    if cfg.get("va_calibration", False):
        raise ValueError("dataset-specific VA calibration is outside this protocol")
    if (cfg.get("save_full_backbone_checkpoint") is not True
            and cfg.get("checkpoint_storage", {}).get("format") != "pinned_base_overlay_v1"):
        raise ValueError("text-only checkpoints must include the complete backbone and routers")
    if cfg.get("input_format", "instruction_va_v1") != "instruction_va_v1":
        raise ValueError("text-only inference requires instruction_va_v1")
    template = (cfg.get("small_llm", {}) or {}).get("prompt_template") or DEFAULT_PROMPT
    fields = set()
    for _, field, spec, conversion in Formatter().parse(template):
        if field is None:
            continue
        if field not in {"text", "aspect"} or spec or conversion:
            raise ValueError("prompt templates may use only plain {text} and {aspect} fields")
        fields.add(field)
    if fields != {"text", "aspect"}:
        raise ValueError("the prompt must include both {text} and {aspect}")
    supervision = cfg.get("soft_condition_label_supervision", {}) or {}
    if not supervision.get("enabled") or float(supervision.get("loss_weight", 0)) <= 0:
        raise ValueError("the supervised soft-text protocol requires a positive auxiliary label loss")
    if min(float(supervision.get(k, 1)) for k in ("lang_weight", "domain_weight")) <= 0:
        raise ValueError("both language and domain supervision must be enabled")


def validate_text_only_model(model) -> None:
    from src.models.heads import VADualHeadMLP

    adapters = getattr(model, "backbone_conditional_adapters", ())
    if not adapters or any(a.condition_source != "soft_text" for a in adapters):
        raise ValueError("model contains missing or metadata-conditioned backbone adapters")
    if any(a.soft_condition_pooling != "masked_mean" for a in adapters):
        raise ValueError("all text routers must respect the token attention mask")
    if type(model.head) is not VADualHeadMLP:
        raise ValueError("model does not have the plain dual MLP head required by this protocol")
    if any(getattr(model, name, None) is not None for name in (
        "text_routing_detector", "pre_pool_repair", "top_block_repair",
    )) or getattr(model, "post_ckpt_repair_adapters", ()):
        raise ValueError("unexpected extra inference module")


def validate_normalization(stats: dict | None, enabled: bool) -> None:
    if not enabled:
        return
    if not isinstance(stats, dict):
        raise ValueError("training VA normalization statistics are missing from the checkpoint")
    for key in ("v_mean", "v_std", "a_mean", "a_std"):
        if key not in stats or not math.isfinite(float(stats[key])):
            raise ValueError(f"invalid VA normalization statistic: {key}")
    if min(float(stats["v_std"]), float(stats["a_std"])) <= 0:
        raise ValueError("VA normalization standard deviations must be positive")


def load_text_only_weights(model, checkpoint: dict, cfg: dict) -> dict | None:
    """Deployment is strict; Stage-1 warm-start loading stays in the trainer."""
    validate_text_only_cfg(cfg, required=True)
    validate_text_only_model(model)
    if checkpoint.get("inference_protocol") != PROTOCOL:
        raise ValueError("checkpoint is not a recorded text-only run; do not reuse legacy main-table weights")
    stats = checkpoint.get("va_norm_stats")
    validate_normalization(stats, bool(cfg.get("va_target_normalization", False)))
    if checkpoint.get("checkpoint_type") == "pinned_base_overlay":
        from src.train.matched_runtime import restore_matched_checkpoint
        restore_matched_checkpoint(model, checkpoint, cfg, from_stage1=False)
        return stats
    # strict=True rejects absent/randomly initialized routing weights, even when
    # a checkpoint marker was copied incorrectly.
    model.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    model.head.load_state_dict(checkpoint["head"], strict=True)
    if model.pool_gate is not None:
        model.pool_gate.load_state_dict(checkpoint["pool_gate"], strict=True)
    return stats


def export_soft_condition_classifier_config(head) -> dict:
    """Save construction state for the separately trained auxiliary classifier."""
    from src.models.heads import SoftConditionGroupClassifier

    if type(head) is not SoftConditionGroupClassifier:
        raise TypeError("expected a SoftConditionGroupClassifier")
    return {
        "embed_dim": int(head.lang_head[0].normalized_shape[0]),
        "hidden_dim": int(head.lang_head[1].out_features),
        "dropout": float(head.lang_head[3].p),
        "langs": [v for v, i in sorted(head.lang_to_idx.items(), key=lambda item: item[1]) if i != 0],
        "domains": [v for v, i in sorted(head.domain_to_idx.items(), key=lambda item: item[1]) if i != 0],
        "lang_to_idx": dict(head.lang_to_idx),
        "domain_to_idx": dict(head.domain_to_idx),
        "lang_alias_map": dict(head.lang_alias_map),
        "domain_alias_map": dict(head.domain_alias_map),
        "metric_role": "condition_representation_label_predictability",
    }


def restore_soft_condition_classifier(checkpoint: dict, *, device="cpu"):
    """Strictly restore a saved diagnostic head; never report random-head metrics.

    This classifier reads continuous condition embeddings. Its accuracy is
    label predictability, not an actual discrete routing accuracy. Unknown
    labels retain index 0 and require a separately reported unseen-label scope.
    """
    from src.models.heads import SoftConditionGroupClassifier

    construction = checkpoint.get("soft_condition_label_head_config")
    state = checkpoint.get("soft_condition_label_head")
    if not isinstance(construction, dict) or not isinstance(state, dict):
        raise ValueError("saved auxiliary classifier weights and construction/vocabulary state are required")
    required = ("embed_dim", "hidden_dim", "dropout", "langs", "domains", "lang_to_idx", "domain_to_idx")
    if any(key not in construction for key in required):
        raise ValueError("incomplete saved auxiliary classifier construction/vocabulary state")
    head = SoftConditionGroupClassifier(
        embed_dim=int(construction["embed_dim"]),
        langs=list(construction["langs"]),
        domains=list(construction["domains"]),
        hidden_dim=int(construction["hidden_dim"]),
        dropout=float(construction["dropout"]),
        lang_alias_map=construction.get("lang_alias_map", {}),
        domain_alias_map=construction.get("domain_alias_map", {}),
    )
    if head.lang_to_idx != construction["lang_to_idx"] or head.domain_to_idx != construction["domain_to_idx"]:
        raise ValueError("saved auxiliary classifier vocabulary does not reproduce the trained index mapping")
    head.load_state_dict(state, strict=True)
    head.to(device)
    head.eval()
    head.diagnostic_metric_role = "condition_representation_label_predictability"
    return head


@torch.inference_mode()
def predict_text_aspect(model, tokenizer, records: list[dict], cfg: dict,
                        va_norm_stats: dict | None = None, batch_size: int = 1,
                        *, return_continuous: bool = False) -> list[dict]:
    """Predict from text and aspect only; any extra record fields are ignored.

    Gold VA, language, domain, record ID, dataset name and routing labels never
    reach tokenization or model.forward. Normalization is TRAINING-set state.
    With return_continuous=True, diagnostics include model-space outputs and
    denormalized, pre-clamp/pre-round VA values alongside unchanged formatted
    official outputs. Invariance tests must compare the continuous values.
    """
    validate_text_only_cfg(cfg, required=True)
    validate_text_only_model(model)
    normalized = bool(cfg.get("va_target_normalization", False))
    validate_normalization(va_norm_stats, normalized)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError("this regressor's last-token readout requires right padding")
    template = (cfg.get("small_llm", {}) or {}).get("prompt_template") or DEFAULT_PROMPT
    prompts = []
    for record in records:
        if not isinstance(record.get("text"), str) or not isinstance(record.get("aspect"), str):
            raise ValueError("each record must contain string text and aspect fields")
        prompts.append(template.format(text=record["text"], aspect=record["aspect"]))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    predictions = []
    try:
        for offset in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[offset:offset + batch_size], return_tensors="pt",
                                padding=True, truncation=True, max_length=int(cfg.get("max_len", 192)))
            raw = model(input_ids=encoded["input_ids"].to(device),
                        attention_mask=encoded["attention_mask"].to(device)).float()
            model_values = raw.cpu().tolist() if return_continuous else None
            if normalized:
                raw = raw * raw.new_tensor([va_norm_stats["v_std"], va_norm_stats["a_std"]])
                raw = raw + raw.new_tensor([va_norm_stats["v_mean"], va_norm_stats["a_mean"]])
            values = (raw.clamp(1.0, 9.0) * 100).round().div(100).cpu().tolist()
            continuous = raw.cpu().tolist() if return_continuous else None
            for index, (v, a) in enumerate(values):
                item = {"v_pred": v, "a_pred": a}
                if return_continuous:
                    item.update(
                        v_continuous=continuous[index][0], a_continuous=continuous[index][1],
                        v_model_raw=model_values[index][0], a_model_raw=model_values[index][1],
                    )
                predictions.append(item)
    finally:
        model.train(was_training)
    return predictions
