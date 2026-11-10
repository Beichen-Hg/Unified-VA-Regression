"""Lossless overlays over one pinned base; no tensor deltas or hashes.

Only frozen base embeddings/decoder layers may be omitted. Every other state
tensor, including all buffers and newly introduced adapter parameters, is saved
in its current dtype. The base snapshot remains an essential dependency.
"""
from __future__ import annotations

import re

FORMAT = "pinned_base_overlay_v1"


def uses_base_overlay(cfg: dict) -> bool:
    return cfg.get("checkpoint_storage", {}).get("format") == FORMAT


def base_reference(cfg: dict) -> dict:
    llm = cfg.get("small_llm", {})
    revision = llm.get("model_revision", llm.get("revision"))
    repository = llm.get("model_repository")
    if not isinstance(revision, str) or not revision or not isinstance(repository, str) or not repository:
        raise ValueError("base overlay requires an explicit pinned model repository/revision")
    if llm.get("local_files_only") is not True or not llm.get("model_name"):
        raise ValueError("base overlay requires the retained local pinned base snapshot")
    return {"repository": repository, "revision": revision,
            "tokenizer_revision": llm.get("tokenizer_revision", revision),
            "torch_dtype": llm.get("torch_dtype"), "freeze_until": int(cfg.get("freeze_until", 0))}


def _eligible_base_key(key: str, freeze_until: int) -> bool:
    # This deliberately narrow format supports the Qwen3 AutoModel namespace.
    # Unknown/frozen custom parameters are saved rather than assumed rebuildable.
    if key in {"embed_tokens.weight", "model.embed_tokens.weight"}:
        return freeze_until > 0
    match = re.fullmatch(r"(?:model\.)?layers\.(\d+)\.(.+)", key)
    return bool(match and int(match[1]) < freeze_until
                and not any(part in match[2].split(".") for part in ("adapter", "repair_adapter", "layer")))


def _omitted_parameters(model, cfg: dict) -> dict:
    freeze_until = base_reference(cfg)["freeze_until"]
    return {key: value for key, value in model.backbone.named_parameters(remove_duplicate=False)
            if not value.requires_grad and _eligible_base_key(key, freeze_until)}


def _nonpersistent_buffers(backbone) -> dict:
    persistent_keys = set(backbone.state_dict())
    return {key: value for key, value in backbone.named_buffers(remove_duplicate=False)
            if key not in persistent_keys}


def _overlay_buffer_plan(model, payload: dict) -> dict:
    incoming = payload.get("checkpoint_storage", {}).get("nonpersistent_buffers")
    target = _nonpersistent_buffers(model.backbone)
    if not isinstance(incoming, dict) or set(incoming) != set(target):
        raise ValueError("base overlay nonpersistent buffer set differs from the target model")
    for key, value in incoming.items():
        if tuple(value.shape) != tuple(target[key].shape) or value.dtype != target[key].dtype:
            raise ValueError(f"base overlay nonpersistent buffer shape/dtype differs: {key}")
    return incoming


def restore_overlay_buffers(model, payload: dict) -> None:
    """Called only after every state component was validated by the restore plan."""
    incoming = _overlay_buffer_plan(model, payload)
    target = _nonpersistent_buffers(model.backbone)
    for key, value in incoming.items():
        target[key].copy_(value)


def initialize_checkpoint_storage(model, cfg: dict) -> dict:
    """Arm a cheap mutation guard after strict initialization and partial freeze.

    Tensor version counters detect normal in-place changes to omitted parameters;
    they are not hashes or independent provenance verification. Direct .data
    writes remain prohibited by the fixed training implementation/recipe.
    """
    if not uses_base_overlay(cfg):
        return {"format": "full_backbone"}
    if not cfg.get("matched_protocol", {}).get("enabled"):
        raise ValueError("base overlays are restricted to the fixed matched protocol")
    omitted = _omitted_parameters(model, cfg)
    if not omitted:
        raise ValueError("base overlay has no eligible frozen base parameters to omit")
    model._matched_overlay_guard = {key: (id(value), value._version) for key, value in omitted.items()}
    state = model.backbone.state_dict()
    nonpersistent = _nonpersistent_buffers(model.backbone)
    saved_bytes = sum(value.numel() * value.element_size() for key, value in state.items() if key not in omitted)
    saved_bytes += sum(value.numel() * value.element_size() for value in nonpersistent.values())
    omitted_bytes = sum(value.numel() * value.element_size() for value in omitted.values())
    return {"format": FORMAT, "base_reference": base_reference(cfg),
            "saved_backbone_tensor_bytes": saved_bytes, "omitted_frozen_tensor_bytes": omitted_bytes,
            "saved_backbone_keys": len(state) - len(omitted), "omitted_frozen_keys": len(omitted),
            "nonpersistent_buffer_keys": len(nonpersistent),
            "scope": "backbone tensors only; heads, auxiliary modules and serialization overhead additional"}


def export_base_overlay(model, cfg: dict) -> tuple[dict, dict]:
    omitted = _omitted_parameters(model, cfg)
    guard = getattr(model, "_matched_overlay_guard", None)
    current = {key: (id(value), value._version) for key, value in omitted.items()}
    if not guard or guard != current:
        raise ValueError("omitted frozen base parameters changed or storage guard was not initialized")
    state = model.backbone.state_dict()
    saved = {key: value for key, value in state.items() if key not in omitted}
    schema = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in state.items()}
    nonpersistent = _nonpersistent_buffers(model.backbone)
    metadata = {"format": FORMAT, "base_reference": base_reference(cfg),
                "base_snapshot_path_at_save": cfg["small_llm"]["model_name"],
                "backbone_schema": schema, "omitted_frozen_base_keys": sorted(omitted),
                "nonpersistent_buffers": nonpersistent,
                "saved_tensor_bytes": sum(value.numel() * value.element_size() for value in saved.values())
                                      + sum(value.numel() * value.element_size() for value in nonpersistent.values()),
                "tensor_encoding": "absolute current values in original dtype; no subtraction, quantization or compression",
                "base_provenance_scope": "pinned loader/configuration and frozen-state contract; no independent content hash"}
    return saved, metadata


def reconstruct_overlay_state(model, payload: dict, cfg: dict) -> dict:
    """Validate the complete saved/omitted partition before loading any tensor.

    The caller has built the model from the specified local base. During a
    same-run best-checkpoint reload, omitted tensors are protected by the frozen
    parameter guard. S->M/T only adds wrappers in the unfrozen decoder layers.
    """
    metadata = payload.get("checkpoint_storage", {})
    if metadata.get("format") != FORMAT or metadata.get("base_reference") != base_reference(cfg):
        raise ValueError("base overlay reference differs from the pinned effective configuration")
    _overlay_buffer_plan(model, payload)
    saved, schema = payload.get("backbone"), metadata.get("backbone_schema")
    omitted = metadata.get("omitted_frozen_base_keys")
    if not isinstance(saved, dict) or not isinstance(schema, dict) or not isinstance(omitted, list):
        raise ValueError("incomplete base overlay partition/schema")
    if len(omitted) != len(set(omitted)) or set(saved) & set(omitted) or set(saved) | set(omitted) != set(schema):
        raise ValueError("base overlay saved/omitted keys do not partition the complete backbone schema")
    target = model.backbone.state_dict()
    freeze_until = base_reference(cfg)["freeze_until"]
    complete = dict(saved)
    for key in omitted:
        if not _eligible_base_key(key, freeze_until) or key not in target:
            raise ValueError(f"base overlay attempts to omit an unsupported base parameter: {key}")
        complete[key] = target[key]
    for key, value in complete.items():
        if schema[key] != {"shape": list(value.shape), "dtype": str(value.dtype)}:
            raise ValueError(f"base overlay shape/dtype schema mismatch: {key}")
    guard = getattr(model, "_matched_overlay_guard", None)
    if guard is not None:
        params = dict(model.backbone.named_parameters(remove_duplicate=False))
        if any(key not in params or guard.get(key) != (id(params[key]), params[key]._version) for key in omitted):
            raise ValueError("target omitted frozen base parameters changed before restoration")
    provenance = getattr(model, "loading_provenance", {})
    requested = provenance.get("requested_model_revision")
    resolved = provenance.get("resolved_model_commit")
    expected = base_reference(cfg)["revision"]
    if requested is not None and requested != expected or resolved is not None and resolved != expected:
        raise ValueError("target loader provenance differs from the pinned overlay base")
    return complete
