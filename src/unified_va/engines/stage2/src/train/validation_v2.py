"""Bounded validation protocol and cheap parameter-update observations.

No hashes, device access, test-data reads, or performance verdicts are made here.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import torch

STRATEGIES = {"partial_backbone", "frozen_backbone", "adapter_only"}


def validate_formal_release(cfg, report_id):
    root = Path(__file__).resolve().parents[2]
    freeze = json.loads((root / "protocol/FROZEN_V2.json").read_text())
    protocol = cfg.get("matched_protocol", {})
    validation = cfg.get("method_validation", {})
    if report_id != freeze["report_id"] or freeze.get("state") != "recipe_frozen_before_v2_test":
        raise ValueError("the frozen v2 report identifier is required")
    arm = protocol.get("arm")
    split_id = protocol.get("split_id", "original")
    reusable_s42 = arm == "S" and split_id == "original" and cfg.get("seed") == 42
    valid_s = (arm == "S" and (reusable_s42 or protocol.get("protocol_version") == "method_validation_v2"))
    if (cfg.get("seed") not in freeze["seeds"] or split_id not in freeze["splits"]
            or (not valid_s and (protocol.get("protocol_version") != "method_validation_v2"
                or validation.get("stage") != "formal"
                or validation.get("candidate_id") != freeze["selected_candidate"]))):
        raise ValueError("exploratory or incompatible configurations cannot enter the formal v2 report")
    split = freeze["splits"][split_id]
    if protocol.get("training_targets") != split["training"] or protocol.get("held_out_targets") != split["held_out"]:
        raise ValueError("formal partition differs from the prespecified split")
    if not valid_s:
        recipe = freeze["recipe"]
        if cfg.get("lr") != recipe["lr"] or validation.get("update_strategy") != recipe["update_strategy"]:
            raise ValueError("formal Stage2 update strategy or learning rate differs from the frozen recipe")
        variant = validation.get("variant")
        if variant != "full":
            raise ValueError("the complete-model recipe is required")
        adapter = cfg.get("backbone_conditional_adapter", {})
        if not adapter.get("use_composition"):
            raise ValueError("the complete model requires the composition branch")
        if arm == "T":
            supervision = cfg.get("soft_condition_label_supervision", {})
            if not supervision.get("enabled"):
                raise ValueError("the complete model requires auxiliary supervision")
            if supervision.get("loss_weight") != recipe["aux_weight"]:
                raise ValueError("auxiliary loss weight differs from the frozen recipe")
            from src.models.text_only_protocol import validate_text_only_cfg
            validate_text_only_cfg(cfg, required=True)
        validate_v2_config(cfg)
    if protocol.get("author_approval") != "approved" or protocol.get("execution_authorized") is not True:
        raise PermissionError("formal execution has not been released")


def validate_v2_config(cfg):
    from src.data.matched_data import protocol_targets
    train, held = protocol_targets(cfg)
    protocol = cfg["matched_protocol"]
    if protocol.get("arm") not in {"S", "M", "T"}:
        raise ValueError("v2 formal scope is S/M/T; C remains internal v1 history")
    if cfg.get("run_test") is not False:
        raise ValueError("v2 training is development-only")
    if int(cfg.get("freeze_until", -1)) != 24:
        raise ValueError("keep freeze_until=24: newly frozen trained S layers must still be saved")
    validation = cfg.get("method_validation", {})
    strategy = validation.get("update_strategy")
    if strategy not in STRATEGIES:
        raise ValueError("unknown v2 update strategy")
    if protocol["arm"] == "S":
        if strategy != "partial_backbone":
            raise ValueError("S recipe is unchanged, with the original partial backbone update")
    else:
        adapter = cfg.get("backbone_conditional_adapter", {})
        expected = (strategy != "partial_backbone", strategy == "adapter_only")
        actual = (bool(adapter.get("freeze_backbone")), bool(adapter.get("freeze_head")))
        if actual != expected or bool(adapter.get("freeze_pool_gate")) != expected[1]:
            raise ValueError("freeze flags do not implement the named update strategy")
        if adapter.get("keep_adapter_trainable") is not True:
            raise ValueError("all v2 adapter paths must remain trainable")
    if cfg.get("save_all_checkpoints") is not True or cfg.get("max_step_checkpoints_to_keep", 0):
        raise ValueError("candidate checkpoints must remain until the selection/retention decision")
    if len(train) != 12 or len(held) != 3:
        raise ValueError("v2 uses explicitly defined 12/3 partitions")


def validate_stage1_parent(payload, cfg):
    """Reject a wrong seed/partition/prompt before any common weights are loaded."""
    parent = payload.get("effective_inference_config", {})
    if parent.get("seed") != cfg.get("seed"):
        raise ValueError("v2 continuation requires its own seed's S checkpoint")
    for key in ("training_targets", "held_out_targets"):
        if payload.get("matched_protocol", {}).get(key) != cfg["matched_protocol"].get(key):
            raise ValueError(f"v2 Stage-1 parent differs in {key}")
    for key in ("max_len", "input_format", "head_type", "va_target_normalization", "freeze_until"):
        if parent.get(key) != cfg.get(key):
            raise ValueError(f"v2 Stage-1 parent differs in {key}")
    for key in ("model_revision", "tokenizer_revision", "prompt_template", "pooling"):
        if parent.get("small_llm", {}).get(key) != cfg.get("small_llm", {}).get(key):
            raise ValueError(f"v2 Stage-1 parent differs in small_llm.{key}")


def parameter_role(name):
    if name.startswith("soft_condition_label_head."):
        return "auxiliary_head"
    if ".adapter." in name:
        local = name.split(".adapter.", 1)[1]
        if local.startswith("soft_condition_proj."):
            return "condition_projection"
        for prefix, role in (("shared_", "shared"), ("lang_", "language"),
                             ("domain_", "domain"), ("comp_", "composition")):
            if local.startswith(prefix):
                return role
        return "adapter_other"
    if name.startswith("head."):
        return "readout"
    if name.startswith("pool_gate."):
        return "pooling"
    return "shared_backbone"


def assert_effective_freeze(model, extra_modules, cfg):
    strategy = cfg["method_validation"]["update_strategy"]
    counts = {}
    for name, param in model.named_parameters():
        role = parameter_role(name)
        bucket = counts.setdefault(role, {"trainable": 0, "frozen": 0, "tensors": 0})
        bucket["trainable" if param.requires_grad else "frozen"] += param.numel()
        bucket["tensors"] += 1
        if strategy != "partial_backbone" and role == "shared_backbone" and param.requires_grad:
            raise ValueError(f"shared backbone unexpectedly trainable: {name}")
        if role == "shared_backbone" and name.startswith("backbone."):
            layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            expected = strategy == "partial_backbone"
            if layer and int(layer[1]) < int(cfg["freeze_until"]):
                expected = False
            if "embed_tokens." in name:
                expected = False
            if bool(param.requires_grad) != expected:
                raise ValueError(f"partial backbone freeze mismatch: {name}")
        if role in {"readout", "pooling"} and param.requires_grad != (strategy != "adapter_only"):
            raise ValueError(f"readout/pooling freeze mismatch: {name}")
        if ".adapter." in name and not param.requires_grad:
            raise ValueError(f"adapter unexpectedly frozen: {name}")
    for key, module in (extra_modules or {}).items():
        if module is None:
            continue
        for name, param in module.named_parameters():
            role = parameter_role(key + "." + name)
            bucket = counts.setdefault(role, {"trainable": 0, "frozen": 0, "tensors": 0})
            bucket["trainable" if param.requires_grad else "frozen"] += param.numel()
            bucket["tensors"] += 1
    return {"update_strategy": strategy, "actual_parameter_groups": counts,
            "scope": "effective requires_grad state; observed updates are reported separately"}


class UpdateObserver:
    """Observe deterministic slices of real optimizer updates without extra forwards."""
    def __init__(self, model, extras, cfg, output):
        settings = cfg.get("method_validation", {}).get("update_observer", {})
        self.updates = set(settings.get("updates", [1, 2, 10, 100])) if settings.get("enabled") else set()
        self.output = Path(output)
        self.parameters = list(model.named_parameters())
        for key, module in (extras or {}).items():
            if module is not None:
                self.parameters += [(key + "." + name, p) for name, p in module.named_parameters()]
        self.before_values = None

    def before(self, update):
        if update not in self.updates:
            return
        self.before_values = {name: p.detach().reshape(-1)[:32].float().clone()
                              for name, p in self.parameters}
        self.gradients = {}
        for name, p in self.parameters:
            role = parameter_role(name)
            row = self.gradients.setdefault(role, {"with_gradient": 0, "without_gradient": 0,
                                                   "nonfinite_gradient_tensors": 0})
            row["with_gradient" if p.grad is not None else "without_gradient"] += 1
            if p.grad is not None and not torch.isfinite(p.grad.detach().reshape(-1)[:32]).all():
                row["nonfinite_gradient_tensors"] += 1

    def after(self, update):
        if update not in self.updates:
            return
        rows = {}
        with torch.no_grad():
            for name, p in self.parameters:
                role = parameter_role(name)
                row = rows.setdefault(role, {"sampled_tensors": 0, "tensors_with_sampled_changes": 0,
                                             "max_abs_sampled_update": 0.0})
                change = (p.detach().reshape(-1)[:32].float() - self.before_values[name]).abs().max().item()
                row["sampled_tensors"] += 1
                row["tensors_with_sampled_changes"] += int(change > 0)
                row["max_abs_sampled_update"] = max(row["max_abs_sampled_update"], change)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("a") as stream:
            stream.write(json.dumps({"update": update, "groups": rows, "gradients": self.gradients,
                                     "scope": "first 32 scalars per tensor; total-loss gradient; no claim of whole-tensor equality"}) + "\n")
        self.before_values = None
