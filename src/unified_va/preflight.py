"""Restore the complete selected parent, verify identity, discard scratch gradients."""

from pathlib import Path
import copy
import gc
import torch
from . import storage
from .io import read, write, now
from .models import regressor
from .inference import predict_rows, autocast
from .data import loader, normalize
from .evaluation.metrics import metrics


def check_parent(model, tokenizer, cfg, norm, development):
    root = Path(cfg["experiment"]["run_dir"])
    parent = Path(cfg["experiment"]["parent_run"])
    result = read(parent / "RESULT.json")
    if result.get("passed") is not True or result["stage"] != "stage1":
        raise ValueError("The selected Stage1 run is incomplete")
    value = storage.load(parent / result["checkpoint"])
    parent_cfg = copy.deepcopy(value["config"])
    parent_cfg["small_llm"]["model_name"] = cfg["small_llm"]["model_name"]
    device = torch.device(cfg["device"])
    source, source_tokenizer = regressor.build(parent_cfg)
    source.to(device)
    storage.restore(source, value, parent_cfg)
    expected = read(
        parent
        / "development"
        / f'update{result["selected_update"]:06d}'
        / "METRICS.json"
    )["per_target"]
    differences = {}
    for target, rows in development.items():
        measured = metrics(
            predict_rows(source, source_tokenizer, rows, parent_cfg, norm, device)
        )
        differences[target] = abs(measured["rmse_va"] - expected[target]["rmse_va"])
    if len(differences) != 12 or max(differences.values()) > 1e-6:
        raise RuntimeError("Selected Stage1 failed complete development restoration")
    probes = [row for rows in development.values() for row in rows[:2]]

    def raw(current, current_tokenizer, rows):
        current.eval()
        chunks = []
        with torch.no_grad():
            for batch in loader(rows, current_tokenizer, cfg):
                with autocast(cfg, device):
                    chunks.append(
                        current(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                        )
                        .float()
                        .cpu()
                    )
        return torch.cat(chunks)

    reference = raw(source, source_tokenizer, probes)
    del source, source_tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    storage.warm_start(model, value, cfg, norm)
    initial = raw(model, tokenizer, probes)
    perturbed = [
        dict(row, lang="foreign", domain="foreign", v=-999, a=999) for row in probes
    ]
    identity = float((initial - reference).abs().max())
    metadata = float((initial - raw(model, tokenizer, perturbed)).abs().max())
    if identity > 1e-6 or metadata != 0:
        raise RuntimeError("Zero-adapter identity or text/aspect boundary failed")
    model.train()
    batch = next(iter(loader(probes[:1], tokenizer, cfg, training=True)))
    with autocast(cfg, device):
        prediction = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        loss = torch.nn.functional.smooth_l1_loss(
            prediction.float(), normalize(batch["va"].to(device), norm)
        )
        loss = loss + model.auxiliary.loss(
            model.condition_embeddings(),
            batch["lang"],
            batch["domain"],
            cfg["soft_condition_label_supervision"],
        )
    loss.backward()
    from .training.trainer import gradient_audit

    gradients = gradient_audit(model)
    if gradients.get("upper", {}).get("nonzero_gradient_tensors", 0) == 0:
        raise RuntimeError("Stage2 upper blocks have no effective gradient")
    if gradients.get("adapter", {}).get("nonzero_gradient_tensors", 0) == 0:
        raise RuntimeError("Stage2 adapters have no effective gradient")
    model.zero_grad(set_to_none=True)
    write(
        root / "PREFLIGHT.json",
        dict(
            passed=True,
            checked_at=now(),
            optimizer_updates=0,
            full_development_per_target_abs_difference=differences,
            probes=len(probes),
            zero_adapter_raw_max_abs_difference=identity,
            metadata_raw_max_abs_difference=metadata,
            gradient_groups=gradients,
            scratch_discarded=True,
        ),
    )
    return value
