"""One explicit AdamW/Huber loop with complete development checkpoint selection."""

from pathlib import Path
import gc
import math
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .. import storage
from ..config import validate_effective
from ..data import audit, loader, normalize
from ..evaluation.metrics import metrics
from ..inference import predict_rows, autocast
from ..io import read, write, now
from ..models import regressor
from .dp_selection import compute_macro_dev_selection


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_audit(model):
    groups = {}
    for name, parameter in model.named_parameters():
        group = (
            "adapter"
            if ".adapter." in name
            else (
                "upper"
                if name.startswith("backbone.layers.")
                else "auxiliary" if name.startswith("auxiliary.") else "readout"
            )
        )
        record = groups.setdefault(
            group,
            dict(
                requires_grad_tensors=0, gradient_tensors=0, nonzero_gradient_tensors=0
            ),
        )
        record["requires_grad_tensors"] += int(parameter.requires_grad)
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Nonfinite gradient")
            record["gradient_tensors"] += 1
            record["nonzero_gradient_tensors"] += int(
                torch.count_nonzero(parameter.grad).item() > 0
            )
    return groups


def development_evaluation(model, tokenizer, development, cfg, norm, device, update):
    per_target = {}
    root = Path(cfg["experiment"]["run_dir"])
    for name, rows in development.items():
        predictions = predict_rows(model, tokenizer, rows, cfg, norm, device)
        per_target[name] = metrics(predictions)
        write(
            root / "development" / f"update{update:06d}" / f"{name}.json", predictions
        )
    score, detail = compute_macro_dev_selection(
        per_target, cfg["macro_dev_eval"].get("selection")
    )
    event = dict(
        update=update,
        score=score,
        macro_dev_rmse_va=detail["macro_mean"],
        per_target=per_target,
        selection_details=detail,
    )
    write(root / "development" / f"update{update:06d}" / "METRICS.json", event)
    return event


def fit(model, tokenizer, training, development, cfg, norm):
    root = Path(cfg["experiment"]["run_dir"])
    device, stage = torch.device(cfg["device"]), cfg["experiment"]["stage"]
    batches = loader(training, tokenizer, cfg, training=True)
    expected_updates = cfg["n_epochs"] * math.ceil(len(batches) / cfg["grad_accum"])
    if expected_updates != cfg["experiment"]["expected_updates"]:
        raise ValueError(
            "Training data does not implement the registered update budget"
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    warmup = int(expected_updates * cfg["warmup_ratio"])

    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, expected_updates - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    best, events, update, started = None, [], 0, time.monotonic()
    checkpoint = root / "checkpoints" / (stage + ".pt")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg["n_epochs"]):
        model.train()
        for index, batch in enumerate(batches, 1):
            with autocast(cfg, device):
                prediction = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                target = normalize(batch["va"].to(device), norm)
                loss = F.smooth_l1_loss(prediction.float(), target)
                if stage == "stage2":
                    loss = loss + model.auxiliary.loss(
                        model.condition_embeddings(),
                        batch["lang"],
                        batch["domain"],
                        cfg["soft_condition_label_supervision"],
                    )
            (loss / cfg["grad_accum"]).backward()
            if index % cfg["grad_accum"] and index != len(batches):
                continue
            if update == 0:
                groups = gradient_audit(model)
                write(root / "FIRST_OPTIMIZER_GRADIENTS.json", groups)
                if groups.get("upper", {}).get("nonzero_gradient_tensors", 0) == 0:
                    raise RuntimeError(
                        "Upper decoder blocks must receive effective gradients in every stage"
                    )
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update == 1 or update % 10 == 0 or update == expected_updates:
                write(
                    root / "PROGRESS.json",
                    dict(
                        checked_at=now(),
                        completed_optimizer_updates=update,
                        registered_updates=expected_updates,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
            if update in cfg["experiment"]["development_events"]:
                event = development_evaluation(
                    model, tokenizer, development, cfg, norm, device, update
                )
                events.append(event)
                write(root / "DEVELOPMENT_EVENTS.json", events)
                if best is None or (event["score"], update) < (
                    best["score"],
                    best["update"],
                ):
                    storage.save(
                        checkpoint, storage.payload(model, cfg, norm, update, epoch)
                    )
                    best = event
                model.train()
    if (
        update != expected_updates
        or [e["update"] for e in events] != cfg["experiment"]["development_events"]
    ):
        raise RuntimeError("Incomplete training or development schedule")
    storage.save(
        root / "checkpoints" / "final.pt",
        storage.payload(model, cfg, norm, update, cfg["n_epochs"] - 1),
    )
    storage.restore(model, storage.load(checkpoint), cfg)
    differences = {}
    for name, rows in development.items():
        actual = metrics(predict_rows(model, tokenizer, rows, cfg, norm, device))
        differences[name] = abs(actual["rmse_va"] - best["per_target"][name]["rmse_va"])
    if max(differences.values()) > 1e-6:
        raise RuntimeError("Selected model failed complete development restoration")
    write(
        root / "FINAL_DEV_RESTORE.json",
        dict(passed=True, per_target_abs_difference=differences),
    )
    result = dict(
        passed=True,
        stage=stage,
        split=cfg["experiment"]["split"],
        seed=cfg["seed"],
        updates=update,
        development_checks=len(events),
        selected_update=best["update"],
        selected_score=best["score"],
        checkpoint=str(checkpoint.relative_to(root)),
        selection="DP" if stage == "stage2" else "12-target macro development RMSE",
        test_used_for_selection=False,
        elapsed_seconds=time.monotonic() - started,
    )
    write(root / "RESULT.json", result)
    return result


def run(config_path):
    cfg = read(config_path)
    validate_effective(cfg)
    root, device = Path(cfg["experiment"]["run_dir"]), torch.device(cfg["device"])
    if (root / "STARTED.json").exists():
        raise FileExistsError("Run already started; choose a fresh directory")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    report, training, development = audit(cfg)
    norm = report["normalization"]
    write(
        root / "STARTED.json", dict(started_at=now(), stage=cfg["experiment"]["stage"])
    )
    parent = None
    if cfg["experiment"]["stage"] == "stage2":
        from ..preflight import check_parent

        set_seed(cfg["seed"])
        scratch, tokenizer = regressor.build(cfg)
        scratch.to(device)
        parent = check_parent(scratch, tokenizer, cfg, norm, development)
        del scratch, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    set_seed(cfg["seed"])
    model, tokenizer = regressor.build(cfg)
    model.to(device)
    if parent is not None:
        storage.warm_start(model, parent, cfg, norm)
        del parent
    return fit(model, tokenizer, training, development, cfg, norm)
