"""Offline, development-only local checkpoint-selection sensitivity.

This module reads scalar development archives and checks linked file existence
and byte length. Explicit retention-aware mode also accepts recorded releases
of finalized, nonselected V2 weights, while still requiring all scalar records.
It never loads a checkpoint, imports a model, hashes a file,
reads test scores, or searches weights/thresholds. Results describe only the
complete saved candidate trajectory; they establish no independent DP benefit.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

from src.train.macro_selection import (
    compute_macro_dev_selection,
    normalize_macro_dev_selection_cfg,
)

TRAINING_TARGETS = (
    "eng_restaurant", "deu_politics", "eng_laptop", "zho_laptop",
    "ukr_restaurant", "zho_restaurant", "jpn_hotel", "jpn_finance",
    "rus_restaurant", "zho_finance", "zho_environmental_protection", "pcm_politics",
)
HELD_OUT_TARGETS = ("eng_environmental_protection", "tat_restaurant", "swa_politics")
REQUIRED_METRICS = (
    "rmse_va", "rmse_v", "rmse_a", "pcc_macro", "pcc_v", "pcc_a",
    "ccc_macro", "ccc_v", "ccc_a", "pred_std_v", "pred_std_a",
    "true_std_v", "true_std_a", "std_gap_v", "std_gap_a",
)
WEIGHT_GROUPS = {
    "tail": ("worst_k_weight",),
    "pseudo_ood": ("pseudo_ood_weight", "pseudo_ood_worst_k_weight"),
    "spread": ("std_gap_v_weight", "std_gap_a_weight"),
    "floor": ("ccc_macro_floor_weight", "pcc_macro_floor_weight"),
}
ARCHIVE_ROLE = "all scheduled development candidates; never sensitivity-pruned"
LIMITATION = (
    "Local development-only sensitivity on all scheduled saved candidates; "
    "no threshold search, test-based selection, model execution, independent "
    "DP performance claim, or historical-provenance validation."
)


def _integer(value, name: str, *, minimum=0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def fixed_sensitivity_rules(selection: dict, training_targets=TRAINING_TARGETS) -> dict[str, dict]:
    """Exactly ten pre-fixed rules, changing one weight group at a time."""
    if not isinstance(selection, dict):
        raise ValueError("original DP selection must be an explicit mapping")
    cfg = normalize_macro_dev_selection_cfg(selection)
    unknown = set(selection) - set(cfg) - {"metric", "lambda"}
    if unknown:
        raise ValueError(f"unknown DP selection keys: {sorted(unknown)}")
    for name, value in cfg.items():
        if isinstance(value, (int, float)):
            _finite(value, name)
    if cfg["mode"] != "macro_plus_worst_k":
        raise ValueError("original DP must explicitly include the fixed tail rule")
    if any(float(value) != 0 for name, value in cfg.items()
           if name.startswith("protected_") and name.endswith("weight")):
        raise ValueError("protected-target additions are outside the fixed four-group protocol")
    if not cfg["pseudo_ood_targets"] or not set(cfg["pseudo_ood_targets"]) <= set(training_targets):
        raise ValueError("pseudo-OOD targets must be a nonempty subset of the 12 training targets")
    if len(cfg["pseudo_ood_targets"]) != len(set(cfg["pseudo_ood_targets"])):
        raise ValueError("pseudo-OOD targets must be unique")
    for group, names in WEIGHT_GROUPS.items():
        if not any(cfg[name] > 0 for name in names):
            raise ValueError(f"original DP {group} group must have a positive fixed weight")
    for metric in ("ccc", "pcc"):
        if cfg[f"{metric}_macro_floor_weight"] > 0 and cfg[f"{metric}_macro_floor"] is None:
            raise ValueError(f"{metric} floor weight requires its pre-fixed threshold")
    rules = {"dp_original": cfg}
    for group, names in WEIGHT_GROUPS.items():
        for factor in (0.5, 1.5):
            variant = copy.deepcopy(cfg)
            for name in names:
                variant[name] *= factor
            rules[f"dp_{group}_x{factor:.1f}"] = variant
    # Build from an empty policy, not a deep merge retaining DP penalties.
    rules["rmse_only"] = normalize_macro_dev_selection_cfg({"mode": "macro_mean"})
    return rules


def expected_schedule(cfg: dict) -> set[tuple[str, int, int]]:
    protocol = cfg.get("matched_protocol", {})
    if protocol.get("enabled") is not True:
        raise ValueError("development archive requires the explicit new matched protocol")
    from src.data.matched_data import protocol_targets
    protocol_targets(cfg)
    if cfg.get("run_test") is not False:
        raise ValueError("development archive must identify a no-test training run")
    if protocol.get("require_complete_dev") is not True or protocol.get("save_evaluation_checkpoints") is not True:
        raise ValueError("development archives require complete metrics and every candidate checkpoint")
    per_epoch = _integer(protocol.get("updates_per_epoch"), "updates_per_epoch", minimum=1)
    epochs = _integer(cfg.get("n_epochs"), "n_epochs", minimum=1)
    total = _integer(protocol.get("expected_total_updates"), "expected_total_updates", minimum=1)
    if total != epochs * per_epoch:
        raise ValueError("planned total updates differ from epochs * updates_per_epoch")
    every = _integer(cfg.get("eval_every_updates", 0), "eval_every_updates")
    schedule = {("epoch", epoch, (epoch + 1) * per_epoch) for epoch in range(epochs)}
    if every:
        schedule.update(("step", (update - 1) // per_epoch, update)
                        for update in range(every, total + 1, every))
    return schedule


def load_development_trajectory(archive_dir: str | Path, *, allow_retained=False) -> list[dict]:
    """Validate all records in one complete trajectory, without reading weights."""
    archive_dir = Path(archive_dir).resolve()
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"development archive directory missing: {archive_dir}")
    paths = sorted(archive_dir.glob("eval_*.json"))
    if not paths:
        raise ValueError("no development candidate archives; no experiment has been run")
    from src.train.checkpoint_retention import read_retention_record, validate_retained_scalars
    retention = read_retention_record(archive_dir) if allow_retained else None
    records = []
    reference_cfg = reference_selection = None
    expected = None
    events = set()
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("archive_role") != ARCHIVE_ROLE:
            raise ValueError(f"{path.name}: not an unpruned development archive")
        if any("test" in str(key).lower() for key in record):
            raise ValueError(f"{path.name}: test fields are forbidden in DP inputs")
        if "split" in record and record["split"] != "dev":
            raise ValueError(f"{path.name}: only dev archives are accepted")
        cfg = record.get("inference_config")
        if not isinstance(cfg, dict):
            raise ValueError(f"{path.name}: full effective inference_config is required")
        if reference_cfg is None:
            reference_cfg = cfg
            expected = expected_schedule(cfg)
            reference_selection = record.get("selection_cfg")
            if not isinstance(reference_selection, dict) or normalize_macro_dev_selection_cfg(reference_selection) != normalize_macro_dev_selection_cfg({"mode": "macro_mean"}):
                raise ValueError("primary trajectory must use a fully reset RMSE-only rule")
        elif cfg != reference_cfg or record.get("selection_cfg") != reference_selection:
            raise ValueError("mixed configurations or selection policies do not form one fixed trajectory")
        event = record.get("event")
        epoch = _integer(record.get("epoch"), "epoch")
        update = _integer(record.get("global_update"), "global_update", minimum=1)
        key = (event, epoch, update)
        if key not in expected or key in events:
            raise ValueError(f"{path.name}: duplicate or unscheduled development candidate {key}")
        events.add(key)
        stem = f"eval_{event}_epoch{epoch:03d}_update{update:08d}"
        if path.name != stem + ".json" or record.get("checkpoint") != stem + ".pt":
            raise ValueError(f"{path.name}: archive/checkpoint names must identify the same scheduled event")
        checkpoint = archive_dir / record["checkpoint"]
        expected_bytes = _integer(record.get("checkpoint_bytes"), "checkpoint_bytes", minimum=1)
        retained_item = retention["candidates"].get(checkpoint.name) if retention else None
        if not checkpoint.is_file() and (retained_item is None or retained_item["retain"]):
            raise FileNotFoundError(f"linked candidate checkpoint missing: {checkpoint}")
        if checkpoint.is_file() and checkpoint.stat().st_size != expected_bytes:
            raise ValueError(f"{path.name}: linked checkpoint byte length differs from its archive")
        from src.data.matched_data import protocol_targets
        train_targets, _ = protocol_targets(cfg)
        if tuple(record.get("targets", [])) != train_targets:
            raise ValueError(f"{path.name}: targets must be exactly the ordered 12 training targets")
        metrics = record.get("per_target")
        if not isinstance(metrics, dict) or set(metrics) != set(train_targets):
            raise ValueError(f"{path.name}: missing/extra development targets")
        for target, values in metrics.items():
            if not isinstance(values, dict) or not set(REQUIRED_METRICS) <= set(values):
                raise ValueError(f"{path.name}/{target}: incomplete development metric fields")
            _integer(values.get("num_examples"), f"{target}/num_examples", minimum=1)
            for metric in REQUIRED_METRICS:
                value = _finite(values[metric], f"{target}/{metric}")
                if metric.startswith(("rmse", "pred_std", "true_std", "std_gap")) and value < 0:
                    raise ValueError(f"{target}/{metric}: nonnegative diagnostic required")
                if metric.startswith(("pcc", "ccc")) and not -1.000001 <= value <= 1.000001:
                    raise ValueError(f"{target}/{metric}: correlation outside valid range")
        record["archive_path"] = str(path)
        record["checkpoint_path"] = str(checkpoint)
        record["checkpoint_retained"] = checkpoint.is_file()
        records.append(record)
    if events != expected:
        raise ValueError(f"missing scheduled development candidates: {sorted(expected - events)}")
    if retention is not None:
        validate_retained_scalars(retention, records)
    return sorted(records, key=lambda row: (row["global_update"], row["event"], row["archive_path"]))


def average_ranks(values: list[float]) -> list[float]:
    """Ascending average ranks; checkpoint tie-breaking does not alter ranks."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman_rank_stability(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("rank vectors must have the same nonzero length")
    x, y = average_ranks(left), average_ranks(right)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx = sum((value - mx) ** 2 for value in x)
    vy = sum((value - my) ** 2 for value in y)
    if vx == 0 or vy == 0:
        return None  # Undefined, rather than claiming perfect stability for ties.
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def analyze_trajectory(records: list[dict], selection: dict) -> dict:
    if not records:
        raise ValueError("a complete development trajectory is required")
    from src.data.matched_data import protocol_targets
    train_targets, _ = protocol_targets(records[0].get("inference_config", {}))
    rules = fixed_sensitivity_rules(selection, training_targets=train_targets)
    scores = {}
    for rule, cfg in rules.items():
        scores[rule] = []
        for record in records:
            score, _ = compute_macro_dev_selection(record["per_target"], cfg)
            scores[rule].append(_finite(score, f"{rule}/score"))
    def selected_index(rule):
        return min(range(len(records)), key=lambda i: (
            scores[rule][i], records[i]["global_update"], records[i]["event"], records[i]["archive_path"],
        ))
    original_best = selected_index("dp_original")
    score_rows, summary_rows, target_rows = [], [], []
    for rule, cfg in rules.items():
        selected = selected_index(rule)
        record = records[selected]
        ranks = average_ranks(scores[rule])
        corr = spearman_rank_stability(scores["dp_original"], scores[rule])
        summary_rows.append({
            "rule": rule, "selected_archive": Path(record["archive_path"]).name,
            "checkpoint": record["checkpoint_path"], "epoch": record["epoch"],
            "global_update": record["global_update"], "selection_score": scores[rule][selected],
            "top1_changed_from_original": selected != original_best,
            "spearman_vs_original": corr,
            "rank_stability_status": "defined" if corr is not None else "undefined_constant_ranks",
            "n_candidates": len(records),
            **{f"dev_macro_{metric}": sum(row[metric] for row in record["per_target"].values()) / 12
               for metric in REQUIRED_METRICS},
        })
        for index, candidate in enumerate(records):
            score_rows.append({
                "rule": rule, "archive": Path(candidate["archive_path"]).name,
                "epoch": candidate["epoch"], "event": candidate["event"],
                "global_update": candidate["global_update"], "score": scores[rule][index],
                "average_rank": ranks[index], "selected": index == selected,
            })
        for target in train_targets:
            target_rows.append({"rule": rule, "selected_archive": Path(record["archive_path"]).name,
                                "target": target, **record["per_target"][target]})
    return {"limitation": LIMITATION, "tie_policy": "lowest score; earliest global_update; event; archive_path",
            "rank_policy": "Spearman correlation of ascending average ranks; exact score ties; constant ranks undefined",
            "rules": rules, "summary": summary_rows, "scores": score_rows, "selected_dev_metrics": target_rows}


def write_analysis(result: dict, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = {"summary": "selection_sensitivity.tsv", "scores": "candidate_dev_scores_ranks.tsv",
             "selected_dev_metrics": "selected_development_metrics.tsv"}
    for key, name in names.items():
        rows = result[key]
        fields = list(dict.fromkeys(field for row in rows for field in row))
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    (output / "fixed_rules.json").write_text(json.dumps(result["rules"], indent=2, allow_nan=False) + "\n")
    (output / "ANALYSIS_SCOPE.txt").write_text(
        LIMITATION + "\n" + result["tie_policy"] + "\n" + result["rank_policy"] + "\n"
        "No hashes or checkpoint-content/model reads were performed. Test results are not inputs or outputs.\n",
        encoding="utf-8",
    )
