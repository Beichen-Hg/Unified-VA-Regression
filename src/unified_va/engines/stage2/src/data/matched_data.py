"""Fixed-split data preparation for new, unstarted S/C/M/T experiments.

No model loading, hashing, test-data reading, or historical-result processing.
Vocabulary and VA normalization come exclusively from the actual TRAIN records.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
from string import Formatter

import numpy as np

from src.data.dimabsa2026 import expand_to_aspect_level, load_samples

TRAINING_TARGETS = (
    "eng_restaurant", "deu_politics", "eng_laptop", "zho_laptop",
    "ukr_restaurant", "zho_restaurant", "jpn_hotel", "jpn_finance",
    "rus_restaurant", "zho_finance", "zho_environmental_protection", "pcm_politics",
)
HELD_OUT_TARGETS = ("eng_environmental_protection", "tat_restaurant", "swa_politics")
DEFAULT_DATA_ROOT = Path("data/DimABSA2026/task-dataset")
MATCHED_PROMPT = (
    "Predict valence and arousal for the aspect from the text.\n\n"
    "Text: {text}\nAspect: {aspect}"
)
UNKNOWN_CATEGORY_POLICY = "unknown_index_0; not a trained known class; nonzero residuals remain possible"
ALL_TARGETS = TRAINING_TARGETS + HELD_OUT_TARGETS


def protocol_targets(cfg: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve v2 membership explicitly while retaining the v1 reference default."""
    protocol = cfg.get("matched_protocol", {}) or {}
    train = tuple(protocol.get("training_targets", TRAINING_TARGETS))
    held = tuple(protocol.get("held_out_targets", HELD_OUT_TARGETS))
    if protocol.get("protocol_version") in ("stage2_confirmation_v9_20261101", "stage2_K_confirmation_v14_20261103", "original_source_stage2_textfix_20261104", "recovered_original_stage1_source_20261104"):
        split = protocol.get("split_id")
        if split == "original":
            expected_held = HELD_OUT_TARGETS
        elif split == "alt_language_combination_1":
            expected_held = ("rus_restaurant", "ukr_restaurant", "zho_finance")
        else:
            raise ValueError("unregistered frozen-confirmation partition")
        expected_train = tuple(t for t in ALL_TARGETS if t not in expected_held)
        if train != expected_train or held != expected_held:
            raise ValueError("confirmation partition/order differs from its registration")
    elif protocol.get("protocol_version") != "method_validation_v2":
        if train != TRAINING_TARGETS or held != HELD_OUT_TARGETS:
            raise ValueError("v1 requires the original ordered 12/3 partition")
    if (len(train) != 12 or len(set(train)) != 12 or len(held) != 3
            or len(set(held)) != 3 or set(train) & set(held)
            or set(train) | set(held) != set(ALL_TARGETS)):
        raise ValueError("partition must cover exactly the 15 known targets with disjoint 12/3 membership")
    return train, held


def target_config(target: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> dict:
    """Describe one target without opening any split file."""
    if target not in TRAINING_TARGETS + HELD_OUT_TARGETS:
        raise ValueError(f"target outside the fixed protocol: {target}")
    lang, domain = target.split("_", 1)
    track = "B" if domain in {"politics", "environmental_protection"} else "A"
    multi_task_train = domain in {"restaurant", "laptop", "hotel"}
    return {
        "task": "small_llm_va", "track": track, "lang": lang, "domain": domain,
        "data_dir": str(Path(data_root).resolve() / f"track_{track.lower()}" / "subtask_1" / lang),
        "train_file": f"{target}_train_{'alltasks' if multi_task_train else 'task1'}.jsonl",
        "dev_file": f"{target}_dev_task1.jsonl", "test_file": f"{target}_test_task1.jsonl",
        "input_format": "instruction_va_v1", "max_len": 192, "filter_unclear": False,
    }


def validate_matched_data_config(cfg: dict) -> None:
    """Validate split membership and input roles before any data are opened."""
    protocol = cfg.get("matched_protocol", {}) or {}
    if not protocol.get("enabled"):
        return
    train_targets, held_targets = protocol_targets(cfg)
    if f"{cfg.get('lang')}_{cfg.get('domain')}" not in train_targets:
        raise ValueError("primary development target is held out or unknown")
    if os.environ.get("DIMABSA2026_DATA_DIR"):
        raise ValueError("unset DIMABSA2026_DATA_DIR: matched runs use explicit absolute target paths")
    if cfg.get("run_test") is not False:
        raise ValueError("matched training does not run test; use the frozen evaluation entry later")
    sources = (cfg.get("auxiliary_data", {}) or {}).get("sources", [])
    source_names = tuple(f"{src.get('lang')}_{src.get('domain')}" for src in sources)
    if source_names != train_targets:
        raise ValueError("auxiliary sources must be precisely the ordered training targets")
    for source in sources:
        if float(source.get("weight", 1.0)) != 1.0:
            raise ValueError("matched arms require equal unweighted training examples")
        path = Path(source["data_dir"]) / source["train_file"]
        if not path.is_absolute() or path.name != target_config(f"{source['lang']}_{source['domain']}")["train_file"]:
            raise ValueError("matched data source must use its explicit original train path")
    if cfg.get("train_file") != "empty_placeholder.jsonl":
        raise ValueError("the primary train placeholder must stay empty; the 12 sources supply training")
    macro = cfg.get("macro_dev_eval", {}) or {}
    specs = macro.get("targets", [])
    if isinstance(specs, str):
        specs = [x.strip() for x in specs.split(",") if x.strip()]
    target_names = tuple(Path(x).stem for x in specs)
    if target_names != train_targets:
        raise ValueError("development selection may use only the ordered 12 training targets")
    template = (cfg.get("small_llm", {}) or {}).get("prompt_template", "")
    fields = []
    for _, name, format_spec, conversion in Formatter().parse(template):
        if name is not None:
            if name not in {"text", "aspect"} or format_spec or conversion:
                raise ValueError("matched prompts allow only plain text/aspect, never VA/metadata/ID")
            fields.append(name)
    if set(fields) != {"text", "aspect"}:
        raise ValueError("matched prompts must contain both text and aspect")
    if cfg.get("teacher", {}).get("enabled") or cfg.get("teacher", {}).get("explanation_as_input"):
        raise ValueError("teacher material is outside the matched prompt/data interface")
    for name in ("group_balanced_sampling", "error_aware_sampling", "lds", "synthetic_data"):
        if (cfg.get(name, {}) or {}).get("enabled"):
            raise ValueError(f"{name} is outside the unweighted matched protocol")


def training_record_audit(train_data: list[dict], *, training_targets=None, held_out_targets=None) -> dict:
    """Derive condition vocabulary/counts and float32 population norm from train only."""
    if not train_data:
        raise ValueError("empty actual training records")
    training_targets = tuple(TRAINING_TARGETS if training_targets is None else training_targets)
    held_out_targets = tuple(HELD_OUT_TARGETS if held_out_targets is None else held_out_targets)
    counts = Counter()
    values = []
    for sample in train_data:
        target = f"{sample.get('lang')}_{sample.get('domain')}"
        if target not in training_targets:
            raise ValueError(f"training record belongs outside training targets: {target}")
        if not all(sample.get(key) is not None and math.isfinite(float(sample[key])) for key in ("v", "a")):
            raise ValueError(f"training record has invalid VA labels in {target}")
        counts[target] += 1
        values.append((float(sample["v"]), float(sample["a"])))
    if set(counts) != set(training_targets):
        raise ValueError("all 12 training targets must have actual labelled records")
    values = np.asarray(values, dtype=np.float32)
    return {
        "vocabulary_source": "actual expanded training records only; dev/test excluded",
        "training_aspects": len(train_data), "target_aspect_counts": dict(counts),
        "langs": sorted({sample["lang"] for sample in train_data}),
        "domains": sorted({sample["domain"] for sample in train_data}),
        "seen_combinations": sorted(counts),
        "held_out_combinations": list(held_out_targets),
        "held_out_unseen_languages": sorted({t.split('_', 1)[0] for t in held_out_targets} - {s['lang'] for s in train_data}),
        "unknown_category_policy": UNKNOWN_CATEGORY_POLICY,
        "normalization": {
            "v_mean": float(values[:, 0].mean()), "v_std": max(float(values[:, 0].std()), 1e-6),
            "a_mean": float(values[:, 1].mean()), "a_std": max(float(values[:, 1].std()), 1e-6),
        },
        "roles": {
            "model_text": ["text", "aspect"],
            "VA_gold": "training loss and post-hoc evaluation only; never prompt or routing input",
            "M_metadata": "language/domain condition inputs, explicitly separate from VA gold",
            "T_metadata": "independent training auxiliary labels only; no prediction input",
        },
    }


def apply_training_vocabulary(cfg: dict, train_data: list[dict]) -> dict:
    """Bind effective adapter labels to observed training records, before model init."""
    train_targets, held_targets = protocol_targets(cfg)
    audit = training_record_audit(train_data, training_targets=train_targets, held_out_targets=held_targets)
    adapter = cfg.setdefault("backbone_conditional_adapter", {})
    if adapter.get("enabled"):
        for field in ("langs", "domains"):
            supplied = adapter.get(field)
            if supplied and supplied != audit[field]:
                raise ValueError(f"configured adapter {field} differ from actual TRAIN-only vocabulary")
            adapter[field] = list(audit[field])
    cfg.setdefault("matched_protocol", {})["training_record_audit"] = copy.deepcopy(audit)
    return audit


def audit_training_splits(training_targets=TRAINING_TARGETS, held_out_targets=HELD_OUT_TARGETS,
                          data_root: str | Path = DEFAULT_DATA_ROOT) -> dict:
    """Read each of the 12 train/dev files once; never open any test or held-out split."""
    if os.environ.get("DIMABSA2026_DATA_DIR"):
        raise ValueError("unset DIMABSA2026_DATA_DIR before the explicit-path audit")
    rows = []
    train_data = []
    files_read = []
    for name in training_targets:
        cfg = target_config(name, data_root)
        row = {"target": name}
        for split in ("train", "dev"):
            path = Path(cfg["data_dir"]) / cfg[f"{split}_file"]
            if not path.is_file():
                raise FileNotFoundError(f"required {split} source missing: {path}")
            samples = load_samples(cfg, split)
            aspects = expand_to_aspect_level(samples)
            if not aspects:
                raise ValueError(f"required {split} source has no aspects: {name}")
            row[f"{split}_records"] = len(samples)
            row[f"{split}_aspects"] = len(aspects)
            files_read.append(str(path))
            if split == "train":
                train_data.extend(aspects)
        rows.append(row)
    return {
        "status": "CPU data preparation only; no experiment launched or historical result validated",
        "targets": rows, "files_read": files_read,
        "held_out_train_dev_files_read": 0, "test_files_read": 0,
        "hash_scans": 0, "training": training_record_audit(train_data, training_targets=training_targets,
                                                           held_out_targets=held_out_targets),
    }


def audit_original_training_splits(data_root: str | Path = DEFAULT_DATA_ROOT) -> dict:
    return audit_training_splits(data_root=data_root)


def write_numeric_audit(audit: dict, output_dir: str | Path) -> None:
    """Export aggregate counts/state, without any private text/aspect/record IDs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "TRAIN_DATA_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    fields = ("target", "train_records", "train_aspects", "dev_records", "dev_aspects")
    lines = ["\t".join(fields)]
    lines += ["\t".join(str(row[key]) for key in fields) for row in audit["targets"]]
    (output_dir / "TRAIN_DEV_COUNTS.tsv").write_text("\n".join(lines) + "\n")
