"""Teacher artifact loading and joining for VA distillation."""

import json
import logging
from pathlib import Path

from src.data.sample_keys import to_legacy_sample_key

logger = logging.getLogger(__name__)


def _iter_teacher_paths(path_or_paths: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[Path]:
    """Normalize one or many teacher artifact paths into a flat ordered list."""
    if isinstance(path_or_paths, (str, Path)):
        paths = [path_or_paths]
    else:
        paths = list(path_or_paths)

    normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("teacher artifact path list is empty")
    return normalized


def load_teacher_map(path_or_paths: str | Path | list[str | Path] | tuple[str | Path, ...]) -> dict:
    """Load teacher predictions from one or many JSONL files.

    Later files override earlier ones on duplicate ``sample_key``.

    Returns:
        dict keyed by sample_key with teacher prediction fields.
    """
    teacher_map = {}
    paths = _iter_teacher_paths(path_or_paths)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"teacher artifact not found: {path}")
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = rec.get("sample_key", "")
                if key:
                    teacher_map[key] = rec
    return teacher_map


def join_teacher_to_samples(
    sample_keys: list[str],
    teacher_map: dict,
    lang_weights: dict | None = None,
    use_confidence: bool = False,
    confidence_clip_min: float = 0.0,
    confidence_clip_max: float = 1.0,
    confidence_threshold: float = 0.0,
    sample_langs: list[str] | None = None,
) -> tuple[dict, dict]:
    """Join teacher predictions to training sample keys.

    Returns:
        (joined_map, audit) where joined_map is {sample_key: {teacher_v,
        teacher_a, teacher_mask, teacher_weight, teacher_expl_quality, ...}}
    """
    joined = {}
    n_matched = 0
    n_missing = 0
    langs_seen = set()
    expl_count = 0
    expl_quality_sum = 0.0

    for i, key in enumerate(sample_keys):
        lang = sample_langs[i] if sample_langs else ""
        rec = teacher_map.get(key)
        if rec is None:
            legacy_key = to_legacy_sample_key(key)
            if legacy_key != key:
                rec = teacher_map.get(legacy_key)

        if rec is not None and rec.get("teacher_v") is not None:
            weight = 1.0
            # Language-based weighting
            if lang_weights and lang:
                weight *= lang_weights.get(lang, 1.0)
            # Confidence weighting
            if use_confidence:
                conf = rec.get("teacher_confidence", 1.0)
                conf = max(confidence_clip_min, min(confidence_clip_max, conf))
                if conf < confidence_threshold:
                    weight = 0.0
                else:
                    weight *= conf

            entry = {
                "teacher_v": rec["teacher_v"],
                "teacher_a": rec["teacher_a"],
                "teacher_mask": 1 if weight > 0 else 0,
                "teacher_weight": weight,
                "teacher_expl_quality": rec.get("teacher_expl_quality", 1.0),
                "teacher_confidence": rec.get("teacher_confidence", 1.0),
                "teacher_explanation": rec.get("teacher_explanation", ""),
            }
            joined[key] = entry
            n_matched += 1
            if lang:
                langs_seen.add(lang)
            # Track explanation coverage
            if rec.get("teacher_explanation"):
                expl_count += 1
                eq = rec.get("teacher_expl_quality")
                if eq is not None:
                    expl_quality_sum += eq
        else:
            joined[key] = {
                "teacher_v": 0.0,
                "teacher_a": 0.0,
                "teacher_mask": 0,
                "teacher_weight": 0.0,
                "teacher_expl_quality": 0.0,
                "teacher_confidence": 0.0,
                "teacher_explanation": "",
            }
            n_missing += 1

    n_total = len(sample_keys)
    ratio = round(n_matched / max(n_total, 1), 4)

    # Determine artifact variant
    variant = "plain"
    if expl_count > 0:
        variant = "explanation_aware"

    audit = {
        "num_samples_total": n_total,
        "num_teacher_matched": n_matched,
        "num_teacher_missing": n_missing,
        "teacher_match_ratio": ratio,
        "langs_present": sorted(langs_seen),
        "use_confidence": use_confidence,
        "teacher_artifact_variant": variant,
        "num_with_explanation": expl_count,
        "explanation_coverage_ratio": round(expl_count / max(n_matched, 1), 4) if expl_count > 0 else 0,
    }

    if expl_count > 0:
        audit["expl_quality_mean"] = round(expl_quality_sum / expl_count, 4)

    return joined, audit


def save_teacher_audit(audit: dict, path: str | Path):
    """Save teacher audit dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
