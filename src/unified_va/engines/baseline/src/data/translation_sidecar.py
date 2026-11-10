"""Utilities for loading translation-pivot sidecars."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.sample_keys import candidate_sample_keys_from_sample, make_sample_key, to_legacy_sample_key


def _normalize_translation_record(rec: dict) -> dict:
    text_pivot = rec.get("text_en", rec.get("pivot_text", rec.get("text_pivot", "")))
    aspect_pivot = rec.get("aspect_en", rec.get("pivot_aspect", rec.get("aspect_pivot", "")))
    pivot_lang = rec.get("pivot_lang", rec.get("target_lang", "eng"))
    mt_conf = rec.get("mt_conf", rec.get("confidence", rec.get("score", 1.0)))

    return {
        "text_en": str(text_pivot or "").strip(),
        "aspect_en": str(aspect_pivot or "").strip(),
        "pivot_lang": str(pivot_lang or "eng").strip() or "eng",
        "mt_conf": float(mt_conf),
    }


def load_translation_pivot_map(
    path_or_paths: str | Path | list[str | Path] | tuple[str | Path, ...]
) -> dict[str, dict]:
    """Load translation sidecar records keyed by sample key.

    Supported JSONL record fields:
      - preferred: ``sample_key``, ``text_en``, ``aspect_en``, ``mt_conf``
      - fallback key fields: ``id``, ``aspect_index``, ``lang``
      - accepted aliases: ``pivot_text`` / ``text_pivot``, ``pivot_aspect`` / ``aspect_pivot``
    """
    if isinstance(path_or_paths, (str, Path)):
        paths = [path_or_paths]
    else:
        paths = list(path_or_paths)

    translation_map: dict[str, dict] = {}
    for path in paths:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sample_key = rec.get("sample_key")
                if not sample_key:
                    if "id" not in rec or "aspect_index" not in rec:
                        continue
                    sample_key = make_sample_key(
                        sample_id=str(rec["id"]),
                        aspect_index=int(rec["aspect_index"]),
                        lang=rec.get("lang"),
                    )
                norm = _normalize_translation_record(rec)
                translation_map[sample_key] = norm
                legacy_key = to_legacy_sample_key(sample_key)
                translation_map.setdefault(legacy_key, norm)
    return translation_map


def lookup_translation_record(translation_map: dict[str, dict] | None, sample: dict) -> tuple[dict | None, str | None]:
    if translation_map is None:
        return None, None
    for sample_key in candidate_sample_keys_from_sample(sample):
        rec = translation_map.get(sample_key)
        if rec is not None:
            return rec, sample_key
    return None, None


def audit_translation_coverage(translation_map: dict[str, dict] | None, samples: list[dict]) -> dict:
    if translation_map is None:
        return {
            "n_dataset": len(samples),
            "n_non_english": sum(1 for s in samples if (s.get("lang") or "").strip() != "eng"),
            "n_hit": 0,
            "hit_rate": 0.0,
            "n_non_english_hit": 0,
            "non_english_hit_rate": 0.0,
        }

    n_hit = 0
    n_non_english = 0
    n_non_english_hit = 0
    for sample in samples:
        lang = (sample.get("lang") or "").strip()
        rec, _ = lookup_translation_record(translation_map, sample)
        if rec is not None:
            n_hit += 1
        if lang != "eng":
            n_non_english += 1
            if rec is not None:
                n_non_english_hit += 1

    return {
        "n_dataset": len(samples),
        "n_non_english": n_non_english,
        "n_hit": n_hit,
        "hit_rate": round(n_hit / max(len(samples), 1), 4),
        "n_non_english_hit": n_non_english_hit,
        "non_english_hit_rate": round(n_non_english_hit / max(n_non_english, 1), 4),
    }
