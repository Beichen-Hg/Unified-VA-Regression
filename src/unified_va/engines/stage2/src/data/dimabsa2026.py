"""DimABSA2026 data loader: JSONL → sample-level & aspect-level."""
import json
import os
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_data_dir(cfg: dict) -> Path:
    env = os.environ.get("DIMABSA2026_DATA_DIR")
    if env:
        return Path(env)
    return Path(cfg["data_dir"])


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_va(va_str: str) -> tuple[float, float]:
    """Parse 'V#A' string to (v, a) floats."""
    v, a = va_str.split("#")
    return float(v), float(a)


def _extract_entries(row: dict) -> list[dict]:
    """Extract aspect entries from Quadruplet, Triplet, or Aspect_VA key."""
    for key in ("Aspect_VA", "Quadruplet", "Triplet"):
        if key in row and row[key]:
            return row[key]
    return []


def load_samples(cfg: dict, split: str) -> list[dict]:
    """Load sample-level data from JSONL.

    Returns list of:
      {id, text, aspects, gold_va (list[(v,a)]|None), track, lang, domain,
       opinions (list[str]|None), categories (list[str]|None)}
    """
    data_dir = _resolve_data_dir(cfg)
    file_key = f"{split}_file"
    fname = cfg.get(file_key, f"{split}.jsonl")
    path = data_dir / fname

    if not path.exists():
        logger.warning(f"{path} not found, returning empty")
        return []

    raw = load_jsonl(path)
    samples = []
    for row in raw:
        sid = str(row["ID"])
        text = row["Text"]
        entries = _extract_entries(row)

        if entries:
            aspects = [e["Aspect"] for e in entries]
            gold_va = [parse_va(e["VA"]) for e in entries] if "VA" in entries[0] else None
            opinions = [e["Opinion"] for e in entries] if "Opinion" in entries[0] else None
            categories = [e["Category"] for e in entries] if "Category" in entries[0] else None
        else:
            aspects = row.get("Aspect", [])
            gold_va = None
            opinions = None
            categories = None

        samples.append({
            "id": sid,
            "text": text,
            "aspects": aspects,
            "gold_va": gold_va,
            "opinions": opinions,
            "categories": categories,
            "track": cfg.get("track", "A"),
            "lang": cfg.get("lang", "unknown"),
            "domain": cfg.get("domain", "unknown"),
        })

    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples


def expand_to_aspect_level(samples: list[dict]) -> list[dict]:
    """Expand sample-level to aspect-level for training/inference.

    Returns list of:
      {id, text, aspect, v, a, aspect_index, sample_aspects,
       opinion, category, ...meta}
    """
    out = []
    for s in samples:
        for i, asp in enumerate(s["aspects"]):
            v, a = None, None
            if s["gold_va"] is not None and i < len(s["gold_va"]):
                v, a = s["gold_va"][i]
            opinion = s["opinions"][i] if s.get("opinions") and i < len(s["opinions"]) else None
            category = s["categories"][i] if s.get("categories") and i < len(s["categories"]) else None
            out.append({
                "id": s["id"],
                "text": s["text"],
                "aspect": asp,
                "v": v,
                "a": a,
                "aspect_index": i,
                "sample_aspects": s["aspects"],
                "opinion": opinion,
                "category": category,
                "track": s["track"],
                "lang": s.get("lang", "unknown"),
                "domain": s.get("domain", "unknown"),
            })
    return out


def load_splits(cfg: dict) -> dict[str, list[dict]]:
    """Load train/dev/test and expand to aspect-level."""
    splits = {}
    for split in ("train", "dev", "test"):
        samples = load_samples(cfg, split)
        smoke_cfg = cfg.get("smoke_test", {})
        limit = smoke_cfg.get(f"max_{split}_samples")
        if limit is not None:
            limit = int(limit)
            if limit >= 0:
                samples = samples[:limit]
                logger.info(f"Smoke test: truncated {split} to {len(samples)} sample-level rows")
        splits[f"{split}_samples"] = samples
        splits[split] = expand_to_aspect_level(samples)
        logger.info(f"{split}: {len(samples)} samples → {len(splits[split])} aspect-level")
    return splits
