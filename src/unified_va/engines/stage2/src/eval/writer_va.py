"""Write aspect-level predictions to official DimABSA2026 JSONL format."""
import json
from pathlib import Path
from collections import defaultdict
from src.models.losses import format_va
from src.utils.logging import get_logger

logger = get_logger(__name__)


def write_predictions(predictions: list[dict], out_path: str | Path):
    """Regroup aspect-level predictions back to sample-level and write JSONL.

    Each prediction dict must have:
      id, aspect, aspect_index, sample_aspects, v_pred, a_pred
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by sample ID
    grouped = defaultdict(list)
    for p in predictions:
        grouped[p["id"]].append(p)

    lines = []
    for sid, preds in grouped.items():
        preds.sort(key=lambda x: x["aspect_index"])
        sample_aspects = preds[0]["sample_aspects"]
        assert len(preds) == len(sample_aspects), \
            f"ID={sid}: {len(preds)} preds vs {len(sample_aspects)} aspects"

        aspect_va = []
        for p, expected_asp in zip(preds, sample_aspects):
            assert p["aspect"] == expected_asp, \
                f"ID={sid}: aspect mismatch '{p['aspect']}' vs '{expected_asp}'"
            entry = {
                "Aspect": p["aspect"],
                "VA": format_va(p["v_pred"], p["a_pred"]),
            }
            if "sigma_v" in p:
                entry["sigma_v"] = round(p["sigma_v"], 4)
                entry["sigma_a"] = round(p["sigma_a"], 4)
            aspect_va.append(entry)

        lines.append(json.dumps({"ID": sid, "Aspect_VA": aspect_va}, ensure_ascii=False))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Wrote {len(lines)} predictions to {out_path}")


def write_verbose_predictions(predictions: list[dict], out_path: str | Path):
    """Write aspect-level verbose predictions as JSONL.

    Each line preserves the full prediction dict for downstream analysis or
    post-hoc blending. This is intentionally separate from the official output
    format so we can keep auxiliary fields such as sigma / TTA uncertainty.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _normalize_value(v):
        if isinstance(v, float):
            return float(v)
        return v

    with out_path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            payload = {k: _normalize_value(v) for k, v in pred.items()}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(predictions)} verbose predictions to {out_path}")
