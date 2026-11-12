"""Official JSONL expansion and a text/aspect-only tokenizer boundary."""

from pathlib import Path
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from ..config import PROMPT, split_targets, target_config
from ..io import write


def load_target(data_root, target, split):
    cfg = target_config(target, data_root)
    path = Path(cfg["data_dir"]) / cfg[f"{split}_file"]
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            entries = next(
                (
                    record[k]
                    for k in ("Aspect_VA", "Quadruplet", "Triplet")
                    if record.get(k)
                ),
                None,
            )
            if entries is None:
                aspects = record.get("Aspect", [])
                if isinstance(aspects, str):
                    aspects = [aspects]
                entries = [dict(Aspect=a) for a in aspects]
            for index, entry in enumerate(entries):
                row = dict(
                    id=str(record["ID"]),
                    text=record["Text"],
                    aspect=entry["Aspect"],
                    aspect_index=index,
                    target=target,
                    lang=cfg["lang"],
                    domain=cfg["domain"],
                )
                if entry.get("VA") is not None:
                    row["v"], row["a"] = map(float, entry["VA"].split("#"))
                    if not all(math.isfinite(row[k]) for k in ("v", "a")):
                        raise ValueError("Nonfinite VA label")
                rows.append(row)
    if not rows:
        raise ValueError("Empty target: " + target)
    return rows


def normalization(rows):
    values = torch.tensor([[r["v"], r["a"]] for r in rows], dtype=torch.float32)
    means, scales = values.mean(0), values.std(0, unbiased=False).clamp_min(1e-6)
    return dict(
        v_mean=means[0].item(),
        v_std=scales[0].item(),
        a_mean=means[1].item(),
        a_std=scales[1].item(),
    )


def normalize(values, stats):
    return (
        values - values.new_tensor([stats["v_mean"], stats["a_mean"]])
    ) / values.new_tensor([stats["v_std"], stats["a_std"]])


class AspectDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=192):
        self.rows, self.tokenizer, self.max_len = rows, tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if not isinstance(row.get("text"), str) or not isinstance(
            row.get("aspect"), str
        ):
            raise ValueError("Each input requires string text and aspect")
        tokens = self.tokenizer(
            PROMPT.format(text=row["text"], aspect=row["aspect"]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        out = {k: tokens[k].squeeze(0) for k in ("input_ids", "attention_mask")}
        out["index"] = index
        if "v" in row and "a" in row:
            out["va"] = torch.tensor([row["v"], row["a"]], dtype=torch.float32)
        out["lang"], out["domain"] = row.get("lang", ""), row.get("domain", "")
        return out


def loader(rows, tokenizer, cfg, *, training=False):
    return DataLoader(
        AspectDataset(rows, tokenizer, cfg["max_len"]),
        batch_size=cfg["batch_size"] if training else cfg["eval_batch_size"],
        shuffle=training,
        num_workers=0,
        drop_last=False,
    )


def audit(cfg):
    train, _ = split_targets(cfg["experiment"]["split"])
    training, development, counts = [], {}, {}
    for target in train:
        current = load_target(cfg["experiment"]["data_root"], target, "train")
        training.extend(current)
        counts[target] = len(current)
        development[target] = load_target(cfg["experiment"]["data_root"], target, "dev")
    for row in training + [r for rows in development.values() for r in rows]:
        if "v" not in row or "a" not in row:
            raise ValueError("Training and development require VA labels")
    actual = (len(training), sum(map(len, development.values())))
    expected = (cfg["experiment"]["train_aspects"], cfg["experiment"]["dev_aspects"])
    if actual != expected:
        raise ValueError(f"Train/dev counts differ: {actual} versus {expected}")
    report = dict(
        passed=True,
        training_aspects=actual[0],
        development_aspects=actual[1],
        training_target_aspects=counts,
        normalization=normalization(training),
    )
    write(Path(cfg["experiment"]["run_dir"]) / "DATA_AUDIT.json", report)
    return report, training, development
