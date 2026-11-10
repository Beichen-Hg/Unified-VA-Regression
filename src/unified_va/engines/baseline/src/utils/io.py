"""I/O helpers: config loading, run directory management."""
import os
import json
import hashlib
import datetime
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(*cfgs: dict) -> dict:
    merged = {}
    for c in cfgs:
        for k, v in c.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = merge_configs(merged[k], v)
            else:
                merged[k] = v
    return merged


def make_run_dir(base: str = "outputs/runs", tag: str = "", seed: int = None) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed is not None:
        rid = f"{ts}_{tag}_seed{seed}" if tag else f"{ts}_seed{seed}"
    else:
        rid = f"{ts}_{tag}" if tag else ts
    p = Path(base) / rid
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_config(cfg: dict, run_dir: Path):
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def save_metrics(metrics: dict, run_dir: Path):
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def cache_path(name: str, base: str = "outputs/cache") -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p / name
