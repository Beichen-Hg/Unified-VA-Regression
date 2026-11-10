"""Reproduce a registered experiment, or print its commands with --dry-run."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys
import yaml
import _common


def commands(args):
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if set(config) - {"experiment", "split", "seeds"}:
        raise ValueError("Unknown experiment configuration field")
    if config["split"] not in ("original", "alt1"):
        raise ValueError("Choose original or alt1")
    if config["experiment"] != "main":
        raise ValueError("Choose the main experiment recipe")
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    if len(seeds) != len(set(seeds)) or any(s not in (42, 13, 7) for s in seeds):
        raise ValueError("Registered seeds are 42, 13 and 7")
    root = Path(args.output_dir).resolve()
    common = ["--split", config["split"], "--data-root", str(Path(args.data_root).resolve()),
              "--model-dir", str(Path(args.model_dir).resolve()), "--device", args.device]
    evaluation_common = common[2:]
    jobs = []
    for seed in seeds:
        options = [*common, "--seed", str(seed)]
        if config["experiment"] == "main":
            source, target = root / str(seed) / "stage1", root / str(seed) / "stage2"
            jobs.append([sys.executable, str(_common.ROOT/"scripts/train.py"), "--stage", "stage1", *options, "--run-dir", str(source)])
            jobs.append([sys.executable, str(_common.ROOT/"scripts/train.py"), "--stage", "stage2", *options, "--parent", str(source), "--run-dir", str(target)])
            for run in (source, target):
                jobs.append([sys.executable, str(_common.ROOT/"scripts/evaluate.py"), *evaluation_common, "--run-dir", str(run)])
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, choices=[42, 13, 7])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="JSON commands; use with --dry-run")
    args = parser.parse_args()
    if args.json and not args.dry_run:
        parser.error("--json requires --dry-run")
    jobs = commands(args)
    if args.json:
        print(json.dumps(jobs))
        return
    for job in jobs:
        print(shlex.join(job), flush=True)
        if not args.dry_run:
            subprocess.run(job, cwd=_common.ROOT, check=True)


if __name__ == "__main__":
    main()
