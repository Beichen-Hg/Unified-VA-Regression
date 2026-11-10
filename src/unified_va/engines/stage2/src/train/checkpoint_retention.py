"""Authorized weight retention after complete dev selection; no tensor reads/hashes.

All scalar candidates remain immutable. Only unselected weights in completed
new V2 runs may be released. A durable record distinguishes intentional release
from missing artifacts and supports an interrupted cleanup without rerunning ML.
"""
from __future__ import annotations

from datetime import datetime
import csv
import fcntl
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "completed_dev_rule_union_v1"
RECORD_NAME = "CHECKPOINT_RETENTION.json"


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def policy():
    path = ROOT / "protocol/CHECKPOINT_RETENTION.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("authorized") is not True:
        return None
    if (value.get("version") != FORMAT or value.get("root") != str(ROOT)
            or not value.get("authorization_source") or not value.get("protect_active_runs")
            or not value.get("keep_all_scalar_archives")):
        raise PermissionError("incomplete checkpoint retention instruction")
    return value


def read_retention_record(archive):
    archive = Path(archive).resolve()
    path = archive.parent / RECORD_NAME
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if (value.get("version") != FORMAT or value.get("run_dir") != str(archive.parent)
            or value.get("state") not in {"applying", "completed"}
            or not value.get("authorization_source")):
        raise ValueError("invalid completed-development retention record")
    candidates = value.get("candidates")
    rules = value.get("selected_rule_checkpoints")
    if not isinstance(candidates, dict) or not candidates or not isinstance(rules, dict) or len(rules) != 10:
        raise ValueError("retention record must preserve every candidate and all ten rules")
    for name, item in candidates.items():
        if Path(name).name != name or not name.endswith(".pt") or type(item.get("retain")) is not bool:
            raise ValueError("invalid retained/released checkpoint name")
    if any(name not in candidates or not candidates[name]["retain"] for name in rules.values()):
        raise ValueError("retention record attempts to release a rule-selected weight")
    return value


def validate_retained_scalars(retention, records):
    candidates = retention["candidates"]
    if set(candidates) != {r["checkpoint"] for r in records}:
        raise ValueError("retention record and complete scalar trajectory differ")
    for record in records:
        item = candidates[record["checkpoint"]]
        if (item["checkpoint_bytes"] != record["checkpoint_bytes"]
                or item["archive"] != Path(record["archive_path"]).name):
            raise ValueError("retention record differs from original scalar archive")
        if record["inference_config"]["matched_protocol"].get("protocol_version") != "method_validation_v2":
            raise ValueError("retention cannot excuse missing historical/V1 weights")


def retain_completed_run(run, completion_record):
    """Release only finalized, nonselected V2 candidate weights; idempotent."""
    instruction = policy()
    if instruction is None:
        return None
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT / "outputs/runs") or run == ROOT / "outputs/runs":
        raise ValueError("retention is restricted to new V2 run directories")
    completion_record = Path(completion_record).resolve()
    if not completion_record.is_relative_to(ROOT / "records"):
        raise ValueError("completion proof must belong to this V2 execution")
    with (run / ".checkpoint_retention.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _retain_locked(run, completion_record, instruction)


def _retain_locked(run, completion_record, instruction):
    from src.train.dp_sensitivity import analyze_trajectory, load_development_trajectory, write_analysis
    import yaml
    result = json.loads(completion_record.read_text())
    if Path(result.get("run_dir", "")).resolve() != run or result.get("run_test") is not False:
        raise ValueError("completed training result does not identify this development-only run")
    archive = run / "evaluation_archive"
    records = load_development_trajectory(archive, allow_retained=True)
    cfg = records[0]["inference_config"]
    protocol = cfg["matched_protocol"]
    if (protocol.get("protocol_version") != "method_validation_v2"
            or result.get("num_optimizer_updates") != protocol["expected_total_updates"]):
        raise ValueError("training has not completed the full registered V2 budget")
    dp_cfg = yaml.safe_load((ROOT / f"configs/{protocol['split_id']}/dp_original.yaml").read_text())
    analysis = analyze_trajectory(records, dp_cfg)
    selected = {row["rule"]: Path(row["checkpoint"]).name for row in analysis["summary"]}
    if set(selected) != set(instruction["selected_rules"]):
        raise ValueError("all ten predeclared development rules must be finalized")
    analysis_dir = run / "dp_local_sensitivity"
    table = analysis_dir / "selection_sensitivity.tsv"
    if table.exists():
        with table.open() as stream:
            previous = {r["rule"]: Path(r["checkpoint"]).name for r in csv.DictReader(stream, delimiter="\t")}
        if previous != selected:
            raise ValueError("existing selection differs; no deletion or reselection is allowed")
    else:
        write_analysis(analysis, analysis_dir)
    if not all((analysis_dir / name).is_file() for name in (
            "selection_sensitivity.tsv", "candidate_dev_scores_ranks.tsv", "selected_development_metrics.tsv", "fixed_rules.json")):
        raise ValueError("complete sensitivity/scalar evidence is required before retention")
    protected = set(selected.values())
    best = run / "checkpoint.pt"
    if not best.is_file() or best.is_symlink():
        raise ValueError("protect the existing best checkpoint before cleanup")
    best_stat = best.stat()
    if protocol["arm"] == "S":
        shared = ROOT / f"checkpoints/{protocol['split_id']}/seed{cfg['seed']}/stage1_shared"
        handoff = json.loads((shared / "HANDOFF_RECORD.json").read_text())
        shared_checkpoint = shared / "checkpoint.pt"
        expected = archive / selected["rmse_only"]
        if (handoff["source_run"] != str(run) or Path(handoff["source_checkpoint"]) != expected
                or not shared_checkpoint.is_file() or not os.path.samefile(shared_checkpoint, expected)):
            raise ValueError("S shared starting point must be preserved before candidate cleanup")
    previous = read_retention_record(archive)
    candidates = {}
    for row in records:
        name = row["checkpoint"]
        path = archive / name
        if path.is_symlink() or path.resolve().parent != archive.resolve():
            raise ValueError("checkpoint path escapes its V2 archive")
        stat = path.stat() if path.exists() else None
        if stat and (stat.st_dev, stat.st_ino) == (best_stat.st_dev, best_stat.st_ino):
            protected.add(name)  # Also retain the root alias if exact ties chose a different event.
        old = previous["candidates"].get(name) if previous else None
        candidates[name] = {
            "archive": Path(row["archive_path"]).name, "checkpoint_bytes": row["checkpoint_bytes"],
            "allocated_bytes": stat.st_blocks * 512 if stat else old["allocated_bytes"],
            "initial_link_count": stat.st_nlink if stat else old["initial_link_count"],
            "retain": name in protected,
        }
    if previous and (previous["selected_rule_checkpoints"] != selected
            or any(previous["candidates"][name]["retain"] != item["retain"] for name, item in candidates.items())):
        raise ValueError("retention decisions are immutable once recorded")
    record = previous or {"version": FORMAT, "run_dir": str(run), "created_at": now(),
        "authorization_source": instruction["authorization_source"], "completion_record": str(completion_record),
        "selected_rule_checkpoints": selected, "candidates": candidates,
        "scalar_candidates_preserved": len(records), "root_best_preserved": str(best),
        "weight_retention_scope": "all ten rule-selected weights plus shared/best dependencies",
        "hash_scans": 0, "model_execution": False}
    record["state"] = "applying"
    atomic(run / RECORD_NAME, record)  # Durable authorization and intended releases precede unlink.
    for name, item in record["candidates"].items():
        path = archive / name
        if item["retain"]:
            if not path.is_file() or path.stat().st_size != item["checkpoint_bytes"]:
                raise ValueError("a protected selected checkpoint is missing or changed")
            continue
        if path.exists():
            stat = path.stat()
            if path.is_symlink() or stat.st_size != item["checkpoint_bytes"] or stat.st_nlink != 1:
                raise ValueError("unselected checkpoint changed or has an unexpected dependency")
            path.unlink()
    record.update(state="completed", completed_at=now(),
        released_weight_files=sum(not x["retain"] for x in record["candidates"].values()),
        released_allocated_bytes=sum(x["allocated_bytes"] for x in record["candidates"].values() if not x["retain"]))
    atomic(run / RECORD_NAME, record)
    load_development_trajectory(archive, allow_retained=True)
    update_summary()
    return record


def update_summary():
    """Small sidecar inventory only, used by status reporting; no weight scan."""
    with (ROOT / "records/checkpoint_retention_summary.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runs = []
        for path in sorted((ROOT / "outputs/runs").glob("*/" + RECORD_NAME)):
            row = json.loads(path.read_text())
            runs.append({key: row.get(key) for key in (
                "run_dir", "state", "released_weight_files", "released_allocated_bytes", "completed_at")})
        done = [row for row in runs if row["state"] == "completed"]
        atomic(ROOT / "records/CHECKPOINT_RETENTION_SUMMARY.json", {
            "updated_at": now(), "runs": runs, "completed_runs": len(done),
            "released_weight_files": sum(row["released_weight_files"] for row in done),
            "released_allocated_gib": sum(row["released_allocated_bytes"] for row in done) / 1024**3,
            "scope": "authorized completed V2 runs; selected weights and all scalar candidates retained",
            "hash_scans": 0,
        })


def publish_best_alias(run, *, event, epoch, update, cfg):
    """Atomic best alias to an already serialized immutable dev candidate."""
    run = Path(run)
    stem = f"eval_{event}_epoch{epoch:03d}_update{update:08d}"
    archive = run / "evaluation_archive"
    record = json.loads((archive / (stem + ".json")).read_text())
    source = archive / (stem + ".pt")
    if (event not in {"step", "epoch"} or record.get("inference_config") != cfg
            or record.get("checkpoint") != source.name or source.is_symlink()
            or not source.is_file() or source.stat().st_size != record["checkpoint_bytes"]):
        raise ValueError("best must reference the exact archived current development event")
    temporary = run / ".best_checkpoint_link.tmp"
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    temporary.replace(run / "checkpoint.pt")
    atomic(run / "BEST_CHECKPOINT_ALIAS.json", {
        "archive": str(archive / (stem + ".json")), "checkpoint": str(source),
        "alias": str(run / "checkpoint.pt"), "event": event, "epoch": epoch, "global_update": update,
        "effective_at": now(), "format": "hardlink_to_immutable_dev_candidate", "additional_tensor_bytes": 0,
    })
