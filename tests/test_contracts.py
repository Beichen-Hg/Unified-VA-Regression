from pathlib import Path
import ast
import copy
import hashlib
import inspect
import json
import math
import subprocess
import sys
import pytest
from unified_va import config
from unified_va.cli import parser
from unified_va.paired_statistics import bootstrap
from unified_va.training.dp_selection import compute_macro_dev_selection

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", ["baseline", "stage1", "stage2"])
@pytest.mark.parametrize("split", ["original", "alt1"])
def test_six_profiles_unique_prompt_checkpointing_partitions_and_frozen_settings(
    tmp_path, stage, split
):
    cfg = config.prepare(
        stage,
        split,
        13,
        tmp_path / "data",
        tmp_path / "model",
        tmp_path / stage,
        device="cpu",
        parent=tmp_path / "stage1" if stage == "stage2" else None,
    )
    assert config.validate_effective(cfg)
    assert cfg["small_llm"]["prompt_template"] == config.PROMPT
    assert cfg["small_llm"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    training, held = config.split_targets(split)
    assert len(training) == 12 and len(held) == 3 and not set(training) & set(held)
    retired = {
        "distill_weight",
        "goemo_ratio",
        "n_rounds",
        "n_epochs_stage1",
        "n_epochs_stage2",
        "n_clusters",
        "cluster_weight",
        "arousal_anti_collapse",
        "teacher",
        "int_weight",
        "int_mode",
        "legacy_prediction_rounding",
        "author_approval",
        "author_approval_scope",
        "execution_authorized",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()

    assert not keys(cfg) & retired
    assert not (tmp_path / "data").exists()
    changed = copy.deepcopy(cfg)
    changed["lr"] *= 2
    with pytest.raises(ValueError):
        config.validate_effective(changed)


@pytest.mark.parametrize("recipe", ["main_original", "main_alt1"])
def test_original_script_arguments_and_dry_run_create_no_runs(tmp_path, recipe):
    output = tmp_path / "fresh runs"
    result = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts/reproduce.py"),
            "--config",
            str(ROOT / "configs" / (recipe + ".yaml")),
            "--data-root",
            str(tmp_path / "data"),
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(output),
            "--dry-run",
            "--json",
        ],
        text=True,
    )
    jobs = json.loads(result)
    assert len(jobs) == 12 and not output.exists()
    for job in jobs:
        command = (
            ["evaluate", *job[2:]]
            if Path(job[1]).name == "evaluate.py"
            else ["plan", *job[2:]]
        )
        parser().parse_args(command)
    for seed in (42, 13, 7):
        training = [
            job
            for job in jobs
            if Path(job[1]).name == "train.py"
            and job[job.index("--seed") + 1] == str(seed)
        ]
        assert len(training) == 2
        assert (
            training[1][training[1].index("--parent") + 1]
            == training[0][training[0].index("--run-dir") + 1]
        )


def test_dp_all_penalties_intersection_and_macro_only():
    cfg = config.template("stage2", "alt1")["macro_dev_eval"]["selection"]
    training, _ = config.split_targets("alt1")
    values = {
        t: dict(
            rmse_va=1 + i / 10,
            std_gap_v=0.2,
            std_gap_a=0.4,
            ccc_macro=0.1,
            pcc_macro=0.2,
        )
        for i, t in enumerate(training)
    }
    score, detail = compute_macro_dev_selection(values, cfg)
    errors = sorted((v["rmse_va"] for v in values.values()), reverse=True)
    pseudo = sorted(
        (values[t]["rmse_va"] for t in cfg["pseudo_ood_targets"] if t in values),
        reverse=True,
    )
    expected = sum(errors) / 12 + 0.3 * sum(errors[:4]) / 4
    expected += 0.25 * sum(pseudo) / len(pseudo) + 0.15 * sum(pseudo[:2]) / 2
    expected += 0.05 * 0.2 + 0.15 * 0.4 + 0.25 * (0.4 - 0.1) + 0.1 * (0.45 - 0.2)
    assert score == pytest.approx(expected, abs=1e-12)
    assert detail["n_targets"] == 12
    assert compute_macro_dev_selection(values, {"mode": "macro_mean"})[
        0
    ] == pytest.approx(sum(errors) / 12)


def test_record_cluster_bootstrap_preserved_and_ten_thousand_draws():
    assert inspect.signature(bootstrap).parameters["resamples"].default == 10000
    pairs = []
    for seed in (42, 13):
        pair = {"seed": seed}
        for role, error in [("S", 2.0), ("T", 1.0)]:
            pair[role] = [
                dict(
                    target=t,
                    id=record,
                    aspect_index=i,
                    v_true=4.0,
                    a_true=5.0,
                    v_pred=4.0 + error,
                    a_pred=5.0 + error,
                )
                for t in ("a", "b", "c")
                for record in ("x", "y")
                for i in (0, 1)
            ]
        pairs.append(pair)
    result = bootstrap(
        pairs,
        ["a", "b", "c"],
        ["a", "b"],
        {"a": 4, "b": 4, "c": 4},
        random_seed=20261101,
    )
    overall = next(r for r in result["group_seed_statistics"] if r["group"] == "full")
    assert overall["T_minus_S_mean"] == pytest.approx(-math.sqrt(2))
    assert result["bootstrap_intervals"]


def test_result_files_and_statistical_module_are_byte_preserved():
    expected = {
        "results/stage_results.csv": "1a696030f46ecf74c680de38fec076f41639d1afc75d2794f784b9dc7bb4408a",
        "results/per_target_results.csv": "ed327c5243f1ebcb2ebcdf9039b179ff547ff52ddcae98b8ff0ef6075148b356",
        "src/unified_va/resources/mainline_summary.json": "564501682f5eb21da06cda2a2903862734e850974ef0f7bcf01e81d5f8254637",
        "src/unified_va/paired_statistics.py": "6419d40e8cae571afc1d1d4178b846b2f3c81b60da27b499ef26f6bc0300818d",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_python_sources_parse_without_dynamic_source_rewriting():
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in ("exec", "eval")
            for n in ast.walk(tree)
        )
    assert not (ROOT / "src/unified_va/engines").exists()
