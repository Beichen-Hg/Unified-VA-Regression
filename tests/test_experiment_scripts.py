"""The documented recipes use a same-seed parent and do not run during planning."""
from pathlib import Path
import json
import subprocess
import sys
import pytest
from unified_va.cli import parser as command_parser, add_run_arguments
import argparse

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("recipe", ["main_original", "main_alt1"])
def test_documented_experiment_commands(tmp_path, recipe):
    output = tmp_path/"new runs"
    invocation = [sys.executable, str(ROOT/"scripts/reproduce.py"), "--config", str(ROOT/"configs"/(recipe+".yaml")),
                  "--data-root", str(tmp_path/"official data"), "--model-dir", str(tmp_path/"local model"),
                  "--output-dir", str(output), "--dry-run", "--json"]
    jobs = json.loads(subprocess.check_output(invocation, text=True, cwd=tmp_path))
    assert len(jobs) == 12
    train_parser = argparse.ArgumentParser()
    add_run_arguments(train_parser)
    for job in jobs:
        if Path(job[1]).name == "train.py":
            train_parser.parse_args(job[2:])
        else:
            command_parser().parse_args(["evaluate", *job[2:]])
    assert not output.exists()
    for seed in (42, 13, 7):
        training = [job for job in jobs if "train.py" in Path(job[1]).name and job[job.index("--seed")+1] == str(seed)]
        assert len(training) == 2
        stages = [job[job.index("--stage")+1] for job in training]
        assert stages == ["stage1", "stage2"]
        assert training[1][training[1].index("--parent")+1] == training[0][training[0].index("--run-dir")+1]
