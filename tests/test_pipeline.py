import copy
import pytest
import torch
from unified_va import config, storage
from unified_va.training.trainer import run
from unified_va.inference import Predictor
from unified_va.io import read
from unified_va.analysis import dp_sensitivity
from unified_va.diagnostics import run as diagnose


def train_fixture(miniature, stage, split="original", parent=None):
    root = miniature["root"] / (split + "_" + stage)
    cfg = config.prepare(
        stage, split, 13, miniature["data"], miniature["model"], root, "cpu", parent
    )
    result = run(root / "config.json")
    return root, cfg, result


@pytest.mark.parametrize("stage", ["baseline", "stage1"])
@pytest.mark.parametrize("split", ["original", "alt1"])
def test_actual_shared_training_loop_selected_restore_and_upper_update(
    miniature, stage, split
):
    root, cfg, result = train_fixture(miniature, stage, split)
    assert (
        result["passed"]
        and result["updates"] == 6
        and result["development_checks"] == 6
    )
    assert read(root / "FINAL_DEV_RESTORE.json")["passed"]
    assert (
        read(root / "FIRST_OPTIMIZER_GRADIENTS.json")["upper"][
            "nonzero_gradient_tensors"
        ]
        > 0
    )
    value = storage.load(root / result["checkpoint"])
    assert not torch.equal(
        value["state"]["backbone.layers.24.weight"], miniature["base"].layers[24].weight
    )
    model, _ = miniature["build"](cfg)
    storage.restore(model, value, cfg)
    for key, expected in value["state"].items():
        assert torch.equal(model.state_dict()[key], expected)
    broken = copy.deepcopy(value)
    broken["state"].pop(next(iter(broken["state"])))
    with pytest.raises(ValueError):
        storage.restore(model, broken, cfg)


@pytest.mark.parametrize("split", ["original", "alt1"])
def test_full_stage1_to_stage2_preflight_training_evaluation_and_diagnostics(
    miniature, monkeypatch, split
):
    parent, _, _ = train_fixture(miniature, "stage1", split)
    root, cfg, result = train_fixture(miniature, "stage2", split, parent)
    preflight = read(root / "PREFLIGHT.json")
    assert preflight["passed"] and preflight["optimizer_updates"] == 0
    assert preflight["zero_adapter_raw_max_abs_difference"] == 0
    assert result["updates"] == 2
    from unified_va.evaluation import runner

    monkeypatch.setattr(runner, "TEST_ASPECTS", 30)
    evaluation = runner.run(root, miniature["data"], miniature["model"], "cpu")
    assert (
        evaluation["passed"]
        and evaluation["aspects"] == 30
        and len(evaluation["per_target"]) == 15
    )
    sensitivity = dp_sensitivity(root, root / "dp.json")
    assert len(sensitivity) == 14
    predictor = Predictor(root / result["checkpoint"], miniature["model"], "cpu")
    rows = [dict(text="clear", aspect="screen", lang="eng", domain="restaurant")]
    assert diagnose(predictor, rows)["language"]["known"] == 1
    assert len(predictor.predict(rows)) == 1


def test_parent_seed_mismatch_is_rejected(miniature):
    parent, _, result = train_fixture(miniature, "stage1")
    cfg = config.prepare(
        "stage2",
        "original",
        42,
        miniature["data"],
        miniature["model"],
        miniature["root"] / "wrong",
        "cpu",
        parent,
    )
    model, _ = miniature["build"](cfg)
    value = storage.load(parent / result["checkpoint"])
    with pytest.raises(ValueError, match="seed"):
        storage.warm_start(model, value, cfg, value["normalization"])
