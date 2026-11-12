"""Complete selected-checkpoint dev restoration followed by 15-target testing."""

from pathlib import Path
import statistics
from ..config import TARGETS, split_targets
from ..data import load_target
from ..inference import Predictor
from ..io import read, write, now
from .metrics import metrics

TEST_ASPECTS = 19977


def run(run_dir, data_root, model_dir, device, output=None):
    run_dir = Path(run_dir).resolve()
    result = read(run_dir / "RESULT.json")
    if result.get("passed") is not True:
        raise ValueError("Training is incomplete")
    destination = Path(output).resolve() if output else run_dir / "evaluation"
    if (destination / "RESULT.json").exists():
        raise FileExistsError("Evaluation already completed")
    predictor = Predictor(run_dir / result["checkpoint"], model_dir, device)
    expected = read(
        run_dir
        / "development"
        / f'update{result["selected_update"]:06d}'
        / "METRICS.json"
    )["per_target"]
    differences = {}
    for name in split_targets(result["split"])[0]:
        actual = metrics(predictor.predict(load_target(data_root, name, "dev")))
        differences[name] = abs(actual["rmse_va"] - expected[name]["rmse_va"])
    if len(differences) != 12 or max(differences.values()) > 1e-6:
        raise RuntimeError(
            "Selected checkpoint failed complete development restoration"
        )
    write(
        destination / "DEV_RESTORE.json",
        dict(passed=True, per_target_abs_difference=differences),
    )
    values, aspects = {}, 0
    for name in TARGETS:
        predictions = predictor.predict(load_target(data_root, name, "test"))
        write(destination / "predictions" / f"{name}.json", predictions)
        values[name] = metrics(predictions)
        aspects += len(predictions)
        write(
            destination / "PROGRESS.json",
            dict(checked_at=now(), completed_targets=list(values), aspects=aspects),
        )
    if aspects != TEST_ASPECTS:
        raise ValueError("Unexpected test cohort size")
    final = dict(
        passed=True,
        stage=result["stage"],
        seed=result["seed"],
        split=result["split"],
        aspects=aspects,
        Full=statistics.mean(v["rmse_va"] for v in values.values()),
        per_target_rmse_va={k: v["rmse_va"] for k, v in values.items()},
        per_target=values,
        precision="native Torch float32 two-decimal values",
        test_used_for_selection=False,
    )
    write(destination / "RESULT.json", final)
    return final
