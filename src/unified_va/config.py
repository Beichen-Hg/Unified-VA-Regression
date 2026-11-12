"""One input template, six profiles, and two ordered 12/3 partitions."""

from pathlib import Path
import copy
from .io import read, write

PACKAGE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-8B-Base"
REVISION = "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
PROMPT = "Predict valence and arousal for the aspect from the text.\n\nText: {text}\nAspect: {aspect}"
ORIGINAL_TRAIN = (
    "eng_restaurant",
    "deu_politics",
    "eng_laptop",
    "zho_laptop",
    "ukr_restaurant",
    "zho_restaurant",
    "jpn_hotel",
    "jpn_finance",
    "rus_restaurant",
    "zho_finance",
    "zho_environmental_protection",
    "pcm_politics",
)
ORIGINAL_HELD = ("eng_environmental_protection", "tat_restaurant", "swa_politics")
TARGETS = ORIGINAL_TRAIN + ORIGINAL_HELD
ALT_HELD = ("rus_restaurant", "ukr_restaurant", "zho_finance")


def split_targets(split):
    if split not in ("original", "alt1"):
        raise ValueError("Choose original or alt1")
    held = ORIGINAL_HELD if split == "original" else ALT_HELD
    return tuple(t for t in TARGETS if t not in held), held


def target_config(target, data_root):
    if target not in TARGETS:
        raise ValueError("Unregistered target: " + target)
    language, domain = target.split("_", 1)
    track = "B" if domain in ("politics", "environmental_protection") else "A"
    suffix = "alltasks" if domain in ("restaurant", "laptop", "hotel") else "task1"
    return dict(
        lang=language,
        domain=domain,
        track=track,
        data_dir=str(
            Path(data_root).resolve()
            / f"track_{track.lower()}"
            / "subtask_1"
            / language
        ),
        train_file=f"{target}_train_{suffix}.jsonl",
        dev_file=f"{target}_dev_task1.jsonl",
        test_file=f"{target}_test_task1.jsonl",
    )


def template(stage, split):
    if stage not in ("baseline", "stage1", "stage2"):
        raise ValueError("Choose baseline, stage1, or stage2")
    split_targets(split)
    return read(PACKAGE / "configs" / f"{stage}_{split}.json")


def validate_effective(cfg):
    record = cfg["experiment"]
    expected = template(record["stage"], record["split"])
    if set(cfg) - set(expected) or set(record) - set(expected["experiment"]) - {
        "run_dir",
        "data_root",
        "parent_run",
    }:
        raise ValueError("Unknown configuration field")
    for key, value in expected.items():
        if key in ("seed", "device", "experiment", "small_llm", "macro_dev_eval"):
            continue
        if cfg.get(key) != value:
            raise ValueError("Registered hyperparameter changed: " + key)
    for key, value in expected["small_llm"].items():
        if key != "model_name" and cfg["small_llm"].get(key) != value:
            raise ValueError("Registered model/input setting changed: " + key)
    if cfg["small_llm"]["prompt_template"] != PROMPT:
        raise ValueError("All stages use the same text/aspect template")
    if cfg["macro_dev_eval"] != expected["macro_dev_eval"]:
        raise ValueError("Development selection changed")
    for key, value in expected["experiment"].items():
        if record.get(key) != value:
            raise ValueError("Registered training budget changed: " + key)
    return True


def prepare(
    stage, split, seed, data_root, model_dir, run_dir, device="cuda:0", parent=None
):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("Specify cpu or cuda:N")
    if stage == "stage2" and not parent:
        raise ValueError("Stage2 requires the selected same-seed Stage1 run")
    cfg = copy.deepcopy(template(stage, split))
    cfg["seed"], cfg["device"] = seed, device
    cfg["small_llm"]["model_name"] = str(Path(model_dir).resolve())
    run = Path(run_dir).resolve()
    if (run / "STARTED.json").exists() or (run / "RESULT.json").exists():
        raise FileExistsError("Choose a fresh run directory")
    cfg["experiment"].update(
        run_dir=str(run),
        data_root=str(Path(data_root).resolve()),
        parent_run=str(Path(parent).resolve()) if parent else None,
    )
    validate_effective(cfg)
    destination = run / "config.json"
    if destination.exists() and read(destination) != cfg:
        raise ValueError("A different plan already exists in this directory")
    write(destination, cfg)
    return cfg
