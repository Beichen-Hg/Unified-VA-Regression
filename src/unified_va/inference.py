"""Text/aspect inference; identities and labels are associated after forward."""

from contextlib import nullcontext
from pathlib import Path
import copy
import torch
from .data import loader
from .evaluation.metrics import report_values
from .models import regressor
from . import storage


def autocast(cfg, device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if cfg["amp"] and device.type == "cuda"
        else nullcontext()
    )


def input_rows(rows):
    result = []
    for row in rows:
        if not isinstance(row.get("text"), str) or not isinstance(
            row.get("aspect"), str
        ):
            raise ValueError("Each input needs string text and aspect fields")
        result.append(dict(text=row["text"], aspect=row["aspect"]))
    return result


def associate(predictions, rows):
    if len(predictions) != len(rows):
        raise ValueError("Input/prediction count differs")
    result = []
    for index, (prediction, row) in enumerate(zip(predictions, rows)):
        current = dict(
            id=str(row.get("id", index)),
            aspect=row["aspect"],
            aspect_index=int(row.get("aspect_index", 0)),
            **prediction,
        )
        for key in ("v", "a"):
            if row.get(key) is not None:
                current[key + "_true"] = float(row[key])
        if "target" in row:
            current["target"] = row["target"]
        elif row.get("lang") and row.get("domain"):
            current["target"] = row["lang"] + "_" + row["domain"]
        result.append(current)
    return result


def predict_rows(model, tokenizer, rows, cfg, norm, device, *, observer=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader(input_rows(rows), tokenizer, cfg):
            with autocast(cfg, device):
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                if observer is not None:
                    observer(model)
            values = report_values(output, norm).cpu().tolist()
            predictions.extend(dict(v_pred=v, a_pred=a) for v, a in values)
    return associate(predictions, rows)


class Predictor:
    def __init__(self, checkpoint, model_dir, device="cuda:0"):
        self.device = torch.device(device)
        value = storage.load(checkpoint)
        self.cfg = copy.deepcopy(value["config"])
        self.cfg["small_llm"]["model_name"] = str(Path(model_dir).resolve())
        self.cfg["device"] = device
        self.stage = self.cfg["experiment"]["stage"]
        torch.manual_seed(self.cfg["seed"])
        self.model, self.tokenizer = regressor.build(self.cfg)
        self.model.to(self.device)
        self.norm = storage.restore(self.model, value, self.cfg)
        self.auxiliary = self.model.auxiliary
        self.model.eval()

    def predict(self, rows, *, development=False, observer=None):
        return predict_rows(
            self.model,
            self.tokenizer,
            rows,
            self.cfg,
            self.norm,
            self.device,
            observer=observer,
        )
