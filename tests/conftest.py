from pathlib import Path
from types import SimpleNamespace
import copy
import json
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from unified_va import config, storage
from unified_va.models import regressor


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embed_tokens = nn.Embedding(64, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(36)])
        self.norm = nn.LayerNorm(8)
        self.checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        assert gradient_checkpointing_kwargs == {"use_reentrant": False}
        self.checkpointing = True

    def forward(self, input_ids, attention_mask, **kwargs):
        value = self.embed_tokens(input_ids)
        for layer in self.layers:
            # Each residual block has a real parameter path after frozen inputs.
            output = (
                checkpoint(layer, value, use_reentrant=False)
                if self.training and self.checkpointing
                else layer(value)
            )
            value = value + 0.03 * torch.tanh(output)
        return SimpleNamespace(last_hidden_state=self.norm(value))


class Tokenizer:
    def __init__(self):
        self.seen = []

    def __call__(self, text, **kwargs):
        self.seen.append(text)
        length = kwargs["max_length"]
        values = [ord(c) % 63 + 1 for c in text][:length]
        return dict(
            input_ids=torch.tensor([values + [0] * (length - len(values))]),
            attention_mask=torch.tensor(
                [[1] * len(values) + [0] * (length - len(values))]
            ),
        )


@pytest.fixture
def miniature(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(1337)
    base = TinyBackbone()
    original_template = config.template

    def profile(stage, split):
        value = original_template(stage, split)
        value["small_llm"]["torch_dtype"] = "float32"
        value["max_len"] = 24
        value["experiment"].update(
            train_aspects=24,
            dev_aspects=24,
            expected_updates=2 if stage == "stage2" else 6,
            development_events=[1, 2] if stage == "stage2" else list(range(1, 7)),
        )
        return value

    monkeypatch.setattr(config, "template", profile)

    def build(cfg):
        backbone = copy.deepcopy(base)
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        return regressor.VARegressor(backbone, cfg), Tokenizer()

    monkeypatch.setattr(regressor, "build", build)
    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 1024**3)
    )
    data = tmp_path / "official-data"
    for target in config.TARGETS:
        target_cfg = config.target_config(target, data)
        folder = Path(target_cfg["data_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        records = [
            dict(
                ID=str(i),
                Text="A clear screen and useful service " + str(i),
                Aspect_VA=[dict(Aspect="screen", VA=f"{4+i}#{5+0.3*i}")],
            )
            for i in range(2)
        ]
        for split in ("train", "dev", "test"):
            (folder / target_cfg[split + "_file"]).write_text(
                "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
            )
    return dict(
        root=tmp_path, data=data, model=tmp_path / "base", base=base, build=build
    )
