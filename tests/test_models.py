import pytest
import torch
from unified_va.config import template, PROMPT
from unified_va.models.conditional_adapter import ConditionalAdapter
from unified_va.models.regressor import VARegressor
from unified_va.models.heads import ConditionClassifier
from unified_va.evaluation.metrics import report_values
from unified_va.inference import input_rows, predict_rows
from conftest import TinyBackbone, Tokenizer


def test_four_branch_zero_identity_masked_conditions_and_nonzero_reference():
    torch.manual_seed(42)
    adapter = ConditionalAdapter(
        8, bottleneck_dim=5, embed_dim=4, dropout=0.0, soft_condition_hidden=6
    )
    value, mask = torch.randn(2, 5, 8), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert torch.equal(adapter(value, mask), value)
    expected = adapter.condition_embeddings[0].clone()
    changed = value.clone()
    changed[0, 3:] = 999
    adapter(changed, mask)
    assert torch.equal(adapter.condition_embeddings[0], expected)
    for branch in ("shared", "lang", "domain", "comp"):
        torch.nn.init.normal_(getattr(adapter, branch + "_up").weight, std=0.1)
    actual = adapter(value, mask)
    conditions = torch.cat(adapter.condition_embeddings, -1)
    gates = adapter.gate_proj(conditions).sigmoid()
    h = adapter.pre_norm(value)
    delta = adapter.shared_up(torch.nn.functional.gelu(adapter.shared_down(h)))
    for i, branch in enumerate(("lang", "domain", "comp")):
        embedding = conditions if branch == "comp" else adapter.condition_embeddings[i]
        scaled = torch.nn.functional.gelu(getattr(adapter, branch + "_down")(h))
        scaled = scaled * getattr(adapter, branch + "_scale")(
            embedding
        ).tanh().unsqueeze(1)
        delta = delta + gates[:, i, None, None] * getattr(adapter, branch + "_up")(
            scaled
        )
    assert torch.equal(actual, value + delta)


@pytest.mark.parametrize("stage", ["baseline", "stage1", "stage2"])
def test_nonreentrant_upper_gradients_and_frozen_lower_blocks(stage):
    cfg = template(stage, "original")
    backbone = TinyBackbone()
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = VARegressor(backbone, cfg)
    out = model(torch.tensor([[1, 2, 3, 4]]), torch.ones(1, 4, dtype=torch.long))
    loss = out.square().mean()
    if stage == "stage2":
        loss += model.auxiliary.loss(
            model.condition_embeddings(),
            ["eng"],
            ["restaurant"],
            cfg["soft_condition_label_supervision"],
        )
    loss.backward()
    assert model.backbone.layers[24].weight.grad is not None
    assert torch.count_nonzero(model.backbone.layers[24].weight.grad)
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.backbone.layers[23].parameters()
    )
    if stage == "stage2":
        assert len(model.backbone_conditional_adapters) == 12
        assert model.auxiliary.lang_head[1].weight.grad is not None


def test_metadata_and_gold_cannot_enter_prediction_or_tokenization():
    cfg = template("stage2", "original")
    model, tokenizer = VARegressor(TinyBackbone(), cfg), Tokenizer()
    norm = dict(v_mean=5.0, a_mean=5.0, v_std=1.0, a_std=1.0)
    rows = [
        dict(
            text="Good service",
            aspect="service",
            lang="eng",
            domain="restaurant",
            v=2,
            a=9,
        )
    ]
    expected = predict_rows(model, tokenizer, rows, cfg, norm, torch.device("cpu"))
    changed = [dict(rows[0], lang="other", domain="other", v=9, a=1)]
    actual = predict_rows(model, tokenizer, changed, cfg, norm, torch.device("cpu"))
    assert [(r["v_pred"], r["a_pred"]) for r in expected] == [
        (r["v_pred"], r["a_pred"]) for r in actual
    ]
    assert set(input_rows(rows)[0]) == {"text", "aspect"}
    assert set(tokenizer.seen) == {PROMPT.format(text="Good service", aspect="service")}
    with pytest.raises(TypeError):
        model(
            torch.ones(1, 3, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.long),
            lang=["eng"],
        )


def test_auxiliary_loss_is_exactly_weighted_and_ignores_unknown_labels():
    head = ConditionClassifier(["eng"], ["restaurant"], embed_dim=4, dropout=0.0)
    embeddings = torch.randn(2, 4), torch.randn(2, 4)
    logits = head(*embeddings)
    settings = dict(loss_weight=0.1, lang_weight=1.0, domain_weight=1.0)
    actual = head.loss(
        embeddings, ["eng", "unknown"], ["restaurant", "unknown"], settings
    )
    expected = 0.1 * sum(
        torch.nn.functional.cross_entropy(v[:1], torch.tensor([1])) for v in logits
    )
    assert torch.equal(actual, expected)


def test_native_rounding_and_clamping_are_the_only_output_path():
    values = torch.tensor([[6.483624458312988, 6.045000076293945], [0.0, 12.0]])
    result = report_values(values, dict(v_mean=0.0, a_mean=0.0, v_std=1.0, a_std=1.0))
    assert torch.equal(result, (values.float().clamp(1, 9) * 100).round() / 100)
    assert round(result[0, 1].item(), 2) == 6.04
    assert result[1].tolist() == [1.0, 9.0]


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_real_transformers_qwen_checkpointing_without_downloads(stage):
    from transformers import Qwen3Config, Qwen3Model

    cfg = template(stage, "original")
    backbone = Qwen3Model(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=36,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = VARegressor(backbone, cfg).train()
    loss = (
        model(torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.long))
        .square()
        .mean()
    )
    if stage == "stage2":
        loss += model.auxiliary.loss(
            model.condition_embeddings(),
            ["eng"],
            ["restaurant"],
            cfg["soft_condition_label_supervision"],
        )
    loss.backward()
    assert backbone.layers[24].self_attn.q_proj.weight.grad is not None
    assert torch.count_nonzero(backbone.layers[24].self_attn.q_proj.weight.grad)
    if stage == "stage2":
        assert torch.count_nonzero(model.auxiliary.lang_head[1].weight.grad)
        assert torch.count_nonzero(
            model.backbone_conditional_adapters[0].soft_condition_proj[0].weight.grad
        )


def test_local_transformers_factory_and_bfloat16_backward(tmp_path):
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
    from unified_va.models.regressor import build

    backbone = Qwen3Model(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=36,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    backbone.save_pretrained(tmp_path)
    tokens = FastTokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "screen": 2}, unk_token="[UNK]")
    )
    tokens.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokens, unk_token="[UNK]", pad_token="[PAD]"
    )
    tokenizer.save_pretrained(tmp_path)
    cfg = template("stage1", "original")
    cfg["small_llm"]["model_name"] = str(tmp_path)
    model, actual_tokenizer = build(cfg)
    batch = actual_tokenizer(
        PROMPT.format(text="good", aspect="screen"), return_tensors="pt"
    )
    prediction = model(batch["input_ids"], batch["attention_mask"])
    prediction.square().mean().backward()
    assert prediction.shape == (1, 2)
    assert next(model.backbone.parameters()).dtype == torch.bfloat16
    assert torch.count_nonzero(model.backbone.layers[24].self_attn.q_proj.weight.grad)
    assert all(p.grad is None for p in model.backbone.layers[23].parameters())
