<div align="center">

<h1>Unified Multilingual and Multidomain<br>Valence–Arousal Regression</h1>

<p><b>Beichen Huang<sup>1</sup> · Zongxi Li<sup>1</sup> · Weiming Wang<sup>2</sup> · Haoran Xie<sup>1</sup> · Fu Lee Wang<sup>2</sup></b></p>
<p><sup>1</sup> Division of Artificial Intelligence, Lingnan University, Hong Kong SAR<br>
<sup>2</sup> School of Science and Technology, The Hong Kong Metropolitan University, Hong Kong SAR</p>

<p><a href="README.md"><b>English</b></a> &nbsp;|&nbsp; <a href="README_zh.md"><b>简体中文</b></a></p>

<p>
<img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.11+">
<img src="https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.9">
<img src="https://img.shields.io/badge/Backbone-Qwen3--8B--Base-7C3AED" alt="Qwen3-8B-Base">
<img src="https://img.shields.io/badge/Benchmark-15_Multilingual_Datasets-0891B2" alt="15 multilingual datasets">
<a href="https://github.com/Beichen-Hg/Unified-VA-Regression/actions/workflows/cpu-tests.yml"><img src="https://github.com/Beichen-Hg/Unified-VA-Regression/actions/workflows/cpu-tests.yml/badge.svg" alt="CI status"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-059669" alt="MIT License"></a>
</p>

<p><a href="#overview">Overview</a> &nbsp;·&nbsp; <a href="#main-results">Main Results</a> &nbsp;·&nbsp; <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp; <a href="#code-structure">Code Structure</a> &nbsp;·&nbsp; <a href="#reproducibility">Reproducibility</a> &nbsp;·&nbsp; <a href="#citation-and-license">Citation and License</a></p>
</div>

## Overview

A unified reference implementation for the IEEE TAFFC manuscript: one model for aspect-level valence–arousal regression across languages and domains.

1. **Shared foundation learning.** Stage 1 learns a shared representation and dual VA readout on 12 in-pool targets.
2. **Conditional residual specialization.** Stage 2 starts from the same seed's selected Stage 1 checkpoint and adds zero-initialized Global, Language, Domain, and Compound residuals at decoder layers 24–35. Inference uses text and aspect only; condition labels supervise an auxiliary training objective.
3. **Distribution-preserving selection.** DP ranks development checkpoints using mean error, tail risk, pseudo-OOD, spread, and correlation-floor terms.

![Two-stage architecture](assets/framework.svg)

Every stage uses the same input template:

```text
Predict valence and arousal for the aspect from the text.

Text: {text}
Aspect: {aspect}
```

## Main Results

Lower RMSE-VA is better. Both complementary partitions have 12 in-pool and 3 held-out targets and evaluate the same 15 targets / 19,977 aspects. Entries are mean ± sample SD across seeds 42, 13, and 7.

| Partition | Baseline | Stage 1 | Stage 2 | Δ (Stage 2 − Stage 1) | Paired 95% Bootstrap CI |
|:--|--:|--:|--:|--:|:--|
| Original | 1.6795 ± 0.0214 | 1.6274 ± 0.0159 | **1.2555 ± 0.0084** | −0.3719 | [−0.3938, −0.3505] |
| Alternative (Alt1) | 1.6425 ± 0.0025 | 1.5740 ± 0.0073 | **1.2118 ± 0.0127** | −0.3623 | [−0.3862, −0.3397] |

Intervals use 10,000 paired record-cluster resamples within targets, conditional on the fitted models. Seed SD separately describes training-seed variation. Negative Δ favors Stage 2.

> These are retained manuscript experiment measurements. The unified reference implementation standardizes inputs and gradient checkpointing; its full-scale 8B training has not been re-benchmarked.

[Stage/seed scores](results/stage_results.csv) · [Per-target V/A](results/per_target_results.csv) · [Baseline scores](results/baseline_results.csv) · [Paired intervals](results/paired_intervals.csv)

## Quick Start

### Step 1: Environment Installation

Tested on Linux with Python 3.11, PyTorch 2.9, and Transformers 4.56.1. Install the CUDA-compatible PyTorch build for your GPU.


```bash
git clone https://github.com/Beichen-Hg/Unified-VA-Regression.git
cd Unified-VA-Regression

conda create -n unified-va python=3.11
conda activate unified-va
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

### Step 2: Data & Base Model Preparation

Obtain the official data from [DimABSA2026](https://github.com/DimABSA/DimABSA2026) and a local [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) snapshot at revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. See the [data guide](data/README.md) for files and target membership.

```bash
export DATA_ROOT=/path/to/DimABSA2026/task-dataset
export MODEL_DIR=/path/to/Qwen3-8B-Base
```

```text
task-dataset/                       Qwen3-8B-Base/
├── track_a/subtask_1/...            ├── config.json
└── track_b/subtask_1/...            ├── tokenizer.json
                                   ├── model.safetensors.index.json
                                   └── model-*.safetensors
```

### Step 3: Training & Three-Seed Reproduction

Train one paired seed in a separate demo directory:

```bash
python scripts/train.py --stage stage1 --split original --seed 42 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --run-dir runs/demo/stage1 --device cuda:0

python scripts/train.py --stage stage2 --split original --seed 42 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --parent runs/demo/stage1 --run-dir runs/demo/stage2 --device cuda:0
```

Reproduce the paired three-seed Stage 1 → Stage 2 pipeline:

```bash
python scripts/reproduce.py --config configs/main_original.yaml \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir runs/original --device cuda:0

python scripts/reproduce.py --config configs/main_alt1.yaml \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir runs/alt1 --device cuda:0
```

The Baseline uses the same training entry:

```bash
python scripts/train.py --stage baseline --split original --seed 42 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --run-dir runs/original/42/baseline --device cuda:0
```

Use one GPU per run and fresh output directories. `--seed 42` selects one seed; `--dry-run` only prints commands. Each partition trains its own Stage 1 parents.

### Step 4: Evaluation, Diagnostics & Custom Inference

Evaluation restores all 12 development targets before testing all 15 targets. Auxiliary labels are joined only after the VA forward, never supplied as inference conditions.

```bash
python scripts/evaluate.py --run-dir runs/demo/stage2 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" --device cuda:0

python scripts/analyze.py diagnose \
  --checkpoint runs/demo/stage2/checkpoints/stage2.pt \
  --model-dir "$MODEL_DIR" --input examples/diagnostics.json \
  --output condition_diagnostics.json --device cuda:0

python scripts/analyze.py dp-sensitivity \
  --run-dir runs/demo/stage2 --output dp_sensitivity.json

python scripts/predict.py \
  --checkpoint runs/demo/stage2/checkpoints/stage2.pt \
  --model-dir "$MODEL_DIR" --input examples/inputs.json \
  --output predictions.json --device cuda:0
```

Synthetic input and illustrative output format for `examples/inputs.json`:

```json
[
  {"id":"demo","aspect_index":0,"text":"The screen is bright, but the battery is disappointing.","aspect":"screen"},
  {"id":"demo","aspect_index":1,"text":"The screen is bright, but the battery is disappointing.","aspect":"battery"}
]
```

```json
[
  {
    "id": "demo",
    "aspect": "screen",
    "aspect_index": 0,
    "v_pred": 7.12,
    "a_pred": 5.46
  },
  {
    "id": "demo",
    "aspect": "battery",
    "aspect_index": 1,
    "v_pred": 2.34,
    "a_pred": 6.78
  }
]
```

```bash
python scripts/analyze.py compare --split original \
  --left runs/original/{42,13,7}/stage1/evaluation \
  --right runs/original/{42,13,7}/stage2/evaluation \
  --output stage_comparison.json
```

## Code Structure

```text
configs/                         Two three-seed experiment recipes
scripts/                         Compatible train/evaluate/predict/analyze entry points
src/unified_va/
├── data/                        Official JSONL loading and aspect expansion
├── models/
│   ├── regressor.py             Backbone, pooling, and gradient checkpointing
│   ├── heads.py                 VA heads and training-only classifiers
│   └── conditional_adapter.py   Four-branch conditional residual adapter
├── training/
│   ├── trainer.py               One AdamW/Huber training loop
│   └── dp_selection.py          Distribution-preserving model selection
├── evaluation/                  Native-Torch reporting and complete evaluation
├── configs/                     Six complete hyperparameter profiles
├── storage.py                   Lossless updated-tensor checkpoints
└── paired_statistics.py         Paired record-cluster bootstrap
results/                         Archived manuscript measurements
tests/                           Offline CPU contracts and pipeline tests
```

| Paper component | Source |
|:--|:--|
| Unified Prompt and partitions | [config.py](src/unified_va/config.py) |
| Backbone, pooling, upper-block gradients | [regressor.py](src/unified_va/models/regressor.py) |
| Four conditional residual branches | [conditional_adapter.py](src/unified_va/models/conditional_adapter.py) |
| VA readout and training-only classifiers | [heads.py](src/unified_va/models/heads.py) |
| Huber/AdamW training | [trainer.py](src/unified_va/training/trainer.py) |
| DP development selection | [dp_selection.py](src/unified_va/training/dp_selection.py) |
| Prediction rounding and metrics | [metrics.py](src/unified_va/evaluation/metrics.py) |
| Record-cluster paired bootstrap | [paired_statistics.py](src/unified_va/paired_statistics.py) |

## Reproducibility

[Experiments and hyperparameters](docs/experiments.md) · [Reproducibility conventions](docs/reproducibility.md) · [Six profiles](configs/README.md)

Offline CPU tests cover actual training loops, non-reentrant checkpoint backward, input isolation, lossless checkpoints, same-seed continuation, DP, and record-cluster statistics:

```bash
python -m pytest -q
```

## Citation and License

The manuscript is under review. A citation is available in [CITATION.bib](CITATION.bib):

```bibtex
@article{huang2026unifiedva,
  title   = {Unified Multilingual and Multidomain Valence--Arousal Regression},
  author  = {Huang, Beichen and Li, Zongxi and Wang, Weiming and Xie, Haoran and Wang, Fu Lee},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Code is released under the [MIT License](LICENSE). The pretrained model and datasets retain their respective licenses.
