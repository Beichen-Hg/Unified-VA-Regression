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

<p><a href="#方法概览">方法概览</a> &nbsp;·&nbsp; <a href="#主实验结果">主实验结果</a> &nbsp;·&nbsp; <a href="#快速开始">快速开始</a> &nbsp;·&nbsp; <a href="#代码结构">代码结构</a> &nbsp;·&nbsp; <a href="#复现与测试">复现与测试</a> &nbsp;·&nbsp; <a href="#引用与许可">引用与许可</a></p>
</div>

## 方法概览

本仓库提供 IEEE TAFFC 稿件的统一参考实现：一个模型完成多语言、多领域的方面级效价（Valence）与唤醒度（Arousal）回归。

1. **共享基础学习**：Stage 1 在 12 个池内目标上学习共享表示与双分支 VA 读出。
2. **条件残差特化**：Stage 2 从同 seed 选中的 Stage 1 权重出发，在第 24–35 层加入零初始化的 Global、Language、Domain、Compound 四分支残差。推理仅依赖文本与 aspect，语言/领域标签只用于训练辅助监督。
3. **分布保持选模**：DP 综合目标均值、尾部误差、pseudo-OOD、输出分布与相关性，在开发集上选择 checkpoint。

![两阶段框架](assets/framework.svg)

所有阶段统一使用以下输入模板：

```text
Predict valence and arousal for the aspect from the text.

Text: {text}
Aspect: {aspect}
```

## 主实验结果

RMSE-VA 越低越好。每个划分均含 12 个池内与 3 个留出目标，测试覆盖相同的 15 个目标、19,977 个 aspect。表中为 seed 42、13、7 的均值 ± 样本 SD。

| 划分 | Baseline | Stage 1 | Stage 2 | Δ（Stage 2 − Stage 1） | 95% 配对 Bootstrap CI |
|:--|--:|--:|--:|--:|:--|
| Original | 1.6795 ± 0.0214 | 1.6274 ± 0.0159 | **1.2555 ± 0.0084** | −0.3719 | [−0.3938, −0.3505] |
| Alternative (Alt1) | 1.6425 ± 0.0025 | 1.5740 ± 0.0073 | **1.2118 ± 0.0127** | −0.3623 | [−0.3862, −0.3397] |

区间采用目标内 record 整簇配对 Bootstrap（10,000 次），条件于已训练模型；seed SD 单独描述训练种子的波动。Δ 为负表示 Stage 2 误差更低。

> 表中数值为保留的论文实验记录。当前统一实现标准化了输入与梯度检查点，尚未重新完成 8B 规模训练基准。

[阶段逐 seed 结果](results/stage_results.csv) · [逐目标 V/A](results/per_target_results.csv) · [Baseline 明细](results/baseline_results.csv) · [完整配对区间](results/paired_intervals.csv)

## 快速开始

### Step 1：安装环境

测试环境为 Linux、Python 3.11、PyTorch 2.9 和 Transformers 4.56.1。请为 GPU 安装匹配的 CUDA 版 PyTorch。


```bash
git clone https://github.com/Beichen-Hg/Unified-VA-Regression.git
cd Unified-VA-Regression

conda create -n unified-va python=3.11
conda activate unified-va
python -m pip install -r requirements.txt
python -m pip install -e '.[test]'
```

### Step 2：准备数据与基础模型

从 [DimABSA2026 官方仓库](https://github.com/DimABSA/DimABSA2026)准备数据，使用 [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) 的固定 revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`。文件布局与划分见[数据指南](data/README.md)。

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

### Step 3：训练与三 seed 复现

单 seed 阶段训练（使用独立的 demo 目录）：

```bash
python scripts/train.py --stage stage1 --split original --seed 42 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --run-dir runs/demo/stage1 --device cuda:0

python scripts/train.py --stage stage2 --split original --seed 42 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --parent runs/demo/stage1 --run-dir runs/demo/stage2 --device cuda:0
```

三 seed 完整阶段流程：

```bash
python scripts/reproduce.py --config configs/main_original.yaml \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir runs/original --device cuda:0

python scripts/reproduce.py --config configs/main_alt1.yaml \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --output-dir runs/alt1 --device cuda:0
```

Baseline 使用同一入口：

```bash
python scripts/train.py --stage baseline --split original --seed 42 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --run-dir runs/original/42/baseline --device cuda:0
```

每次运行使用一个 GPU 和新的输出目录。`--seed 42` 选择单个 seed，`--dry-run` 只打印命令。两划分使用各自的 Stage 1 父模型。

### Step 4：评估、诊断与自定义推理

评估先恢复全部 12 个开发目标，再测试 15 个目标。条件诊断的标签在 VA 前向后才关联，不能作为推理路由输入。

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

`examples/inputs.json` 中的合成输入与输出格式如下（数值仅用于说明格式）：

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

## 代码结构

```text
configs/                         两个三 seed 实验配方
scripts/                         训练、评估、推理与分析入口
src/unified_va/
├── data/                        官方 JSONL 读取与 aspect 展开
├── models/
│   ├── regressor.py             Backbone、池化与梯度检查点
│   ├── heads.py                 VA 读出与训练辅助分类器
│   └── conditional_adapter.py   四分支条件残差 Adapter
├── training/
│   ├── trainer.py               统一 AdamW/Huber 训练循环
│   └── dp_selection.py          分布保持选模
├── evaluation/                  原生 Torch 输出与完整评测
├── configs/                     六份完整超参数配置
├── storage.py                   更新张量无损检查点
└── paired_statistics.py         记录簇配对 Bootstrap
results/                         已完成的论文实验数据
tests/                           离线 CPU 契约与全流程测试
```

| 论文模块 | 实现 |
|:--|:--|
| 统一 Prompt 与双划分 | [config.py](src/unified_va/config.py) |
| Backbone、池化与上层梯度 | [regressor.py](src/unified_va/models/regressor.py) |
| 四分支条件残差 | [conditional_adapter.py](src/unified_va/models/conditional_adapter.py) |
| VA 读出与训练辅助分类 | [heads.py](src/unified_va/models/heads.py) |
| Huber/AdamW 训练 | [trainer.py](src/unified_va/training/trainer.py) |
| DP 开发选模 | [dp_selection.py](src/unified_va/training/dp_selection.py) |
| 统一推理舍入与指标 | [metrics.py](src/unified_va/evaluation/metrics.py) |
| 记录簇配对 Bootstrap | [paired_statistics.py](src/unified_va/paired_statistics.py) |

## 复现与测试

[实验与超参数](docs/experiments.md) · [复现约定](docs/reproducibility.md) · [六份配置](configs/README.md)

CPU 测试不下载模型，覆盖真实训练循环、非重入检查点反向传播、输入隔离、checkpoint 无损恢复、Stage 1 → Stage 2 接续、DP 和记录簇统计：

```bash
python -m pytest -q
```

## 引用与许可

论文目前处于审稿阶段。引用信息见 [CITATION.bib](CITATION.bib)：

```bibtex
@article{huang2026unifiedva,
  title   = {Unified Multilingual and Multidomain Valence--Arousal Regression},
  author  = {Huang, Beichen and Li, Zongxi and Wang, Weiming and Xie, Haoran and Wang, Fu Lee},
  journal = {IEEE Transactions on Affective Computing},
  year    = {2026},
  note    = {Manuscript under review}
}
```

代码采用 [MIT License](LICENSE)。基础模型和数据集分别遵循各自的许可条款。
