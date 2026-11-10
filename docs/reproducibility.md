# Implementation and reproducibility

## Environment and inputs

The reference environment is Linux/Python 3.11, PyTorch 2.9.0 with CUDA 12.8 and Transformers 4.56.1. Complete versions are in `requirements.txt`. The local Qwen3-8B-Base revision is `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.

Use one process per run. `--device cuda:0` selects the logical CUDA device; allocation to physical hardware can be supplied through `CUDA_VISIBLE_DEVICES` or a scheduler. Dataset lookup uses the explicit `--data-root`; unset the legacy `DIMABSA2026_DATA_DIR` variable to avoid overriding it.

## Stage implementations

`src/unified_va/engines/` contains the stage-specific implementations. Each runs in a separate process, allowing its original `src` imports and model-construction order to remain consistent. Public scripts provide one interface over these implementations.

Baseline uses last-token pooling and a linear head. Stage 1 uses gated last/mean pooling and a dual-branch MLP. Stage 2 continues its pooling/readout and attaches factorized residual adapters after decoder layers 24–35. The condition representation consists of two continuous 16-dimensional vectors computed from input hidden states. A language/domain auxiliary classifier is trained with weight 0.1; VA inference uses the continuous vectors directly.

Each residual branch follows pre-normalization, down projection, GELU, conditional tanh scaling where applicable, dropout, up projection and branch gating. The shared branch has no conditional scale. The composition condition concatenates the language and domain vectors. Up-projection weights and biases are initialized to zero.

## Trainable paths and input templates

Baseline and Stage 1 retain reentrant gradient checkpointing without an input-gradient hook. Under this execution path, upper-block `requires_grad` flags do not imply an effective update: the reference Stage 1 updates final normalization, pooling and readout, while its upper-block gradients are absent. First-step gradient records are saved with each run. Stage 2 has an effective upper-block, adapter and readout gradient path; it is not adapter-only tuning.

Baseline, Stage 1 and Stage 2 retain their respective prompt templates, available in the complete profiles under `src/unified_va/configs/`. Stage 1 uses a longer foundation prompt and Stage 2 uses a short regression prompt. Stage comparisons evaluate the complete procedure. Zero-adapter identity checks use the same Stage 2 prompt on both S weights and the initialized T model.

## Data order and optimization

The optimizer is AdamW with Huber VA loss, batch size 1, gradient accumulation 16, BF16, maximum length 192, weight decay 0.01 and cosine scheduling with 0.1 warmup. Baseline/Stage 1 use learning rate 1.5e-4; Stage 2 uses 5e-5. The source global-Torch-RNG shuffle is retained. Last windows contain 14 or 15 examples and retain the accumulation divisor 16.

Normalization and condition vocabularies are computed from the chosen partition's training examples. The alternative partition trains its own foundation models.

## Selection and prediction precision

Baseline/Stage 1 select the minimum 12-target macro development RMSE. Stage 2 uses the distribution-preserving selector, including worst-target, pseudo-OOD, spread and correlation-floor penalties. Its fixed coefficients are in the complete Stage 2 profiles. Selection never reads test scores.

Stage 2 development/DP selection retains Python two-decimal rounding. Final reporting uses native float32 Torch `(va * 100).round() / 100` after denormalization and clamping. Stage 1/Baseline also use native Torch reporting. The two paths can differ at half-cent values; the precision distinction is covered by a CPU regression test.

## Checkpoints and outputs

Checkpoints contain all updated tensors, normalization, effective configuration and the pinned base reference. Frozen base parameters are restored from the local base snapshot. The Stage 1 → Stage 2 bridge copies identical tensors without an optimization step or quantization.

Each run saves configuration, training-data counts, first-step gradients, development predictions/metrics, selected and final checkpoints. Stage 2 additionally checks parent development restoration, zero-residual identity, input metadata perturbations and an independent scratch backward pass before formal training.

Allow approximately 35 GiB of working checkpoint space per Stage 2 run, plus model/data storage and the 40 GiB free-space reserve enforced at writes. A completed run retains selected/final weights and all development predictions/metrics; other checkpoints created within that run are retired. Use fresh run directories for new experiments.

## Validation scope

The recorded CPU checks exercise the actual training loops with tiny local backbones, lossless save/restore, identical-tensor bridging, input boundaries and paired statistics. Existing selected 8B Stage 1/2 checkpoints have also been loaded on CPU with synthetic inputs. The script layout and packaged interface have not been validated by repeating the full 8B training/evaluation cohort; the recorded result tables refer to the completed experiments described in `results/README.md`.
