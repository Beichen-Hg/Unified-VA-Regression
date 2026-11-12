# Reproducibility

## Environment and inputs

The tested environment is Linux with Python 3.11, PyTorch 2.9, and Transformers 4.56.1. Dependencies are pinned in [requirements.txt](../requirements.txt). Prepare a local Qwen3-8B-Base snapshot at revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` and official files as described in the [data guide](../data/README.md).

All stages share the single template in [config.py](../src/unified_va/config.py). Only text and aspect reach tokenization and model forward. IDs, VA labels, language labels, and domain labels are carried separately. Language/domain labels supervise only the Stage 2 training classifiers.

## Execution and checkpointing

Use one explicit device and a fresh output directory per run. Run each partition and seed independently. Stage 2 loads its own seed and partition's selected Stage 1 checkpoint, restores all 12 development targets, checks zero-adapter identity, and validates a scratch backward pass with zero optimizer updates. It then constructs a fresh model and optimizer for formal training.

Non-reentrant activation checkpointing preserves gradients through trainable decoder blocks 24–35 when earlier blocks are frozen. VA normalization uses only the training pool's float32 population moments. The global Torch RNG determines shuffled training order. Incomplete final accumulation windows retain the configured divisor of 16.

Checkpoints store every updated tensor and persistent buffer without quantization, together with the effective configuration, normalization, and base-model reference. Frozen tensors come from the pinned local snapshot. The run retains selected and final checkpoints, development predictions, and metrics. Writes enforce a 40 GiB free-space reserve.

## Evaluation and reporting

Evaluation restores the complete selected development cohort before testing the fixed 15 targets. Development and test predictions follow the same path: denormalize, clamp to [1, 9], then apply native Torch `(va * 100).round() / 100`. Checkpoint tensors retain their original precision.

Auxiliary decoding reports known-class coverage, accuracy, macro-F1, and confusion counts. Unknown labels are excluded from classification scores; they do not affect VA prediction. [examples/diagnostics.json](../examples/diagnostics.json) illustrates labelled diagnostic inputs. DP sensitivity re-scores saved development trajectories while leaving the selected model unchanged.

## Result provenance and tests

The released result files retain the completed manuscript measurements. This unified reference implementation standardizes inputs and checkpointing; full-scale 8B training has not been re-benchmarked. See the [result guide](../results/README.md).

Run `python -m pytest -q`. Offline CPU tests exercise complete training/evaluation on tiny backbones, real Transformers Qwen layer checkpointing, four-branch computations, upper-block gradients, metadata isolation, lossless save/restore, same-seed continuation, dry-run scripts, DP, and paired record-cluster statistics. No test downloads the pretrained model.
