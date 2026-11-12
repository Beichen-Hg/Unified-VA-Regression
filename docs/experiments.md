# Experiments and hyperparameters

## Complementary partitions

Both partitions use 12 in-pool training/development targets and 3 held-out targets. Evaluation covers the same 15 targets and 19,977 aspects. Ordered target definitions are centralized in [config.py](../src/unified_va/config.py).

| Partition | Train aspects | Dev aspects | Held-out targets |
|:--|--:|--:|:--|
| Original | 40,126 | 3,532 | `eng_environmental_protection`, `tat_restaurant`, `swa_politics` |
| Alternative (Alt1) | 38,687 | 3,372 | `rus_restaurant`, `ukr_restaurant`, `zho_finance` |

## Training

| Setting | Baseline | Stage 1 | Stage 2 |
|:--|:--|:--|:--|
| Backbone initialization | Qwen3-8B-Base | Qwen3-8B-Base | Same-seed selected Stage 1 |
| Pooling | Last token | Gated last/mean | Gated last/mean |
| VA head | Linear | Shared 512 → separate 256 → scalar | Shared 512 → separate 256 → scalar |
| Epochs | 3 | 3 | 1 |
| Learning rate | 1.5e-4 | 1.5e-4 | 5e-5 |
| Effective batch | 16 | 16 | 16 |
| Loss | Huber | Huber | Huber + 0.10 × auxiliary CE |
| Selection | Macro dev RMSE-VA | Macro dev RMSE-VA | DP |

Common settings: microbatch 1, gradient accumulation 16, maximum sequence length 192, AdamW, weight decay 0.01, gradient clipping 1.0, cosine schedule with 10% warmup, and feature/head dropout 0.10. Decoder layers 0–23 are frozen; layers 24–35 and the final normalization/readout are trainable. All stages use non-reentrant gradient checkpointing.

| Partition | Baseline / Stage 1 updates | Stage 2 updates | Seeds |
|:--|--:|--:|:--|
| Original | 7,524 | 2,508 | 42, 13, 7 |
| Alternative (Alt1) | 7,254 | 2,418 | 42, 13, 7 |

Exact development checkpoints are listed in the [configuration guide](../configs/README.md).

Stage 2 inserts 12 zero-initialized four-branch adapters with bottleneck width 64, condition embedding dimension 16, and dropout 0.10. Each layer infers a 32-dimensional concatenated language/domain condition from masked-mean hidden states through an MLP of width 128. The Global branch is unconditioned; Language, Domain, and Compound residuals use tanh condition scaling and sigmoid gates. Training-only classifiers decode layer-averaged conditions; the sum of language and domain cross-entropy losses has weight 0.10.

## Distribution-preserving checkpoint selection

Let R be mean development RMSE-VA over the 12 in-pool targets. The DP score is R plus the weighted terms below.

| Term | Definition | Weight |
|:--|:--|--:|
| Tail | Mean of the 4 largest target RMSE-VA values | 0.30 |
| Pseudo-OOD | Mean over configured targets present in the development pool | 0.25 |
| Pseudo-OOD tail | Mean of the 2 largest errors in that subset | 0.15 |
| V spread | Mean absolute predicted/gold population-SD gap | 0.05 |
| A spread | Mean absolute predicted/gold population-SD gap | 0.15 |
| CCC floor | Mean max(0, 0.40 − target CCC-macro) | 0.25 |
| PCC floor | Mean max(0, 0.45 − target PCC-macro) | 0.10 |

The pseudo-OOD subset is `deu_politics`, `ukr_restaurant`, `jpn_finance`, `pcm_politics`, and `zho_environmental_protection`, intersected with the current development pool. Checkpoints minimize DP; ties select the earlier update. Test scores never enter selection.

## Metrics and intervals

Per-target RMSE-VA is sqrt(mean(error_V² + error_A²)). Overall is the equal-weight mean over 15 targets. In-pool and held-out results average their respective target subsets. Seed summaries average fitted-model scalar scores rather than predictions.

Paired 95% confidence intervals use 10,000 record-cluster bootstrap draws within each target. Draws are shared between compared systems, and all aspects of a record stay together. Intervals condition on the fitted models; sample SD separately describes training-seed variation. The unchanged implementation is [paired_statistics.py](../src/unified_va/paired_statistics.py).
