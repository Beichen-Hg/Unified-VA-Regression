# Manuscript experiment results

| File | Contents |
|:--|:--|
| [stage_results.csv](stage_results.csv) | Two partitions × Stage 1/Stage 2 × 3 seeds × Overall/in-pool/held-out |
| [per_target_results.csv](per_target_results.csv) | 90 seed/target rows with V, A, and VA RMSE |
| [baseline_results.csv](baseline_results.csv) | 90 completed Baseline seed/target scores |
| [paired_intervals.csv](paired_intervals.csv) | Saved Stage 2 − Stage 1 record-cluster bootstrap intervals |
| [mainline_summary.json](../src/unified_va/resources/mainline_summary.json) | Machine-readable archived stage summaries |

These files archive the completed three-seed (42, 13, 7) Qwen3-8B evaluation records and paired bootstrap summaries reported in the manuscript across the Original and Alt1 partitions.

Scores average target RMSE values within each seed and then summarize seed scalars. They are not prediction ensembles. Confidence intervals use 10,000 within-target record-cluster paired resamples and condition on the fitted models. Training-seed sample SD is reported separately. Obtain raw corpus text and labels from the official benchmark.
