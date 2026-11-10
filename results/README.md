# Experiment results

[stage_results.csv](stage_results.csv) records the completed Stage 1/2 experiments on two partitions, with three paired seeds each. [per_target_results.csv](per_target_results.csv) contains target-level VA/V/A RMSE values. The complete machine-readable summary is in [mainline_summary.json](../src/unified_va/resources/mainline_summary.json).

Result files contain aggregate scores, not source text or per-aspect gold labels. The original baseline summary in the JSON is recomputed from retained historical predictions; its historical weights are unavailable. The included CSV tables retain the completed Stage 1/2 snapshot. Additional Baseline and auxiliary diagnostic result tables are not included here.

The code exposes commands to train the Baseline and both complete stages. Reported stage differences concern the complete training procedures, including prompt and effective optimization-path differences described in [reproducibility.md](../docs/reproducibility.md). Statistics are seed-scalar summaries, not prediction ensembles.
