# Configuration guide

[main_original.yaml](main_original.yaml) and [main_alt1.yaml](main_alt1.yaml) execute Stage 1 → Stage 2 with seeds 42, 13, and 7. Use `scripts/reproduce.py --config <recipe> --data-root <data> --model-dir <model> --output-dir <runs>`; `--seed` selects one seed and `--dry-run` prints commands without creating runs. Train Baseline through `scripts/train.py --stage baseline`.

| Stage | Original | Alternative |
|:--|:--|:--|
| Baseline | [JSON](../src/unified_va/configs/baseline_original.json) | [JSON](../src/unified_va/configs/baseline_alt1.json) |
| Stage 1 | [JSON](../src/unified_va/configs/stage1_original.json) | [JSON](../src/unified_va/configs/stage1_alt1.json) |
| Stage 2 | [JSON](../src/unified_va/configs/stage2_original.json) | [JSON](../src/unified_va/configs/stage2_alt1.json) |

The six profiles specify architecture, the shared Prompt, optimizer schedule, development updates, and selector weights. Runtime fills explicit model/data/run paths and saves `config.json`. Registered algorithm settings are validated before execution. See [experiments.md](../docs/experiments.md) for hyperparameters.

| Partition | Stage | Development updates |
|:--|:--|:--|
| Original | Baseline / Stage 1 | 2500, 2508, 5000, 5016, 7500, 7524 |
| Original | Stage 2 | 500, 1000, 1500, 2000, 2500, 2508 |
| Alternative | Baseline / Stage 1 | 2418, 2500, 4836, 5000, 7254 |
| Alternative | Stage 2 | 500, 1000, 1500, 2000, 2418 |
