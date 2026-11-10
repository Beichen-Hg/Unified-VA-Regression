# Configuration guide

- [main_original.yaml](main_original.yaml): paired Stage 1 → Stage 2, original partition, seeds 42/13/7.
- [main_alt1.yaml](main_alt1.yaml): paired Stage 1 → Stage 2, alternative partition, seeds 42/13/7.

Run recipes with `python scripts/reproduce.py --config <file> --data-root <data> --model-dir <model> --output-dir <runs>`. `--seed` selects one registered seed and `--dry-run` prints commands without loading data or models. Use the individual scripts to run one stage or evaluate an existing run.

## Complete hyperparameter profiles

| Stage | Original | Alternative |
|:--|:--|:--|
| Baseline | [profile](../src/unified_va/configs/baseline_original.json) | [profile](../src/unified_va/configs/baseline_alt1.json) |
| Stage 1 | [profile](../src/unified_va/configs/stage1_original.json) | [profile](../src/unified_va/configs/stage1_alt1.json) |
| Stage 2 | [profile](../src/unified_va/configs/stage2_original.json) | [profile](../src/unified_va/configs/stage2_alt1.json) |

These JSON files contain the actual architecture, prompt, optimization and development-selection settings. Model/data/run placeholders are resolved by the scripts; the resolved `config.json` is saved with each run. Algorithm settings are checked against the profiles before training.

Numerical conventions and actual trainable paths are specified in [reproducibility.md](../docs/reproducibility.md).
