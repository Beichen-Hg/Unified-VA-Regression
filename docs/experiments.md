# Experiments

## Partitions and training budgets

Both partitions select checkpoints on 12 training/development targets and evaluate the same 15 targets. Target order and membership are defined in [config.py](../src/unified_va/config.py).

| Partition | Train aspects | Dev aspects | Baseline / Stage 1 | Stage 2 |
|:--|--:|--:|:--|:--|
| original | 40,126 | 3,532 | 3 epochs; 7,524 updates; 6 dev checks | 1 epoch; 2,508 updates; 6 dev checks |
| alt1 | 38,687 | 3,372 | 3 epochs; 7,254 updates; 5 dev checks | 1 epoch; 2,418 updates; 5 dev checks |

Seeds are 42, 13 and 7. The complete pipeline is run separately for each partition and seed.

## Baseline

```bash
python scripts/train.py --stage baseline --split alt1 --seed 42 \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" \
  --run-dir runs/alt1/42/baseline --device cuda:0
python scripts/evaluate.py --run-dir runs/alt1/42/baseline \
  --data-root "$DATA_ROOT" --model-dir "$MODEL_DIR" --device cuda:0
```

The baseline retains its own pooling, readout and prompt. Baseline → Stage 1 is a procedural comparison, including these differences.

## Auxiliary diagnostics

```bash
python scripts/analyze.py diagnose \
  --checkpoint runs/original/42/stage2/checkpoints/stage2.pt \
  --model-dir "$MODEL_DIR" --input labelled_aspect_rows.json \
  --output results/condition_diagnostics.json

python scripts/analyze.py dp-sensitivity \
  --run-dir runs/original/42/stage2 --output results/dp_sensitivity42.json
```

Auxiliary accuracy/macro F1 describe the saved condition classifier. Known-class coverage is reported, excluding unknown and configured-but-untrained labels. This classifier does not choose the VA adapter route. DP sensitivity reweights saved development archives without changing the released selected checkpoint.

## Metrics and statistics

For each target, RMSE-VA is `sqrt(mean((v_pred-v_true)^2 + (a_pred-a_true)^2))`. Overall is the equal-weight mean of the 15 target values. Training-covered and held-out groups are reported separately. Seed means and sample standard deviations summarize individual model scalars.

Paired bootstrap uses 10,000 draws with seed 20261101. It resamples original records within targets, retaining all aspects of a record together and sharing draws between compared systems and fixed training seeds. These intervals are conditional on the fitted models/targets and are separate from training-seed variation. The test data were previously examined; they are not described as a new blind test.
