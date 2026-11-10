# Data preparation

Obtain the regression datasets from the [official DimABSA2026 repository](https://github.com/DimABSA/DimABSA2026). Set `DATA_ROOT` to its `task-dataset` directory:

```text
task-dataset/
  track_a/subtask_1/eng/eng_restaurant_train_alltasks.jsonl
  track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl
  track_a/subtask_1/eng/eng_restaurant_test_task1.jsonl
  ...
  track_b/subtask_1/deu/deu_politics_train_task1.jsonl
  track_b/subtask_1/deu/deu_politics_dev_task1.jsonl
  track_b/subtask_1/deu/deu_politics_test_task1.jsonl
```

Restaurant/laptop/hotel use `train_alltasks`; finance/politics/environmental_protection use `train_task1`. The loader expands each record to aspect-level examples. The official files are read in place.

## Targets

The collection contains `eng_restaurant`, `deu_politics`, `eng_laptop`, `zho_laptop`, `ukr_restaurant`, `zho_restaurant`, `jpn_hotel`, `jpn_finance`, `rus_restaurant`, `zho_finance`, `zho_environmental_protection`, `pcm_politics`, `eng_environmental_protection`, `tat_restaurant` and `swa_politics`.

| Partition | Excluded from training and development selection |
|:--|:--|
| original | `eng_environmental_protection`, `tat_restaurant`, `swa_politics` |
| alt1 | `rus_restaurant`, `ukr_restaurant`, `zho_finance` |

All other 12 targets supply training/development examples. Both partitions evaluate 19,977 aspects from the same 15 test targets. Fixed ordering is specified in [config.py](../src/unified_va/config.py). VA normalization is recomputed from the selected partition's training records.

## Application inputs

Prediction accepts one aspect per row:

```json
[{"id": "demo", "text": "The screen is bright.", "aspect": "screen"}]
```

`id` and `aspect_index` can associate outputs with original records. Optional `v`, `a`, `lang` and `domain` fields support post-forward metrics/diagnostics; tokenization receives text/aspect only. Repository examples are synthetic. Dataset licensing follows the official resources.
