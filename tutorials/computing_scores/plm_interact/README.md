# PLM-Interact: compute prediction scores

Use the released humanV11 and humanV12 checkpoints to score protein pairs. No fine-tuning is needed.

## Setup

Install the [root dependencies](../../../README.md#requirements), then prepare the sequence datasets and download the model checkpoints:

```bash
python data/download_data.py --only-upstream
python data/preprocess_data.py --only-upstream
python tutorials/computing_scores/plm_interact/download_models.py
```

The downloader saves the two PLM-Interact checkpoints and the ESM-2 backbone to `data/checkpoints/plm_interact/`. Their repository versions are pinned in [assets.json](assets.json).

## Compute scores

Run each model on the five species test sets:

```bash
python tutorials/computing_scores/plm_interact/score.py --model humanv11
python tutorials/computing_scores/plm_interact/score.py --model humanv12
```

Scores are saved to `results/plm-interact/scores/<model>/<species>/predictions.csv`. Each table contains `row_idx`, `label`, `len_pair_ab`, and `p_ab`. Use `p_ab` as the score column in PPI-Caliper.

The model reads `query,text,label` from the sequence tables. Pairs exceeding 1,603 combined tokens, including special tokens, are excluded. `row_idx` keeps each pair’s original row number.

Use `--species` to select test sets, `--output-dir` to choose the output folder for one model, and `--cache-root` to use model checkpoints stored elsewhere. The default device is `cuda:0` and the default precision is `float32`. Each script lists its options with `--help`.
