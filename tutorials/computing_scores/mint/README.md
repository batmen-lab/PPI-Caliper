# MINT: fine-tuning and inference

These adapters use the official MINT encoder and prediction head. The encoder converts protein pairs to embeddings; fine-tuning trains the small MLP head on labeled human pairs while keeping the encoder frozen.

## Setup

With the [root dependencies](../../../README.md#requirements) installed, install the pinned MINT package from the repository root:

```bash
python -m pip install -r tutorials/computing_scores/mint/requirements.txt
```

The dependency uses [official MINT commit `06694b7`](https://github.com/VarunUllanat/mint/tree/06694b7606e2d00b76ec58daf5c7aecdaf7cd283). Download the authors' [encoder checkpoint](https://huggingface.co/varunullanat2012/mint/blob/main/mint.ckpt) to `data/checkpoints/mint/mint.ckpt`, or provide its location with `--encoder`.

Prepare the sequence datasets:

```bash
python data/download_data.py --only-upstream
python data/preprocess_data.py --only-upstream
```

V11 uses the human training split from Dan Liu's cross-species benchmark; V12 uses the STRING V12 human training set. Both use the V11 human validation split and the same five species test sets.

## Fine-tune and compute scores

Extract embeddings, train the head, and score the species test sets:

```bash
python tutorials/computing_scores/mint/extract_embeddings.py --version v11
python tutorials/computing_scores/mint/train.py --version v11
python tutorials/computing_scores/mint/score.py --version v11
```

Repeat with `--version v12` for the other training set. Training uses 100 epochs, three repetitions with seeds 0–2, batch size 16, and AdamW with learning rate `1e-4`. The saved head has the highest validation accuracy across epochs and repetitions.

| Output | Default location |
| --- | --- |
| Training and validation embeddings | `results/mint/embeddings/<version>/` |
| Trained heads and epoch metrics | `data/checkpoints/mint/<version>/` |
| Prediction scores | `results/mint/scores/<version>/<species>/predictions.csv` |

The prediction table contains `row_idx`, `label`, and `p_ab`. Use `p_ab` as the score column in PPI-Caliper.

## Use an existing head or your own pairs

Pass `--head` to score with an existing trained head. A custom sequence table contains `query,text`, with an optional `label` column:

```bash
python tutorials/computing_scores/mint/score.py --version v11 --head path/to/head.pth --dataset pairs.csv
```

The scripts default to `--device cuda:0`. Use `--output-dir` to choose an output location. Embedding extraction and scoring use `--max-seq-length 2048`, including the two special tokens, and `--seed 0` for random cropping of longer sequences. Training options include `--epochs`, `--repeats`, and `--learning-rate`; each script lists its options with `--help`.

The MINT implementation is maintained in the [upstream repository](https://github.com/VarunUllanat/mint). This folder contains the dataset, training, and scoring adapters for our experiments; attribution is in [LICENSE](LICENSE).
