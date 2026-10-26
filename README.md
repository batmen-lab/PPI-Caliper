# PPI-Caliper

**PPI-Caliper** selects interaction-score thresholds for requested false discovery rate (FDR) levels. Give it a **labeled calibration set** and **unlabeled target scores**; it returns a **two-column threshold table**.

The repository also contains the PLM-Interact and MINT workflows used to compute scores, run within-species calibration, and evaluate cross-species threshold transfer. PLM-Interact uses the authors' released checkpoints while MINT trains a downstream MLP head on frozen encoder embeddings.

## Installation

### Requirements

PPI-Caliper requires Python 3.11 or newer. The same environment supports the [Quick Start](#quick-start) and [paper reproduction](#reproducing-the-paper-results) workflows. Install the required dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```



### Data

The following data are available from [Hugging Face](https://huggingface.co/datasets/lidehan/PPI-Caliper-predictions):

- precomputed prediction scores (PLM-Interact, MINT)
- reference results.

Use these scores to reproduce the calibration experiments, or supply your own scores for the Quick Start. See [Download and preprocess the paper data](#download-and-preprocess-the-paper-data) for setup. Original benchmark sequences are downloaded separately from [Dan Liu’s cross-species benchmark](https://huggingface.co/datasets/danliu1226/cross_species_benchmarking/tree/19dbed25184466fda5341b5ba70c2560ade1bf85) and [STRING V12 training dataset](https://huggingface.co/datasets/danliu1226/STRING_V12_TrainingSet/tree/819f938b1c21b8ec4990bb58c1506cff2537e3c2), only for optional score generation and training in the tutorial. Downloaded data are stored in `data/` alongside the data utilities:

```text
PPI-Caliper/
├── data/
│   ├── download_data.py
│   ├── preprocess_data.py
│   └── data_sources.json
├── results/                      # generated outputs; excluded from Git
├── requirements.txt
├── src/
│   ├── PPICaliper.py              # reusable calibration class
│   └── run_PPICaliper.py          # command-line runner
└── tutorials/
    ├── tutorial.ipynb            # paper walkthrough
    ├── computing_scores/
    │   ├── plm_interact/
    │   └── mint/
    ├── within_species/
    │   ├── run_experiments.py
    │   └── plot.py
    ├── transfer_species/
    │   ├── run_experiments.py
    │   └── plot.py
    └── utils/                    # shared experiment utilities
```

The `data/` directory contains the download and preprocessing utilities and their pinned source configuration. Each experiment folder contains `run_experiments.py` for calibration and evaluation, and `plot.py` for its figures. Both use `src/PPICaliper.py` and the shared utilities in `tutorials/utils/`, including `figure_style.py` for plotting colors and figure export.

## Quick Start

Commands below use default calibration settings. See the tutorial for the sampling procedure and use `--help` to view available options.

### Same-species use

After [installation](#installation), prepare labeled calibration scores and a separate set of target scores from the same species. From the repository root, run the command below to select thresholds for your requested FDR levels:

```bash
python src/run_PPICaliper.py calibrate --calibration calibration.csv --target target.csv --target-fdr 0.1 0.2 0.3 --out thresholds.csv
```

- `calibration.csv` contains `score,label`
  - Calibration labels are 0 or 1;
  - Higher scores indicate stronger evidence for interaction.
- `target.csv` contains `score`.
- `thresholds.csv` has two columns, `target_fdr` and `cutoff_score`.

Use `--score-column p_ab` for the supplied prediction tables or `--score-column mint_score` for original MINT prediction files. Set `--target-score-column` if the target scores use a different column name.

Select target pairs whose prediction scores are greater than or equal to the returned threshold. The threshold applies directly to the original prediction scores in your target file. Run PPI-Caliper again when using a different target file. A threshold of `inf` means no pairs are selected for that requested FDR level.

### Cross-species use

For cross-species use, provide a separate sample of target-species scores with `--target-mapping`. PPI-Caliper learns the target species’ mapping from this sample and applies the quantile threshold selected using the labeled source-species calibration data:

```bash
python src/run_PPICaliper.py calibrate --calibration source_calibration.csv --target-mapping target_mapping.csv --target target_test.csv --target-fdr 0.1 0.2 0.3 --out thresholds.csv
```

- `source_calibration.csv` contains `score,label` from the source species.
  - Calibration labels are 0 or 1.
  - Higher scores indicate stronger evidence for interaction.
- `target_mapping.csv` contains `score` from the target species, used to learn its score-to-quantile mapping.
- `target_test.csv` contains `score` for the target-species pairs you want to select.
- `thresholds.csv` has two columns, `target_fdr` and `cutoff_score`. Apply each threshold directly to the scores in `target_test.csv`.

Use separate samples from the target species for `target_mapping.csv` and `target_test.csv` when reproducing the paper’s held-out evaluation. Neither target file requires labels; keep the test labels separately if you want to measure FDP and power afterward. Cross-species FDR control is evaluated empirically and is not guaranteed for arbitrary differences between species.

## Reproducing the Paper Results

### Tutorial

Run [tutorials/tutorial.ipynb](tutorials/tutorial.ipynb). It starts with data setup and same-species and cross-species examples, then reproduces within-species and across-species results. An optional section covers score generation and MINT head training.

Launch the notebook tutorial:

```bash
python -m pip install jupyterlab
python -m jupyter lab tutorials/tutorial.ipynb
```

The notebook includes saved tables and figures from 100 random splits with PLM-Interact scores and 30 with MINT scores. Running it regenerates the results in `results/tutorial/paper/`.

For optional score generation, follow the [PLM-Interact scoring guide](tutorials/computing_scores/plm_interact/README.md) or the [MINT training and scoring guide](tutorials/computing_scores/mint/README.md). MINT V12 results currently differ from the paper.

### Download and preprocess the paper data

This step prepares the inputs for the tutorial and paper experiments. The notebook can run it for you, or you can run the commands below before using the analysis scripts directly. It downloads and verifies the experiment inputs. Skip it when using your own scores.

The prediction scores and saved reference results are hosted in [lidehan/PPI-Caliper-predictions on Hugging Face](https://huggingface.co/datasets/lidehan/PPI-Caliper-predictions). **(TODO: The dataset is currently private.)**

Run these commands to authenticate your Hugging Face account, download and preprocess the pinned release:

```bash
hf auth login
python data/download_data.py
python data/preprocess_data.py
```

Downloads use the repositories, commit IDs, and file paths in `data/data_sources.json`. Preprocessing checks the prediction tables and prepares the example inputs. The tutorial downloads original sequences from Dan Liu’s datasets for optional score generation and training.


| Downloaded files                                                 | Purpose                                                                       |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `data/scores/{humanv11,humanv12}/{species}/predictions.csv`      | Ten PLM-Interact prediction tables                                            |
| `data/scores/mint/{humanv11,humanv12}/{species}/predictions.csv` | Ten normalized MINT prediction tables                                         |
| `data/reference/{within,transfer}/`                           | Frozen PLM-Interact results, summaries, split hashes and experiment protocols |
