# PPI-Caliper

**PPI-Caliper** selects interaction-score thresholds for requested false discovery rate (FDR) levels. Give it a **labeled calibration set** and **unlabeled target scores**; it selects a raw-score cutoff for each requested FDR level.

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
│   └── PPICaliper.py             # reusable calibration class
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

Run the Python examples from the repository root. Supply finite scores, with higher values indicating stronger evidence of interaction. The examples use a column named `score`; the supplied prediction tables name it `p_ab`.

### Same-species use

Read a labeled calibration set (`score` and binary `label`) and the target pairs you want to screen (`score`). The paths below use the E. coli examples created by `data/preprocess_data.py`. Substitute your own CSVs to screen another dataset.

```python
import pandas as pd
from src.PPICaliper import PPICaliper

source = pd.read_csv("data/examples/calibration.csv")
target = pd.read_csv("data/examples/target.csv")
caliper = PPICaliper()
caliper.calibration(source["score"], source["label"], target["score"])

cutoffs = {}
for q in [0.1, 0.2, 0.3]:
    threshold = caliper.select_quantile_threshold(q)
    cutoffs[q] = caliper.find_raw_score_cutoff(threshold)

selected_pairs = target.loc[target["score"] >= cutoffs[0.1]]
thresholds = pd.DataFrame(cutoffs.items(), columns=["target_fdr", "cutoff_score"])
thresholds.to_csv("thresholds.csv", index=False)
```

The raw cutoff applies to the target batch passed to `calibration()`. A cutoff of `inf` selects zero pairs. Call `calibration()` again when using a different target batch.

### Cross-species use

Use source-species scores and labels to select the quantile threshold. Supply a separate sample of unlabeled target-species scores to fit the target mapping. The raw cutoff applies to the held-out target pairs.

```python
import pandas as pd
from src.PPICaliper import PPICaliper

source = pd.read_csv("source_calibration.csv")
mapping = pd.read_csv("target_mapping.csv")
target = pd.read_csv("target_test.csv")

caliper = PPICaliper()
caliper.calibration(
    source["score"], source["label"], target["score"],
    target_mapping_scores=mapping["score"],
)
threshold = caliper.select_quantile_threshold(0.1)
cutoff = caliper.find_raw_score_cutoff(threshold)
selected_pairs = target.loc[target["score"] >= cutoff]
```

Keep target test labels for evaluating FDP and power after selection. Cross-species FDR control is evaluated empirically and can fail under distribution shift.

## Reproducing the Paper Results

### Tutorial

The [tutorial notebook](tutorials/tutorial.ipynb) starts with two threshold examples. Section 4 saves the benchmark tables. Sections 5.1 and 5.2 use those tables to draw the within-species and across-species figures. Change the source and target species in the transfer example to explore another direction. The notebook and command-line experiments share the [within-species](tutorials/within_species/plot.py) and [across-species](tutorials/transfer_species/plot.py) plotting functions.

Launch the notebook tutorial:

```bash
python -m pip install jupyterlab
python -m jupyter lab tutorials/tutorial.ipynb
```

The benchmark uses 100 splits for PLM-Interact and 30 for MINT and writes the tables to `results/tutorial/paper/`. Set `RERUN_EXPERIMENTS = False` after the first run to redraw figures from those tables. To export the grids and their individual panels as PNGs and PDFs, run the corresponding experiment script with `--plots-only`, the predictor, and its existing output directory.

For optional score generation, follow the [PLM-Interact scoring guide](tutorials/computing_scores/plm_interact/README.md) or the [MINT training and scoring guide](tutorials/computing_scores/mint/README.md).

### Download and preprocess the paper data

This step prepares the inputs for the tutorial and paper experiments. The notebook can run it for you, or you can run the commands below before using the analysis scripts directly. It downloads and verifies the experiment inputs. Skip it when using your own scores.

The prediction scores and saved reference results are hosted in [lidehan/PPI-Caliper-predictions on Hugging Face](https://huggingface.co/datasets/lidehan/PPI-Caliper-predictions). **(TODO: The dataset is currently private.)**

Run these commands to authenticate your Hugging Face account, download and preprocess the pinned release:

```bash
hf auth login
python data/download_data.py
python data/preprocess_data.py
```

Downloads use the repositories, commit IDs, and file paths in `data/data_sources.json`. Preprocessing checks row indices, labels, and score ranges and prepares the example inputs. The experiment scripts read those verified tables directly. The tutorial downloads original sequences from Dan Liu’s datasets for optional score generation and training.


| Downloaded files                                                 | Purpose                                                                       |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `data/scores/{humanv11,humanv12}/{species}/predictions.csv`      | Ten PLM-Interact prediction tables                                            |
| `data/scores/mint/{humanv11,humanv12}/{species}/predictions.csv` | Ten normalized MINT prediction tables                                         |
| `data/reference/{within,transfer}/`                              | PLM-Interact reference results, summaries, split hashes and protocols (100 splits) |
| `data/reference/mint/{within,transfer}/`                         | MINT reference results, summaries, split hashes and protocols (30 splits) |
