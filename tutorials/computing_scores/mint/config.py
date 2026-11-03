"""Paths for MINT training inputs, checkpoints, and prediction scores."""
from pathlib import Path

from utils.inputs import PROJECT_ROOT, SPECIES

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "checkpoints" / "mint"
RESULTS_DIR = PROJECT_ROOT / "results" / "mint"
TASK_NAMES = {"v11": "StringV11HumanPPI", "v12": "PLMhumanPPI"}


def training_files(version, raw_root=RAW_ROOT):
    raw_root = Path(raw_root)
    train = (raw_root / "cross_species_benchmarking/train/human.ppi.qrels.seq.train.csv"
             if version == "v11" else
             raw_root / "STRING_V12_TrainingSet/protein.pairs_human_V12_train.csv")
    return train, raw_root / "cross_species_benchmarking/val/human.ppi.qrels.seq.test.csv"


def head_path(version):
    return CHECKPOINT_ROOT / version / f"{TASK_NAMES[version]}_mint_sep_final_overall.pth"
