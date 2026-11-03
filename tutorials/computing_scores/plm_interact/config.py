"""PLM-Interact model settings and benchmark paths."""

from pathlib import Path

from utils.inputs import CHECKPOINTS, PROJECT_ROOT, SPECIES

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "checkpoints" / "plm_interact"
RESULTS_DIR = PROJECT_ROOT / "results" / "plm-interact"
MODEL_REPOSITORIES = CHECKPOINTS["plm-interact"]
BASE_MODEL = "facebook/esm2_t33_650M_UR50D"
EMBEDDING_SIZE = 1280
SPECIAL_TOKENS = 3
MAX_LENGTH = 1603


def model_directory(cache_root: Path, repo_id: str) -> Path:
    return cache_root / repo_id.split("/")[-1]


def dataset_path(raw_root: Path, species: str) -> Path:
    return raw_root / "cross_species_benchmarking" / "test" / f"{species}.ppi.qrels.seq.test.csv"


def checkpoint_path(cache_root: Path, model_key: str) -> Path:
    return model_directory(cache_root, MODEL_REPOSITORIES[model_key]) / "pytorch_model.bin"
