"""Shared benchmark inputs and predictor metadata."""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from utils import paths
from utils.paths import DATA_DIR, data_path

SPECIES = ("ecoli", "fly", "mouse", "worm", "yeast")
SPECIES_LABELS = {
    "ecoli": "E. coli",
    "fly": "Fly",
    "mouse": "Mouse",
    "worm": "Worm",
    "yeast": "Yeast",
}
MODEL_REPOSITORIES = {
    "humanv11": "danliu1226/PLM-interact-650M-humanV11",
    "humanv12": "danliu1226/PLM-interact-650M-humanV12",
}
DEFAULT_Q_LEVELS = tuple(round(float(q), 1) for q in np.arange(0.1, 1.0, 0.1))


PREDICTOR = "plm-interact"
PLM_CHECKPOINTS = dict(MODEL_REPOSITORIES)


def configure_inputs(predictor: str = "plm-interact", data_root: Path | None = None) -> None:
    """Select the prediction tables for one predictor."""
    global PREDICTOR, DATA_DIR
    if predictor not in {"plm-interact", "mint"}:
        raise ValueError(f"Unknown predictor: {predictor}")
    PREDICTOR = predictor
    if data_root is not None:
        paths.set_data_root(data_root)
    DATA_DIR = paths.DATA_DIR
    MODEL_REPOSITORIES.clear()
    MODEL_REPOSITORIES.update(
        PLM_CHECKPOINTS
        if predictor == "plm-interact"
        else {
            "humanv11": "MINT-humanV11; checkpoint provenance pending",
            "humanv12": "MINT-humanV12; checkpoint provenance pending",
        }
    )


def predictor_label() -> str:
    return "MINT" if PREDICTOR == "mint" else "PLM-Interact"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def prediction_path(
    model: str, species: str, *, predictor: str | None = None, data_root: Path | None = None
) -> Path:
    """Locate a released prediction table for one model and species."""
    score_root = (DATA_DIR if data_root is None else Path(data_root)) / "scores"
    if (PREDICTOR if predictor is None else predictor) == "mint":
        score_root = score_root / "mint"
    return score_root / model / species / "predictions.csv"


def load_prediction(path: Path, model: str, species: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python data/download_data.py && python data/preprocess_data.py"
        )
    df_scores = pd.read_csv(path, usecols=["row_idx", "label", "p_ab"])
    df_scores = df_scores.rename(columns={"p_ab": "score"})
    df_scores["row_idx"] = pd.to_numeric(df_scores["row_idx"], errors="raise").astype(np.int64)
    df_scores["label"] = pd.to_numeric(df_scores["label"], errors="raise").astype(np.int8)
    df_scores["score"] = pd.to_numeric(df_scores["score"], errors="raise").astype(np.float64)
    if df_scores.empty:
        raise ValueError(f"{path} is empty")
    if df_scores["row_idx"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row_idx values")
    if not set(df_scores["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} contains non-binary labels")
    if not np.isfinite(df_scores["score"].to_numpy()).all():
        raise ValueError(f"{path} contains non-finite scores")
    if ((df_scores["score"] < 0) | (df_scores["score"] > 1)).any():
        raise ValueError(f"{path} contains scores outside [0, 1]")
    df_scores["model"] = model
    df_scores["species"] = species
    return df_scores


def load_inputs() -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    score_tables: dict[tuple[str, str], pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []
    for model in MODEL_REPOSITORIES:
        for species in SPECIES:
            path = prediction_path(model, species)
            df_scores = load_prediction(path, model, species)
            score_tables[(model, species)] = df_scores
            labels = df_scores["label"].to_numpy()
            scores = df_scores["score"].to_numpy()
            manifest_rows.append(
                {
                    "model": model,
                    "checkpoint_repository": MODEL_REPOSITORIES[model],
                    "species": species,
                    "path": data_path(path),
                    "sha256": sha256_file(path),
                    "n_rows": int(len(df_scores)),
                    "n_positive": int((labels == 1).sum()),
                    "n_negative": int((labels == 0).sum()),
                    "positive_fraction": float((labels == 1).mean()),
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "score_mean": float(scores.mean()),
                }
            )

    for species in SPECIES:
        v11 = score_tables[("humanv11", species)]
        v12 = score_tables[("humanv12", species)]
        if not np.array_equal(v11["row_idx"].to_numpy(), v12["row_idx"].to_numpy()):
            raise ValueError(f"{species}: v11 and v12 row_idx values do not match")
        if not np.array_equal(v11["label"].to_numpy(), v12["label"].to_numpy()):
            raise ValueError(f"{species}: v11 and v12 labels do not match")
    return score_tables, pd.DataFrame(manifest_rows)
