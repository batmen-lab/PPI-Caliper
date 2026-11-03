"""Shared benchmark inputs and predictor metadata."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

SPECIES = ("ecoli", "fly", "mouse", "worm", "yeast")
CHECKPOINTS = {
    "plm-interact": {
        "humanv11": "danliu1226/PLM-interact-650M-humanV11",
        "humanv12": "danliu1226/PLM-interact-650M-humanV12",
    },
    "mint": {
        "humanv11": "MINT-humanV11; checkpoint provenance pending",
        "humanv12": "MINT-humanV12; checkpoint provenance pending",
    },
}
DEFAULT_Q_LEVELS = tuple(round(float(q), 1) for q in np.arange(0.1, 1.0, 0.1))


def parse_experiment_args(description: str, kind: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, help=f"Default: results/<predictor>/{kind}")
    parser.add_argument("--predictor", choices=CHECKPOINTS, default="plm-interact")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--q-levels", nargs="+", type=float, default=DEFAULT_Q_LEVELS)
    parser.add_argument("--quantile-anchors", type=int, default=1000)
    parser.add_argument("--lambda-grid-size", type=int, default=100)
    parser.add_argument("--delta", type=float, default=0.5)
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--plots-only", action="store_true", help="Redraw figures from saved results.")
    plots.add_argument("--no-plots", action="store_true", help="Run the experiment and save just its data.")
    args = parser.parse_args()
    if not args.plots_only and args.n_seeds < 1:
        parser.error("--n-seeds must be positive")
    args.output_dir = args.output_dir or PROJECT_ROOT / "results" / args.predictor / kind
    return args


def sha256_indices(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def prediction_path(
    model: str, species: str, *, predictor: str = "plm-interact", data_root: Path = DATA_DIR
) -> Path:
    """Locate a released prediction table for one model and species."""
    score_root = Path(data_root) / "scores"
    if predictor == "mint":
        score_root = score_root / "mint"
    return score_root / model / species / "predictions.csv"


def load_inputs(
    predictor: str, data_root: Path
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Read the predictor's score tables and build its input manifest."""
    data_root = Path(data_root).resolve()
    columns = {"row_idx": np.int64, "label": np.int8, "p_ab": np.float64}
    score_tables: dict[tuple[str, str], pd.DataFrame] = {}
    manifest_rows: list[dict[str, object]] = []
    for model, checkpoint in CHECKPOINTS[predictor].items():
        for species in SPECIES:
            path = prediction_path(model, species, predictor=predictor, data_root=data_root)
            df_scores = pd.read_csv(path, usecols=list(columns), dtype=columns)
            df_scores = df_scores.rename(columns={"p_ab": "score"}).assign(model=model, species=species)
            score_tables[(model, species)] = df_scores
            labels = df_scores["label"].to_numpy()
            scores = df_scores["score"].to_numpy()
            with path.open("rb") as handle:
                sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            manifest_rows.append(
                {
                    "model": model,
                    "checkpoint_repository": checkpoint,
                    "species": species,
                    "path": (Path("data") / path.relative_to(data_root)).as_posix(),
                    "sha256": sha256,
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
        if not v11[["row_idx", "label"]].equals(v12[["row_idx", "label"]]):
            raise ValueError(f"{species}: v11 and v12 must have matching row indices and labels")
    return score_tables, pd.DataFrame(manifest_rows)
