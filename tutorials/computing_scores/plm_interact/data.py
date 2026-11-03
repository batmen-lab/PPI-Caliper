"""Read sequence pairs and apply the PLM-Interact length limit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from computing_scores.plm_interact.config import SPECIAL_TOKENS


def normalize_sequence(value: object) -> str:
    return "".join(str(value).split()).upper()


def load_ppi_frame(path: Path, max_length: int) -> tuple[pd.DataFrame, int]:
    """Return eligible pairs with original row indices, and the total input count."""
    frame = pd.read_csv(path, usecols=["query", "text", "label"])
    frame["query"] = frame["query"].map(normalize_sequence)
    frame["text"] = frame["text"].map(normalize_sequence)
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{path}: labels must be binary 0/1")
    frame.insert(0, "row_idx", np.arange(len(frame), dtype=np.int64))
    frame["len_pair_ab"] = (
        frame["query"].str.len() + frame["text"].str.len() + SPECIAL_TOKENS
    )
    keep = frame["len_pair_ab"] <= max_length
    return frame.loc[keep].reset_index(drop=True), len(frame)
