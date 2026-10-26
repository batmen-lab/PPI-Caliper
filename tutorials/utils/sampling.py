"""Shared data splits for benchmark experiments."""

import numpy as np


def stratified_half_split(
    labels: np.ndarray, seed: int, species_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create one deterministic 50/50 split while preserving class balance."""

    labels = np.asarray(labels, dtype=np.int8)
    rng = np.random.default_rng(np.random.SeedSequence([seed, species_index, 0]))
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError(f"label {label} needs at least two rows")
        indices = rng.permutation(indices)
        n_train = len(indices) // 2
        train_parts.append(indices[:n_train])
        test_parts.append(indices[n_train:])
    train_idx = np.sort(np.concatenate(train_parts)).astype(np.int64)
    test_idx = np.sort(np.concatenate(test_parts)).astype(np.int64)
    return train_idx, test_idx
