"""Selection metrics and confidence intervals for benchmark experiments."""

import numpy as np
from scipy.stats import t as student_t


def evaluate_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, object]:
    selected = np.asarray(scores) >= threshold
    labels = np.asarray(labels, dtype=np.int8)
    n_selected = int(selected.sum())
    true_positives = int((selected & (labels == 1)).sum())
    false_positives = int((selected & (labels == 0)).sum())
    total_positive = int((labels == 1).sum())
    return {
        "n_selected": n_selected,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "observed_FDP": float(false_positives / max(n_selected, 1)),
        "recall": float(true_positives / max(total_positive, 1)),
        "empty_discovery_set": bool(n_selected == 0),
    }


def mean_confidence_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return the mean, sample SD, and 95% Student-t interval for proportions."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, mean, mean
    sd = float(values.std(ddof=1))
    margin = student_t.ppf(0.975, df=n - 1) * sd / np.sqrt(n)
    return mean, sd, max(0.0, mean - margin), min(1.0, mean + margin)
