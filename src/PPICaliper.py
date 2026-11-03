import numpy as np
from scipy.stats import norm


class PPICaliper:
    """Select protein interaction score cutoffs for requested false discovery rates.

    Usage
    -----
    >>> from src.PPICaliper import PPICaliper
    >>> caliper = PPICaliper()
    >>> caliper.calibration(calibration_scores, calibration_labels, target_scores)
    >>> for q in [0.1, 0.2, 0.3]:
    ...     threshold = caliper.select_quantile_threshold(q)
    ...     cutoff = caliper.find_raw_score_cutoff(threshold)
    ...     print(q, cutoff)

    Parameters
    ----------
    seed, species_index : int
        Random seed and species index for bootstrap sampling (both default to 0).
    n_pseudoqueries : int
        Number of bootstrap batches (default 10).
    query_size : int or None
        Pairs sampled with replacement per batch. None uses one fifth of the
        calibration set, rounded to the nearest integer and at least one.
    n_anchors, n_lambdas : int
        Number of quantile anchors (default 1000) and candidate thresholds (100).
    delta : float
        Maximum passing p-value for the suffix rule (default 0.5).

    Data
    ----
    Supply nonempty, finite, one-dimensional score arrays. Higher scores indicate
    stronger evidence of interaction. Calibration labels must be binary (0 or 1)
    and match the calibration scores in length and order. For species transfer,
    pass independent, unlabeled target_mapping_scores to fit the target mapping.

    Returns
    -------
    calibration() prepares the instance for the supplied target batch and returns
    None. select_quantile_threshold(q) returns a quantile threshold and records
    its mean bootstrap FDP in calibration_risk_. find_raw_score_cutoff(threshold)
    returns the lowest selected score in this target batch. A raw cutoff of
    infinity means that no pairs were selected.
    """

    def __init__(
        self, seed=0, species_index=0, n_pseudoqueries=10, query_size=None,
        n_anchors=1000, n_lambdas=100, delta=0.5,
    ):
        self.seed = seed
        self.species_index = species_index
        self.n_pseudoqueries = n_pseudoqueries
        self.query_size = query_size
        self.n_anchors = n_anchors
        self.n_lambdas = n_lambdas
        self.delta = delta

        self._thresholds = None
        self.calibration_risk_ = None

    def calibration(
        self, calibration_scores, calibration_labels, target_scores, *, target_mapping_scores=None
    ) -> None:
        """Map scores to quantiles and estimate FDP for each candidate threshold."""
        calibration_scores = np.asarray(calibration_scores, dtype=float)
        labels = np.asarray(calibration_labels)
        target_scores = np.array(target_scores, dtype=float, copy=True)

        # Within species, both batches use the source map. Transfer fits a second map.
        reference_samples = [calibration_scores]
        if target_mapping_scores is not None:
            reference_samples.append(np.asarray(target_mapping_scores, dtype=float))
        quantile_levels = np.linspace(0.0, 1.0, self.n_anchors)
        maps = []
        for reference in reference_samples:
            scores = np.quantile(reference, quantile_levels, method="linear")
            anchors, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
            # Each tied score maps to the mean of its associated quantile levels.
            quantiles = np.bincount(inverse, weights=quantile_levels) / counts
            maps.append((anchors, quantiles))
        source_quantiles = np.interp(calibration_scores, *maps[0], left=0.0, right=1.0)
        target_quantiles = np.interp(target_scores, *maps[-1], left=0.0, right=1.0)

        batch_size = self.query_size or max(1, round(len(labels) / 5))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.species_index, 1]))
        indices = rng.choice(len(labels), size=(self.n_pseudoqueries, batch_size), replace=True)
        batch_quantiles = source_quantiles[indices]
        batch_negatives = labels[indices] == 0
        thresholds = np.linspace(batch_quantiles.min(), batch_quantiles.max(), self.n_lambdas)
        mean_fdps = np.empty_like(thresholds)
        std_fdps = np.empty_like(thresholds)
        for i, threshold in enumerate(thresholds):
            selected = batch_quantiles >= threshold
            false_counts = (selected & batch_negatives).sum(axis=1)
            batch_fdps = false_counts / np.maximum(selected.sum(axis=1), 1)
            mean_fdps[i] = batch_fdps.mean()
            std_fdps[i] = batch_fdps.std()

        self._thresholds = thresholds
        self._mean_fdps = mean_fdps
        self._standard_errors = np.maximum(std_fdps, 1e-6) / np.sqrt(self.n_pseudoqueries)
        self._target_scores = target_scores
        self._target_quantiles = target_quantiles
        self.calibration_risk_ = None

    def select_quantile_threshold(self, target_fdr: float) -> float:
        """Select the lowest grid threshold whose entire suffix passes; otherwise return inf."""
        if self._thresholds is None:
            raise RuntimeError("Call calibration() before selecting a threshold.")
        if not 0 < target_fdr < 1:
            raise ValueError("target_fdr must be in (0, 1)")

        # CLT/LTT rule adapted from https://github.com/ronboger/conformal-protein-retrieval/blob/509888fec5eaf50ca401163c4f61c26ce59f4147/protein_conformal/util.py
        p_values = norm.cdf((self._mean_fdps - target_fdr) / self._standard_errors)
        # The first passing suffix starts immediately after the last failed test.
        failed = np.flatnonzero(p_values > self.delta)
        first = failed[-1] + 1 if failed.size else 0
        if first == len(self._thresholds):
            self.calibration_risk_ = 0.0
            return float("inf")
        self.calibration_risk_ = float(self._mean_fdps[first])
        return float(self._thresholds[first])

    def find_raw_score_cutoff(self, quantile_threshold: float) -> float:
        """Return the raw cutoff that selects exactly the same pairs in this target batch."""
        if self._thresholds is None:
            raise RuntimeError("Call calibration() before finding a raw cutoff.")
        selected = self._target_quantiles >= quantile_threshold
        return float(self._target_scores[selected].min(initial=np.inf))
