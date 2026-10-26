"""PPI-Caliper: score calibration and FDR cutoffs for protein interaction pairs.

Use PPICaliper(...).ppicaliper(calibration_scores, calibration_labels,
                             target_scores, q_levels).
The labeled calibration set fits the quantile map and bootstrap risk grid.
An optional independent target mapping sample supports species transfer.
No target labels enter calibration. Raw cutoffs apply to the supplied batch.

The CLT/LTT rule is adapted from protein_conformal/util.py at OFFICIAL_COMMIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import norm

OFFICIAL_REPOSITORY = "https://github.com/ronboger/conformal-protein-retrieval"
OFFICIAL_COMMIT = "509888fec5eaf50ca401163c4f61c26ce59f4147"


def clt_p_value(r_hat: float, std_hat: float, n: int, alpha: float) -> float:
    z = (r_hat - alpha) / (std_hat / np.sqrt(n))
    return float(norm.cdf(z))


def risk_1d(sims: np.ndarray, labels: np.ndarray, lam: float) -> float:
    total_discoveries = int((sims >= lam).sum())
    false_discoveries = int(((1 - labels) * (sims >= lam)).sum())
    return float(false_discoveries / max(total_discoveries, 1))


def risk(sims: np.ndarray, labels: np.ndarray, lam: float) -> float:
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return float((false_discoveries / total_discoveries).mean())


def std_loss(sims: np.ndarray, labels: np.ndarray, lam: float) -> float:
    total_discoveries = (sims >= lam).sum(axis=1)
    false_discoveries = ((1 - labels) * (sims >= lam)).sum(axis=1)
    total_discoveries = np.maximum(total_discoveries, 1)
    return float((false_discoveries / total_discoveries).std())


def std_loss_1d(sims: np.ndarray, labels: np.ndarray, lam: float) -> float:
    del sims, labels, lam
    return 0.01


@dataclass(frozen=True)
class FDRGrid:
    """The alpha-independent calculations shared by all target FDR levels."""

    labels: np.ndarray
    sims: np.ndarray
    lambdas: np.ndarray
    risks: np.ndarray
    stds: np.ndarray
    n_observations: int
    is_1d: bool


@dataclass(frozen=True)
class FDRSelection:
    target_fdr: float
    quantile_threshold: float
    calibration_risk: float
    p_value: float
    grid_index: int
    suffix_condition_exists: bool


def _validate_inputs(
    labels: np.ndarray, sims: np.ndarray, n_lambdas: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    sims = np.asarray(sims, dtype=np.float64)
    if labels.shape != sims.shape:
        raise ValueError(
            f"labels and sims must have identical shapes, got {labels.shape} and {sims.shape}"
        )
    if labels.ndim not in (1, 2):
        raise ValueError("labels and sims must be one- or two-dimensional")
    if labels.size == 0:
        raise ValueError("labels and sims cannot be empty")
    if n_lambdas <= 0:
        raise ValueError("n_lambdas must be positive")
    if not np.isfinite(sims).all():
        raise ValueError("sims contains non-finite values")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1, False, True}):
        raise ValueError(f"labels must be binary, found {sorted(unique_labels)}")
    return labels.astype(np.int8, copy=False), sims


@dataclass(frozen=True)
class UniformQuantileMapper:
    """Training-only score-to-quantile map fitted at 1,000 uniform anchors."""

    score_anchors: np.ndarray
    quantile_anchors: np.ndarray
    requested_anchor_count: int

    @classmethod
    def fit(cls, scores: np.ndarray, n_anchors: int = 1000) -> "UniformQuantileMapper":
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError("scores must be a non-empty one-dimensional array")
        if not np.isfinite(scores).all():
            raise ValueError("scores contains non-finite values")
        if n_anchors < 2:
            raise ValueError("n_anchors must be at least 2")

        uniform_quantiles = np.linspace(0.0, 1.0, n_anchors)
        sampled_scores = np.quantile(scores, uniform_quantiles, method="linear")

        # Tied score anchors cannot be separate x values for interpolation. Map
        # each tied score to the average of the quantiles assigned to it, which
        # preserves a single monotone function without using any test data.
        unique_scores, inverse, counts = np.unique(
            sampled_scores, return_inverse=True, return_counts=True
        )
        quantile_sums = np.bincount(inverse, weights=uniform_quantiles)
        unique_quantiles = quantile_sums / counts
        return cls(
            score_anchors=unique_scores.astype(np.float64),
            quantile_anchors=unique_quantiles.astype(np.float64),
            requested_anchor_count=int(n_anchors),
        )

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("scores contains non-finite values")
        shape = scores.shape
        mapped = np.interp(
            scores.reshape(-1),
            self.score_anchors,
            self.quantile_anchors,
            left=0.0,
            right=1.0,
        )
        return mapped.reshape(shape)

    def inverse_threshold(self, quantile_threshold: float) -> float:
        if np.isposinf(quantile_threshold):
            return float("inf")
        return float(
            np.interp(
                quantile_threshold,
                self.quantile_anchors,
                self.score_anchors,
                left=float(self.score_anchors[0]),
                right=float(self.score_anchors[-1]),
            )
        )


@dataclass(frozen=True)
class CutoffResult:
    target_fdr: float
    cutoff_score: float
    quantile_threshold: float
    n_selected: int
    suffix_condition_exists: bool
    calibration_risk: float


def _scores(values: Iterable[float], name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite scores")
    return values


def raw_batch_threshold(
    target_scores: np.ndarray, target_quantiles: np.ndarray, quantile_threshold: float
) -> float:
    """Return a score >= cutoff with exactly the same selections on this batch.

    Taking the smallest selected observed score handles clipped quantile tails
    and tied anchors exactly; an interpolated inverse can include extra rows at
    those boundaries. The resulting cutoff is intentionally batch-specific.
    Positive infinity represents an empty discovery set for finite scores.
    """
    selected = target_quantiles >= quantile_threshold
    return float(np.min(target_scores[selected])) if selected.any() else float("inf")


@dataclass(frozen=True)
class Calibration:
    """Prepared risk grid and target scores on the raw and quantile scales."""

    grid: FDRGrid
    target_scores: np.ndarray
    target_quantiles: np.ndarray


class PPICaliper:
    """Configure calibration once, then run it on score arrays.

    Ten bootstrap pseudoqueries use SeedSequence([seed, species_index, 1]).
    By default each samples round(n_calibration / 5) rows with replacement.
    Higher scores must indicate stronger evidence of interaction.
    Cross-species control is empirical, not guaranteed under arbitrary shift.
    """

    def __init__(
        self,
        seed=0,
        species_index=0,
        n_pseudoqueries=10,
        query_size=None,
        n_anchors=1000,
        n_lambdas=100,
        delta=0.5,
    ):
        for name, value, minimum in (
            ("seed", seed, 0),
            ("species_index", species_index, 0),
            ("n_pseudoqueries", n_pseudoqueries, 2),
            ("n_anchors", n_anchors, 2),
            ("n_lambdas", n_lambdas, 2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if query_size is not None and (
            isinstance(query_size, bool)
            or not isinstance(query_size, (int, np.integer))
            or query_size < 1
        ):
            raise ValueError("query_size must be a positive integer")
        if not 0 < delta < 1:
            raise ValueError("delta must be strictly between 0 and 1")
        self.seed = seed
        self.species_index = species_index
        self.n_pseudoqueries = n_pseudoqueries
        self.query_size = query_size
        self.n_anchors = n_anchors
        self.n_lambdas = n_lambdas
        self.delta = delta

    def calibration(
        self, calibration_scores, calibration_labels, target_scores, target_mapping_scores=None
    ) -> Calibration:
        """Fit mappings and bootstrap the calibration grid without target labels.

        For held-out transfer evaluation, target_mapping_scores must come from
        a sample separate from target_scores. Otherwise the source map is used.
        """
        calibration_scores = _scores(calibration_scores, "calibration_scores")
        target_scores = _scores(target_scores, "target_scores")
        labels = np.asarray(calibration_labels)
        if labels.shape != calibration_scores.shape:
            raise ValueError("calibration_labels must match calibration_scores shape")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("calibration_labels must be binary 0 or 1")
        source_map = UniformQuantileMapper.fit(calibration_scores, self.n_anchors)
        target_map = source_map
        if target_mapping_scores is not None:
            mapping_scores = _scores(target_mapping_scores, "target_mapping_scores")
            target_map = UniformQuantileMapper.fit(mapping_scores, self.n_anchors)
        source_quantiles = source_map.transform(calibration_scores)
        target_quantiles = target_map.transform(target_scores)
        query_size = self.query_size or max(1, round(len(labels) / 5))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.species_index, 1]))
        indices = rng.choice(len(labels), size=(self.n_pseudoqueries, query_size), replace=True)
        grid = self.prepare_fdr_grid(labels[indices], source_quantiles[indices], self.n_lambdas)
        return Calibration(grid, target_scores, target_quantiles)

    @staticmethod
    def prepare_fdr_grid(labels: np.ndarray, sims: np.ndarray, n_lambdas: int = 100) -> FDRGrid:
        """Compute the official lambda/risk/std grid once for multiple alphas."""

        labels, sims = _validate_inputs(labels, sims, n_lambdas)
        is_1d = labels.ndim == 1
        risk_fn = risk_1d if is_1d else risk
        std_fn = std_loss_1d if is_1d else std_loss
        lambdas = np.linspace(float(sims.min()), float(sims.max()), n_lambdas)
        risks = np.array([risk_fn(sims, labels, lam) for lam in lambdas], dtype=np.float64)
        stds = np.array([std_fn(sims, labels, lam) for lam in lambdas], dtype=np.float64)
        return FDRGrid(
            labels=labels,
            sims=sims,
            lambdas=lambdas,
            risks=risks,
            stds=stds,
            n_observations=len(labels),
            is_1d=is_1d,
        )

    @staticmethod
    def compute_cutoff(grid: FDRGrid, target_fdr: float, delta: float = 0.5) -> FDRSelection:
        """Apply the CLT/LTT suffix rule; return infinity when no quantile_threshold passes."""

        if not 0 < target_fdr < 1:
            raise ValueError("target_fdr must be in (0, 1)")
        if not 0 < delta < 1:
            raise ValueError("delta must be in (0, 1)")
        stds = np.maximum(grid.stds, 1e-6)
        pvals = np.array(
            [clt_p_value(r, s, grid.n_observations, target_fdr) for r, s in zip(grid.risks, stds)],
            dtype=np.float64,
        )
        below = pvals <= delta
        suffix_condition = np.array([np.all(below[i:]) for i in range(len(grid.lambdas))])

        accepted = np.flatnonzero(suffix_condition)
        if len(accepted) == 0:
            return FDRSelection(
                target_fdr=float(target_fdr),
                quantile_threshold=float("inf"),
                calibration_risk=0.0,
                p_value=0.0,
                grid_index=-1,
                suffix_condition_exists=False,
            )
        idx = int(accepted[0])
        quantile_threshold = min(float(grid.lambdas[idx]), 1.0)
        risk_fn = risk_1d if grid.is_1d else risk
        calibration_risk = risk_fn(grid.sims, grid.labels, quantile_threshold)
        return FDRSelection(
            target_fdr=float(target_fdr),
            quantile_threshold=quantile_threshold,
            calibration_risk=float(calibration_risk),
            p_value=float(pvals[idx]),
            grid_index=idx,
            suffix_condition_exists=bool(suffix_condition.any()),
        )

    def ppicaliper(
        self,
        calibration_scores,
        calibration_labels,
        target_scores,
        q_levels,
        *,
        target_mapping_scores=None,
    ) -> list[CutoffResult]:
        """Return batch-specific raw cutoffs; infinity means no selected pairs."""
        q_levels = _scores(list(q_levels), "q_levels")
        if ((q_levels <= 0) | (q_levels >= 1)).any():
            raise ValueError("q_levels must be strictly between 0 and 1")
        if len(np.unique(q_levels)) != len(q_levels):
            raise ValueError("q_levels must not contain duplicates")
        calibrated = self.calibration(
            calibration_scores, calibration_labels, target_scores, target_mapping_scores
        )
        cutoffs = []
        for target_fdr in q_levels:
            selected = self.compute_cutoff(calibrated.grid, float(target_fdr), self.delta)
            cutoff_score = raw_batch_threshold(
                calibrated.target_scores, calibrated.target_quantiles, selected.quantile_threshold
            )
            cutoffs.append(
                CutoffResult(
                    target_fdr=float(target_fdr),
                    cutoff_score=cutoff_score,
                    quantile_threshold=selected.quantile_threshold,
                    n_selected=int((calibrated.target_scores >= cutoff_score).sum()),
                    suffix_condition_exists=selected.suffix_condition_exists,
                    calibration_risk=selected.calibration_risk,
                )
            )
        return cutoffs

    @staticmethod
    def obtain_hits(target_scores, cutoffs) -> dict[float, np.ndarray]:
        """Return selected row indices for each FDR level, preserving pair order.

        Pass the same target batch used by ppicaliper. Use these indices to
        select rows from the original pair table, retaining its identifiers.
        """
        target_scores = _scores(target_scores, "target_scores")
        return {
            row.target_fdr: np.flatnonzero(target_scores >= row.cutoff_score) for row in cutoffs
        }
