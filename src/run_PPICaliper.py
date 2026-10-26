"""Command-line input/output for PPI-Caliper."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

from PPICaliper import PPICaliper


def read_columns(path: Path, columns: list[str]) -> list[list[float]]:
    """Read named columns only; extra columns (including target labels) are ignored."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"{path}: empty table")
        handle.seek(0)
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: duplicate column names")
        missing = set(columns) - set(names)
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        values = [[] for _ in columns]
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"{path}:{line}: extra fields")
            for column, output in zip(columns, values):
                try:
                    number = float(row[column])
                except (ValueError, TypeError):
                    raise ValueError(f"{path}:{line}: {column} must be numeric") from None
                if not math.isfinite(number):
                    raise ValueError(f"{path}:{line}: {column} must be finite")
                output.append(number)
        return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select PPI-Caliper FDR thresholds from labeled calibration scores and an unlabeled target batch.",
        epilog="Output CSV columns: target_fdr, cutoff_score. Select target rows with score >= threshold; inf means none. "
        "Thresholds are equivalent on this target batch only. Target labels, if present, are ignored. "
        "This empirical calibration is not a guarantee under arbitrary distribution shift.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate", help="Select PPI score cutoffs")
    p.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="CSV containing score and binary label columns",
    )
    p.add_argument(
        "--target", type=Path, required=True, help="CSV containing unlabeled target scores"
    )
    p.add_argument("--out", type=Path, required=True, help="Two-column threshold CSV to create")
    p.add_argument(
        "--score-column", default="score", help="Calibration score column (default: score)"
    )
    p.add_argument(
        "--label-column", default="label", help="Calibration label column (default: label)"
    )
    p.add_argument(
        "--target-score-column", help="Target/mapping score column; defaults to --score-column"
    )
    p.add_argument(
        "--target-mapping",
        type=Path,
        help="Independent unlabeled target mapping sample for species transfer; omit for same-species calibration",
    )
    p.add_argument(
        "--target-fdr",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Requested FDR levels, strictly between 0 and 1",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--species-index",
        type=int,
        default=0,
        help="SeedSequence species index used by reproduction drivers",
    )
    p.add_argument("--pseudoqueries", type=int, default=10)
    p.add_argument(
        "--query-size",
        type=int,
        help="Rows sampled per pseudoquery; default round(calibration rows / 5), at least 1",
    )
    p.add_argument("--quantile-anchors", type=int, default=1000)
    p.add_argument("--lambda-grid-size", type=int, default=100)
    p.add_argument("--delta", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    try:
        inputs = [args.calibration, args.target] + (
            [args.target_mapping] if args.target_mapping else []
        )
        if args.out.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("output cannot overwrite an input table")
        calibration_scores, labels = read_columns(
            args.calibration, [args.score_column, args.label_column]
        )
        target_column = args.target_score_column or args.score_column
        (target_scores,) = read_columns(args.target, [target_column])
        mapping_scores = None
        if args.target_mapping:
            (mapping_scores,) = read_columns(args.target_mapping, [target_column])
        caliper = PPICaliper(
            seed=args.seed,
            species_index=args.species_index,
            n_pseudoqueries=args.pseudoqueries,
            query_size=args.query_size,
            n_anchors=args.quantile_anchors,
            n_lambdas=args.lambda_grid_size,
            delta=args.delta,
        )
        cutoffs = caliper.ppicaliper(
            calibration_scores,
            labels,
            target_scores,
            args.target_fdr,
            target_mapping_scores=mapping_scores,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["target_fdr", "cutoff_score"])
            writer.writerows((r.target_fdr, r.cutoff_score) for r in cutoffs)
        failed = [r.target_fdr for r in cutoffs if not r.suffix_condition_exists]
        if failed:
            print(
                f"No accepted FDR-grid suffix for levels {failed}; wrote inf (no discoveries).",
                file=sys.stderr,
            )
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
