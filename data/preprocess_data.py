#!/usr/bin/env python3
"""Prepare released scores, Quick Start examples, and optional training data."""

import argparse
import csv
import filecmp
import io
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

from download_data import DATA_DIR, safe_output, upstream_sources

SPECIES = ('ecoli', 'fly', 'mouse', 'worm', 'yeast')
PREDICTION_COLUMNS = ['row_idx', 'label', 'len_pair_ab', 'p_ab']


def convert_mint(archive, version, destination):
    """Convert all five species without reordering rows or replacing other data."""
    with zipfile.ZipFile(archive) as source_zip, tempfile.TemporaryDirectory(prefix='ppi-mint-') as temp:
        names = source_zip.namelist()
        if len(names) != len(set(names)):
            raise ValueError('MINT archive contains duplicate member names')

        staged = []
        for species in SPECIES:
            name = f'result_{version}/{species}.ppi.qrels.seq.test_result.csv'
            output = safe_output(destination, f'scores/mint/human{version}/{species}/predictions.csv')
            temporary = Path(temp) / f'{species}.csv'
            with (
                io.TextIOWrapper(source_zip.open(name), encoding='utf-8-sig') as source,
                temporary.open('w', newline='') as target,
            ):
                reader = csv.DictReader(source)
                columns = reader.fieldnames or []
                if len(columns) != len(set(columns)):
                    raise ValueError(f'Duplicate columns: {name}')
                writer = csv.writer(target)
                writer.writerow(PREDICTION_COLUMNS)
                rows_written = 0
                for row in reader:
                    if None in row or row['label'] not in ('0', '1') or not 0 <= float(row['mint_score']) <= 1:
                        raise ValueError(f'Invalid label or score: {name}, row {rows_written}')
                    if not row['query'] or not row['text']:
                        raise ValueError(f'Empty sequence: {name}, row {rows_written}')
                    pair_length = len(row['query']) + len(row['text'])
                    writer.writerow([rows_written, row['label'], pair_length, row['mint_score']])
                    rows_written += 1
                if rows_written == 0:
                    raise ValueError(f'Empty MINT prediction table: {name}')
            staged.append((temporary, output))

        # Check every destination before copying the first species.
        for temporary, output in staged:
            if output.exists() and (not output.is_file() or not filecmp.cmp(temporary, output, shallow=False)):
                raise ValueError(f'Existing file differs: {output}')
            if any(parent.exists() and not parent.is_dir() for parent in output.parents):
                raise ValueError(f'Parent path is not a directory: {output}')
        for temporary, output in staged:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                shutil.copyfile(temporary, output)


def verify_prediction_tables(destination):
    """Verify the released prediction tables and return the number checked."""
    score_dir = Path(destination) / 'scores'
    checked = 0
    for checkpoint_dir in ('humanv11', 'humanv12', 'mint/humanv11', 'mint/humanv12'):
        for species in SPECIES:
            path = score_dir / checkpoint_dir / species / 'predictions.csv'
            with path.open() as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != PREDICTION_COLUMNS:
                    raise ValueError(f'Unexpected prediction columns: {path}')
                row_count = 0
                for row in reader:
                    if (
                        None in row or int(row['row_idx']) != row_count or int(row['len_pair_ab']) <= 0
                        or row['label'] not in ('0', '1') or not 0 <= float(row['p_ab']) <= 1
                    ):
                        raise ValueError(f'Invalid prediction row {row_count}: {path}')
                    row_count += 1
            expected = 22_000 if species == 'ecoli' else 55_000
            if row_count != expected:
                raise ValueError(f'Unexpected row count: {path}: {row_count}')
            checked += 1
    return checked


def prepare_sequence_inputs(destination, config):
    """Read upstream sequence headers and prepare the V12 training CSV."""
    prepared = 0
    for source, directory in upstream_sources(config, destination):
        for name in source["files"]:
            original = safe_output(directory, name)
            with original.open(newline="", encoding="utf-8-sig") as handle:
                columns = next(csv.reader(handle), [])
            if not {"query", "text", "label"}.issubset(columns):
                raise ValueError(f"Missing sequence-pair columns: {original}")
            if original.suffix == ".tsv":
                # The released V12 training file is comma-delimited despite its suffix.
                output = safe_output(directory, Path(name).with_suffix(".csv").as_posix())
                shutil.copyfile(original, output)
                prepared += 1
    return prepared


def prepare_quick_start_examples(destination, seed):
    source = destination / "scores/humanv11/ecoli/predictions.csv"
    scores_by_label = {"0": [], "1": []}
    with source.open(newline="") as handle:
        for row in csv.DictReader(handle):
            scores_by_label[row["label"]].append(row["p_ab"])

    examples = destination / "examples"
    examples.mkdir(exist_ok=True)
    calibration_path = safe_output(examples, "calibration.csv")
    target_path = safe_output(examples, "target.csv")
    rng = random.Random(seed)
    with (
        calibration_path.open("w", newline="") as calibration_file,
        target_path.open("w", newline="") as target_file,
    ):
        calibration = csv.writer(calibration_file)
        target = csv.writer(target_file)
        calibration.writerow(["score", "label"])
        target.writerow(["score"])
        for label, scores in scores_by_label.items():
            rng.shuffle(scores)
            midpoint = len(scores) // 2
            calibration.writerows((score, label) for score in scores[:midpoint])
            target.writerows((score,) for score in scores[midpoint:])

    safe_output(examples, "README.txt").write_text(
        f"Quick Start example, seed {seed}: separate calibration and target halves of humanV11 E. coli scores.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--with-upstream', action='store_true', help='Also prepare original sequence inputs.')
    group.add_argument('--only-upstream', action='store_true', help='Prepare only original sequence inputs.')
    parser.add_argument('--mint-v11', type=Path, help='Convert original MINT V11 predictions.')
    parser.add_argument('--mint-v12', type=Path, help='Convert original MINT V12 predictions.')
    parser.add_argument('--example-seed', type=int, default=0)
    args = parser.parse_args()
    destination = args.data_dir.absolute()
    if args.with_upstream or args.only_upstream:
        config = json.loads((DATA_DIR / 'data_sources.json').read_text())
        prepared = prepare_sequence_inputs(destination, config)
        print(f'Prepared {prepared} training CSV files.')
    if not args.only_upstream:
        for version, archive in (("v11", args.mint_v11), ("v12", args.mint_v12)):
            if archive:
                convert_mint(archive, version, destination)
        count = verify_prediction_tables(destination)
        prepare_quick_start_examples(destination, args.example_seed)
        print(f'Checked {count} prediction tables. Quick Start examples: {destination / "examples"}')


if __name__ == '__main__':
    main()
