#!/usr/bin/env python3
"""Check prediction tables and prepare labeled/unlabeled Quick Start examples.

Optional --mint-v11/--mint-v12 converts original MINT ZIPs into the common
row_idx,label,len_pair_ab,p_ab schema without shuffling or deduplicating pairs.
"""
import argparse
import csv
import io
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from download_data import safe_output, upstream_files

DATA_DIR = Path(__file__).resolve().parent
SPECIES = ('ecoli', 'fly', 'mouse', 'worm', 'yeast')

def convert_mint(archive, version, destination):
    """Convert all five species without reordering rows or replacing other data."""
    destination = Path(destination).absolute()
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix='ppi-mint-') as temp:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('MINT archive contains duplicate member names')
        staged = []
        for species in SPECIES:
            name = f'result_{version}/{species}.ppi.qrels.seq.test_result.csv'
            with z.open(name) as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig'))
                columns = reader.fieldnames or []
                if len(columns) != len(set(columns)):
                    raise ValueError(f'Duplicate columns: {name}')
                output = safe_output(destination, f'scores/mint/human{version}/{species}/predictions.csv')
                temporary = Path(temp) / f'{species}.csv'
                with temporary.open('w', newline='') as out:
                    writer = csv.writer(out); writer.writerow(['row_idx','label','len_pair_ab','p_ab'])
                    count = 0
                    for idx, row in enumerate(reader):
                        if None in row or row['label'] not in ('0','1') or not 0 <= float(row['mint_score']) <= 1:
                            raise ValueError(f'Invalid label or score: {name}, row {idx}')
                        if not row['query'] or not row['text']: raise ValueError(f'Empty sequence: {name}, row {idx}')
                        writer.writerow([idx,row['label'],len(row['query'])+len(row['text']),row['mint_score']])
                        count += 1
                    if not count:
                        raise ValueError(f'Empty MINT prediction table: {name}')
                staged.append((temporary, output))
        for temporary, output in staged:
            if output.exists() and (not output.is_file() or output.read_bytes() != temporary.read_bytes()):
                raise ValueError(f'Existing file differs: {output}')
            if any(parent.exists() and not parent.is_dir() for parent in output.parents):
                raise ValueError(f'Parent path is not a directory: {output}')
        for temporary, output in staged:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists(): shutil.copyfile(temporary, output)

def verify_data(destination):
    destination = Path(destination)
    for sub in ['', 'mint/']:
        for version in ['humanv11', 'humanv12']:
            for species in SPECIES:
                path = destination / 'scores' / sub / version / species / 'predictions.csv'
                with path.open() as handle:
                    reader = csv.DictReader(handle); count = 0
                    if reader.fieldnames != ['row_idx','label','len_pair_ab','p_ab']:
                        raise ValueError(f'Unexpected prediction columns: {path}')
                    for idx, row in enumerate(reader):
                        if (None in row or int(row['len_pair_ab']) <= 0 or int(row['row_idx']) != idx
                                or row['label'] not in ('0','1') or not 0 <= float(row['p_ab']) <= 1):
                            raise ValueError(f'Invalid prediction row {idx}: {path}')
                        count += 1
                    expected = 22000 if species == 'ecoli' else 55000
                    if count != expected: raise ValueError(f'Unexpected row count: {path}: {count}')
    return 20

def prepare_csv_inputs(destination, config):
    """Prepare the V12 training CSV filename used by the scoring scripts."""
    prepared = []
    for source, directory in upstream_files(config, destination):
        for name in source['files']:
            original = safe_output(directory, name)
            with original.open(newline='', encoding='utf-8-sig') as handle:
                columns = next(csv.reader(handle), [])
            if not {'query', 'text', 'label'}.issubset(columns):
                raise ValueError(f'Missing sequence-pair columns: {original}')
            if original.suffix == '.tsv':
                output = original.with_suffix('.csv')
                shutil.copyfile(original, output)
                prepared.append(output)
    return prepared


def prepare_example(destination, seed):
    source = destination / 'scores/humanv11/ecoli/predictions.csv'
    with source.open() as handle: rows = list(csv.DictReader(handle))
    groups = {label: [row for row in rows if row['label'] == label] for label in ['0','1']}
    rng = random.Random(seed); calibration=[]; target=[]
    for group in groups.values():
        rng.shuffle(group); midpoint = len(group)//2
        calibration.extend(group[:midpoint]); target.extend(group[midpoint:])
    out = destination / 'examples'; out.mkdir(exist_ok=True)
    for name, items, columns in [('calibration.csv',calibration,['score','label']), ('target.csv',target,['score'])]:
        path = safe_output(out, name)
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader()
            for row in items: writer.writerow({key: row['p_ab'] if key == 'score' else row[key] for key in columns})
    safe_output(out,'README.txt').write_text(f'Quick Start example, seed {seed}: separate calibration and target halves of humanV11 E. coli scores.\n')

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
        prepared = prepare_csv_inputs(destination, config)
        print(f'Prepared {len(prepared)} training CSV files.')
    if not args.only_upstream:
        for version in ('v11', 'v12'):
            archive = getattr(args, 'mint_' + version)
            if archive:
                convert_mint(archive, version, destination)
        count = verify_data(destination)
        prepare_example(destination, args.example_seed)
        print(f'Checked {count} prediction tables. Quick Start examples: {destination / "examples"}')

if __name__ == '__main__':
    main()
