#!/usr/bin/env python3
"""Download PPI-Caliper scores and optional sequence data from pinned Hugging Face commits."""
import argparse
import json
import re
from pathlib import Path, PurePosixPath

DATA_DIR = Path(__file__).resolve().parent


def safe_output(root, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or not relative.parts
            or '\\' in name or relative.as_posix() != name):
        raise ValueError(f'Unsafe file path: {name}')
    path = Path(root).absolute() / relative
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f'Symbolic link in destination: {path}')
    return path


def upstream_files(config, destination):
    for source in config['upstream']:
        directory = source['directory']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', directory):
            raise ValueError(f'Invalid dataset directory: {directory}')
        yield source, Path(destination) / 'raw' / directory


def download_files(source, destination):
    from huggingface_hub import hf_hub_download
    if not re.fullmatch(r'[0-9a-f]{40}', source['revision']):
        raise ValueError('A full Hugging Face commit ID is required.')
    for name in source['files']:
        safe_output(destination, name)
        hf_hub_download(repo_id=source['repo_id'], repo_type='dataset',
                        revision=source['revision'], filename=name, local_dir=destination)
    print(f"Downloaded {len(source['files'])} files from {source['repo_id']} at {source['revision']}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DATA_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--with-upstream', action='store_true', help='Also download original sequence datasets.')
    group.add_argument('--only-upstream', action='store_true', help='Download only original sequence datasets.')
    args = parser.parse_args()
    config = json.loads((DATA_DIR / 'data_sources.json').read_text())
    if not args.only_upstream:
        download_files(config['predictions'], args.destination)
    if args.with_upstream or args.only_upstream:
        for source, destination in upstream_files(config, args.destination):
            download_files(source, destination)


if __name__ == '__main__':
    main()
