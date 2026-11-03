#!/usr/bin/env python3
"""Download PPI-Caliper scores and optional sequence data from pinned Hugging Face commits."""

import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def safe_output(root, name):
    """Resolve a relative manifest filename without following symbolic links."""
    if '\\' in name or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError(f'Unsafe file path: {name}')
    path = Path(root).absolute() / name
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError(f'Symbolic link in destination: {path}')
    return path


def upstream_sources(config, destination):
    for source in config['upstream']:
        directory = source['directory']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', directory):
            raise ValueError(f'Invalid dataset directory: {directory}')
        yield source, Path(destination) / 'raw' / directory


def download_files(source, destination):
    from huggingface_hub import hf_hub_download

    repo_id = source['repo_id']
    revision = source['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('A full Hugging Face commit ID is required.')
    for name in source['files']:
        safe_output(destination, name)
        hf_hub_download(
            repo_id=repo_id, repo_type='dataset', revision=revision,
            filename=name, local_dir=destination,
        )
    print(f"Downloaded {len(source['files'])} files from {repo_id} at {revision}.")


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
        for source, destination in upstream_sources(config, args.destination):
            download_files(source, destination)


if __name__ == '__main__':
    main()
