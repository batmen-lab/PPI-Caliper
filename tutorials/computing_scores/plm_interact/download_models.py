#!/usr/bin/env python3
"""Download pinned PLM-Interact checkpoints and the ESM-2 tokenizer/backbone."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from computing_scores.plm_interact.config import CHECKPOINT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=CHECKPOINT_ROOT)
    return parser.parse_args()


def main(args):
    from huggingface_hub import snapshot_download

    assets = json.loads(Path(__file__).with_name("assets.json").read_text())
    destination = args.destination.resolve()
    for asset in assets:
        snapshot_download(
            repo_id=asset["repo_id"], repo_type=asset["repo_type"],
            revision=asset["revision"], local_dir=destination / asset["directory"],
            allow_patterns=asset.get("allow_patterns"),
        )


if __name__ == "__main__":
    main(parse_args())
