#!/usr/bin/env python3
"""Compute one continuous MINT PPI score per sequence pair."""
import argparse
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from computing_scores.mint.config import CHECKPOINT_ROOT, RAW_ROOT, RESULTS_DIR, SPECIES, head_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v11", "v12"], required=True)
    parser.add_argument("--species", nargs="+", choices=SPECIES, default=list(SPECIES))
    parser.add_argument("--test-dir", type=Path, default=RAW_ROOT / "cross_species_benchmarking/test")
    parser.add_argument("--dataset", type=Path, help="Score a custom CSV instead of the five species")
    parser.add_argument("--encoder", type=Path, default=CHECKPOINT_ROOT / "mint.ckpt")
    parser.add_argument("--head", type=Path, help="Matched MLP checkpoint; overrides version default")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0", help="One device, for example cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0, help="Random sequence-crop seed, reset for each input")
    return parser.parse_args()


def main(args):
    head_file = args.head or head_path(args.version)
    inputs = {"custom": args.dataset} if args.dataset else {
        species: args.test_dir / f"{species}.ppi.qrels.seq.test.csv" for species in args.species
    }
    output = args.output_dir or RESULTS_DIR / "scores" / args.version

    import numpy as np
    import pandas as pd
    import torch
    from mint.helpers.extract import CollateFn
    from tqdm import tqdm
    from computing_scores.mint.data import PairDataset
    from computing_scores.mint.model import load_encoder, load_head

    encoder = load_encoder(args.encoder, args.device)
    head = load_head(head_file, args.device)
    with torch.inference_mode():
        for name, path in inputs.items():
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            dataset = PairDataset(path, args.max_seq_length)
            loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                               collate_fn=CollateFn())
            scores = []
            for chains, chain_ids in tqdm(loader, desc=name):
                embeddings = encoder(chains.to(args.device), chain_ids.to(args.device))
                scores.append(torch.sigmoid(head(embeddings).squeeze(-1)).cpu().numpy())
            values = np.concatenate(scores)
            result = pd.DataFrame({"row_idx": np.arange(len(dataset)), "p_ab": values})
            if dataset.has_labels:
                result.insert(1, "label", dataset.frame["label"].to_numpy())
            destination = output / name / "predictions.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(destination, index=False)
            print(f"Wrote {destination}")


if __name__ == "__main__":
    main(parse_args())
