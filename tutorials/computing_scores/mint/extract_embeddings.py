#!/usr/bin/env python3
"""Extract frozen MINT embeddings for human head training and validation."""
import argparse
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from computing_scores.mint.config import CHECKPOINT_ROOT, RAW_ROOT, RESULTS_DIR, training_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v11", "v12"], required=True)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--validation-csv", type=Path)
    parser.add_argument("--encoder", type=Path, default=CHECKPOINT_ROOT / "mint.ckpt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0", help="One device, for example cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0, help="Random sequence-crop seed")
    return parser.parse_args()


def main(args):
    train, validation = training_files(args.version, args.raw_root)
    inputs = {"train": args.train_csv or train, "validation": args.validation_csv or validation}
    output = args.output_dir or RESULTS_DIR / "embeddings" / args.version

    import torch
    from computing_scores.mint.data import PairDataset
    from computing_scores.mint.model import embedding_batches, load_encoder

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder = load_encoder(args.encoder, args.device)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for split, path in inputs.items():
            dataset = PairDataset(path, args.max_seq_length)
            batches = embedding_batches(encoder, dataset, args.device, args.batch_size, split)
            embeddings = torch.cat([batch.cpu() for batch in batches])
            torch.save({"embeddings": embeddings,
                        "labels": torch.tensor(dataset.frame["label"].to_numpy())}, output / f"{split}.pt")
    print(f"Wrote embeddings to {output}")


if __name__ == "__main__":
    main(parse_args())
