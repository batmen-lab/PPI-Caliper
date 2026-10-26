#!/usr/bin/env python3
"""Train the official MINT prediction head on human protein-pair embeddings."""
import argparse
import csv
import random
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from computing_scores.mint.config import CHECKPOINT_ROOT, RESULTS_DIR, TASK_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v11", "v12"], required=True)
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main(args):
    import numpy as np
    import torch
    from mint.helpers.predict import SimpleMLP

    embeddings = args.embeddings_dir or RESULTS_DIR / "embeddings" / args.version
    output = args.output_dir or CHECKPOINT_ROOT / args.version
    output.mkdir(parents=True, exist_ok=True)
    bundles = {name: torch.load(embeddings / f"{name}.pt", map_location="cpu", weights_only=True)
               for name in ("train", "validation")}
    datasets = {name: torch.utils.data.TensorDataset(bundle["embeddings"], bundle["labels"])
                for name, bundle in bundles.items()}
    validation_loader = torch.utils.data.DataLoader(datasets["validation"], batch_size=args.batch_size)
    best_accuracy = -1.0
    for seed in range(args.seed_start, args.seed_start + args.repeats):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        loader = torch.utils.data.DataLoader(datasets["train"], batch_size=args.batch_size,
                                            shuffle=True, drop_last=True)
        model = SimpleMLP(bundles["train"]["embeddings"].shape[1]).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        checkpoint = output / f"{TASK_NAMES[args.version]}_mint_sep_rep{seed}_best.pth"
        repeat_accuracy = -1.0
        history = []
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for features, labels in loader:
                optimizer.zero_grad()
                logits = model(features.to(args.device)).squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.to(args.device).float())
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))
            scheduler.step(train_loss)
            model.eval()
            correct = 0
            with torch.inference_mode():
                for features, labels in validation_loader:
                    predictions = model(features.to(args.device)).squeeze(-1) >= 0
                    correct += (predictions.cpu() == labels).sum().item()
            accuracy = correct / len(datasets["validation"])
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "validation_accuracy": accuracy})
            if accuracy >= repeat_accuracy:
                repeat_accuracy = accuracy
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                            "seed": seed, "validation_accuracy": accuracy}, checkpoint)
            print(f"seed={seed} epoch={epoch + 1}/{args.epochs} loss={train_loss:.6f} validation_accuracy={accuracy:.6f}")
        with (output / f"epoch_metrics_seed{seed}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if repeat_accuracy > best_accuracy:
            best_accuracy, best_checkpoint = repeat_accuracy, checkpoint
    selected = output / f"{TASK_NAMES[args.version]}_mint_sep_final_overall.pth"
    shutil.copy2(best_checkpoint, selected)
    print(f"Selected {selected} (validation accuracy {best_accuracy:.6f})")


if __name__ == "__main__":
    main(parse_args())
