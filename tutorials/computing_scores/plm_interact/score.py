#!/usr/bin/env python3
"""Compute PLM-Interact prediction scores from released checkpoints."""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from computing_scores.plm_interact.config import (
    BASE_MODEL, CHECKPOINT_ROOT, MAX_LENGTH, MODEL_REPOSITORIES,
    RAW_ROOT, RESULTS_DIR, SPECIES, checkpoint_path, dataset_path, model_directory,
)
from computing_scores.plm_interact.data import eligible_frame, load_ppi_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_REPOSITORIES)
    parser.add_argument("--species", nargs="+", choices=SPECIES, default=list(SPECIES))
    parser.add_argument("--cache-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--data-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bf16", "float16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--row-limit", type=int)
    return parser.parse_args()


def _autocast_context(torch, device, dtype: str):
    if device.type != "cuda" or dtype == "float32":
        return nullcontext()
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def _adaptive_batch_size(max_tokens: int, requested: int) -> int:
    if max_tokens <= 400:
        return min(requested, 64)
    if max_tokens <= 800:
        return min(requested, 32)
    if max_tokens <= 1200:
        return min(requested, 16)
    return min(requested, 8)


def _score_sorted_chunk(
    frame: pd.DataFrame,
    *,
    model,
    tokenizer,
    device,
    requested_batch_size: int,
    dtype: str,
) -> pd.DataFrame:
    import torch
    ordered = frame.sort_values(["len_pair_ab", "row_idx"]).reset_index(drop=True)
    records: list[pd.DataFrame] = []
    cursor = 0
    with torch.inference_mode():
        while cursor < len(ordered):
            remaining = ordered.iloc[cursor : cursor + requested_batch_size]
            batch_size = _adaptive_batch_size(int(remaining["len_pair_ab"].max()), requested_batch_size)
            batch = ordered.iloc[cursor : cursor + batch_size].copy()
            while True:
                try:
                    tokens = tokenizer(
                        batch["query"].tolist(),
                        batch["text"].tolist(),
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    ).to(device)
                    with _autocast_context(torch, device, dtype):
                        probs = model.forward_test(tokens).float().cpu().numpy()
                    batch["p_ab"] = probs.astype(np.float64)
                    records.append(batch.loc[:, ["row_idx", "label", "len_pair_ab", "p_ab"]])
                    cursor += len(batch)
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or len(batch) == 1:
                        raise
                    torch.cuda.empty_cache()
                    batch = batch.iloc[: max(1, len(batch) // 2)].copy()
    result = pd.concat(records, ignore_index=True).sort_values("row_idx").reset_index(drop=True)
    if result["p_ab"].isna().any() or not result["p_ab"].between(0, 1).all():
        raise ValueError("invalid p_ab values")
    return result


def score_species(
    *,
    model_key: str,
    species: str,
    data_root: Path,
    out_root: Path,
    model,
    tokenizer,
    device,
    batch_size: int,
    chunk_size: int,
    max_length: int,
    row_limit: int | None,
    dtype: str,
) -> Path:
    input_path = dataset_path(data_root, species)
    full = load_ppi_frame(input_path)
    eligible, excluded = eligible_frame(full, max_length)
    if row_limit is not None:
        if row_limit <= 0:
            raise ValueError("row_limit must be positive")
        eligible = eligible.iloc[:row_limit].copy().reset_index(drop=True)
    out_dir = out_root / species
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    expected_parts: list[Path] = []
    iterator = range(0, len(eligible), chunk_size)
    for start in tqdm(iterator, desc=f"{model_key}/{species}", unit="chunk"):
        stop = min(start + chunk_size, len(eligible))
        part = parts_dir / f"rows_{start:06d}_{stop:06d}.csv"
        expected_parts.append(part)
        chunk = eligible.iloc[start:stop].copy()
        scored = _score_sorted_chunk(
            chunk,
            model=model,
            tokenizer=tokenizer,
            device=device,
            requested_batch_size=batch_size,
            dtype=dtype,
        )
        scored.to_csv(part, index=False)

    merged = pd.concat([pd.read_csv(path) for path in expected_parts], ignore_index=True)
    merged = merged.sort_values("row_idx").reset_index(drop=True)
    if len(merged) != len(eligible) or merged["row_idx"].duplicated().any():
        raise AssertionError("merged score parts are incomplete or duplicated")
    prediction_path = out_dir / "predictions.csv"
    merged.to_csv(prediction_path, index=False)
    summary = {
        "model": model_key,
        "species": species,
        "dataset": input_path.name,
        "n_input": int(len(full)),
        "n_scored": int(len(eligible)),
        "n_excluded_length": excluded,
        "n_positive": int((eligible["label"] == 1).sum()),
        "n_negative": int((eligible["label"] == 0).sum()),
        "max_length": max_length,
        "row_limit": row_limit,
        "dtype": dtype,
        "predictions": prediction_path.name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return prediction_path


def main(args):
    import torch
    from transformers import AutoTokenizer
    from computing_scores.plm_interact.model import load_ppi_model

    device = torch.device(args.device)
    print("DEVICE:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device.index or 0))
    model = load_ppi_model(
        cache_root=args.cache_root,
        checkpoint_path=checkpoint_path(args.cache_root, args.model),
        device=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_directory(args.cache_root, BASE_MODEL)))
    output_dir = args.output_dir or RESULTS_DIR / "scores" / args.model
    for species in args.species:
        output = score_species(
            model_key=args.model,
            species=species,
            data_root=args.data_root,
            out_root=output_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            max_length=args.max_length,
            row_limit=args.row_limit,
            dtype=args.dtype,
        )
        print("SAVED:", output)


if __name__ == "__main__":
    main(parse_args())
