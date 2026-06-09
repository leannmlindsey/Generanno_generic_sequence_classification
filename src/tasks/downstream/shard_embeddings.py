#!/usr/bin/env python
"""Parallel sharded embedding extraction for the LAMBDA embedding analysis.

The single-process ``embedding_analysis.py`` extracts embeddings for the full
train/val/test splits, twice (pretrained model + random baseline). For long
windows (e.g. 8k) that is GPU-compute-bound and can take ~30h on one GPU.

This module splits the *extraction* across many GPU jobs (each handles one
contiguous row-slice of one split for one model), then concatenates the shards
— in original row order — into the EXACT files the analysis step caches:

    <output_dir>/embeddings_pretrained.npz
    <output_dir>/embeddings_random.npz

Once those exist, re-running ``embedding_analysis.py`` finds them, SKIPS all
extraction, and only runs the (fast) linear-probe + 3-layer-NN analysis.

It reuses the very same helpers (extract_embeddings, create_random_model,
load_csv_data) so the sharded embeddings are bit-for-bit equivalent to a
single-pass run: contiguous slices + in-order concat preserve row/label order,
and create_random_model is deterministic given the seed (so every random shard
sees the identical random model).

Modes
-----
  --mode extract  : extract ONE shard of one split (train|val|test) from one
                    model (pretrained|random). Writes a shard .npz.
  --mode combine  : concatenate all shards for one model into
                    embeddings_<which>.npz (validates full, gap-free coverage).
"""
import argparse
import glob
import os
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.tasks.downstream.embedding_analysis import (
    create_random_model,
    extract_embeddings,
    load_csv_data,
)

SPLITS = ("train", "val", "test")


def shard_bounds(n, num_shards, idx):
    """Contiguous balanced bounds for shard ``idx`` of ``num_shards`` over ``n``
    rows (same partition as numpy.array_split). Requires num_shards <= n, which
    the launcher guarantees, so every shard is non-empty."""
    base = n // num_shards
    rem = n % num_shards
    start = idx * base + min(idx, rem)
    end = start + base + (1 if idx < rem else 0)
    return start, end


def get_split_df(csv_dir, split):
    train_df, val_df, test_df = load_csv_data(csv_dir)
    return {"train": train_df, "val": val_df, "test": test_df}[split]


def do_extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = get_split_df(args.csv_dir, args.split)
    n = len(df)
    if args.shard_index >= args.num_shards:
        raise SystemExit(
            f"shard_index {args.shard_index} >= num_shards {args.num_shards}")
    if args.num_shards > n:
        raise SystemExit(
            f"num_shards {args.num_shards} > rows {n} for split '{args.split}' "
            f"(would create empty shards)")
    start, end = shard_bounds(n, args.num_shards, args.shard_index)
    seqs = df["sequence"].tolist()[start:end]
    labels = df["label"].tolist()[start:end]
    print(f"[shard] which={args.which} split={args.split} "
          f"idx={args.shard_index}/{args.num_shards} rows[{start}:{end}] of {n}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.which == "pretrained":
        print(f"Loading pretrained model from: {args.model_path}")
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
        model = model.to(device)
        model.eval()
    else:
        # Same seed offset as embedding_analysis.py so the random model is
        # identical across all shards.
        model = create_random_model(args.model_path, device, seed=args.seed + 1000)

    emb, lab = extract_embeddings(
        model, tokenizer, seqs, labels,
        args.batch_size, args.max_length, args.pooling, device,
    )

    shard_dir = os.path.join(args.output_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    out = os.path.join(
        shard_dir,
        f"{args.which}_{args.split}_{args.shard_index:04d}of{args.num_shards:04d}.npz",
    )
    np.savez(out, embeddings=emb, labels=lab,
             start=start, end=end, n=n, num_shards=args.num_shards)
    print(f"[shard] wrote {out}  emb={emb.shape}")


def _load_split_shards(shard_dir, which, split):
    pattern = os.path.join(shard_dir, f"{which}_{split}_*.npz")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No shards found: {pattern}")

    rx = re.compile(rf"{re.escape(which)}_{re.escape(split)}_(\d+)of(\d+)\.npz$")

    def idx_of(p):
        m = rx.search(os.path.basename(p))
        return int(m.group(1))

    files.sort(key=idx_of)

    embs, labs = [], []
    expected_start = 0
    total = None
    num_shards = None
    for p in files:
        d = np.load(p)
        s, e, nn, ns = int(d["start"]), int(d["end"]), int(d["n"]), int(d["num_shards"])
        if total is None:
            total, num_shards = nn, ns
        if s != expected_start:
            raise ValueError(
                f"{which}/{split}: shard gap/overlap — expected start "
                f"{expected_start}, got {s} ({os.path.basename(p)})")
        embs.append(d["embeddings"])
        labs.append(d["labels"])
        expected_start = e
    if len(files) != num_shards:
        raise ValueError(
            f"{which}/{split}: found {len(files)} shards, expected {num_shards} "
            f"— some shard jobs did not complete")
    if expected_start != total:
        raise ValueError(
            f"{which}/{split}: shards cover {expected_start} rows, expected {total}")
    return np.vstack(embs), np.concatenate(labs)


def do_combine(args):
    shard_dir = os.path.join(args.output_dir, "_shards")
    data = {}
    for split in SPLITS:
        emb, lab = _load_split_shards(shard_dir, args.which, split)
        data[f"{split}_embeddings"] = emb
        data[f"{split}_labels"] = lab
        print(f"[combine] {args.which}/{split}: emb={emb.shape} labels={lab.shape}")
    out = os.path.join(args.output_dir, f"embeddings_{args.which}.npz")
    np.savez(out, **data)
    print(f"[combine] wrote {out}")


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["extract", "combine"])
    p.add_argument("--which", required=True, choices=["pretrained", "random"])
    p.add_argument("--output_dir", required=True,
                   help="Same dir embedding_analysis.py uses (…/embedding/<variant>)")
    # extract-only
    p.add_argument("--csv_dir")
    p.add_argument("--model_path")
    p.add_argument("--split", choices=list(SPLITS))
    p.add_argument("--num_shards", type=int)
    p.add_argument("--shard_index", type=int)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_arguments()
    if args.mode == "extract":
        missing = [r for r in ("csv_dir", "model_path", "split",
                               "num_shards", "shard_index")
                   if getattr(args, r) is None]
        if missing:
            raise SystemExit("extract mode requires: " + ", ".join(missing))
        do_extract(args)
    else:
        do_combine(args)


if __name__ == "__main__":
    main()
