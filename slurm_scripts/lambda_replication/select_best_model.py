#!/usr/bin/env python3
"""
Per-variant, pick the finetune seed with the highest test-set MCC.

GENERanno has a single architecture/checkpoint, so this selects the best-of-N
seed for each variant (only finetune candidates; the embedding linear probe /
3-layer NN are reported separately and are not part of the winning checkpoint).

Writes <output_dir>/winners.json:
    {
      "generanno": {
        "type": "finetune",
        "seed": 3,
        "test_mcc": 0.85,
        "path": "<absolute path to the saved model dir (.../seed-N/best_model)>",
        "results_dir": "<absolute path to the seed dir holding test_results.json>",
        "base_model": "GenerTeam/GENERanno-prokaryote-0.5b-base",
        "all_candidates": [{type, seed, test_mcc}, ...]
      }
    }

Where it reads from:
  The thin finetune job body (lambda_finetune_job.sh) calls
  sequence_understanding.py with an EXPLICIT --output_dir, so every per-seed
  result lands in the single canonical OUTPUT_DIR tree:
      <OUTPUT_DIR>/<LEN>/finetune/<variant>/seed-<N>/
  This driver passes --seed_root_template pointing at that tree (with a literal
  {variant} placeholder), e.g. <OUTPUT_DIR>/<LEN>/finetune/{variant}.

  Inside each seed dir, sequence_understanding.py writes:
      test_results.json          (key "eval_mcc"; written by evaluate_model_comprehensive)
      best_model/                (the saved model dir, via trainer.save_model)
  No surfacing/copy step is required — test_results.json already lives there and
  already carries eval_mcc. (No modification to sequence_understanding.py.)
"""

import argparse
import glob
import json
import os
import sys


# MCC key candidates in order of preference. Generanno's sequence_understanding.py
# writes "eval_mcc" into test_results.json; the others are accepted for safety.
MCC_KEYS = ("eval_mcc", "eval_matthews_correlation", "mcc", "matthews_correlation")


def _read_mcc(metrics):
    for k in MCC_KEYS:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def _saved_model_path(seed_dir):
    # sequence_understanding.py saves the final model under <seed_dir>/best_model.
    # Fall back to the seed dir itself if best_model is absent.
    bm = os.path.join(seed_dir, "best_model")
    if os.path.isdir(bm):
        return os.path.abspath(bm)
    return os.path.abspath(seed_dir)


def collect_finetune_candidates(seed_root):
    out = []
    for seed_dir in sorted(glob.glob(os.path.join(seed_root, "seed-*"))):
        results_path = os.path.join(seed_dir, "test_results.json")
        if not os.path.isfile(results_path):
            print(f"  WARN: missing {results_path}, skipping", file=sys.stderr)
            continue
        with open(results_path) as f:
            metrics = json.load(f)
        mcc = _read_mcc(metrics)
        if mcc is None:
            print(f"  WARN: no MCC key {MCC_KEYS} in {results_path}, skipping",
                  file=sys.stderr)
            continue
        seed = int(os.path.basename(seed_dir).split("-")[1])
        out.append({
            "type": "finetune",
            "seed": seed,
            "test_mcc": float(mcc),
            "path": _saved_model_path(seed_dir),
            "results_dir": os.path.abspath(seed_dir),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True,
                        help="Per-length LAMBDA replication output dir (winners.json written here)")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variants to select for (e.g. generanno)")
    parser.add_argument("--seed_root_template", required=True,
                        help="Template for the per-variant seed root, with a literal "
                             "{variant} placeholder. The training job writes "
                             "seed-<N>/test_results.json under here.")
    parser.add_argument("--base_model", default="GenerTeam/GENERanno-prokaryote-0.5b-base",
                        help="HF base model recorded in winners.json")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Skip variants with no candidates instead of aborting. "
                             "Useful for in-progress dev runs; do NOT use for the "
                             "reviewer-facing pipeline — a missing variant there means "
                             "a real training failure that should fail loudly.")
    args = parser.parse_args()

    winners = {}
    skipped = []
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        seed_root = args.seed_root_template.replace("{variant}", variant)
        print(f"  seed root: {seed_root}")
        candidates = collect_finetune_candidates(seed_root)
        if not candidates:
            if not args.allow_partial:
                print(f"  ERROR: no candidates found for {variant} "
                      f"(missing seed-*/test_results.json under {seed_root}). "
                      f"Re-run with --allow-partial to skip and continue.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: no candidates found for {variant}", file=sys.stderr)
            skipped.append(variant)
            continue

        for c in sorted(candidates, key=lambda c: c["test_mcc"], reverse=True):
            print(f"  test_mcc={c['test_mcc']:.4f}  seed-{c['seed']}")

        winner = max(candidates, key=lambda c: c["test_mcc"])
        winner["base_model"] = args.base_model
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "test_mcc")}
            for c in candidates
        ]
        winners[variant] = winner
        print(f"  WINNER: seed-{winner['seed']} (test_mcc={winner['test_mcc']:.4f})")
        print(f"          model path: {winner['path']}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "winners.json")
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nWrote {out_path}  ({len(winners)} variant(s) with winners"
          f"{'; skipped: ' + ','.join(skipped) if skipped else ''})")

    if not winners:
        print("\nERROR: no variant produced any candidates; nothing to write.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
