#!/usr/bin/env python3
"""
Per-variant, pick the best model to deploy for genome-wide + diagnostic inference:
the highest-scoring of {fine-tuning, 3-layer NN probe, linear probe}. Fine-tuning
is scored by the MEAN test MCC of its 5 seeds and, if it wins, is deployed via its
single best seed; the probes are scored by their test MCC. FT is chosen only when
its 5-seed average is >= both probes (mirrors the ProkBERT reference).

When a probe wins, run_lambda_inference.sh deploys it via inference_embedding_head.py
(base model + the saved probe artifacts from embedding_analysis.py); no FT checkpoint.

Writes <output_dir>/winners.json, e.g.:
    { "generanno": {                         # FT winner
        "type": "finetune", "seed": 3, "test_mcc": 0.93 (=5-seed mean),
        "best_seed_test_mcc": 0.94,
        "path": ".../seed-3/best_model", "base_model": "...", "all_candidates": [...] } }
    { "generanno": {                         # probe winner
        "type": "three_layer_nn" | "linear_probe", "test_mcc": 0.97,
        "head_path": ".../three_layer_nn_pretrained.pt",
        "scaler_path": ".../three_layer_nn_pretrained_scaler.pkl",  # NN only
        "base_model": "...", "all_candidates": [...] } }

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


def read_embedding_scores(embedding_dir):
    """Linear-probe / 3-layer-NN candidates from embedding_analysis_results.json.

    embedding_analysis.py writes FLAT keys (pretrained_linear_probe_mcc /
    pretrained_nn_mcc) — the Table 2 test-MCC numbers — plus the deployable probe
    artifacts saved alongside (linear_probe_pretrained.pkl,
    three_layer_nn_pretrained.pt + three_layer_nn_pretrained_scaler.pkl).
    """
    results_path = os.path.join(embedding_dir, "embedding_analysis_results.json")
    if not os.path.isfile(results_path):
        print(f"  WARN: missing {results_path} (no probe candidates)", file=sys.stderr)
        return []
    with open(results_path) as f:
        r = json.load(f)
    out = []
    lp = r.get("pretrained_linear_probe_mcc")
    if lp is not None:
        out.append({"type": "linear_probe", "seed": None, "test_mcc": float(lp),
                    "head_path": os.path.abspath(
                        os.path.join(embedding_dir, "linear_probe_pretrained.pkl"))})
    nn = r.get("pretrained_nn_mcc")
    if nn is not None:
        out.append({"type": "three_layer_nn", "seed": None, "test_mcc": float(nn),
                    "head_path": os.path.abspath(
                        os.path.join(embedding_dir, "three_layer_nn_pretrained.pt")),
                    "scaler_path": os.path.abspath(
                        os.path.join(embedding_dir, "three_layer_nn_pretrained_scaler.pkl"))})
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
        embedding_dir = os.path.join(args.output_dir, "embedding", variant)
        print(f"  seed root: {seed_root}")
        print(f"  embedding: {embedding_dir}")

        ft = collect_finetune_candidates(seed_root)
        emb = read_embedding_scores(embedding_dir)

        # Build selection candidates: fine-tuning scored by the MEAN of its 5 seeds
        # (deployed via the single best seed); each probe scored by its test MCC.
        sel = []
        if ft:
            ft_avg = sum(c["test_mcc"] for c in ft) / len(ft)
            best_seed = max(ft, key=lambda c: c["test_mcc"])
            sel.append({"type": "finetune", "score": float(ft_avg), "seed": best_seed["seed"],
                        "test_mcc": float(ft_avg), "best_seed_test_mcc": best_seed["test_mcc"],
                        "path": best_seed["path"], "results_dir": best_seed["results_dir"]})
        for cand in emb:
            c = dict(cand); c["score"] = c["test_mcc"]
            sel.append(c)

        if not sel:
            if not args.allow_partial:
                print(f"  ERROR: no candidates for {variant} (no finetune seeds AND no "
                      f"embedding_analysis_results.json). Re-run with --allow-partial to skip.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: no candidates for {variant}", file=sys.stderr)
            skipped.append(variant)
            continue

        def _tag(c):
            return c["type"] + (f"/seed-{c['seed']}" if c.get("seed") is not None else "")
        for c in sorted(sel, key=lambda c: c["score"], reverse=True):
            note = "  (mean of 5 seeds)" if c["type"] == "finetune" else ""
            print(f"  score={c['score']:.4f}  {_tag(c)}{note}")

        # Highest score wins; ties prefer finetune (deploy FT only when its 5-seed
        # average is >= both probes), per the LAMBDA design.
        winner = max(sel, key=lambda c: (c["score"], c["type"] == "finetune"))
        winner["base_model"] = args.base_model
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "test_mcc", "score")}
            for c in sel
        ]
        winners[variant] = winner
        print(f"  WINNER: {_tag(winner)} (score={winner['score']:.4f})")

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
