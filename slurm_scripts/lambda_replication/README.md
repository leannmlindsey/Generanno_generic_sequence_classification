# GENERanno — LAMBDA_v1 replication

An orchestration layer with **thin job bodies** that call the repo's **existing**
module entry points directly — with explicit output control — over the LAMBDA_v1
benchmark: finetune across seeds per window, pick the best seed by test-set MCC,
then run all diagnostic + genome-wide inference and a pretrained embedding
analysis. It does **not** modify any experiment code (the
`src/tasks/downstream/*.py` modules).

The thin job bodies invoke:

- `python -m src.tasks.downstream.sequence_understanding` (finetune)
- `python -m src.tasks.downstream.inference` (inference)
- `python -m src.tasks.downstream.embedding_analysis` (embedding analysis)

each from the repo root with `PYTHONPATH=<repo root>` and an **explicit
`--output_dir` / `--output_csv`**, so every output lands in the single canonical
`OUTPUT_DIR` tree under canonical, harvest-compatible names.

## Two-step workflow

```bash
# 0. (one time) pre-warm the HF cache from a LOGIN node — jobs run offline.
#    Set HF_HOME=/data/lindseylm/.cache/huggingface and download
#    GenerTeam/GENERanno-prokaryote-0.5b-base once.

# 1. Edit lambda_replication.conf — confirm LAMBDA_BASE + OUTPUT_DIR.
bash slurm_scripts/lambda_replication/run_lambda_training.sh   # finetune × seeds × windows
#    wait — squeue -u $USER
bash slurm_scripts/lambda_replication/check_training.sh        # confirm all seeds healthy

# 2. pick winners + run all inference/embeddings
bash slurm_scripts/lambda_replication/run_lambda_inference.sh
#    wait — squeue -u $USER
bash slurm_scripts/lambda_replication/check_inference.sh       # confirm all outputs landed
```

## Files

| File | Role |
|------|------|
| `lambda_replication.conf` | the only file you normally edit — paths + hyperparameters |
| `run_lambda_training.sh` | submit one finetune job per (variant × window × seed) via `lambda_finetune_job.sh` |
| `lambda_finetune_job.sh` | thin sbatch body: one finetune run (`sequence_understanding`) into the canonical tree |
| `select_best_model.py` | pick best-of-N seed per variant by **test-set MCC** (`eval_mcc`) → `winners.json` |
| `run_lambda_inference.sh` | select winners, submit embedding + one inference job per diagnostic / genome-wide CSV with **canonical** output names |
| `lambda_inference_job.sh` | thin sbatch body: one inference run (`inference`) with an explicit `--output_csv` |
| `lambda_embedding_job.sh` | thin sbatch body: pretrained-embedding analysis (`embedding_analysis`) |
| `print_winner_exports.py` | emit shell exports for the winning checkpoint (ad-hoc reuse) |
| `check_training.sh` / `check_inference.sh` | post-hoc verification helpers |

## Single canonical output tree

Everything lands under the conf's `OUTPUT_DIR` (no repo-internal
`results/csv_binary/...` split layout):

```
$OUTPUT_DIR/<LEN>/finetune/<variant>/seed-<N>/   test_results.json + best_model/
$OUTPUT_DIR/<LEN>/embedding/<variant>/           embedding_analysis_results.json + .npz
$OUTPUT_DIR/<LEN>/winners.json                   picked by run_lambda_inference.sh
$OUTPUT_DIR/<LEN>/inference/<variant>/           canonical *_predictions.csv (+ _metrics.json)
$OUTPUT_DIR/logs/                                SLURM stdout/stderr (shared)
```

## Canonical output names (required by central `harvest`)

A central aggregator called **harvest** consumes results from ALL model repos
and REQUIRES identical naming across them. The inference driver controls the
output filename per item (it does NOT name by input basename), producing exactly:

| Diagnostic | Input CSV | Output CSV (+ `_predictions_metrics.json`) |
|------------|-----------|---------------------------------------------|
| test | `train_val_test/<LEN>/test.csv` | `test_predictions.csv` |
| fpr | `fpr_test/<LEN>/bacteria_segments_<LEN>.csv` | `fpr_predictions.csv` |
| gc_control | `shuffled_controls/<LEN>/test_shuffled.csv` | `gc_control_predictions.csv` |
| fnr | `FNR_<LEN>` (if set) | `fnr_predictions.csv` |
| genome-wide | each `*.csv` in `GENOME_WIDE_<LEN>` | `genome_wide_<stem>_predictions.csv` |

`<stem>` is the genome-wide input CSV basename without `.csv`. The
`genome_wide_` PREFIX is what harvest globs for.

**Aggregation / clustering is done centrally by harvest**, not per-repo. This
layer only emits the prediction CSVs (+ their `_metrics.json`); there is no local
clustering / aggregation / analysis job.

## GENERanno-specific notes

- **Module-run + PYTHONPATH.** The thin job bodies `cd` to the repo root, set
  `PYTHONPATH=<repo root>`, and call the modules as
  `python -m src.tasks.downstream.{sequence_understanding,inference,embedding_analysis}`.
- **Single variant.** GENERanno has one architecture/checkpoint
  (`GenerTeam/GENERanno-prokaryote-0.5b-base`), so `VARIANTS="generanno"`. The
  variant loop is kept only to mirror the shared
  `outputs/<LEN>/finetune/<variant>/` layout used by the other model repos.
- **Windows 2k / 4k / 8k.** GENERanno has an **8k base-pair context**, so all
  three windows are supported. The 8k LAMBDA inputs (`FNR_8k`,
  `GENOME_WIDE_8k`) **exist in LAMBDA_v1** and are wired in. The inference driver
  still warn-and-skips any diagnostic file/dir that happens to be missing
  (defensive; not fatal).
- **Per-window MAX_LENGTH = bp length.** GENERanno uses a single-nucleotide
  (character/byte-level) tokenizer over A/C/G/T/N, so 1 bp ≈ 1 token. The conf
  sets `MAX_LENGTH_2k=2048`, `MAX_LENGTH_4k=4096`, `MAX_LENGTH_8k=8192` (8192 is
  the model's full context).
- **Explicit output control (no path indirection).** The finetune job body passes
  an explicit `--output_dir`, so each seed lands at
  `$OUTPUT_DIR/<LEN>/finetune/<variant>/seed-<N>/` (`test_results.json` +
  `best_model/`). `select_best_model.py` and the checkers read from there;
  `winners.json`, inference outputs, and logs live under the same `OUTPUT_DIR`.
- **Metric key.** `sequence_understanding.py` already writes per-seed
  `test_results.json` containing `eval_mcc` directly into `--output_dir` — no
  surfacing/copy step is needed.
- **`val.csv` is read natively.** `sequence_understanding.py` and
  `embedding_analysis.py` accept either `dev.csv` or `val.csv` in `--csv_dir`, so
  no staging/symlink step is needed (LAMBDA_v1 ships `val.csv`).
- **Diagnostic interpretation.** Inference metrics are uniform across datasets;
  interpret per dataset: MCC for the mixed **test** set; **FPR = 1 − accuracy**
  on the bacteria-only fpr set; **FNR = 1 − accuracy** on the phage-only fnr set.
- **Bare job scripts.** No `set -euo pipefail`, no `2>/dev/null` masking in the
  job bodies — `source activate` under `set -e` silently kills SLURM jobs.
- Outputs go under `/data/lindseylm/...`, never `/gpfs/gsfs12/...`.
