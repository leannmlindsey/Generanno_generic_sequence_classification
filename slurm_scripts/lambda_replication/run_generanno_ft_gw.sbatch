#!/bin/bash
#SBATCH --account=bfzj-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=gen_ft_gw
#SBATCH --output=gen_ft_gw_%j.log
#SBATCH --error=gen_ft_gw_%j.log
#
# GENERanno FT genome-wide (2k, 4k, 8k) — the FT head was never deployed
# genome-wide because a probe won winner-selection. Deploys the BEST fine-tuned
# seed's checkpoint across the genome-wide CSVs via the EXISTING inference entry
# point (python -m src.tasks.downstream.inference), writing to a dedicated dir so
# nothing is overwritten. Best seeds + test-MCC fingerprints (verify on Delta
# against each seed's test_results.json before trusting):
#   2k -> seed 2 (0.8973)   4k -> seed 3 (0.9271)   8k -> seed 3 (0.9450)
#
# Submit FROM the GENERanno repo root on Delta, with the generanno env active:
#   conda activate /work/hdd/bfzj/llindsey1/conda/envs/generanno
#   sbatch slurm_scripts/lambda_replication/run_generanno_ft_gw.sbatch
set -uo pipefail

echo "conda env: ${CONDA_DEFAULT_ENV:-none}   python: $(command -v python)  node $(hostname)  $(date)"
python -c "import torch, transformers" 2>/dev/null || { echo "ERROR: torch/transformers not importable — activate the generanno env before sbatch"; exit 1; }

export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/hf_cache}

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"; export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTROOT=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/Generanno_generic_sequence_classification/outputs
GWROOT=/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/LAMBDA_v1/genome_wide
VARIANT=generanno

seed_for () { case "$1" in 2k) echo 2;; 4k) echo 3;; 8k) echo 3;; esac; }
maxlen_for () { case "$1" in 2k) echo 2048;; 4k) echo 4096;; 8k) echo 8192;; esac; }

for w in 2k 4k 8k; do
  seed=$(seed_for "$w"); ML=$(maxlen_for "$w")
  CKPT="$OUTROOT/$w/finetune/$VARIANT/seed-$seed/best_model"
  OUT="$OUTROOT/$w/genome_wide_ft/$VARIANT"; mkdir -p "$OUT"
  [ -d "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }
  echo "===== $w  seed $seed  max_length $ML  ckpt $CKPT ====="
  n=0
  for csv in "$GWROOT/$w"/*.csv; do
    stem=$(basename "$csv" .csv)
    python -m src.tasks.downstream.inference \
      --input_csv="$csv" \
      --model_path="$CKPT" \
      --output_csv="$OUT/genome_wide_${stem}_predictions.csv" \
      --max_length=$ML --batch_size=16 --threshold=0.5 --save_metrics
    n=$((n+1))
  done
  echo "  $w done: $n genome(s) -> $OUT"
done
echo "DONE $(date)"
