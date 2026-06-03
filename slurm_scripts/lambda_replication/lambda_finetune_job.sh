#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Stage 1 of GENERanno LAMBDA replication: finetune ONE (variant, seed).
# Submitted by run_lambda_training.sh. All paths/resources come via --export.
#
# This is a THIN orchestration job body: it calls the EXISTING module entry
# point `python -m src.tasks.downstream.sequence_understanding` directly with an
# EXPLICIT --output_dir, so every per-seed result lands in the single canonical
# OUTPUT_DIR tree (NOT the repo's internal results/csv_binary/... tree). It does
# NOT modify any experiment code.
#
# Required env:
#   REPO_ROOT          repo root (holds src/, configs/)
#   REPL_OUTPUT_DIR    per-length replication output dir (OUTPUT_DIR/<LEN>)
#   CSV_DIR            train/dev(or val)/test CSV directory (LAMBDA_v1 train_val_test/<LEN>)
#   VARIANT            generanno
#   SEED               integer
#   MAX_LENGTH         max token length for this window (bp == tokens)
# Optional env (with defaults):
#   BASE_MODEL, LR, BATCH_SIZE, D_OUTPUT, MAIN_METRICS, EARLY_STOPPING_PATIENCE,
#   GRAD_ACCUM, PROBLEM_TYPE, CONDA_ENV (generanno_env), HF_HOME


echo "=== finetune ${VARIANT} seed=${SEED} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# Activate conda (bare style — no set -e; `source activate` under set -e silently
# kills SLURM jobs). No 2>/dev/null masking either.
module load conda
module load CUDA/12.8
if [ -z "${CUDA_HOME}" ]; then
    NVCC_PATH=$(which nvcc)
    if [ -n "${NVCC_PATH}" ]; then
        export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
    fi
fi
source activate "${CONDA_ENV:-generanno_env}"
echo "  conda env: ${CONDA_DEFAULT_ENV:-<none>}   python: $(command -v python || echo none)"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Stay offline so the Biowulf HTTPS proxy can't 503 us mid-run. Cache must be
# pre-warmed from a login node (see lambda_replication/README.md).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/data/lindseylm/.cache/huggingface}

# REPO_ROOT is supplied by the launcher via --export. SLURM stages this script
# to /var/spool/slurm/... so BASH_SOURCE[0] would not resolve to the real repo.
# The modules are invoked as `python -m src.tasks...` so we must cd to the repo
# root and put it on PYTHONPATH.
if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-GenerTeam/GENERanno-prokaryote-0.5b-base}
LR=${LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-16}
D_OUTPUT=${D_OUTPUT:-2}
MAIN_METRICS=${MAIN_METRICS:-mcc}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}
GRAD_ACCUM=${GRAD_ACCUM:-1}
PROBLEM_TYPE=${PROBLEM_TYPE:-single_label_classification}
MAX_LENGTH=${MAX_LENGTH:-8192}

# Canonical per-seed output dir in the single OUTPUT_DIR tree.
OUTPUT_DIR="${REPL_OUTPUT_DIR}/finetune/${VARIANT}/seed-${SEED}"
mkdir -p "${OUTPUT_DIR}"

echo "  base model:   ${BASE_MODEL}"
echo "  csv dir:      ${CSV_DIR}"
echo "  output:       ${OUTPUT_DIR}"
echo "  lr=${LR}  batch=${BATCH_SIZE}  grad_accum=${GRAD_ACCUM}  max_length=${MAX_LENGTH}"
echo "  d_output=${D_OUTPUT}  main_metrics=${MAIN_METRICS}  patience=${EARLY_STOPPING_PATIENCE}  problem_type=${PROBLEM_TYPE}"

# sequence_understanding.py reads {train,test}.csv plus dev.csv OR val.csv from
# --csv_dir (it natively accepts val.csv — no staging needed), writes test-set
# metrics to <output_dir>/test_results.json (key eval_mcc) and saves the model to
# <output_dir>/best_model. select_best_model.py reads test_results.json directly.
python -m src.tasks.downstream.sequence_understanding \
    --csv_dir="${CSV_DIR}" \
    --model_name="${BASE_MODEL}" \
    --output_dir="${OUTPUT_DIR}" \
    --learning_rate=${LR} \
    --batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --max_length=${MAX_LENGTH} \
    --d_output=${D_OUTPUT} \
    --seed=${SEED} \
    --main_metrics="${MAIN_METRICS}" \
    --early_stopping_patience=${EARLY_STOPPING_PATIENCE} \
    --problem_type="${PROBLEM_TYPE}"

# sequence_understanding.py writes test_results.json directly into --output_dir,
# so it already lives at the per-seed dir the selector reads. Confirm it.
if [ -f "${OUTPUT_DIR}/test_results.json" ]; then
    echo "  wrote ${OUTPUT_DIR}/test_results.json"
else
    echo "  WARNING: ${OUTPUT_DIR}/test_results.json not found — training/eval may have failed"
fi

echo "Done: $(date)"
echo "Job completed at: $(date)"
