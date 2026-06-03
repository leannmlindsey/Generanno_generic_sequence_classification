#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Pretrained-embedding analysis (Surface D) for one (length, variant). Runs the
# EXISTING module entry point `python -m src.tasks.downstream.embedding_analysis`
# on the BASE_MODEL (pretrained, NOT a finetuned checkpoint — independent of the
# winners). Thin orchestration body; it does NOT modify any experiment code.
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR    per-length replication output dir (OUTPUT_DIR/<LEN>)
#   CSV_DIR            train/dev(or val)/test CSV directory (LAMBDA_v1 train_val_test/<LEN>)
#   VARIANT            generanno
#   MAX_LENGTH         max token length for this window (bp == tokens)
# Optional env:
#   BASE_MODEL, POOLING, EMB_SEED, NN_EPOCHS, NN_HIDDEN_DIM, NN_LR, BATCH_SIZE,
#   INCLUDE_RANDOM_BASELINE, CONDA_ENV (generanno_env), HF_HOME


echo "=== embedding ${VARIANT} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load CUDA/12.8
source /data/lindseylm/conda/etc/profile.d/conda.sh
if [ -z "${CUDA_HOME}" ]; then
    NVCC_PATH=$(which nvcc)
    if [ -n "${NVCC_PATH}" ]; then
        export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
    fi
fi
conda activate "${CONDA_ENV:-generanno_env}"
if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-generanno_env}" ]; then
    echo "ERROR: could not activate conda env '${CONDA_ENV:-generanno_env}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
    exit 1
fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/data/lindseylm/.cache/huggingface}

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-GenerTeam/GENERanno-prokaryote-0.5b-base}
POOLING=${POOLING:-mean}
EMB_SEED=${EMB_SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

# embedding_analysis.py reads {train,test}.csv plus dev.csv OR val.csv from
# --csv_dir (it natively accepts val.csv — no staging needed).
OUTPUT_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

echo "  base model:   ${BASE_MODEL}"
echo "  csv dir:      ${CSV_DIR}"
echo "  output:       ${OUTPUT_DIR}"
echo "  pooling=${POOLING}  max_length=${MAX_LENGTH}  nn_epochs=${NN_EPOCHS}  random_baseline=${INCLUDE_RANDOM_BASELINE}"

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

# Writes embedding_analysis_results.json to OUTPUT_DIR.
python -m src.tasks.downstream.embedding_analysis \
    --csv_dir="${CSV_DIR}" \
    --model_path="${BASE_MODEL}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --seed=${EMB_SEED} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_hidden_dim=${NN_HIDDEN_DIM} \
    --nn_lr=${NN_LR} \
    ${RANDOM_BASELINE_FLAG}

echo "Done: $(date)"
echo "Job completed at: $(date)"
