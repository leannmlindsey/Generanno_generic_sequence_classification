#!/bin/bash
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# Extract embeddings for ONE shard of one split (train|val|test) from one model
# (pretrained|random). Designed to be submitted as a SLURM job ARRAY — the array
# task id IS the shard index. Thin orchestration body; calls
# `python -m src.tasks.downstream.shard_embeddings --mode extract`.
#
# Required env (passed via --export):
#   REPO_ROOT, CSV_DIR, OUTPUT_DIR  (…/<LEN>/embedding/<variant>)
#   WHICH      pretrained | random
#   SPLIT      train | val | test
#   NUM_SHARDS number of shards for THIS split
#   MAX_LENGTH max token length for this window
# Optional env:
#   MODEL_PATH (BASE_MODEL), POOLING, EMB_SEED, BATCH_SIZE,
#   CONDA_ENV (generanno_env), HF_HOME
#
# Shard index comes from SLURM_ARRAY_TASK_ID.

set -uo pipefail

SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?must be run as a job array (SLURM_ARRAY_TASK_ID unset)}"

echo "=== shard-extract which=${WHICH} split=${SPLIT} idx=${SHARD_INDEX}/${NUM_SHARDS} len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

# --- conda env setup: BIOWULF ONLY, disabled for Delta ---------------------
# Delta-AI inherits the submitting shell's environment (sbatch --export=ALL), so
# activate the conda env (and load any needed module) on the LOGIN node BEFORE
# running the driver. The block below was needed on Biowulf, where jobs did not
# inherit the submitting shell's environment.
# module load CUDA/12.8
# source /data/lindseylm/conda/etc/profile.d/conda.sh
# if [ -z "${CUDA_HOME:-}" ]; then
#     NVCC_PATH=$(which nvcc)
#     if [ -n "${NVCC_PATH}" ]; then
#         export CUDA_HOME=$(dirname $(dirname "${NVCC_PATH}"))
#     fi
# fi
# conda activate "${CONDA_ENV:-generanno_env}"
# if [ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV:-generanno_env}" ]; then
#     echo "ERROR: could not activate conda env '${CONDA_ENV:-generanno_env}' (active: '${CONDA_DEFAULT_ENV:-none}'). Aborting." >&2
#     exit 1
# fi
echo "  conda env: ${CONDA_DEFAULT_ENV}   python: $(command -v python)"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/hf_cache}

if [ -z "${REPO_ROOT:-}" ]; then
    echo "ERROR: REPO_ROOT is not set; the launcher must pass it via --export"; exit 1
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-GenerTeam/GENERanno-prokaryote-0.5b-base}
POOLING=${POOLING:-mean}
EMB_SEED=${EMB_SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}

mkdir -p "${OUTPUT_DIR}"

echo "  model:    ${MODEL_PATH}"
echo "  csv dir:  ${CSV_DIR}"
echo "  output:   ${OUTPUT_DIR}"
echo "  pooling=${POOLING}  max_length=${MAX_LENGTH}  batch_size=${BATCH_SIZE}  seed=${EMB_SEED}"

python -m src.tasks.downstream.shard_embeddings \
    --mode=extract \
    --which="${WHICH}" \
    --split="${SPLIT}" \
    --num_shards="${NUM_SHARDS}" \
    --shard_index="${SHARD_INDEX}" \
    --csv_dir="${CSV_DIR}" \
    --model_path="${MODEL_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --seed=${EMB_SEED}

echo "Done: $(date)"
