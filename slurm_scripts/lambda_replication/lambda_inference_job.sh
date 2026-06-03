#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Inference for the winning seed of VARIANT on one CSV. The DRIVER controls the
# output filename so names are canonical (test_predictions.csv, fpr_predictions.csv,
# gc_control_predictions.csv, fnr_predictions.csv, genome_wide_<stem>_predictions.csv),
# which the central "harvest" aggregator requires.
#
# This is a THIN orchestration job body: it calls the EXISTING module entry point
# `python -m src.tasks.downstream.inference` directly with an EXPLICIT --output_csv.
# It does NOT modify any experiment code.
#
# Required env:
#   REPO_ROOT
#   REPL_OUTPUT_DIR    per-length replication output dir (OUTPUT_DIR/<LEN>)
#   VARIANT            generanno
#   INPUT_CSV          path to the CSV to predict on
#   OUTPUT_FILENAME    canonical name for the predictions CSV (e.g. test_predictions.csv)
#   MODEL_PATH         winning checkpoint dir (best_model/) from winners.json
#   MAX_LENGTH         max token length for this window (bp == tokens)
# Optional env:
#   BATCH_SIZE (16), THRESHOLD (0.5), CONDA_ENV (generanno_env), HF_HOME


echo "=== inference ${VARIANT}  input=${INPUT_CSV}  output=${OUTPUT_FILENAME} ==="
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

BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
THRESHOLD=${THRESHOLD:-0.5}

if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is not set; the driver must pass the winning checkpoint"; exit 1
fi
if [ ! -f "${INPUT_CSV:-/nonexistent}" ]; then
    echo "ERROR: INPUT_CSV not found: ${INPUT_CSV:-<unset>}"; exit 1
fi

echo "  model path:   ${MODEL_PATH}"

OUTPUT_DIR="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTPUT_DIR}"

# inference.py writes the predictions CSV at --output_csv and a sibling
# <output_csv with .csv -> _metrics.json> when --save_metrics and labels exist.
python -m src.tasks.downstream.inference \
    --input_csv="${INPUT_CSV}" \
    --model_path="${MODEL_PATH}" \
    --output_csv="${OUTPUT_DIR}/${OUTPUT_FILENAME}" \
    --max_length=${MAX_LENGTH} \
    --batch_size=${BATCH_SIZE} \
    --threshold=${THRESHOLD} \
    --save_metrics

echo "Done: $(date)"
echo "Job completed at: $(date)"
