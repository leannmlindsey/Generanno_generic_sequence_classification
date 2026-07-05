#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Deploy a frozen-embedding probe (linear_probe | three_layer_nn) winner for ONE
# CSV. Parallel to lambda_inference_job.sh, but calls inference_embedding_head.py
# instead of the fine-tuned inference. Used by run_lambda_inference.sh when
# select_best_model.py picks a probe over fine-tuning. The DRIVER controls the
# canonical OUTPUT_FILENAME so names match the fine-tuned path exactly.
#
# Required env:
#   REPO_ROOT, REPL_OUTPUT_DIR, VARIANT, BASE_MODEL, HEAD_TYPE (linear_probe|three_layer_nn),
#   HEAD_PATH, INPUT_CSV, OUTPUT_FILENAME, MAX_LENGTH
# Optional env:
#   SCALER_PATH (required for three_layer_nn), BATCH_SIZE(16), POOLING(mean),
#   THRESHOLD(0.5), HF_HOME, CONDA_ENV

echo "=== probe inference ${HEAD_TYPE}  input=${INPUT_CSV}  output=${OUTPUT_FILENAME} ==="
echo "Started: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"
echo "  conda env: ${CONDA_DEFAULT_ENV:-none}   python: $(command -v python)"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/hf_cache}

if [ -z "${REPO_ROOT:-}" ]; then echo "ERROR: REPO_ROOT not set"; exit 1; fi
if [ -z "${HEAD_PATH:-}" ]; then echo "ERROR: HEAD_PATH not set"; exit 1; fi
if [ ! -f "${INPUT_CSV:-/nonexistent}" ]; then echo "ERROR: INPUT_CSV not found: ${INPUT_CSV:-<unset>}"; exit 1; fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

OUTDIR="${REPL_OUTPUT_DIR}/inference/${VARIANT}"
mkdir -p "${OUTDIR}"

SCALER_ARG=()
if [ -n "${SCALER_PATH:-}" ] && [ "${SCALER_PATH}" != "-" ]; then
    SCALER_ARG=(--scaler_path "${SCALER_PATH}")
fi

python -m src.tasks.downstream.inference_embedding_head \
    --model_path "${BASE_MODEL}" \
    --head_type "${HEAD_TYPE}" \
    --head_path "${HEAD_PATH}" \
    "${SCALER_ARG[@]}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTDIR}/${OUTPUT_FILENAME}" \
    --max_length "${MAX_LENGTH:-8192}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --pooling "${POOLING:-mean}" \
    --threshold "${THRESHOLD:-0.5}" \
    --save_metrics

echo "Done: $(date)"
