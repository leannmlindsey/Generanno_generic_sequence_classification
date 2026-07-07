#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Genome-wide predictions for BOTH frozen-embedding probe heads (linear probe +
# 3-layer NN) in a SINGLE embedding pass, for ONE genome-wide CSV. Reuses the
# saved probe artifacts from the pretrained embedding analysis
# (linear_probe_pretrained.pkl, three_layer_nn_pretrained.pt + _scaler.pkl in
# REPL_OUTPUT_DIR/embedding/<variant>). Writes, under
# REPL_OUTPUT_DIR/genome_wide_heads/<variant>/:
#   lp/genome_wide_<stem>_predictions.csv (+ _metrics.json)
#   nn/genome_wide_<stem>_predictions.csv (+ _metrics.json)
#
# This does NOT re-run any experiment. FT genome-wide already exists (via
# run_lambda_inference.sh); this only fills the missing LP + NN heads so the
# harvest can report best-of-{LP,NN,FT} genome-wide.
#
# Required env: REPO_ROOT, REPL_OUTPUT_DIR, VARIANT, BASE_MODEL, INPUT_CSV, MAX_LENGTH
# Optional env: BATCH_SIZE(16), POOLING(mean), THRESHOLD(0.5), HF_HOME, CONDA_ENV

echo "=== all-heads genome-wide  variant=${VARIANT}  input=${INPUT_CSV} ==="
echo "Started: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"
echo "  conda env: ${CONDA_DEFAULT_ENV:-none}   python: $(command -v python)"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=${HF_HOME:-/work/hdd/bfzj/llindsey1/LAMBDA_REPLICATION/hf_cache}

if [ -z "${REPO_ROOT:-}" ]; then echo "ERROR: REPO_ROOT not set"; exit 1; fi
if [ ! -f "${INPUT_CSV:-/nonexistent}" ]; then echo "ERROR: INPUT_CSV not found: ${INPUT_CSV:-<unset>}"; exit 1; fi
cd "${REPO_ROOT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

EMB_DIR="${REPL_OUTPUT_DIR}/embedding/${VARIANT}"
OUT_DIR="${REPL_OUTPUT_DIR}/genome_wide_heads/${VARIANT}"
GW_DIR="$(dirname "${INPUT_CSV}")"
STEM="$(basename "${INPUT_CSV}" .csv)"

for f in linear_probe_pretrained.pkl three_layer_nn_pretrained.pt three_layer_nn_pretrained_scaler.pkl; do
    if [ ! -f "${EMB_DIR}/${f}" ]; then
        echo "ERROR: missing probe artifact ${EMB_DIR}/${f}"
        echo "       run the embedding analysis (with the probe-save fix) first"
        exit 1
    fi
done

python -m src.tasks.downstream.genome_wide_all_heads \
    --model_path "${BASE_MODEL}" \
    --embedding_dir "${EMB_DIR}" \
    --input_dir "${GW_DIR}" \
    --pattern "${STEM}.csv" \
    --output_dir "${OUT_DIR}" \
    --max_length "${MAX_LENGTH:-8192}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --pooling "${POOLING:-mean}" \
    --threshold "${THRESHOLD:-0.5}" \
    --save_metrics

echo "Done: $(date)"
