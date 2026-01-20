#!/bin/bash

# Interactive script for running CSV binary classification WITHOUT sbatch
# Usage: bash run_generanno_csv_interactive.sh
#
# This script reads configuration from wrapper_run_generanno_csv.sh (or specify another)
# and runs the job directly on the current node.

# Source the wrapper to get all the environment variables
# Change this path if your wrapper has a different name
WRAPPER_SCRIPT="${1:-wrapper_run_generanno_csv.sh}"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: Wrapper script not found: ${WRAPPER_SCRIPT}"
    echo "Usage: bash run_generanno_csv_interactive.sh [wrapper_script.sh]"
    exit 1
fi

echo "============================================================"
echo "Loading configuration from: ${WRAPPER_SCRIPT}"
echo "============================================================"

# Source the wrapper but skip the sbatch line at the end
# We just want the exports
source <(grep "^export" "${WRAPPER_SCRIPT}")

# Now run the main script logic (copied from run_generanno_csv.sh)

echo ""
echo "GENERanno CSV Binary Classification (Interactive Mode)"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Load modules (comment out if not on Biowulf/HPC)
module load conda 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

# Activate conda environment
source activate generanno_env

# Ignore user site-packages to avoid conflicts with ~/.local packages
export PYTHONNOUSERSITE=1

# Check GPU availability
echo ""
echo "GPU Information:"
nvidia-smi
echo ""
echo "Python environment:"
which python
python --version
echo ""

# Set defaults for optional parameters
DATASET_NAME=${DATASET_NAME:-csv_dataset}
MODEL_NAME=${MODEL_NAME:-GenerTeam/GENERanno-prokaryote-0.5b-base}
LR=${LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
D_OUTPUT=${D_OUTPUT:-2}
MAIN_METRICS=${MAIN_METRICS:-mcc}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}
NUM_REPLICATES=${NUM_REPLICATES:-1}

# Validate required parameters
if [ -z "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

# Add to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Determine seeds to run
if [ "${NUM_REPLICATES}" -eq 1 ]; then
    SEEDS="42"
else
    SEEDS=$(seq 1 ${NUM_REPLICATES})
fi

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL_NAME}"
echo "  CSV dir: ${CSV_DIR}"
echo "  Dataset name: ${DATASET_NAME}"
echo "  Learning rate: ${LR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Output classes: ${D_OUTPUT}"
echo "  Main metrics: ${MAIN_METRICS}"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo "  Replicates: ${NUM_REPLICATES}"
echo "============================================================"
echo ""

# Run training for each seed
for SEED in ${SEEDS}; do
    OUTPUT_DIR="./results/csv_binary/${DATASET_NAME}/lr-${LR}_batch-${BATCH_SIZE}/seed-${SEED}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "============================================================"
    echo "Running replicate with seed ${SEED}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "============================================================"
    echo ""

    python -m src.tasks.downstream.sequence_understanding \
        --csv_dir="${CSV_DIR}" \
        --model_name="${MODEL_NAME}" \
        --output_dir="${OUTPUT_DIR}" \
        --learning_rate=${LR} \
        --batch_size=${BATCH_SIZE} \
        --max_length=${MAX_LENGTH} \
        --d_output=${D_OUTPUT} \
        --seed=${SEED} \
        --main_metrics="${MAIN_METRICS}" \
        --early_stopping_patience=${EARLY_STOPPING_PATIENCE} \
        --problem_type="single_label_classification"

    echo ""
    echo "============================================================"
    echo "Replicate ${SEED} completed at: $(date)"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "All ${NUM_REPLICATES} replicate(s) completed!"
echo "============================================================"
