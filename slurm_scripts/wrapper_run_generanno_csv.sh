#!/bin/bash

# Wrapper script for running CSV binary classification with GENERanno on Biowulf
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash wrapper_run_generanno_csv.sh
#
# Or submit directly with environment variables:
#   sbatch --export=ALL,CSV_DIR=/path/to/data,MODEL_NAME=model_name run_generanno_csv.sh

#####################################################################
# CONFIGURATION - Edit this section
#####################################################################

# === REQUIRED: Dataset Configuration ===
# Path to directory containing train.csv, dev.csv, test.csv
export CSV_DIR="/path/to/your/csv/data"
# Name for this dataset (used in output directory structure)
export DATASET_NAME="my_dataset"

# === REQUIRED: Model Configuration ===
# HuggingFace model path or name
export MODEL_NAME="GenerTeam/GENERanno-prokaryote-0.5b-base"

# === OPTIONAL: Hyperparameters ===
export LR="1e-5"
export BATCH_SIZE="16"
export MAX_LENGTH="8192"
export D_OUTPUT="2"
export MAIN_METRICS="mcc"
export EARLY_STOPPING_PATIENCE="5"

# === Replicates ===
# Set NUM_REPLICATES=1 for a single run, or higher for multiple seeds
# Seeds will be 1, 2, 3, ... NUM_REPLICATES (or 42 for single run)
NUM_REPLICATES=1

#####################################################################
# END CONFIGURATION
#####################################################################

# Validate configuration
if [ "${CSV_DIR}" == "/path/to/your/csv/data" ]; then
    echo "ERROR: Please set CSV_DIR to your actual data directory"
    exit 1
fi

# Verify files exist
if [ ! -d "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR does not exist: ${CSV_DIR}"
    exit 1
fi

if [ ! -f "${CSV_DIR}/train.csv" ]; then
    echo "ERROR: train.csv not found in ${CSV_DIR}"
    exit 1
fi

echo "=========================================="
echo "Submitting GENERanno CSV Binary Job(s)"
echo "=========================================="
echo "Dataset: ${DATASET_NAME}"
echo "CSV dir: ${CSV_DIR}"
echo "Model: ${MODEL_NAME}"
echo "LR: ${LR}, Batch: ${BATCH_SIZE}, Max Length: ${MAX_LENGTH}"
echo "Replicates: ${NUM_REPLICATES}"
echo "=========================================="

# Submit job(s)
if [ "${NUM_REPLICATES}" -eq 1 ]; then
    # Single run with default seed
    export SEED=42
    echo "Submitting single job with seed ${SEED}..."
    sbatch --export=ALL run_generanno_csv.sh
else
    # Multiple replicates with seeds 1 to NUM_REPLICATES
    for SEED in $(seq 1 ${NUM_REPLICATES}); do
        export SEED
        echo "Submitting replicate ${SEED}/${NUM_REPLICATES} with seed ${SEED}..."
        sbatch --export=ALL --job-name="gen_${DATASET_NAME}_s${SEED}" run_generanno_csv.sh
    done
fi

echo ""
echo "${NUM_REPLICATES} job(s) submitted. Monitor with: squeue -u \$USER"
