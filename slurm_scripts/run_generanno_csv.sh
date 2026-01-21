#!/bin/bash
#SBATCH --job-name=generanno_csv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=generanno_csv_%j.out
#SBATCH --error=generanno_csv_%j.err

# Biowulf batch script for GENERanno CSV binary classification
# Usage: sbatch run_generanno_csv.sh
#
# Required environment variables (set via --export or edit below):
#   CSV_DIR: Path to directory containing train.csv, dev.csv, test.csv
#   MODEL_NAME: HuggingFace model path or name
#
# Optional environment variables:
#   DATASET_NAME: Name for output directory (default: csv_dataset)
#   LR: Learning rate (default: 1e-5)
#   BATCH_SIZE: Batch size (default: 16)
#   MAX_LENGTH: Max sequence length (default: 8192)
#   D_OUTPUT: Number of classes (default: 2)
#   SEED: Random seed (default: 42)

echo "============================================================"
echo "GENERanno CSV Binary Classification"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load CUDA/12.8

# Set CUDA_HOME if not set
if [ -z "${CUDA_HOME}" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null))) 2>/dev/null || true
fi

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
SEED=${SEED:-42}
MAIN_METRICS=${MAIN_METRICS:-mcc}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5}

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

# Set output directory
OUTPUT_DIR="./results/csv_binary/${DATASET_NAME}/lr-${LR}_batch-${BATCH_SIZE}/seed-${SEED}"
mkdir -p "${OUTPUT_DIR}"

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
echo "  Seed: ${SEED}"
echo "  Main metrics: ${MAIN_METRICS}"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Run training
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
echo "Job completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
