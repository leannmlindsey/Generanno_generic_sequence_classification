#!/bin/bash
#SBATCH --job-name=generanno_emb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=generanno_emb_%j.out
#SBATCH --error=generanno_emb_%j.err

# Biowulf batch script for GENERanno embedding analysis
# Usage: sbatch run_embedding_analysis.sh
#
# Required environment variables:
#   CSV_DIR: Path to directory containing train.csv, dev.csv, test.csv
#   MODEL_PATH: Path to fine-tuned model or HuggingFace model name

echo "============================================================"
echo "GENERanno Embedding Analysis"
echo "============================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda
module load cuda/12.8

# Activate conda environment
source activate generanno_env

# Ignore user site-packages
export PYTHONNOUSERSITE=1

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi
echo ""

# Set defaults
MODEL_PATH=${MODEL_PATH:-GenerTeam/GENERanno-prokaryote-0.5b-base}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
POOLING=${POOLING:-mean}
SEED=${SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}

# Validate required parameters
if [ -z "${CSV_DIR}" ]; then
    echo "ERROR: CSV_DIR is not set"
    exit 1
fi

# Navigate to repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.." || exit
echo "Working directory: $(pwd)"

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Set output directory
OUTPUT_DIR=${OUTPUT_DIR:-./results/embedding_analysis/$(basename ${CSV_DIR})}
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "Configuration:"
echo "============================================================"
echo "  Model: ${MODEL_PATH}"
echo "  CSV dir: ${CSV_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max length: ${MAX_LENGTH}"
echo "  Pooling: ${POOLING}"
echo "  Seed: ${SEED}"
echo "============================================================"
echo ""

# Run embedding analysis
python -m src.tasks.downstream.embedding_analysis \
    --csv_dir="${CSV_DIR}" \
    --model_path="${MODEL_PATH}" \
    --output_dir="${OUTPUT_DIR}" \
    --batch_size=${BATCH_SIZE} \
    --max_length=${MAX_LENGTH} \
    --pooling="${POOLING}" \
    --seed=${SEED} \
    --nn_epochs=${NN_EPOCHS} \
    --nn_hidden_dim=${NN_HIDDEN_DIM} \
    --nn_lr=${NN_LR}

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
