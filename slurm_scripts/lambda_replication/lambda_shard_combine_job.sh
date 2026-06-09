#!/bin/bash
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Combine sharded embeddings into embeddings_pretrained.npz (+ embeddings_random.npz
# if requested), THEN run the normal embedding analysis. Because both npz files
# now exist in OUTPUT_DIR, embedding_analysis.py SKIPS extraction entirely and
# only runs the linear-probe + 3-layer-NN analysis (minutes). Writes the same
# embedding_analysis_results.json the standard pipeline produces.
#
# Required env (passed via --export):
#   REPO_ROOT, CSV_DIR, OUTPUT_DIR  (…/<LEN>/embedding/<variant>)
#   MAX_LENGTH
# Optional env:
#   MODEL_PATH (BASE_MODEL), POOLING, EMB_SEED, NN_EPOCHS, NN_HIDDEN_DIM, NN_LR,
#   BATCH_SIZE, INCLUDE_RANDOM_BASELINE, CONDA_ENV (generanno_env), HF_HOME

set -uo pipefail

echo "=== shard-combine + analysis len=${LEN:-?} ==="
echo "Started at: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID:-N/A}"

module load CUDA/12.8
source /data/lindseylm/conda/etc/profile.d/conda.sh
if [ -z "${CUDA_HOME:-}" ]; then
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

MODEL_PATH=${MODEL_PATH:-GenerTeam/GENERanno-prokaryote-0.5b-base}
POOLING=${POOLING:-mean}
EMB_SEED=${EMB_SEED:-42}
NN_EPOCHS=${NN_EPOCHS:-100}
NN_HIDDEN_DIM=${NN_HIDDEN_DIM:-256}
NN_LR=${NN_LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_LENGTH=${MAX_LENGTH:-8192}
INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false}

echo "  output:   ${OUTPUT_DIR}"
echo "  random_baseline=${INCLUDE_RANDOM_BASELINE}"

# 1) Concatenate shards -> embeddings_<which>.npz, for each model we sharded.
#    COMBINE_WHICH defaults to the full pair; for a random-only rebuild it is just
#    "random" and the cached embeddings_pretrained.npz is left in place + reused.
COMBINE_WHICH="${COMBINE_WHICH:-pretrained random}"
echo "  combining: ${COMBINE_WHICH}"
for w in ${COMBINE_WHICH}; do
    python -m src.tasks.downstream.shard_embeddings \
        --mode=combine --which="${w}" --output_dir="${OUTPUT_DIR}"
done

RANDOM_BASELINE_FLAG=""
if [ "${INCLUDE_RANDOM_BASELINE}" == "true" ]; then
    RANDOM_BASELINE_FLAG="--include_random_baseline"
fi

# 2) Run the standard analysis. Both npz now exist -> extraction is skipped.
python -m src.tasks.downstream.embedding_analysis \
    --csv_dir="${CSV_DIR}" \
    --model_path="${MODEL_PATH}" \
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
