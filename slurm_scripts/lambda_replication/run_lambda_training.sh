#!/bin/bash
#
# GENERanno LAMBDA_v1 replication — STAGE 1: fire off all training jobs.
#
# For each segment length in SEGMENT_LENGTHS, submits one finetune sbatch job
# per (variant, seed) using the THIN job body lambda_finetune_job.sh, which calls
# `python -m src.tasks.downstream.sequence_understanding` with an EXPLICIT
# --output_dir. Every per-seed result therefore lands in the single canonical
# OUTPUT_DIR tree:
#   ${OUTPUT_DIR}/<LEN>/finetune/<variant>/seed-<N>/  (test_results.json + best_model/)
# (No more repo-internal results/csv_binary/... split layout.)
#
# All jobs run in parallel (no --dependency chaining). Once they complete, run
# run_lambda_inference.sh to pick the best seed and run inference + embeddings.
#
# GENERanno has an 8k base-pair context, so 2k / 4k / 8k are all supported.
#
# Usage:
#   1. Edit lambda_replication.conf — confirm LAMBDA_BASE and OUTPUT_DIR.
#   2. bash slurm_scripts/lambda_replication/run_lambda_training.sh
#   3. Wait for jobs: squeue -u $USER ; then bash .../check_training.sh
#   4. bash slurm_scripts/lambda_replication/run_lambda_inference.sh


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# REPO_ROOT is the repo top (holds src/, slurm_scripts/).
# slurm_scripts/lambda_replication -> ../.. == repo root.
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

FT_JOB="${SCRIPT_DIR}/lambda_finetune_job.sh"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [ ! -f "${FT_JOB}" ]; then
    echo "ERROR: missing finetune job body ${FT_JOB}"; exit 1
fi

# --- validate -----------------------------------------------------------------

if [[ "${LAMBDA_BASE}" == /path/to/* ]] || [[ "${OUTPUT_DIR}" == /path/to/* ]]; then
    echo "ERROR: edit ${CONFIG} — LAMBDA_BASE or OUTPUT_DIR still set to placeholder"
    exit 1
fi
[ -d "${LAMBDA_BASE}/train_val_test" ] || {
    echo "ERROR: ${LAMBDA_BASE}/train_val_test not found (expected LAMBDA_v1 layout)"
    exit 1
}
if [ -z "${SEGMENT_LENGTHS}" ]; then
    echo "ERROR: SEGMENT_LENGTHS is empty"; exit 1
fi

# Validate per-length input dirs exist before submitting anything.
RUN_LENGTHS=""
for LEN in ${SEGMENT_LENGTHS}; do
    LDIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    if [ ! -d "${LDIR}" ]; then
        echo "WARNING: ${LDIR} not found — skipping ${LEN}"; continue
    fi
    [ -f "${LDIR}/train.csv" ] || { echo "ERROR: ${LDIR}/train.csv not found"; exit 1; }
    [ -f "${LDIR}/test.csv" ]  || { echo "ERROR: ${LDIR}/test.csv not found"; exit 1; }
    if [ ! -f "${LDIR}/dev.csv" ] && [ ! -f "${LDIR}/val.csv" ]; then
        echo "ERROR: ${LDIR} must contain dev.csv or val.csv"; exit 1
    fi
    RUN_LENGTHS="${RUN_LENGTHS} ${LEN}"
done
RUN_LENGTHS="$(echo "${RUN_LENGTHS}" | xargs)"
if [ -z "${RUN_LENGTHS}" ]; then
    echo "ERROR: no runnable lengths after validation"; exit 1
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

# --- summary ------------------------------------------------------------------

echo "============================================================"
echo "GENERanno LAMBDA replication — Stage 1: training"
echo "============================================================"
echo "  LAMBDA_BASE:     ${LAMBDA_BASE}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  REPO_ROOT:       ${REPO_ROOT}"
echo "  FT_JOB:          ${FT_JOB}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "  BASE_MODEL:      ${BASE_MODEL}"
echo "  SEEDS:           ${SEEDS}"
echo "  FT params:       lr=${LR} batch=${BATCH_SIZE} d_output=${D_OUTPUT} metric=${MAIN_METRICS} patience=${EARLY_STOPPING_PATIENCE}"
echo "============================================================"

# --- common sbatch flags ------------------------------------------------------

FT_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${FT_MEM}" --time="${FT_TIME}" --cpus-per-task=8)

# REPO_ROOT is propagated to every job so they can cd to the real repo — SLURM
# stages each job script to /var/spool/slurm/... where BASH_SOURCE[0] can't
# recover the original location.
FT_ENV_BASE="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},HF_HOME=${HF_HOME},BASE_MODEL=${BASE_MODEL},LR=${LR},BATCH_SIZE=${BATCH_SIZE},D_OUTPUT=${D_OUTPUT},MAIN_METRICS=${MAIN_METRICS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"

NUM_JOBS=0

for LEN in ${RUN_LENGTHS}; do
    CSV_DIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    mkdir -p "${REPL_LEN_DIR}"

    # Resolve per-window max token length (MAX_LENGTH_<LEN>).
    ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-8192}"

    echo ""
    echo "--- length: ${LEN} (max_length=${MAX_LENGTH}) ---"
    echo "    csv dir:    ${CSV_DIR}"
    echo "    output dir: ${REPL_LEN_DIR}"

    for VARIANT in ${VARIANTS}; do
        for SEED in ${SEEDS}; do
            JOB="ft_${LEN}_${VARIANT}_s${SEED}"
            echo "    submitting ${JOB}..."
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${FT_FLAGS[@]}" \
                --export="ALL,REPL_OUTPUT_DIR=${REPL_LEN_DIR},CSV_DIR=${CSV_DIR},${FT_ENV_BASE},VARIANT=${VARIANT},SEED=${SEED},LEN=${LEN},MAX_LENGTH=${MAX_LENGTH}" \
                "${FT_JOB}"
            NUM_JOBS=$((NUM_JOBS + 1))
        done
    done
done

echo ""
echo "Submitted ${NUM_JOBS} jobs. Monitor with: squeue -u \$USER"
echo "When all jobs are done, run:"
echo "  bash ${SCRIPT_DIR}/check_training.sh"
echo "  bash ${SCRIPT_DIR}/run_lambda_inference.sh"
