#!/bin/bash
#
# Genome-wide predictions for BOTH frozen-embedding probe heads (linear probe +
# 3-layer NN) across all genome-wide CSVs, for the fragment-vs-genome-wide
# transfer analysis: best-of-{LP,NN,FT} genome-wide MCC in the main table, full
# per-head breakdown in the appendix.
#
# FT genome-wide already exists (run_lambda_inference.sh deploys the winner). This
# fills ONLY the missing LP + NN heads, extracting embeddings ONCE per CSV and
# applying both heads (roughly halves the compute vs two single-head passes).
#
# Reuses lambda_replication.conf for all paths / env / resources. Submits one
# lambda_allheads_job.sh per (LEN, variant, genome CSV). Requires the saved probe
# artifacts to already exist in OUTPUT_DIR/<LEN>/embedding/<variant> (produced by
# the embedding analysis with the probe-save fix); a job hard-fails if missing.
#
# Usage (from a login node, conda env active, repo pulled):
#   bash slurm_scripts/lambda_replication/run_lambda_allheads.sh [LEN ...]
# LEN defaults to SEGMENT_LENGTHS from the conf.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
JOB="${SCRIPT_DIR}/lambda_allheads_job.sh"
[ -f "${CONFIG}" ] || { echo "ERROR: missing ${CONFIG}"; exit 1; }
[ -f "${JOB}" ]    || { echo "ERROR: missing ${JOB}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

LENS=("$@"); [ "${#LENS[@]}" -gt 0 ] || read -ra LENS <<< "${SEGMENT_LENGTHS}"

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"
FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "GENERanno — all-heads (LP + NN) genome-wide"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  LENGTHS:    ${LENS[*]}"
echo "  VARIANTS:   ${VARIANTS}"
echo "============================================================"

NUM=0
for LEN in "${LENS[@]}"; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-8192}"
    gw_var="GENOME_WIDE_${LEN}"; GW_PATH="${!gw_var:-}"
    if [ -z "${GW_PATH}" ] || [ ! -d "${GW_PATH}" ]; then
        echo "WARNING: no genome-wide dir for ${LEN} (${GW_PATH:-unset}) — skipping"; continue
    fi
    for VARIANT in ${VARIANTS}; do
        EMB_DIR="${REPL_LEN_DIR}/embedding/${VARIANT}"
        if [ ! -f "${EMB_DIR}/linear_probe_pretrained.pkl" ]; then
            echo "WARNING: no saved LP probe in ${EMB_DIR} — run embedding analysis first; skipping ${LEN}/${VARIANT}"; continue
        fi
        shopt -s nullglob
        gw_csvs=("${GW_PATH}"/*.csv)
        shopt -u nullglob
        if [ "${#gw_csvs[@]}" -eq 0 ]; then
            echo "WARNING: ${GW_PATH} has no *.csv — skipping ${LEN}/${VARIANT}"; continue
        fi
        echo "--- ${LEN}/${VARIANT}: ${#gw_csvs[@]} genome CSV(s)  max_length=${MAX_LENGTH} ---"
        for csv in "${gw_csvs[@]}"; do
            stem="$(basename "${csv}" .csv)"
            J="gwheads_${LEN}_${VARIANT}_${stem}"
            sbatch --job-name="${J}" \
                --output="${LOGDIR}/${J}_%j.out" --error="${LOGDIR}/${J}_%j.err" \
                "${FLAGS[@]}" \
                --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},HF_HOME=${HF_HOME},REPL_OUTPUT_DIR=${REPL_LEN_DIR},VARIANT=${VARIANT},BASE_MODEL=${BASE_MODEL},INPUT_CSV=${csv},MAX_LENGTH=${MAX_LENGTH},BATCH_SIZE=${INF_BATCH_SIZE},POOLING=${POOLING},THRESHOLD=${THRESHOLD}" \
                "${JOB}"
            NUM=$((NUM+1))
        done
    done
done
echo ""
echo "Submitted ${NUM} all-heads genome-wide jobs. Monitor: squeue -u \$USER"
echo "Output: ${OUTPUT_DIR}/<LEN>/genome_wide_heads/<variant>/{lp,nn}/"
