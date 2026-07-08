#!/bin/bash
#
# GENERanno 4k genome-wide SEED SWEEP — diagnostic re-run (not part of the main pipeline).
#
# WHY: the final LAMBDA tables' GENERanno 4k genome-wide filtered MCC (0.42) is an
# outlier vs 2k (0.65) and 8k (0.53) and vs the original submission (0.62). The
# cause is that genome-wide inference used ONLY the winners.json seed (seed 3),
# chosen for best FRAGMENT test MCC (0.9444). All 5 finetune seeds are excellent
# on fragments (0.914-0.944) but their GENOME-WIDE calibration was never checked.
# Seed 3 over-calls bacteria genome-wide at 4k (mean prob 0.49 on bacteria windows),
# forcing an aggressive size filter that discards >half the prophages -> recall 0.42.
#
# WHAT: re-run 4k genome-wide inference for the OTHER seeds (default 1 2 4 5) so we
# can compare per-seed genome-wide behavior and either pick a better-calibrated seed
# or report the multi-seed mean/median. Seed 3 is included by default as a control
# (it should reproduce the canonical 0.42, validating the comparison).
#
# NON-DESTRUCTIVE: each seed writes to its OWN output subdir
#   ${OUTPUT_DIR}/4k/inference/generanno_seed<N>/genome_wide_<stem>_predictions.csv
# The canonical ${OUTPUT_DIR}/4k/inference/generanno/ predictions are NOT touched.
#
# REUSES the existing thin job body lambda_inference_job.sh unchanged. No experiment
# code is modified. Only 4k + genome-wide (no diagnostics, no other lengths).
#
# PREREQ (Delta-AI inherit model): activate the conda env on the LOGIN node first,
# then run this script from the login node:
#     conda activate /work/hdd/bfzj/llindsey1/conda/envs/generanno
#     bash slurm_scripts/lambda_replication/run_generanno_4k_gw_seed_sweep.sh
#
# Knobs (env overrides):
#   LEN=4k                    segment window: 2k | 4k | 8k (default 4k). MAX_LENGTH and
#                             the genome-wide CSV dir are derived from it via the conf.
#   SWEEP_SEEDS="1 2 4 5 3"   which seeds to run (default: all five, seed 3 = control).
#                             For the FINAL FT genome-wide deployment (best-of-{LP,NN,FT}
#                             tables), pass the single BEST fine-tuned seed for the window:
#                               LEN=2k SWEEP_SEEDS=2 bash .../run_generanno_4k_gw_seed_sweep.sh
#                               LEN=4k SWEEP_SEEDS=3 bash .../run_generanno_4k_gw_seed_sweep.sh
#                               LEN=8k SWEEP_SEEDS=3 bash .../run_generanno_4k_gw_seed_sweep.sh
#                             Each writes to inference/generanno_seed<N>/ (canonical
#                             inference/generanno/ untouched); one job per genome, parallel.
#   DRY_RUN=true              print the sbatch commands but submit nothing
#
# After the jobs finish, evaluate locally (per seed) with the same grid the paper
# uses; see the companion note the analysis step prints at the end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"
INF_JOB="${SCRIPT_DIR}/lambda_inference_job.sh"

[ -f "${CONFIG}" ]  || { echo "ERROR: missing ${CONFIG}"; exit 1; }
[ -f "${INF_JOB}" ] || { echo "ERROR: missing inference job body ${INF_JOB}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

LEN="${LEN:-4k}"                                       # segment window: 2k | 4k | 8k
SWEEP_SEEDS="${SWEEP_SEEDS:-1 2 4 5 3}"
DRY_RUN="${DRY_RUN:-false}"

REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
FT_ROOT="${REPL_LEN_DIR}/finetune/generanno"           # seed-<N>/best_model live here
# MAX_LENGTH and the genome-wide CSV dir are derived from LEN via the conf, so the
# same launcher serves any window (indirect lookup on MAX_LENGTH_<LEN> / GENOME_WIDE_<LEN>).
ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-8192}"

gw_var="GENOME_WIDE_${LEN}"; GW_PATH="${!gw_var:-}"
[ -n "${GW_PATH}" ] && [ -d "${GW_PATH}" ] || { echo "ERROR: GENOME_WIDE_${LEN} not a dir: ${GW_PATH:-<unset>}"; exit 1; }

shopt -s nullglob
GW_CSVS=( "${GW_PATH}"/*.csv )
shopt -u nullglob
[ "${#GW_CSVS[@]}" -gt 0 ] || { echo "ERROR: no *.csv in ${GW_PATH}"; exit 1; }

mkdir -p "${REPL_LEN_DIR}/logs"
LOGDIR="${REPL_LEN_DIR}/logs"

INF_FLAGS=(--account=bfzj-dtai-gh --partition=ghx4 --gpus-per-node=1 --mem="${INF_MEM}" --time="${INF_TIME}" --cpus-per-task=8)

echo "============================================================"
echo "GENERanno 4k genome-wide SEED SWEEP (diagnostic)"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  GW CSVs:     ${#GW_CSVS[@]} file(s) in ${GW_PATH}"
echo "  seeds:       ${SWEEP_SEEDS}"
echo "  MAX_LENGTH:  ${MAX_LENGTH}   conda: ${CONDA_DEFAULT_ENV:-<none active>}"
echo "  DRY_RUN:     ${DRY_RUN}"
echo "============================================================"

if [ "${DRY_RUN}" != "true" ] && [ "${CONDA_DEFAULT_ENV:-}" == "" ]; then
    echo "WARNING: no conda env active. Delta jobs inherit the login shell (--export=ALL);"
    echo "         activate the env first:  conda activate ${CONDA_ENV}"
fi

NUM_JOBS=0
cd "${REPO_ROOT}"

for SEED in ${SWEEP_SEEDS}; do
    MODEL_PATH="${FT_ROOT}/seed-${SEED}/best_model"
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "  WARNING: seed-${SEED} checkpoint missing (${MODEL_PATH}) — skipping seed ${SEED}"
        continue
    fi
    VARIANT_OUT="generanno_seed${SEED}"     # isolates outputs; canonical 'generanno/' untouched
    echo ""
    echo "--- seed ${SEED}  ->  inference/${VARIANT_OUT}/  (${MODEL_PATH}) ---"

    INF_ENV="REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},HF_HOME=${HF_HOME},REPL_OUTPUT_DIR=${REPL_LEN_DIR},VARIANT=${VARIANT_OUT},MODEL_PATH=${MODEL_PATH},MAX_LENGTH=${MAX_LENGTH},BATCH_SIZE=${INF_BATCH_SIZE},THRESHOLD=${THRESHOLD}"

    for csv in "${GW_CSVS[@]}"; do
        stem="$(basename "${csv}" .csv)"
        JOB="gwseed${SEED}_${LEN}_${stem}"
        OUT_NAME="genome_wide_${stem}_predictions.csv"
        if [ "${DRY_RUN}" == "true" ]; then
            echo "    [dry-run] ${JOB} -> inference/${VARIANT_OUT}/${OUT_NAME}"
        else
            sbatch \
                --job-name="${JOB}" \
                --output="${LOGDIR}/${JOB}_%j.out" \
                --error="${LOGDIR}/${JOB}_%j.err" \
                "${INF_FLAGS[@]}" \
                --export="ALL,${INF_ENV},INPUT_CSV=${csv},OUTPUT_FILENAME=${OUT_NAME}" \
                "${INF_JOB}"
        fi
        NUM_JOBS=$((NUM_JOBS + 1))
    done
done

echo ""
echo "${NUM_JOBS} job(s) $([ "${DRY_RUN}" == "true" ] && echo "would be submitted (dry-run)" || echo "submitted"). Monitor: squeue -u \$USER"
echo "Outputs (per seed): ${OUTPUT_DIR}/${LEN}/inference/generanno_seed<N>/genome_wide_*_predictions.csv"
echo ""
echo "NEXT (after jobs finish): scp the generanno_seed<N> dirs back and run the paper's"
echo "grid on each seed to compare filtered MCC (2k/8k unchanged; only 4k GW re-scored)."
