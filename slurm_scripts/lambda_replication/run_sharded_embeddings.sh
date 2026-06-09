#!/bin/bash
#
# Parallel sharded embedding extraction — a drop-in replacement for the single
# emb_<LEN>_<variant> job when extraction is too slow to finish in one wall
# (e.g. 8k). It shards the train/val/test splits across many GPU array tasks
# (pretrained + random baseline), then a dependent combine job assembles the
# shards into embeddings_pretrained.npz / embeddings_random.npz and runs the
# normal analysis (which finds the npz and skips extraction).
#
# Usage (from the repo root, on a login node, AFTER pushing/pulling the repo):
#   bash slurm_scripts/lambda_replication/run_sharded_embeddings.sh [LEN ...]
#
#   LEN defaults to "8k". Tune shard size / resources via env:
#     SHARD_SIZE=2000        # sequences per shard (smaller = more, shorter jobs)
#     SHARD_TIME=2:00:00     # wall per shard array task
#     MAX_CONCURRENT=        # optional array throttle, e.g. 16 -> --array=...%16
#
# NOTE: if a non-sharded emb_<LEN>_<variant> job is still running for the same
# length, scancel it first — this writes the same output dir.

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

EXTRACT_JOB="${SCRIPT_DIR}/lambda_shard_extract_job.sh"
COMBINE_JOB="${SCRIPT_DIR}/lambda_shard_combine_job.sh"

[ -f "${CONFIG}" ]      || { echo "ERROR: missing ${CONFIG}"; exit 1; }
[ -f "${EXTRACT_JOB}" ] || { echo "ERROR: missing ${EXTRACT_JOB}"; exit 1; }
[ -f "${COMBINE_JOB}" ] || { echo "ERROR: missing ${COMBINE_JOB}"; exit 1; }
# shellcheck disable=SC1090
source "${CONFIG}"

LENS=("$@"); [ "${#LENS[@]}" -gt 0 ] || LENS=("8k")
SHARD_SIZE="${SHARD_SIZE:-2000}"
SHARD_TIME="${SHARD_TIME:-2:00:00}"
ARRAY_THROTTLE=""
[ -n "${MAX_CONCURRENT:-}" ] && ARRAY_THROTTLE="%${MAX_CONCURRENT}"

BASE_MODEL="${BASE_MODEL:-GenerTeam/GENERanno-prokaryote-0.5b-base}"

WHICH_LIST=(pretrained)
if [ "${INCLUDE_RANDOM_BASELINE:-false}" == "true" ]; then
    WHICH_LIST+=(random)
fi

mkdir -p "${OUTPUT_DIR}/logs"
LOGDIR="${OUTPUT_DIR}/logs"

EXTRACT_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${EMB_MEM}" --time="${SHARD_TIME}" --cpus-per-task=8)
COMBINE_FLAGS=(--partition=gpu --gres=gpu:a100:1 --mem="${EMB_MEM}" --time="2:00:00" --cpus-per-task=8)

# data rows in a CSV (minus the header)
rows_of() { echo $(( $(wc -l < "$1") - 1 )); }
# ceil(n / size), clamped to >=1
nshards_of() {
    local n="$1" size="$2" k
    k=$(( (n + size - 1) / size ))
    [ "${k}" -lt 1 ] && k=1
    echo "${k}"
}

echo "============================================================"
echo "GENERanno — parallel sharded embedding extraction"
echo "============================================================"
echo "  REPO_ROOT:   ${REPO_ROOT}"
echo "  LENGTHS:     ${LENS[*]}"
echo "  VARIANTS:    ${VARIANTS}"
echo "  WHICH:       ${WHICH_LIST[*]}"
echo "  SHARD_SIZE:  ${SHARD_SIZE} seq/shard   SHARD_TIME: ${SHARD_TIME}"
echo "============================================================"

TOTAL_JOBS=0

for LEN in "${LENS[@]}"; do
    REPL_LEN_DIR="${OUTPUT_DIR}/${LEN}"
    CSV_DIR="${LAMBDA_BASE}/train_val_test/${LEN}"
    ml_var="MAX_LENGTH_${LEN}"; MAX_LENGTH="${!ml_var:-8192}"

    if [ ! -d "${CSV_DIR}" ]; then
        echo "WARNING: ${CSV_DIR} not found — skipping ${LEN}"; continue
    fi

    # resolve per-split CSV paths (val.csv or dev.csv)
    declare -A CSV_OF
    CSV_OF[train]="${CSV_DIR}/train.csv"
    CSV_OF[test]="${CSV_DIR}/test.csv"
    if   [ -f "${CSV_DIR}/dev.csv" ]; then CSV_OF[val]="${CSV_DIR}/dev.csv"
    elif [ -f "${CSV_DIR}/val.csv" ]; then CSV_OF[val]="${CSV_DIR}/val.csv"
    else echo "WARNING: no dev.csv/val.csv in ${CSV_DIR} — skipping ${LEN}"; unset CSV_OF; continue; fi

    ok=1
    for split in train val test; do
        [ -f "${CSV_OF[$split]}" ] || { echo "WARNING: missing ${CSV_OF[$split]} — skipping ${LEN}"; ok=0; }
    done
    [ "${ok}" -eq 1 ] || { unset CSV_OF; continue; }

    for VARIANT in ${VARIANTS}; do
        EMB_OUTPUT_DIR="${REPL_LEN_DIR}/embedding/${VARIANT}"
        mkdir -p "${EMB_OUTPUT_DIR}"

        # Start clean: stale shards from a prior run (different shard counts)
        # would corrupt the combine's coverage check.
        rm -rf "${EMB_OUTPUT_DIR}/_shards"
        rm -f  "${EMB_OUTPUT_DIR}/embeddings_pretrained.npz" \
               "${EMB_OUTPUT_DIR}/embeddings_random.npz"

        echo ""
        echo "--- ${LEN} / ${VARIANT}  (out: ${EMB_OUTPUT_DIR}) ---"

        DEP_IDS=()
        for WHICH in "${WHICH_LIST[@]}"; do
            for split in train val test; do
                n=$(rows_of "${CSV_OF[$split]}")
                k=$(nshards_of "${n}" "${SHARD_SIZE}")
                JOB="shemb_${LEN}_${VARIANT}_${WHICH}_${split}"
                echo "  ${WHICH}/${split}: ${n} rows -> ${k} shard(s)  [array 0-$((k-1))${ARRAY_THROTTLE}]"
                jid=$(sbatch --parsable \
                    --job-name="${JOB}" \
                    --array=0-$((k-1))${ARRAY_THROTTLE} \
                    --output="${LOGDIR}/${JOB}_%A_%a.out" \
                    --error="${LOGDIR}/${JOB}_%A_%a.err" \
                    "${EXTRACT_FLAGS[@]}" \
                    --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},HF_HOME=${HF_HOME},MODEL_PATH=${BASE_MODEL},WHICH=${WHICH},SPLIT=${split},NUM_SHARDS=${k},CSV_DIR=${CSV_DIR},OUTPUT_DIR=${EMB_OUTPUT_DIR},BATCH_SIZE=${INF_BATCH_SIZE},MAX_LENGTH=${MAX_LENGTH},POOLING=${POOLING},EMB_SEED=${EMB_SEED},LEN=${LEN}" \
                    "${EXTRACT_JOB}")
                DEP_IDS+=("${jid}")
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            done
        done

        # combine + analysis, after ALL shard arrays for this variant finish OK
        DEP=$(IFS=:; echo "${DEP_IDS[*]}")
        CJOB="shembcomb_${LEN}_${VARIANT}"
        echo "  combine: ${CJOB}  (afterok:${DEP})"
        sbatch \
            --job-name="${CJOB}" \
            --dependency="afterok:${DEP}" \
            --output="${LOGDIR}/${CJOB}_%j.out" \
            --error="${LOGDIR}/${CJOB}_%j.err" \
            "${COMBINE_FLAGS[@]}" \
            --export="ALL,REPO_ROOT=${REPO_ROOT},CONDA_ENV=${CONDA_ENV},HF_HOME=${HF_HOME},MODEL_PATH=${BASE_MODEL},CSV_DIR=${CSV_DIR},OUTPUT_DIR=${EMB_OUTPUT_DIR},BATCH_SIZE=${INF_BATCH_SIZE},MAX_LENGTH=${MAX_LENGTH},POOLING=${POOLING},EMB_SEED=${EMB_SEED},NN_EPOCHS=${NN_EPOCHS},NN_HIDDEN_DIM=${NN_HIDDEN_DIM},NN_LR=${NN_LR},INCLUDE_RANDOM_BASELINE=${INCLUDE_RANDOM_BASELINE:-false},LEN=${LEN}" \
            "${COMBINE_JOB}"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
    unset CSV_OF
done

echo ""
echo "Submitted ${TOTAL_JOBS} jobs. Monitor: squeue -u \$USER"
echo "Result lands in: ${OUTPUT_DIR}/<LEN>/embedding/<variant>/embedding_analysis_results.json"
