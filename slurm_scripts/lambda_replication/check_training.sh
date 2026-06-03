#!/bin/bash
#
# GENERanno LAMBDA replication — check that all STAGE 1 training jobs finished.
#
# Reads the same lambda_replication.conf as the launcher, then for every
# (length, variant, seed) cell reports:
#   RESULTS  test_results.json present (training + eval finished)
#   MODEL    saved model present (best_model/ with config.json or weights)
#   MCC      test-set MCC straight from test_results.json (key "eval_mcc")
#   LOG      whether the matching SLURM .out log ended with "Job completed"
# and lists any non-empty .err files (potential failures).
#
# The thin finetune job body writes each seed to the single canonical OUTPUT_DIR
# tree (no more repo-internal results/csv_binary/... split):
#   <OUTPUT_DIR>/<LEN>/finetune/<variant>/seed-<SEED>/
#
# Usage:
#   bash slurm_scripts/lambda_replication/check_training.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG="${SCRIPT_DIR}/lambda_replication.conf"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: missing ${CONFIG}"; exit 1
fi
# shellcheck disable=SC1090
source "${CONFIG}"

if [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: OUTPUT_DIR is empty (check ${CONFIG})"; exit 1
fi

RUN_LENGTHS="$(echo "${SEGMENT_LENGTHS}" | xargs)"
LOGDIR="${OUTPUT_DIR}/logs"

echo "============================================================"
echo "GENERanno LAMBDA replication — training check"
echo "============================================================"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  SEGMENT_LENGTHS: ${RUN_LENGTHS}"
echo "  VARIANTS:        ${VARIANTS}"
echo "  SEEDS:           ${SEEDS}"
echo "============================================================"
echo ""

TOTAL=0
OK=0

printf "%-4s %-10s %-5s  %-8s  %-8s  %-8s  %s\n" LEN VARIANT SEED RESULTS MODEL MCC LOG
for LEN in ${RUN_LENGTHS}; do
    for VARIANT in ${VARIANTS}; do
        SEED_ROOT="${OUTPUT_DIR}/${LEN}/finetune/${VARIANT}"
        for SEED in ${SEEDS}; do
            TOTAL=$((TOTAL + 1))
            D="${SEED_ROOT}/seed-${SEED}"

            if [ -f "${D}/test_results.json" ]; then R=ok; else R=MISSING; fi

            if [ -f "${D}/best_model/config.json" ] || \
               [ -f "${D}/best_model/pytorch_model.bin" ] || \
               [ -f "${D}/best_model/model.safetensors" ]; then
                M=ok
            else
                M=MISSING
            fi

            MCC=$(python - "${D}/test_results.json" 2>/dev/null <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    v = d.get('eval_mcc', d.get('eval_matthews_correlation', d.get('mcc')))
    print(f"{v:.4f}" if isinstance(v, (int, float)) else "?")
except Exception:
    print("-")
PY
)

            LOG=$(ls -t "${LOGDIR}/ft_${LEN}_${VARIANT}_s${SEED}_"*.out 2>/dev/null | head -1)
            if [ -n "${LOG}" ] && grep -q "Job completed at" "${LOG}"; then
                L=done
            elif [ -n "${LOG}" ]; then
                L="NOT done"
            else
                L="no .out"
            fi

            [ "${R}" = ok ] && [ "${M}" = ok ] && [ "${L}" = done ] && OK=$((OK + 1))

            printf "%-4s %-10s %-5s  %-8s  %-8s  %-8s  %s\n" \
                "${LEN}" "${VARIANT}" "${SEED}" "${R}" "${M}" "${MCC}" "${L}"
        done
    done
done

echo ""
echo "Healthy: ${OK} / ${TOTAL}"

echo ""
echo "=== non-empty .err files (potential failures) ==="
ERRS=$(find "${LOGDIR}" -name "ft_*.err" -size +0c -printf "%s  %p\n" 2>/dev/null | sort -rn)
if [ -n "${ERRS}" ]; then
    echo "${ERRS}"
else
    echo "  (none)"
fi
