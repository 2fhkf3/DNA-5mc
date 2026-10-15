#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
INPUT_DIR="${INPUT_DIR:-${OUTPUT_ROOT}/step2_output/chr16_mapq60_cov20}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/step3_output/chr16}"
SEED="${SEED:-42}"
MIN_COVERAGE="${MIN_COVERAGE:-20}"
SITE_POLICY="${SITE_POLICY:-auto}"
WORKERS="${WORKERS:-4}"
CHUNK_MB="${CHUNK_MB:-16}"
GZIP_LEVEL="${GZIP_LEVEL:-3}"
LOG_EVERY="${LOG_EVERY:-100000}"
cmd=("${PYTHON_BIN}" -u "${SCRIPT_DIR}/step3_balance_chr19.py"
     --input-dir "${INPUT_DIR}" --out-dir "${OUT_DIR}" --seed "${SEED}"
     --min-coverage "${MIN_COVERAGE}" --site-policy "${SITE_POLICY}" --workers "${WORKERS}" --chunk-mb "${CHUNK_MB}"
     --gzip-level "${GZIP_LEVEL}" --log-every "${LOG_EVERY}")
if [[ -n "${SPLITS:-}" ]]; then
    read -r -a split_names <<< "${SPLITS}"
    cmd+=(--splits "${split_names[@]}")
fi
cmd+=("$@")
printf '[INFO] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" != "1" ]] || exit 0
exec "${cmd[@]}"
