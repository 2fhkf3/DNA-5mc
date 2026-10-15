#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
cmd=("${PYTHON_BIN:-python3}" -u "${SCRIPT_DIR}/step3.2_filter_chr19_mapq.py"
     --input-dir "${INPUT_DIR:-${OUTPUT_ROOT}/step3_output/chr19}"
     --out-dir "${OUT_DIR:-${OUTPUT_ROOT}/step3_output/chr19_mapq60_cov10}"
     --min-mapq "${MIN_MAPQ:-60}" --min-coverage "${MIN_COVERAGE:-10}"
     --workers "${WORKERS:-4}" --chunk-mb "${CHUNK_MB:-16}"
     --gzip-level "${GZIP_LEVEL:-3}" --log-every "${LOG_EVERY:-250000}")
cmd+=("$@")
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
[[ "${DRY_RUN:-0}" != 1 ]] || exit 0
exec "${cmd[@]}"
