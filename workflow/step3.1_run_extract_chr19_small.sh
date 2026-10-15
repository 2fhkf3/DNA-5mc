#!/usr/bin/env bash
set -euo pipefail
# Python infers train/valid/test from INPUT and writes the same split filename.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
INPUT="${INPUT:-${OUTPUT_ROOT}/step3_output/chr16/test.jsonl.gz}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/step3_output/chr16-small-test}"
POSITIVE_COUNT="${POSITIVE_COUNT:-5000}"
NEGATIVE_COUNT="${NEGATIVE_COUNT:-5000}"
SEED="${SEED:-42}"
MIN_COVERAGE="${MIN_COVERAGE:-5}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"
WORKERS="${WORKERS:-4}"
GZIP_LEVEL="${GZIP_LEVEL:-3}"
LOG_EVERY="${LOG_EVERY:-100000}"
cmd=("${PYTHON_BIN}" -u "${SCRIPT_DIR}/step3.1_extract_chr19_small.py"
     --input "${INPUT}" --out-dir "${OUT_DIR}"
     --positive-count "${POSITIVE_COUNT}" --negative-count "${NEGATIVE_COUNT}"
     --seed "${SEED}" --min-coverage "${MIN_COVERAGE}" --workers "${WORKERS}" --gzip-level "${GZIP_LEVEL}"
     --log-every "${LOG_EVERY}")
[[ "${REPLACE_EXISTING}" != "1" ]] || cmd+=(--replace-existing)
[[ -z "${SPLIT:-}" ]] || cmd+=(--split "${SPLIT}")
cmd+=("$@")
printf '[INFO] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" != "1" ]] || exit 0
exec "${cmd[@]}"
