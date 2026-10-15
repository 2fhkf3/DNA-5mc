#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
SPLIT_SEED="${SPLIT_SEED:-42}"

# Both labels are read from this MOD run's BED. Input already contains Apple
# processed full-read signal; this stage only extracts windows.
INPUT_JSONL="${INPUT_JSONL:-${MOD_JSONL:-${OUTPUT_ROOT}/step1_output/s2_all_reads.genome_signal.preprocessed-2.jsonl}}"
BED_GZ="${BED_GZ:-${OUTPUT_ROOT}/step2_input/bed/HG002_250F500490011.bed.gz}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/step2_output/chr16_bed01_cov20_15nt_test}"

cmd=("${PYTHON_BIN}" "${SCRIPT_DIR}/step2_build_chrom19_dataset.py" \
  --jsonl "${INPUT_JSONL}" \
  --bed-gz "${BED_GZ}" \
  --out-dir "${OUT_DIR}" \
  --chrom chr16 \
  --mod-type m \
  --bed-strand + \
  --read-strand + \
  --center-base C \
  --flank 7 \
  --min-coverage 20 \
  --min-actual-coverage 20 \
  --coverage-op ge \
  --min-mapq 30 \
  --test-only \
  --split-seed "${SPLIT_SEED}" \
  "$@")
printf '[INFO] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" != "1" ]] || exit 0
exec "${cmd[@]}"


# "${PYTHON_BIN}" "${SCRIPT_DIR}/step2_build_chrom19_dataset.py" \
#   --jsonl "${INPUT_JSONL}" \
#   --bed-gz "${BED_GZ}" \
#   --out-dir "${OUT_DIR}" \
#   --chrom chr19 \
#   --mod-type m \
#   --bed-strand + \
#   --read-strand + \
#   --center-base C \
#   --flank 7 \
#   --min-coverage 10 \
#   --min-actual-coverage 10 \
#   --coverage-op ge \
#   --min-mapq 30 \
#   --train-ratio 0.9 \
#   --valid-ratio 0.1 \
#   --split-seed "${SPLIT_SEED}" \
#   "$@"
