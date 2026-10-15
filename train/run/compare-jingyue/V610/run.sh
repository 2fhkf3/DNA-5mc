#!/usr/bin/env bash
# Run from any directory. Arguments after an optional YAML path go to train.py.
# Examples:
#   bash run.sh --dry-run
#   bash run.sh --strategies center_5mer_mlp_raw768 --seeds 1 --device cuda:0
#   bash run.sh /path/to/config.yaml --epochs 2 --experiment quick_check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="${SCRIPT_DIR}"
while [[ ! -f "${TRAIN_DIR}/train.py" ]]; do
  if [[ "${TRAIN_DIR}" == / ]]; then
    echo "[ERROR] Cannot find the project's train.py above ${SCRIPT_DIR}" >&2
    exit 1
  fi
  TRAIN_DIR="$(dirname "${TRAIN_DIR}")"
done
source "${TRAIN_DIR}/../workflow/paths.sh"
CONFIG="${CONFIG:-${SCRIPT_DIR}/config.yaml}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG="$1"
  shift
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Configuration not found: ${CONFIG}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
cmd=("${PYTHON_BIN}" -u "${TRAIN_DIR}/train.py" --config "${CONFIG}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '[INFO] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi
exec "${cmd[@]}"
