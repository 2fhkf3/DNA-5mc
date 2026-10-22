#!/usr/bin/env bash
# Test4共享环境；配置只读，正式输出由协议目录隔离。
set -euo pipefail
TEST4_WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TEST4_WORKFLOW_DIR}/../../../workflow/paths.sh"
CONFIG="${CONFIG:-${TEST4_WORKFLOW_DIR}/config.yaml}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dna5mc-test4-matplotlib-${UID}}"
export OMP_NUM_THREADS="${TEST4_CPU_THREADS:-4}" MKL_NUM_THREADS="${TEST4_CPU_THREADS:-4}" OPENBLAS_NUM_THREADS="${TEST4_CPU_THREADS:-4}"

run_test4_step() {
    local script="$1"
    shift
    local -a command=("${PYTHON_BIN}" "${TEST4_WORKFLOW_DIR}/${script}" --config "${CONFIG}")
    [[ -z "${PROTOCOL_DIR:-}" ]] || command+=(--protocol-dir "${PROTOCOL_DIR}")
    [[ -z "${MAX_RECORDS:-}" ]] || command+=(--max-records "${MAX_RECORDS}")
    [[ "${DRY_RUN:-0}" != 1 ]] || command+=(--dry-run)
    "${command[@]}" "$@"
}
