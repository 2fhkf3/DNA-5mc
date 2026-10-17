#!/usr/bin/env bash
# Test3 共用路径和环境；PYTHON_BIN 可指定含 numpy、torch、PyYAML、matplotlib 的解释器。
set -euo pipefail
TEST3_WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TEST3_WORKFLOW_DIR}/../../../workflow/paths.sh"
# CONFIG：共享配置路径；命令行 --config 可覆盖。启用模型、参数与输出目录都在 YAML 中设置。
CONFIG="${CONFIG:-${TEST3_WORKFLOW_DIR}/config.yaml}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# CPU 线程上限；正式计算以配置 cpu_threads 为准，脚本环境默认4，避免 BLAS 大量抢占线程。
export OMP_NUM_THREADS="${TEST3_CPU_THREADS:-4}" MKL_NUM_THREADS="${TEST3_CPU_THREADS:-4}" OPENBLAS_NUM_THREADS="${TEST3_CPU_THREADS:-4}"

run_test3_step() {
    local script="$1"
    shift
    local -a command=("${PYTHON_BIN}" "${TEST3_WORKFLOW_DIR}/${script}" --config "${CONFIG}")
    # PROTOCOL_DIR：覆盖本次输出根目录；每一步都须使用同一个值，不会改变原训练结果。
    [[ -z "${PROTOCOL_DIR:-}" ]] || command+=(--protocol-dir "${PROTOCOL_DIR}")
    # DRY_RUN=1：只读计划，不创建缓存、抽样或评测结果。
    [[ "${DRY_RUN:-0}" != 1 ]] || command+=(--dry-run)
    "${command[@]}" "$@"
}
