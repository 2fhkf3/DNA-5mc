#!/usr/bin/env bash
# Test2 环境入口；PYTHON_BIN 可覆盖为包含 torch、numpy、PyYAML 的解释器。
set -euo pipefail
TEST2_WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TEST2_WORKFLOW_DIR}/../../../workflow/paths.sh"
# CONFIG：各步骤共享配置；命令行 --config 优先于此环境变量。
CONFIG="${CONFIG:-${TEST2_WORKFLOW_DIR}/config.yaml}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

run_test2_step() {
    local script="$1"
    shift
    local -a command=("${PYTHON_BIN}" "${TEST2_WORKFLOW_DIR}/${script}" --config "${CONFIG}")
    # VERSION：可选，只处理一个已在 YAML 配置的模型；不设置则使用 enabled_versions。
    [[ -z "${VERSION:-}" ]] || command+=(--version "${VERSION}")
    # MAX_RECORDS：只扫描原始输入前 N 行用于联调，输出自动标记 partial_test。
    [[ -z "${MAX_RECORDS:-}" ]] || command+=(--max-records "${MAX_RECORDS}")
    # DRY_RUN=1：调用 Python 的只读计划检查，不创建正式缓存或评测结果。
    [[ "${DRY_RUN:-0}" != 1 ]] || command+=(--dry-run)
    "${command[@]}" "$@"
}
