#!/usr/bin/env bash
# 各步骤共用的路径与环境；所有路径均不依赖调用时所在目录。
set -euo pipefail
TEST1_WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TEST1_WORKFLOW_DIR}/../../../workflow/paths.sh"
# TEST1_OUTPUT_ROOT：正式结果的根目录；测试工具会改到 /tmp 并自动清理。
TEST1_OUTPUT_ROOT="${TEST1_OUTPUT_ROOT:-${PROJECT_ROOT}/eval/Test1/output}"
# VERSION：主干版本，只允许 V011、V543、V610。
VERSION="${VERSION:-V610}"
case "${VERSION}" in V011|V543|V610) ;; *) echo 'Invalid VERSION' >&2; exit 2;; esac
# DEVICE：auto 自动选择可用 GPU/CPU；可以显式设 cpu、cuda:0。
DEVICE="${DEVICE:-auto}"
# CPU_THREADS：每个 Python 进程使用的 CPU 算子线程数。
CPU_THREADS="${CPU_THREADS:-4}"
# HF_MODULES_CACHE：自定义 Hugging Face 模型代码缓存；不修改主干权重。
export HF_MODULES_CACHE="${TEST1_HF_MODULES_CACHE:-${TEST1_OUTPUT_ROOT}/.hf_modules}"
export HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}"

run_test1_command() {
    # DRY_RUN=1：只显示命令，不创建正式数据/特征/评测结果。
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '[DRY RUN]'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}
