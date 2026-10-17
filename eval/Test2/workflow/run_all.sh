#!/usr/bin/env bash
# 默认 --test：仅在临时目录中验证，不导出正式结果。
# --dry-run：只读显示 Step2/Step3 执行计划。
# --run：按 YAML 执行正式流程；features 路线先运行 Step1，predictions 路线跳过 Step1。
# CONFIG 可指定另一份 YAML；VERSION 可缩小范围；完整路径可从任意工作目录调用。
# 一键入口不接受 MAX_RECORDS；限量联调请调用单步脚本，避免把部分数据当正式比较。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ -z "${MAX_RECORDS:-}" ]] || { echo 'Use per-step scripts for MAX_RECORDS smoke tests' >&2; exit 2; }
mode="${1:---test}"
case "${mode}" in
    --test)
        "${PYTHON_BIN}" "${TEST2_WORKFLOW_DIR}/test_workflow.py"
        ;;
    --dry-run)
        DRY_RUN=1 bash "${TEST2_WORKFLOW_DIR}/step2_evaluate.sh"
        DRY_RUN=1 bash "${TEST2_WORKFLOW_DIR}/step3_compare.sh"
        ;;
    --run)
        source_mode="${SOURCE:-$("${PYTHON_BIN}" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["evaluation"]["source"])' "${CONFIG}")}"
        if [[ "${source_mode}" == features ]]; then
            bash "${TEST2_WORKFLOW_DIR}/step1_select_native_positives.sh"
        fi
        bash "${TEST2_WORKFLOW_DIR}/step2_evaluate.sh"
        bash "${TEST2_WORKFLOW_DIR}/step3_compare.sh"
        ;;
    *) echo 'Usage: run_all.sh [--test|--dry-run|--run]; set CONFIG/VERSION as needed' >&2; exit 2 ;;
esac
