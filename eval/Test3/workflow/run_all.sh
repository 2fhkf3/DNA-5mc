#!/usr/bin/env bash
# 默认 --test：仅运行 /tmp 下的小数据测试，不执行正式评测。
# --dry-run：五步只读计划；--run：按当前 CONFIG 正式执行全部步骤。
# --from-step N：从第N步继续，N为1..5；前序必须已完成，配置必须与Step1冻结配置相同。
# CONFIG：YAML路径；PROTOCOL_DIR：协议根目录覆盖值；每次调用须保持一致。
# MAX_RECORDS：只允许单独Step1联调，一键正式流程拒绝限量，防止混淆完整评测。
# 测试：bash run_all.sh --test；计划：bash run_all.sh --dry-run；续跑：bash run_all.sh --run --from-step 4
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mode=--test
first=1
while (($#)); do
    case "$1" in
        --test|--dry-run|--run) mode="$1"; shift ;;
        --from-step) first="${2:?Missing step number}"; shift 2 ;;
        --config) CONFIG="${2:?Missing YAML path}"; export CONFIG; shift 2 ;;
        --protocol-dir) PROTOCOL_DIR="${2:?Missing protocol directory}"; export PROTOCOL_DIR; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "${first}" =~ ^[1-5]$ ]] || { echo '--from-step must be 1..5' >&2; exit 2; }
[[ -z "${MAX_RECORDS:-}" ]] || { echo 'Use standalone Step1 for MAX_RECORDS smoke tests' >&2; exit 2; }
if [[ "${mode}" == --test ]]; then
    exec "${PYTHON_BIN}" "${TEST3_WORKFLOW_DIR}/test_workflow.py"
fi
[[ "${mode}" != --dry-run ]] || export DRY_RUN=1
steps=(step1_audit_predictions.sh step2_build_common_pool.sh step3_sample_site_reads.sh step4_evaluate_coverage.sh step5_summarize.sh)
for ((i=first-1;i<5;i++)); do
    bash "${TEST3_WORKFLOW_DIR}/${steps[i]}"
done
