#!/usr/bin/env bash
# 默认--test只在/tmp运行合成测试；--dry-run只读检查，--run才正式处理。
# --from-step 0..5从已校验的完成阶段继续；每步CONFIG/PROTOCOL_DIR/MAX_RECORDS须一致。
# 1.97TiB源数据不会因默认调用而开始扫描。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mode=--test
first=0
while (($#)); do
    case "$1" in
        --test|--dry-run|--run) mode="$1"; shift ;;
        --from-step) first="${2:?Missing step number}"; shift 2 ;;
        --config) CONFIG="${2:?Missing configuration}"; export CONFIG; shift 2 ;;
        --protocol-dir) PROTOCOL_DIR="${2:?Missing protocol directory}"; export PROTOCOL_DIR; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "${first}" =~ ^[0-5]$ ]] || { echo '--from-step must be 0..5' >&2; exit 2; }
if [[ "${mode}" == --test ]]; then
    exec "${PYTHON_BIN}" "${TEST4_WORKFLOW_DIR}/test_workflow.py"
fi
[[ "${mode}" != --dry-run ]] || export DRY_RUN=1
if [[ "${mode}" == --run && -n "${MAX_RECORDS:-}" ]]; then
    echo 'MAX_RECORDS仅用于独立步骤联调；一键正式运行请使用已冻结配置scan.max_records并明确allow_partial_results。' >&2
    exit 2
fi
steps=(step0_audit_inputs.sh step1_build_windows.sh step2_predict_reads.sh step3_aggregate_sites.sh step4_evaluate_correlation.sh step5_compare.sh)
for ((i=first;i<6;i++)); do
    bash "${TEST4_WORKFLOW_DIR}/${steps[i]}"
done
