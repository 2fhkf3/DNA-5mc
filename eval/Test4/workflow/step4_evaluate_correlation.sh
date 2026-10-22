#!/usr/bin/env bash
# Step4：Pearson/Spearman及误差，覆盖度/beta分层，共享空间区块bootstrap。
# VERSION、RUN_NAME仅分配已冻结运行，不能改变配对位点集合。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
args=()
[[ -z "${VERSION:-}" ]] || args+=(--version "${VERSION}")
[[ -z "${RUN_NAME:-}" ]] || args+=(--run-name "${RUN_NAME}")
run_test4_step step4_evaluate_correlation.py "${args[@]}" "$@"
