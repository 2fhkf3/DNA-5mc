#!/usr/bin/env bash
# Step3：原始集合与各策略共同窗口集合分别汇总，匹配后重算覆盖度；不要求各模型 seed 列表相同。
# RESULTS_ROOT：覆盖 Step2 结果根目录；默认 config.yaml.results_root。
# OUT_DIR：本次比较目录；默认 comparison_dir；禁止覆盖已有结果。
# bootstrap/coverage/threshold 设置须与 Step2 冻结配置一致；正式比较默认拒绝 partial_test。
# 配对差值仅用于相同训练 seed；不相同的 seed 仍纳入 common_aggregate 的均值和样本标准差。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
args=()
[[ -z "${RESULTS_ROOT:-}" ]] || args+=(--results-root "${RESULTS_ROOT}")
[[ -z "${OUT_DIR:-}" ]] || args+=(--out-dir "${OUT_DIR}")
run_test2_step step3_compare.py "${args[@]}" "$@"
