#!/usr/bin/env bash
# Step2：默认复用 test_predictions.tsv；只纳入参考 label=1，输出 read/site FNR 与 Recall。
# SOURCE：predictions（无需 Step1、GPU）或 features（先运行 Step1）；默认由 YAML 决定。
# RUN_DIR：可选，只评测一个已完成 run；须选择单个 VERSION。
# FEATURE_DIR：可选，覆盖单模型 Step1 阳性缓存；仅 SOURCE=features 使用。
# OUT_DIR：结果根目录，实际输出为 OUT_DIR/VERSION/<strategy>_seedN；已有目录拒绝覆盖。
# DEVICE：auto、cpu、cuda:N，仅 features 路线使用；batch_size/cpu_threads 在 YAML 中设置。
# 示例：VERSION=V011 DRY_RUN=1 bash step2_evaluate.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
args=()
[[ -z "${SOURCE:-}" ]] || args+=(--source "${SOURCE}")
[[ -z "${RUN_DIR:-}" ]] || args+=(--run-dir "${RUN_DIR}")
[[ -z "${FEATURE_DIR:-}" ]] || args+=(--feature-dir "${FEATURE_DIR}")
[[ -z "${DEVICE:-}" ]] || args+=(--device "${DEVICE}")
[[ -z "${OUT_DIR:-}" ]] || args+=(--out-dir "${OUT_DIR}")
run_test2_step step2_evaluate.py "${args[@]}" "$@"
