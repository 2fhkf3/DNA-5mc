#!/usr/bin/env bash
# Step4：按固定抽样清单求 read 概率均值，生成每个run/repeat/depth的 site-level 性能。
# CONFIG/--config、PROTOCOL_DIR/--protocol-dir：与前序一致；输出 results/<version>/<run>/。
# VERSION/--version：可选，仅计算冻结清单中的一个模型；不改变共同位点和抽样。
# RUN_NAME/--run-name：可选，如 center_5mer_mlp_raw768_seed1；可与 VERSION 配合拆分任务。
# site_threshold：默认0.5，恰等于阈值判阳性；禁止用 Test3 分深度调参。
# save_all_site_scores：是否额外保存大分数矩阵；save_site_stability：是否保存逐位点稳定性表。
# bootstrap.block_size_bp：区块宽度，当前只累计区块TP/TN/FP/FN，区间在Step5计算。
# DRY_RUN=1/--dry-run：只显示计划；此步骤只汇总已审计概率，无需GPU。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
args=()
[[ -z "${VERSION:-}" ]] || args+=(--version "${VERSION}")
[[ -z "${RUN_NAME:-}" ]] || args+=(--run-name "${RUN_NAME}")
run_test3_step step4_evaluate_coverage.py "${args[@]}" "$@"
