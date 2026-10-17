#!/usr/bin/env bash
# Step5：逐seed/逐模型汇总、配对深度损失与模型差值、区块bootstrap及可选图形。
# CONFIG/--config、PROTOCOL_DIR/--protocol-dir：与前序一致；要求所有冻结run的Step4已完成。
# bootstrap.repeats/seed：区块重采样次数和seed；0次禁用，所有模型/深度共享区块权重。
# bootstrap.metrics：只支持 balanced_accuracy/fpr/fnr，不用区块AUC平均值代替全局AUC。
# plots.enabled：是否输出PNG/PDF曲线；带为抽样经验范围，不能当作生物学置信区间。
# allow_partial_results：默认拒绝联调子集；不同seed列表仍使用各自全部冻结seed汇总。
# DRY_RUN=1/--dry-run：只显示汇总计划；输出 summary/，拒绝覆盖旧结果。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_test3_step step5_summarize.py "$@"
