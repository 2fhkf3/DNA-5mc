#!/usr/bin/env bash
# Step2：所有已冻结模型/策略/训练seed共同观测取交集，之后重算 coverage 并固定测试位点。
# CONFIG/--config：须与 Step1 完全一致；禁止中途更换启用模型或任何配置。
# PROTOCOL_DIR/--protocol-dir：Step1 已完成的实验根目录；输出 common/，拒绝覆盖。
# minimum_common_reads：YAML 中设置，默认20；不足的位点在5/10/20三个深度全部排除。
# site_balance：none 保留全部合格位点；equal 按 balance_seed 一次固定抽等量正负位点。
# allow_partial_results：正式为false；仅临时联调可true，不是补齐缺失 read。
# DRY_RUN=1/--dry-run：只显示输入、输出和平衡策略。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_test3_step step2_build_common_pool.py "$@"
