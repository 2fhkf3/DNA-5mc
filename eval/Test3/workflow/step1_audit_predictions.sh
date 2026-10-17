#!/usr/bin/env bash
# Step1：冻结已完成 runs，审计原始完整正负 test，生成每个模型观测索引及每个 run 的概率向量。
# CONFIG/--config：YAML 路径；models 指定原始特征和训练输出，strategies/training_seeds 选择范围。
# PROTOCOL_DIR/--protocol-dir：新的实验根目录，必须不存在；后续每步使用相同目录。
# MAX_RECORDS/--max-records：每个模型仅扫描原始 test 前 N 行，用于临时联调；标记 partial_test。
# DRY_RUN=1/--dry-run：只显示冻结清单，不导出数据、不加载模型到 GPU。
# prediction_source=saved：无需GPU；inference：按 device/batch_size/cpu_threads 从本主干缓存推理一次。
# 示例：DRY_RUN=1 bash step1_audit_predictions.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
args=()
[[ -z "${MAX_RECORDS:-}" ]] || args+=(--max-records "${MAX_RECORDS}")
run_test3_step step1_audit_predictions.py "${args[@]}" "$@"
