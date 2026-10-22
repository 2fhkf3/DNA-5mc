#!/usr/bin/env bash
# Step0：确认参考FASTA/坐标/预处理证据，冻结连续WGBS位点及模型/分类头内容哈希。
# DRY_RUN=1只列出计划及缺项，不扫描完整BED/JSONL，不加载GPU。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_test4_step step0_audit_inputs.py "$@"
