#!/usr/bin/env bash
# 可选 Step1：按 label=1 同步筛选已有 test 的特征/标签/元数据，禁止重新编码或改变标签。
# INPUT_DIR/OUT_DIR：覆盖单个模型输入/阳性缓存目录，使用时须指定 VERSION 或 --version。
# CHUNK_SIZE 等内存参数在 config.yaml 的 selection 中配置。
# 示例：VERSION=V011 DRY_RUN=1 bash step1_select_native_positives.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
args=()
[[ -z "${INPUT_DIR:-}" ]] || args+=(--input-dir "${INPUT_DIR}")
[[ -z "${OUT_DIR:-}" ]] || args+=(--out-dir "${OUT_DIR}")
run_test2_step step1_select_native_positives.py "${args[@]}" "$@"
