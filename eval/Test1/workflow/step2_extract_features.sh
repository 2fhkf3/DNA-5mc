#!/usr/bin/env bash
# 单独提取 test split；复用原主干实现，BERT 通过适配器只处理 test。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# DATA_DIR：Step1 的窗口目录。
# FEATURES_DIR：当前 VERSION 的 15 nt 特征输出，包含 test.pt/bin/TSV。
# BATCH_SIZE：主干推理批次；CPU 测试用 1，正式 GPU 根据显存调整。
# FEATURE_DTYPE：缓存存储精度，默认 float16，与现有 v2 特征一致。
cmd=("${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/step2_extract_features.py"
     --version "${VERSION}" --data-dir "${DATA_DIR:-${TEST1_OUTPUT_ROOT}/data}"
     --out-dir "${FEATURES_DIR:-${TEST1_OUTPUT_ROOT}/features/${VERSION}}"
     --device "${DEVICE}" --cpu-threads "${CPU_THREADS}"
     --batch-size "${BATCH_SIZE:-4}" --dtype "${FEATURE_DTYPE:-float16}")
# MODEL_ROOT：可选模型根目录；默认使用 train/backbone 中对应版本链接。
# 版本对应的 encoder layer、decoder layer、ODE、token 等在 Python 中固定为原训练协议。
[[ -z "${MODEL_ROOT:-}" ]] || cmd+=(--model-root "${MODEL_ROOT}")
cmd+=("$@")
run_test1_command "${cmd[@]}"
