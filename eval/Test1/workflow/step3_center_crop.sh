#!/usr/bin/env bash
# 只裁剪 test 特征；保留原 15 nt 输入上下文，不重新编码短信号。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# FEATURES_DIR：Step2 的完整 15 nt 表征目录。
# CROPPED_DIR：裁剪后目录，需不存在。
# CROP_WINDOW：输出的中心位置数，当前训练的 v2 模型使用 5。
# CHUNK_SIZE：一次读写的窗口数，影响内存/磁盘吞吐，不影响结果。
cmd=("${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/step3_center_crop.py"
     --input-dir "${FEATURES_DIR:-${TEST1_OUTPUT_ROOT}/features/${VERSION}}"
     --out-dir "${CROPPED_DIR:-${TEST1_OUTPUT_ROOT}/center5/${VERSION}}"
     --window-size "${CROP_WINDOW:-5}" --chunk-size "${CHUNK_SIZE:-4096}")
cmd+=("$@")
run_test1_command "${cmd[@]}"
