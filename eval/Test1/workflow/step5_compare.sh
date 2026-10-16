#!/usr/bin/env bash
# 汇总逐 seed 结果，并按原始 read+位点键进行各主干共同样本比较。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# RESULTS_ROOT：包含 V011、V543、V610 结果子目录的根目录。
# COMPARISON_DIR：新建汇总目录。
# VERSIONS：参与比较的版本，按空格分隔；缺少对应 run 的组会记录在 manifest。
read -r -a versions <<< "${VERSIONS:-V011 V543 V610}"
cmd=("${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/step5_compare.py"
     --results-dir "${RESULTS_ROOT:-${TEST1_OUTPUT_ROOT}/results}"
     --out-dir "${COMPARISON_DIR:-${TEST1_OUTPUT_ROOT}/comparison}"
     --versions "${versions[@]}" --bootstrap "${BOOTSTRAP:-1000}"
     --block-size "${BLOCK_SIZE:-1000000}" --bootstrap-seed "${BOOTSTRAP_SEED:-42}")
cmd+=("$@")
run_test1_command "${cmd[@]}"
