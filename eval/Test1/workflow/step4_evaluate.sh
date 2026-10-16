#!/usr/bin/env bash
# 只加载已训练的 best.pt，不建立优化器，不继续训练，不上传 W&B。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# CROPPED_DIR：Step3 特征；也可指定 15 nt 特征，由推理器按 checkpoint 裁剪。
# RESULTS_DIR：当前 VERSION 的批量结果目录，每个 run 一个子目录。
# EVAL_BATCH_SIZE：分类头推理批次，和主干提取的 BATCH_SIZE 分开。
# SITE_THRESHOLD：位点平均概率的阳性阈值，默认 0.5。
# BOOTSTRAP / BLOCK_SIZE / BOOTSTRAP_SEED：区块重采样次数、区块 bp、随机种子。
cmd=("${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/step4_evaluate.py"
     --version "${VERSION}" --feature-dir "${CROPPED_DIR:-${TEST1_OUTPUT_ROOT}/center5/${VERSION}}"
     --out-dir "${RESULTS_DIR:-${TEST1_OUTPUT_ROOT}/results/${VERSION}}"
     --device "${DEVICE}" --cpu-threads "${CPU_THREADS}" --batch-size "${EVAL_BATCH_SIZE:-2048}"
     --site-threshold "${SITE_THRESHOLD:-0.5}" --bootstrap "${BOOTSTRAP:-1000}"
     --block-size "${BLOCK_SIZE:-1000000}" --bootstrap-seed "${BOOTSTRAP_SEED:-42}")
# RUN_DIR：单个 run；设置时 RESULTS_DIR 就是该单模型的结果目录。
# RUNS_ROOT：批量模式的 experiment 目录，默认 train/output/<VERSION>-v2。
if [[ -n "${RUN_DIR:-}" ]]; then
    cmd+=(--run-dir "${RUN_DIR}")
else
    cmd+=(--runs-root "${RUNS_ROOT:-${PROJECT_ROOT}/train/output/${VERSION}-v2}")
fi
# SEEDS：批量模式默认共同的 seed1/2/3；不会根据 WGA FPR 选择 seed。
read -r -a seed_values <<< "${SEEDS:-1 2 3}"
cmd+=(--seeds "${seed_values[@]}")
# STRATEGIES：可选，以空格分隔的分类头列表；空值表示所有完成的分类头。
if [[ -n "${STRATEGIES:-}" ]]; then
    read -r -a strategy_values <<< "${STRATEGIES}"
    cmd+=(--strategies "${strategy_values[@]}")
fi
# READ_THRESHOLD：空值沿用 checkpoint 阈值；只能使用事先固定/独立校准的阈值。
[[ -z "${READ_THRESHOLD:-}" ]] || cmd+=(--threshold "${READ_THRESHOLD}")
# SAVE_READ_PREDICTIONS=0：不额外写逐窗口 TSV.gz；仍保留共同样本匹配 SQLite。
[[ "${SAVE_READ_PREDICTIONS:-1}" != 0 ]] || cmd+=(--no-save-read-predictions)
cmd+=("$@")
run_test1_command "${cmd[@]}"
