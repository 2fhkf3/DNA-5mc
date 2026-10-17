#!/usr/bin/env bash
# 只加载已训练的 best.pt，不建立优化器，不继续训练，不上传 W&B。
set -euo pipefail

# 本次评测配置：直接运行此脚本即可评测 V610-v2；环境变量仍可临时覆盖。
# VERSION="${VERSION:-V543}"                 # 模型版本，须与 Step2 特征一致。
VERSION="${VERSION:-V011}"
DEVICE="${DEVICE:-cuda:1}"                 # 分类器推理设备；可改为 cuda:1 或 cpu。
CPU_THREADS="${CPU_THREADS:-4}"            # CPU 算子线程数。
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# RUNS_ROOT：已训练模型的 experiment 目录，读取其中 runs/*/best.pt。
# RUNS_ROOT="${RUNS_ROOT:-/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V543-v2}"
RUNS_ROOT="${RUNS_ROOT:-/home/shj/ZJ-lab/Modification/DNA-5mC/train/output/V011-v2}"
# CROPPED_DIR：本次直接读取 Step2 的 15 nt 特征，推理时自动中心裁剪，无需先运行 Step3。
# CROPPED_DIR="${CROPPED_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/data/step2_output/features/V543}"
CROPPED_DIR="${CROPPED_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/data/step2_output/features/V011}"
# RESULTS_DIR：当前 VERSION 的批量结果目录，每个 run 一个子目录。
# RESULTS_DIR="${RESULTS_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/output/results/V543}"
RESULTS_DIR="${RESULTS_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/output/results/V011}"
# 更换模型版本时，请同步修改 VERSION 和上面三个绝对路径，或通过环境变量覆盖。
# SEEDS：评测 seed1–5 中已经完成的 run，不根据 WGA 结果挑选 seed。
SEEDS="${SEEDS:-1 2 3 4 5 6}"
# STRATEGIES：空值表示所有已完成分类头；也可填写空格分隔的分类头名称。
STRATEGIES="${STRATEGIES:-}"
# EVAL_BATCH_SIZE：分类头推理批次，和主干提取的 BATCH_SIZE 分开。
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
# SITE_THRESHOLD：位点平均概率的阳性阈值，默认 0.5。
SITE_THRESHOLD="${SITE_THRESHOLD:-0.5}"
# BOOTSTRAP / BLOCK_SIZE / BOOTSTRAP_SEED：区块重采样次数、区块 bp、随机种子。
BOOTSTRAP="${BOOTSTRAP:-1000}"
BLOCK_SIZE="${BLOCK_SIZE:-1000000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-42}"
# READ_THRESHOLD：空值沿用 checkpoint 保存的阈值。
READ_THRESHOLD="${READ_THRESHOLD:-}"
# SAVE_READ_PREDICTIONS：1 另存逐窗口 TSV.gz；0 只保留 SQLite 索引和其他结果。
SAVE_READ_PREDICTIONS="${SAVE_READ_PREDICTIONS:-1}"

cmd=("${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/step4_evaluate.py"
     --version "${VERSION}"
     --feature-dir "${CROPPED_DIR}"
     --out-dir "${RESULTS_DIR}"
     --device "${DEVICE}"
     --cpu-threads "${CPU_THREADS}"
     --batch-size "${EVAL_BATCH_SIZE}"
     --site-threshold "${SITE_THRESHOLD}"
     --bootstrap "${BOOTSTRAP}"
     --block-size "${BLOCK_SIZE}"
     --bootstrap-seed "${BOOTSTRAP_SEED}")
# RUN_DIR：单个 run；设置时 RESULTS_DIR 就是该单模型的结果目录。
if [[ -n "${RUN_DIR:-}" ]]; then
    cmd+=(--run-dir "${RUN_DIR}")
else
    cmd+=(--runs-root "${RUNS_ROOT}")
fi
read -r -a seed_values <<< "${SEEDS}"
cmd+=(--seeds "${seed_values[@]}")
# STRATEGIES：可选，以空格分隔的分类头列表；空值表示所有完成的分类头。
if [[ -n "${STRATEGIES}" ]]; then
    read -r -a strategy_values <<< "${STRATEGIES}"
    cmd+=(--strategies "${strategy_values[@]}")
fi
# READ_THRESHOLD：空值沿用 checkpoint 阈值；只能使用事先固定/独立校准的阈值。
[[ -z "${READ_THRESHOLD}" ]] || cmd+=(--threshold "${READ_THRESHOLD}")
# SAVE_READ_PREDICTIONS=0：不额外写逐窗口 TSV.gz；仍保留共同样本匹配 SQLite。
[[ "${SAVE_READ_PREDICTIONS}" != 0 ]] || cmd+=(--no-save-read-predictions)
cmd+=("$@")
run_test1_command "${cmd[@]}"
