#!/usr/bin/env bash
# V013 Flow Matching：电流 → Codec → 冻结 BERT → ODE refinement → 碱基窗口特征。
# 正式运行：bash workflow/step4_extract_window_hidden_states_Flow-matching.sh
# 仅预览命令：DRY_RUN=1 bash workflow/step4_extract_window_hidden_states_Flow-matching.sh
# 小测试示例（只读取 2 条，输出到新的临时目录）：
#   SPLITS=test LIMIT=2 DEVICE=cpu DATA_DIR=/path/to/windows OUT_DIR=/tmp/v013_smoke \
#     bash workflow/step4_extract_window_hidden_states_Flow-matching.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
# MODEL_ROOT：模型根目录；新版本可覆盖。encoder/主干路径也可独立指定。
MODEL_ROOT="${MODEL_ROOT:-${BACKBONE_ROOT}/V013-Flow-matching}"
ENCODER_DIR="${ENCODER_DIR:-${MODEL_ROOT}/encoder}"
# DECODER_DIR / DLM_DIR：Flow checkpoint 目录，不是旧 V011 的 ELF 目录。
DECODER_DIR="${DECODER_DIR:-${DLM_DIR:-${MODEL_ROOT}/hf_dlm_step10000}}"
# DATA_DIR：含 train/valid/test.jsonl[.gz] 的窗口目录。
# OUT_DIR：特征输出目录；与 V011/V554 分开存储，默认拒绝覆盖已有 split。
DATA_DIR="${DATA_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mc/output/step3_output/new}"
OUT_DIR="${OUT_DIR:-/mnt/nas_data/shenhaojie/step4-output_dir/V013}"
DEVICE="${DEVICE:-auto}"                  # auto / cpu / cuda:0 / cuda:1。
CPU_THREADS="${CPU_THREADS:-4}"            # CPU 算子线程数。
ENCODER_LAYER="${ENCODER_LAYER:-0}"        # Codec 量化轨道；V013 单码本，只支持 0。
HIDDEN_SOURCE="${HIDDEN_SOURCE:-ode_hidden}" # ode_hidden / context_hidden / last_hidden。
# 下列 ODE 默认值沿用 V013 readme 的编码示例，不代表已经为 5mC 调优。
ODE_STEPS="${ODE_STEPS:-1}"                # refinement 步数；仅 ode_hidden 使用。
ODE_START_T="${ODE_START_T:-0.97}"          # ODE 起始时间，(0,1]；1 表示无积分区间。
ODE_SELF_COND_CFG_SCALE="${ODE_SELF_COND_CFG_SCALE:-0.5}" # ODE 自条件 CFG 系数，>=0。
LAST_T="${LAST_T:-1.0}"                    # last_hidden 单次前向使用的时间，(0,1]。
WINDOW_SIZE="${WINDOW_SIZE:-15}"            # 输入窗口碱基数，奇数；输出 [N,15,768]。
BATCH_SIZE="${BATCH_SIZE:-8}"              # BERT/Flow 主干批次；CUDA OOM 自动拆小。
PREPROCESS="${PREPROCESS:-none}"            # 已预处理电流选 none；auto 使用 Codec 预处理器。
DTYPE="${DTYPE:-float16}"                  # 落盘精度；模型和聚合保持 FP32。
SAVE_EVERY="${SAVE_EVERY:-512}"            # 特征写盘缓冲窗口数。
LOG_EVERY="${LOG_EVERY:-200}"              # 进度日志间隔；0 禁用。
SPLITS="${SPLITS:-train valid test}"        # 需要提取的 split，空格分隔。
LIMIT="${LIMIT:-}"                        # 每个 split 最多读 N 条；正式运行留空。
CODEC_BATCH_SIZE="${CODEC_BATCH_SIZE:-1}"   # 1 逐条编码；>1 等长批量可能改变离散 token。
CODEC_BUFFER_SIZE="${CODEC_BUFFER_SIZE:-1024}" # 等长 Codec 分组缓冲条数。
POOLING_DEVICE="${POOLING_DEVICE:-model}"   # model 在模型设备聚合；cpu 沿用 CPU 聚合。
FORCE="${FORCE:-0}"                        # 1 显式覆盖所选 split；默认 0。
DRY_RUN="${DRY_RUN:-0}"                    # 1 只打印命令，不加载模型和生成特征。
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

cmd=("${PYTHON_BIN}" -u "${SCRIPT_DIR}/step4_extract_window_hidden_states_Flow-matching.py"
     --data-dir "${DATA_DIR}" --encoder-dir "${ENCODER_DIR}" --decoder-dir "${DECODER_DIR}"
     --out-dir "${OUT_DIR}" --device "${DEVICE}" --cpu-threads "${CPU_THREADS}"
     --encoder-layer "${ENCODER_LAYER}" --hidden-source "${HIDDEN_SOURCE}"
     --ode-steps "${ODE_STEPS}" --ode-start-t "${ODE_START_T}"
     --ode-self-cond-cfg-scale "${ODE_SELF_COND_CFG_SCALE}" --last-t "${LAST_T}"
     --window-size "${WINDOW_SIZE}" --batch-size "${BATCH_SIZE}"
     --preprocess "${PREPROCESS}" --dtype "${DTYPE}"
     --save-every "${SAVE_EVERY}" --log-every "${LOG_EVERY}"
     --codec-batch-size "${CODEC_BATCH_SIZE}" --codec-buffer-size "${CODEC_BUFFER_SIZE}"
     --pooling-device "${POOLING_DEVICE}")
read -r -a split_names <<< "${SPLITS}"
cmd+=(--splits "${split_names[@]}")
[[ -z "${LIMIT}" ]] || cmd+=(--limit "${LIMIT}")
[[ "${FORCE}" != 1 ]] || cmd+=(--force)
cmd+=("$@")
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
[[ "${DRY_RUN}" != 1 ]] || exit 0
exec "${cmd[@]}"
