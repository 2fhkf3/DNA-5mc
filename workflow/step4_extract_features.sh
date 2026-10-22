#!/usr/bin/env bash
# 统一 Step4：BERT / Diffusion ELF / Flow Matching / GPT，优先修改 config/step4_config.yaml。
# 正式：bash workflow/step4_extract_features.sh
# 预览：DRY_RUN=1 MODEL=V548 bash workflow/step4_extract_features.sh
# V013 Flow：MODEL=V013 bash workflow/step4_extract_features.sh
# OUTPUT_WINDOW_SIZE=5：提取后直接保存中心5碱基，无需再跑Step5；输入上下文不变。
# 小测试（每个 split 最多 17 条；选择新的输出目录）：
#   MODEL=V610 SPLITS=test LIMIT=17 OUT_DIR=/tmp/step4_gpt_smoke \
#     DATA_DIR=/path/to/windows bash workflow/step4_extract_features.sh
# EXTRACTION_CONFIG：另一份 YAML；PYTHON_BIN：包含 torch/transformers/PyYAML 的解释器。
set -euo pipefail

# Capture only explicit overrides before paths.sh supplies its own defaults.
declare -A overrides=()
for name in MODEL VERSION MODEL_ARCH DATA_DIR OUT_DIR OUTPUT_ROOT MODEL_ROOT ENCODER_DIR \
            DECODER_DIR BERT_DIR DLM_DIR DEVICE CPU_THREADS BATCH_SIZE DTYPE WINDOW_SIZE \
            PREPROCESS SAVE_EVERY LOG_EVERY LIMIT SPLITS CODEC_BATCH_SIZE CODEC_BUFFER_SIZE \
            POOLING_DEVICE ENCODER_LAYER HIDDEN_LAYER TOKEN_OFFSET HIDDEN_SOURCE ODE_STEPS \
            ODE_START_T ODE_SELF_COND_CFG_SCALE LAST_T DECODER_LAYERS SELECTED_LAYER_NAME \
            TOKEN_ID_MODE TOKENIZER_FILE SPECIAL_TOKENS CODEC_MODE MAX_TOKENS OVERLONG_POLICY \
            CODEC_MAX_SAMPLES TOKEN_CACHE_MODE TOKEN_CACHE_DIR INFERENCE_PRECISION \
            OUTPUT_WINDOW_SIZE TRAIN_CONFIG EXPERIMENT SPLIT_DEVICES; do
    if [[ -n "${!name:-}" ]]; then overrides["${name}"]="${!name}"; fi
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

cmd=("${PYTHON_BIN}" -u "${SCRIPT_DIR}/step4_extract_features.py"
     --config "${EXTRACTION_CONFIG:-${SCRIPT_DIR}/config/step4_config.yaml}")
append_override() {
    local name="$1" flag="$2"
    if [[ -n "${overrides[$name]+set}" ]]; then cmd+=("${flag}" "${overrides[$name]}"); fi
}
# MODEL/VERSION：YAML 模型名；同时设置时 MODEL 优先。
# MODEL_ARCH：auto / bert / diffusion / flow_matching / gpt；也可由 YAML 决定。
append_override VERSION --model
append_override MODEL --model
append_override MODEL_ARCH --architecture
# DATA_DIR：输入窗口目录；OUT_DIR：本次新输出目录；OUTPUT_ROOT：默认输出根目录。
# 输出默认为 <output_root>/<模型名>；out_dir 有值时优先，已有目录禁止覆盖。
append_override DATA_DIR --data-dir
append_override OUTPUT_ROOT --output-root
append_override OUT_DIR --out-dir
# MODEL_ROOT：模型根目录；encoder/decoder 可独立指定；DECODER_DIR 优先于两个别名。
append_override MODEL_ROOT --model-root
append_override ENCODER_DIR --encoder-dir
append_override BERT_DIR --decoder-dir
append_override DLM_DIR --decoder-dir
append_override DECODER_DIR --decoder-dir
# DEVICE：auto/cpu/cuda:N；CPU_THREADS：CPU 线程；BATCH_SIZE：主干批次。
# DTYPE：float16/float32 落盘精度；INFERENCE_PRECISION：fp16/fp32主干推理精度。
# WINDOW_SIZE：输入窗口碱基数；OUTPUT_WINDOW_SIZE：输出中心碱基数，无需再跑Step5。
# TRAIN_CONFIG：生成训练配置的模板；EXPERIMENT：训练实验名；不自动启动训练。
for name in DEVICE CPU_THREADS BATCH_SIZE DTYPE WINDOW_SIZE PREPROCESS SAVE_EVERY LOG_EVERY LIMIT \
            CODEC_BATCH_SIZE CODEC_BUFFER_SIZE POOLING_DEVICE ENCODER_LAYER HIDDEN_LAYER TOKEN_OFFSET \
            HIDDEN_SOURCE ODE_STEPS ODE_START_T ODE_SELF_COND_CFG_SCALE LAST_T DECODER_LAYERS \
            SELECTED_LAYER_NAME TOKEN_ID_MODE TOKENIZER_FILE SPECIAL_TOKENS CODEC_MODE MAX_TOKENS OVERLONG_POLICY \
            CODEC_MAX_SAMPLES TOKEN_CACHE_MODE TOKEN_CACHE_DIR INFERENCE_PRECISION \
            OUTPUT_WINDOW_SIZE TRAIN_CONFIG EXPERIMENT; do
    flag="${name,,}"
    append_override "${name}" "--${flag//_/-}"
done
# SAVE_EVERY：落盘缓冲窗口数；LOG_EVERY：日志间隔；LIMIT：每个 split 的测试上限。
# CODEC_BATCH_SIZE：1 保留逐条编码；>1 等长批量编码可能改变离散 token。
# CODEC_BUFFER_SIZE：等长分组缓冲窗口数；POOLING_DEVICE：model/cpu。
# CODEC_MAX_SAMPLES：每个codec批次的采样点预算；TOKEN_CACHE_MODE：readwrite/readonly/off。
# TOKEN_CACHE_DIR：共享raw token缓存根目录；绑定编码器、信号输入和数值协议。
# 隐藏层/ODE/token 映射参数的含义见 YAML 各架构条目。
# SPLITS：空格分隔，如 "train valid test" 或 "test"。
if [[ -n "${overrides[SPLITS]+set}" ]]; then
    read -r -a split_names <<< "${overrides[SPLITS]}"
    cmd+=(--splits "${split_names[@]}")
fi
# SPLIT_DEVICES="train=cuda:0 valid=cuda:1 test=cuda:1"；off关闭按split分配。
# 显式DEVICE会关闭YAML中的分配；同时提供SPLIT_DEVICES时以显式分配为准。
if [[ -n "${overrides[SPLIT_DEVICES]+set}" ]]; then
    read -r -a split_device_names <<< "${overrides[SPLIT_DEVICES]}"
    cmd+=(--split-devices "${split_device_names[@]}")
fi
# 显式 CLI 最后传入：CLI > 环境变量 > 模型条目 > extraction > 默认值。
cmd+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then cmd+=(--print-plan); fi
exec "${cmd[@]}"
