#!/usr/bin/env bash
# Step2 统一提取入口：版本、架构、路径、隐藏层参数主要在同目录 config.yaml 中修改。
# 默认：bash step2_extract_features.sh
# 切换：VERSION=V543 bash step2_extract_features.sh
# V013 Flow Matching：VERSION=V013 bash step2_extract_features.sh
# 仅解析最终配置：DRY_RUN=1 bash step2_extract_features.sh
# 指定配置：EXTRACTION_CONFIG=/path/to/config.yaml bash step2_extract_features.sh
set -euo pipefail

# 只转发用户显式提供的环境变量，避免 common.sh 的默认值覆盖 YAML 中的设置。
declare -A step2_overrides=()
for name in VERSION MODEL_ARCH MODEL_ROOT ENCODER_DIR DECODER_DIR DATA_DIR FEATURES_DIR \
            CODEC_BATCH_SIZE CODEC_BUFFER_SIZE POOLING_DEVICE \
            DEVICE CPU_THREADS BATCH_SIZE FEATURE_DTYPE WINDOW_SIZE PREPROCESS SAVE_EVERY LOG_EVERY LIMIT \
            ENCODER_LAYER HIDDEN_LAYER TOKEN_OFFSET HIDDEN_SOURCE ODE_STEPS ODE_START_T \
            ODE_SELF_COND_CFG_SCALE LAST_T DECODER_LAYERS SELECTED_LAYER_NAME TOKEN_ID_MODE \
            TOKENIZER_FILE SPECIAL_TOKENS CODEC_MODE MAX_TOKENS OVERLONG_POLICY; do
    if [[ -n "${!name:-}" ]]; then step2_overrides["${name}"]="${!name}"; fi
done
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# EXTRACTION_CONFIG：Step2 YAML 文件；相对路径是调用目录的相对路径。
# YAML 文件内部的相对路径统一以 DNA-5mC 项目根目录为基准。
cmd=("${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/step2_extract_features.py"
     --config "${EXTRACTION_CONFIG:-${TEST1_WORKFLOW_DIR}/config.yaml}")

append_override() {
    local name="$1" flag="$2"
    if [[ -n "${step2_overrides[$name]+set}" ]]; then cmd+=("${flag}" "${step2_overrides[$name]}"); fi
}

# VERSION：选择 models 中的条目；MODEL_ARCH：diffusion / flow_matching / bert / gpt / auto。
append_override VERSION --version
append_override MODEL_ARCH --architecture
# MODEL_ROOT：模型根目录；ENCODER_DIR / DECODER_DIR 可分别指定不同 checkpoint。
append_override MODEL_ROOT --model-root
append_override ENCODER_DIR --encoder-dir
append_override DECODER_DIR --decoder-dir
# DATA_DIR：Step1 窗口目录；FEATURES_DIR：本次提取的新输出目录，不覆盖已有结果。
append_override DATA_DIR --data-dir
append_override FEATURES_DIR --out-dir
# DEVICE / CPU_THREADS：设备和 CPU 线程；BATCH_SIZE：主干推理批次。
# FEATURE_DTYPE：缓存精度；WINDOW_SIZE：语料窗口长度；PREPROCESS：是否再次预处理。
append_override DEVICE --device
append_override CPU_THREADS --cpu-threads
append_override BATCH_SIZE --batch-size
# CODEC_BATCH_SIZE：等长信号编码批次；1 恢复逐条编码。
# CODEC_BUFFER_SIZE：分组缓冲的窗口数；POOLING_DEVICE：cpu / model。
append_override CODEC_BATCH_SIZE --codec-batch-size
append_override CODEC_BUFFER_SIZE --codec-buffer-size
append_override POOLING_DEVICE --pooling-device
append_override FEATURE_DTYPE --dtype
append_override WINDOW_SIZE --window-size
append_override PREPROCESS --preprocess
# SAVE_EVERY / LOG_EVERY：写盘、日志间隔；LIMIT：测试时最多处理的窗口数。
append_override SAVE_EVERY --save-every
append_override LOG_EVERY --log-every
append_override LIMIT --limit
# ENCODER_LAYER：codec 参数；HIDDEN_LAYER：BERT hidden_states 索引。
append_override ENCODER_LAYER --encoder-layer
append_override HIDDEN_LAYER --hidden-layer
append_override TOKEN_OFFSET --token-offset
# Diffusion / Flow Matching：表征来源、ODE 步数、时间和自条件系数，含义详见 config.yaml。
append_override HIDDEN_SOURCE --hidden-source
append_override ODE_STEPS --ode-steps
append_override ODE_START_T --ode-start-t
append_override ODE_SELF_COND_CFG_SCALE --ode-self-cond-cfg-scale
append_override LAST_T --last-t
# GPT：目标层、词表映射、特殊 token、codec 路径及超长处理。
append_override DECODER_LAYERS --decoder-layers
append_override SELECTED_LAYER_NAME --selected-layer-name
append_override TOKEN_ID_MODE --token-id-mode
append_override TOKENIZER_FILE --tokenizer-file
append_override SPECIAL_TOKENS --special-tokens
append_override CODEC_MODE --codec-mode
append_override MAX_TOKENS --max-tokens
append_override OVERLONG_POLICY --overlong-policy

# 额外命令行参数最后传入，优先级高于 YAML 与环境变量。
cmd+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then cmd+=(--print-plan); fi
"${cmd[@]}"
