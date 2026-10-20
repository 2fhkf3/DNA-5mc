#!/usr/bin/env bash
set -euo pipefail

# Same data/output contract and environment controls as Diffusion-kx.sh.
# GPT defaults to [BOS, mapped codec tokens, EOS]; specials are excluded from pooling.
# SPECIAL_TOKENS=none explicitly restores the previous signal-only layout.
# Examples:
#   bash step4_extract_window_hidden_states_GPT.sh
#   LIMIT=4 DEVICE=cpu OUT_DIR=/tmp/gpt_smoke bash step4_extract_window_hidden_states_GPT.sh
#   SPLITS=test DATA_DIR=/path/to/test_data OUT_DIR=/path/to/test_features bash step4_extract_window_hidden_states_GPT.sh
#   MODEL_ROOT=/path/to/gpt_checkpoint bash step4_extract_window_hidden_states_GPT.sh

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
MODEL_ROOT="${MODEL_ROOT:-${BACKBONE_ROOT}/V610-GPT}"
DATA_DIR="${DATA_DIR:-${OUTPUT_ROOT}/step3_output/new}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/step4_output/V610}"
ENCODER_DIR="${ENCODER_DIR:-${MODEL_ROOT}/encoder}"
DECODER_DIR="${DECODER_DIR:-${MODEL_ROOT}/base}"
DEVICE="${DEVICE:-auto}"
CPU_THREADS="${CPU_THREADS:-4}"
ENCODER_LAYER="${ENCODER_LAYER:-1}"
DECODER_LAYERS="${DECODER_LAYERS:-last}"
SELECTED_LAYER_NAME="${SELECTED_LAYER_NAME:-}"
TOKEN_ID_MODE="${TOKEN_ID_MODE:-auto}"
TOKENIZER_FILE="${TOKENIZER_FILE:-}"
TOKEN_OFFSET="${TOKEN_OFFSET:-128}"
SPECIAL_TOKENS="${SPECIAL_TOKENS:-bos_eos}"
CODEC_MODE="${CODEC_MODE:-auto}"
PREPROCESS="${PREPROCESS:-none}"
WINDOW_SIZE="${WINDOW_SIZE:-15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-float32}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-200}"
LIMIT="${LIMIT:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
OVERLONG_POLICY="${OVERLONG_POLICY:-skip}"
SPLITS="${SPLITS:-train valid test}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${PROJECT_ROOT}/output/.hf_modules}"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

cmd=(
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/step4_extract_window_hidden_states_GPT.py"
    --data-dir "${DATA_DIR}" --encoder-dir "${ENCODER_DIR}" --decoder-dir "${DECODER_DIR}"
    --out-dir "${OUT_DIR}" --device "${DEVICE}" --cpu-threads "${CPU_THREADS}"
    --encoder-layer "${ENCODER_LAYER}" --decoder-layers "${DECODER_LAYERS}"
    --token-id-mode "${TOKEN_ID_MODE}" --token-offset "${TOKEN_OFFSET}"
    --special-tokens "${SPECIAL_TOKENS}" --codec-mode "${CODEC_MODE}"
    --window-size "${WINDOW_SIZE}" --batch-size "${BATCH_SIZE}"
    --preprocess "${PREPROCESS}" --dtype "${DTYPE}"
    --save-every "${SAVE_EVERY}" --log-every "${LOG_EVERY}"
    --overlong-policy "${OVERLONG_POLICY}"
)
read -r -a split_names <<< "${SPLITS}"
cmd+=(--splits "${split_names[@]}")
[[ -z "${SELECTED_LAYER_NAME}" ]] || cmd+=(--selected-layer-name "${SELECTED_LAYER_NAME}")
[[ -z "${TOKENIZER_FILE}" ]] || cmd+=(--tokenizer-file "${TOKENIZER_FILE}")
[[ -z "${LIMIT}" ]] || cmd+=(--limit "${LIMIT}")
[[ -z "${MAX_TOKENS}" ]] || cmd+=(--max-tokens "${MAX_TOKENS}")
[[ "${FORCE}" == "1" ]] && cmd+=(--force)
cmd+=("$@")
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
[[ "${DRY_RUN}" != "1" ]] || exit 0
exec "${cmd[@]}"
