#!/usr/bin/env bash
# Extract 15-nt base-level window embeddings with the selected BERT backbone.
#
# Input:
#   DATA_DIR/{train,valid,test}.jsonl[.gz]
# Output:
#   OUT_DIR/{train,valid,test}.pt                 (small manifests)
#   OUT_DIR/{train,valid,test}.features.f16.bin   (streamed features)
#   OUT_DIR/{train,valid,test}.labels.f32.bin
#   OUT_DIR/metadata_{train,valid,test}.tsv
#   OUT_DIR/feature_summary.json
#
# Smoke test:
#   PYTHON_BIN=/path/to/python LIMIT=20 DEVICE=cuda:0 \
#     bash step4_extract_window_hidden_states_BERT-js.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/paths.sh"
MODEL_ROOT="${MODEL_ROOT:-${BACKBONE_ROOT}/V554-BERT}"
MODEL_NAME="$(basename "${MODEL_ROOT}")"
DATA_DIR="${DATA_DIR:-${OUTPUT_ROOT}/step3_output/new}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/step4_output/${MODEL_NAME%%-*}}"

ENCODER_DIR="${ENCODER_DIR:-${MODEL_ROOT}/encoder}"
BERT_DIR="${BERT_DIR:-${MODEL_ROOT}/base}"
DEVICE="${DEVICE:-cuda:0}"
ENCODER_LAYER="${ENCODER_LAYER:-0}"
HIDDEN_LAYER="${HIDDEN_LAYER:--1}"
TOKEN_OFFSET="${TOKEN_OFFSET:-128}"
WINDOW_SIZE="${WINDOW_SIZE:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PREPROCESS="${PREPROCESS:-none}"
DTYPE="${DTYPE:-float16}"
LIMIT="${LIMIT:-}"
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-1024}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cmd=(
  "${PYTHON_BIN}" -u "${SCRIPT_DIR}/step4_extract_window_hidden_states_BERT-js.py"
  --data-dir "${DATA_DIR}"
  --encoder-dir "${ENCODER_DIR}"
  --bert-dir "${BERT_DIR}"
  --out-dir "${OUT_DIR}"
  --device "${DEVICE}"
  --encoder-layer "${ENCODER_LAYER}"
  --hidden-layer "${HIDDEN_LAYER}"
  --token-offset "${TOKEN_OFFSET}"
  --window-size "${WINDOW_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --preprocess "${PREPROCESS}"
  --dtype "${DTYPE}"
  --save-every "${SAVE_EVERY}"
  --log-every "${LOG_EVERY}"
)

if [[ -n "${LIMIT}" ]]; then
  cmd+=(--limit "${LIMIT}")
fi

cmd+=("$@")
printf '[INFO] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" != "1" ]] || exit 0
exec "${cmd[@]}"
