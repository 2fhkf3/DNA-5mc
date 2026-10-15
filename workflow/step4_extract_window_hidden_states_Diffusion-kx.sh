#!/usr/bin/env bash
set -euo pipefail

# Extract V011 CLS step65000 embeddings with the integrated Diffusion-kx.py.
# Defaults: ODE 1 step, start_t=0.97, CFG=0.5, batch size 4, 15-nt windows.
# Outputs: streaming [N, WINDOW_SIZE, 768] features, labels, metadata and manifests.
# Examples (from this directory):
#   bash step4_extract_window_hidden_states_Diffusion-kx.sh
#   LIMIT=4 DEVICE=cpu OUT_DIR=/tmp/diffusion_smoke bash step4_extract_window_hidden_states_Diffusion-kx.sh
#   HIDDEN_SOURCE=context_hidden OUT_DIR=/tmp/context_smoke bash step4_extract_window_hidden_states_Diffusion-kx.sh
#   SPLITS=test DATA_DIR=/path/to/test_dataset OUT_DIR=/path/to/test_features bash step4_extract_window_hidden_states_Diffusion-kx.sh

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
MODEL_ROOT="${MODEL_ROOT:-${BACKBONE_ROOT}/V011-Diffusion}"
DATA_DIR="${DATA_DIR:-${OUTPUT_ROOT}/step3_output/new}"
OUT_DIR="${OUT_DIR:-${OUTPUT_ROOT}/step4_output/V011-ode097}"

ENCODER_DIR="${ENCODER_DIR:-${MODEL_ROOT}/encoder}"
DECODER_DIR="${DECODER_DIR:-${DLM_DIR:-${MODEL_ROOT}/hf_dlm_elf_mixed_with_cls_token_step65000-unsharded}}"
DEVICE="${DEVICE:-auto}"
CPU_THREADS="${CPU_THREADS:-4}"
ENCODER_LAYER="${ENCODER_LAYER:-0}"
HIDDEN_SOURCE="${HIDDEN_SOURCE:-${ENCODING_TYPE:-ode_hidden}}"
ODE_STEPS="${ODE_STEPS:-1}"
ODE_START_T="${ODE_START_T:-0.97}"
ODE_SELF_COND_CFG_SCALE="${ODE_SELF_COND_CFG_SCALE:-0.5}"
LAST_T="${LAST_T:-1.0}"
WINDOW_SIZE="${WINDOW_SIZE:-15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREPROCESS="${PREPROCESS:-none}"
DTYPE="${DTYPE:-float32}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-200}"
LIMIT="${LIMIT:-}"
SPLITS="${SPLITS:-train valid test}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${PROJECT_ROOT}/output/.hf_modules}"
export HF_HUB_OFFLINE=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

cmd=(
    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/step4_extract_window_hidden_states_Diffusion-kx.py"
    --data-dir "${DATA_DIR}" --encoder-dir "${ENCODER_DIR}" --decoder-dir "${DECODER_DIR}"
    --out-dir "${OUT_DIR}" --device "${DEVICE}" --cpu-threads "${CPU_THREADS}"
    --encoder-layer "${ENCODER_LAYER}" --hidden-source "${HIDDEN_SOURCE}"
    --ode-steps "${ODE_STEPS}" --ode-start-t "${ODE_START_T}"
    --ode-self-cond-cfg-scale "${ODE_SELF_COND_CFG_SCALE}" --last-t "${LAST_T}"
    --window-size "${WINDOW_SIZE}" --batch-size "${BATCH_SIZE}"
    --preprocess "${PREPROCESS}" --dtype "${DTYPE}"
    --save-every "${SAVE_EVERY}" --log-every "${LOG_EVERY}"
)
read -r -a split_names <<< "${SPLITS}"
cmd+=(--splits "${split_names[@]}")
[[ -z "${LIMIT}" ]] || cmd+=(--limit "${LIMIT}")
[[ "${FORCE}" == "1" ]] && cmd+=(--force)
cmd+=("$@")
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
[[ "${DRY_RUN}" != "1" ]] || exit 0
exec "${cmd[@]}"
