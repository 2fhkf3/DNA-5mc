#!/usr/bin/env bash
# 将 Step-4 特征分块裁剪为中心 5/7/9... 个碱基，并生成 train 可直接使用的配置。
# 例：WINDOW_SIZE=7 INPUT_DIR=/path/to/V543 OUT_DIR=/path/to/V543_center7 bash step5_center_crop_features.sh
# 预览：bash step5_center_crop_features.sh --dry-run
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"

# 输入为 step4_extract_window_hidden_states_BERT-js.sh 的完整输出目录。
INPUT_DIR="${INPUT_DIR:-/mnt/nas_data/shenhaojie/step4-output/V554}"
# 裁剪长度，必须是 >= 5 且不超过输入窗口的奇数。
WINDOW_SIZE="${WINDOW_SIZE:-5}"
# 默认输出到项目内 train/data；也可通过 OUT_DIR 指定其他目录。
# 相同目录存在时拒绝覆盖。
INPUT_NAME="$(basename "${INPUT_DIR}")"
OUT_DIR="${OUT_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/train/data/V554-V2}"
# 分块大小和进度频率；不影响样本顺序及输出数值。
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
LOG_EVERY_CHUNKS="${LOG_EVERY_CHUNKS:-16}"
# 训练参数模板；脚本会生成独立 train_config.yaml，不修改模板或已有运行配置。
TRAIN_CONFIG="${TRAIN_CONFIG:-/home/shj/ZJ-lab/Modification/DNA-5mC/train/run/V554-V2/config.yaml}"
EXPERIMENT="${EXPERIMENT:-}"
export PYTHONUNBUFFERED=1
cmd=("${PYTHON_BIN}" -u "${SCRIPT_DIR}/step5_center_crop_features.py"
     --input-dir "${INPUT_DIR}" --out-dir "${OUT_DIR}" --window-size "${WINDOW_SIZE}"
     --chunk-size "${CHUNK_SIZE}" --log-every-chunks "${LOG_EVERY_CHUNKS}"
     --train-config "${TRAIN_CONFIG}")
if [[ -n "${EXPERIMENT}" ]]; then cmd+=(--experiment "${EXPERIMENT}"); fi
cmd+=("$@")
printf '[INFO] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" != "1" ]] || exit 0
exec "${cmd[@]}"
