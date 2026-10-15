#!/usr/bin/env bash

# ======================================================================
# Nanopore Signal Preprocessing Pipeline
# ======================================================================
#
# 功能：
#
#   input JSONL
#       ↓
#   AutoFeatureExtractor
#       ↓
#   替换 signal
#       ↓
#   output JSONL
#
#
# 支持两种运行模式：
#
#   MODE=single
#
#       单进程、单文件流式处理。
#       最稳定，不产生中间 shard。
#
#
#   MODE=shard
#
#       将输入按固定 read 数分片，
#       多进程并行 preprocessing，
#       最终合并。
#
#
# 推荐策略：
#
#   第一次：
#
#       LIMIT=10000 MODE=single bash step1_preprocess_signal_jsonl.sh
#
#
#   正式：
#
#       LIMIT="" MODE=single bash step1_preprocess_signal_jsonl.sh
#
#
#   如果单进程 CPU 使用率低、磁盘吞吐有余量：
#
#       MODE=shard JOBS=4 bash step1_preprocess_signal_jsonl.sh
#
#
# ======================================================================

set -euo pipefail


# ======================================================================
# 1. Environment
# ======================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/step1_preprocess_signal_jsonl.py}"


# ======================================================================
# 2. Model
# ======================================================================

# 保留原来的 V003 预处理器；可用 MODEL_DIR 指定其他 encoder 的预处理器。
MODEL_DIR="${MODEL_DIR:-${BACKBONE_ROOT}/V003-Preprocess/encoder}"


# ======================================================================
# 3. Input / Output
#
# 可以直接修改这里，
# 也可以通过环境变量覆盖。
# ======================================================================

INPUT_FILE="${INPUT_FILE:-${OUTPUT_ROOT}/step1_input/s2_all_reads.genome_signal.jsonl}"

OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_ROOT}/step1_output/s2_all_reads.genome_signal.preprocessed-2.jsonl}"


# ======================================================================
# 4. Run mode
#
# single:
#     一个 Python process 直接处理整个文件
#
# shard:
#     split + 多 Python process + merge
# ======================================================================

MODE="${MODE:-single}"


# ======================================================================
# 5. QC
# ======================================================================

STRICT_LENGTH="${STRICT_LENGTH:-1}"

FAIL_FAST="${FAIL_FAST:-1}"

LOG_EVERY="${LOG_EVERY:-10000}"


# ======================================================================
# 6. Benchmark
#
# 空字符串：
#     全量运行
#
# 例如：
#
#     LIMIT=10000
#
# 只测试前 1 万条。
# ======================================================================

LIMIT="${LIMIT:-}"


# ======================================================================
# 7. Sharding / Parallel mode
# ======================================================================
#
# 300 万 read：
#
#   100000 read / shard
#
# 大约得到：
#
#   30 shards
#
#
# JOBS:
#   建议先从 2 或 4 开始。
#
# 不建议一开始直接几十个进程，
# 因为 JSON + signal 读写很容易变成 I/O bottleneck。
# ======================================================================

SHARD_LINES="${SHARD_LINES:-100000}"

JOBS="${JOBS:-4}"


# ======================================================================
# 8. Working directories
# ======================================================================

WORK_ROOT="${WORK_ROOT:-${OUTPUT_ROOT}/work_signal_preprocess}"

SHARD_INPUT_DIR="${WORK_ROOT}/input_shards"

SHARD_OUTPUT_DIR="${WORK_ROOT}/output_shards"

SHARD_ERROR_DIR="${WORK_ROOT}/error_shards"

LOG_DIR="${WORK_ROOT}/logs"


if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[PATH PREVIEW]'; printf ' %q' "${PYTHON_BIN}" "${SCRIPT_PATH}" \
        --input "${INPUT_FILE}" --output "${OUTPUT_FILE}" --model-dir "${MODEL_DIR}"
    printf '\n'
    exit 0
fi

mkdir -p "${WORK_ROOT}"
mkdir -p "${LOG_DIR}"


TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

MAIN_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"


# ======================================================================
# 9. Validation
# ======================================================================

if [[ ! -f "${INPUT_FILE}" ]]; then

    echo "[ERROR] Input file does not exist:"
    echo "        ${INPUT_FILE}"

    exit 1

fi


if [[ ! -f "${SCRIPT_PATH}" ]]; then

    echo "[ERROR] Python script does not exist:"
    echo "        ${SCRIPT_PATH}"

    exit 1

fi


if [[ ! -d "${MODEL_DIR}" ]]; then

    echo "[ERROR] Model directory does not exist:"
    echo "        ${MODEL_DIR}"

    exit 1

fi


mkdir -p "$(dirname "${OUTPUT_FILE}")"


# ======================================================================
# 10. Install recommendation
#
# Python 脚本可以没有 orjson。
#
# 但如果环境允许：
#
#     pip install orjson
#
# 对几百万条大型 JSON 的 encode/decode
# 通常会有明显帮助。
# ======================================================================

echo "======================================================================"
echo " Nanopore Signal Preprocessing"
echo "======================================================================"

echo "[INFO] Python        : ${PYTHON_BIN}"
echo "[INFO] Python script : ${SCRIPT_PATH}"

echo "[INFO] Model         : ${MODEL_DIR}"

echo "[INFO] Input         : ${INPUT_FILE}"
echo "[INFO] Output        : ${OUTPUT_FILE}"

echo "[INFO] Mode          : ${MODE}"

echo "[INFO] Strict length : ${STRICT_LENGTH}"
echo "[INFO] Fail fast     : ${FAIL_FAST}"
echo "[INFO] Limit         : ${LIMIT:-FULL}"

if [[ "${MODE}" == "shard" ]]; then

    echo "[INFO] Shard lines   : ${SHARD_LINES}"
    echo "[INFO] Jobs          : ${JOBS}"

fi

echo "[INFO] Main log      : ${MAIN_LOG}"

echo "======================================================================"


# ======================================================================
# Function:
# run one Python preprocessing process
# ======================================================================

run_python()
{
    local input="$1"
    local output="$2"
    local error_file="$3"
    local log_file="$4"

    local cmd=(
        "${PYTHON_BIN}"
        "${SCRIPT_PATH}"

        --input "${input}"
        --output "${output}"

        --model-dir "${MODEL_DIR}"

        --signal-field signal

        --error-file "${error_file}"

        --log-every "${LOG_EVERY}"
    )


    if [[ "${STRICT_LENGTH}" == "1" ]]; then
        cmd+=(--strict-length)
    fi


    if [[ "${FAIL_FAST}" == "1" ]]; then
        cmd+=(--fail-fast)
    fi


    if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
    fi


    echo
    echo "[CMD]"
    printf ' %q' "${cmd[@]}"
    echo


    stdbuf -oL -eL \
        "${cmd[@]}" \
        > "${log_file}" \
        2>&1
}


export -f run_python

export PYTHON_BIN
export SCRIPT_PATH
export MODEL_DIR

export STRICT_LENGTH
export FAIL_FAST
export LOG_EVERY
export LIMIT


# ======================================================================
# SINGLE MODE
# ======================================================================

if [[ "${MODE}" == "single" ]]; then

    ERROR_FILE="${OUTPUT_FILE}.errors.jsonl"

    echo
    echo "[INFO] Running SINGLE mode..."
    echo


    START_TIME="$(date +%s)"


    CMD=(
        "${PYTHON_BIN}"
        "${SCRIPT_PATH}"

        --input "${INPUT_FILE}"
        --output "${OUTPUT_FILE}"

        --model-dir "${MODEL_DIR}"

        --signal-field signal

        --error-file "${ERROR_FILE}"

        --log-every "${LOG_EVERY}"
    )


    if [[ "${STRICT_LENGTH}" == "1" ]]; then
        CMD+=(--strict-length)
    fi


    if [[ "${FAIL_FAST}" == "1" ]]; then
        CMD+=(--fail-fast)
    fi


    if [[ -n "${LIMIT}" ]]; then
        CMD+=(--limit "${LIMIT}")
    fi


    echo "[INFO] Command:"
    printf ' %q' "${CMD[@]}"
    echo
    echo


    set +e

    stdbuf -oL -eL \
        "${CMD[@]}" \
        2>&1 | tee "${MAIN_LOG}"

    EXIT_CODE=${PIPESTATUS[0]}

    set -e


    END_TIME="$(date +%s)"

    ELAPSED=$((END_TIME - START_TIME))


    echo
    echo "======================================================================"

    if [[ "${EXIT_CODE}" -eq 0 ]]; then

        echo "[DONE] SINGLE mode completed successfully."

    else

        echo "[ERROR] SINGLE mode failed."
        echo "[ERROR] Exit code=${EXIT_CODE}"

    fi


    printf \
        "[INFO] Runtime: %02d:%02d:%02d\n" \
        $((ELAPSED / 3600)) \
        $(((ELAPSED % 3600) / 60)) \
        $((ELAPSED % 60))


    echo "[INFO] Output: ${OUTPUT_FILE}"
    echo "[INFO] Log   : ${MAIN_LOG}"

    echo "======================================================================"


    exit "${EXIT_CODE}"

fi


# ======================================================================
# SHARD MODE
# ======================================================================

if [[ "${MODE}" != "shard" ]]; then

    echo "[ERROR] MODE must be:"
    echo "        single"
    echo "        shard"

    exit 1

fi


# ======================================================================
# 11. Prepare shard workspace
# ======================================================================

rm -rf "${SHARD_INPUT_DIR}"
rm -rf "${SHARD_OUTPUT_DIR}"
rm -rf "${SHARD_ERROR_DIR}"


mkdir -p "${SHARD_INPUT_DIR}"
mkdir -p "${SHARD_OUTPUT_DIR}"
mkdir -p "${SHARD_ERROR_DIR}"


echo
echo "[INFO] Splitting input into shards..."
echo


# ======================================================================
# 12. Split
#
# 如果输入是 gzip：
#
#     gzip -dc input.gz | split ...
#
# 如果输入是普通 JSONL：
#
#     split ...
#
#
# 每一行 JSONL 对应一条 read，
# 因此 split -l 不会破坏 JSON record。
# ======================================================================

if [[ "${INPUT_FILE}" == *.gz ]]; then

    gzip -dc "${INPUT_FILE}" |
        split \
            -l "${SHARD_LINES}" \
            -d \
            -a 5 \
            - \
            "${SHARD_INPUT_DIR}/part_"

else

    split \
        -l "${SHARD_LINES}" \
        -d \
        -a 5 \
        "${INPUT_FILE}" \
        "${SHARD_INPUT_DIR}/part_"

fi


SHARD_COUNT="$(
    find "${SHARD_INPUT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'part_*' |
        wc -l
)"


echo "[INFO] Number of shards: ${SHARD_COUNT}"


# ======================================================================
# 13. Build shard task list
# ======================================================================

TASK_FILE="${WORK_ROOT}/tasks.txt"

: > "${TASK_FILE}"


for input_shard in "${SHARD_INPUT_DIR}"/part_*; do

    shard_name="$(basename "${input_shard}")"

    echo "${shard_name}" \
        >> "${TASK_FILE}"

done


# ======================================================================
# 14. Parallel processing
#
# 使用 xargs -P，因此不依赖 GNU parallel。
# ======================================================================

echo
echo "[INFO] Starting parallel preprocessing..."
echo "[INFO] Concurrent processes: ${JOBS}"
echo


export SHARD_INPUT_DIR
export SHARD_OUTPUT_DIR
export SHARD_ERROR_DIR
export LOG_DIR


cat "${TASK_FILE}" |
    xargs \
        -n 1 \
        -P "${JOBS}" \
        bash -c '

            set -euo pipefail

            shard="$1"

            input="${SHARD_INPUT_DIR}/${shard}"

            output="${SHARD_OUTPUT_DIR}/${shard}.processed.jsonl"

            error_file="${SHARD_ERROR_DIR}/${shard}.errors.jsonl"

            log_file="${LOG_DIR}/${shard}.log"


            echo "[START] ${shard}"


            run_python \
                "${input}" \
                "${output}" \
                "${error_file}" \
                "${log_file}"


            echo "[DONE] ${shard}"

        ' _


# ======================================================================
# 15. Validate shard count
# ======================================================================

OUTPUT_SHARD_COUNT="$(
    find "${SHARD_OUTPUT_DIR}" \
        -maxdepth 1 \
        -type f \
        -name '*.processed.jsonl' |
        wc -l
)"


if [[ "${OUTPUT_SHARD_COUNT}" -ne "${SHARD_COUNT}" ]]; then

    echo "[ERROR] Output shard count mismatch."

    echo "[ERROR] Input : ${SHARD_COUNT}"
    echo "[ERROR] Output: ${OUTPUT_SHARD_COUNT}"

    exit 1

fi


# ======================================================================
# 16. Merge
#
# 注意按 part_00000, part_00001 ... 顺序合并。
# ======================================================================

echo
echo "[INFO] Merging processed shards..."
echo


TMP_MERGED="${OUTPUT_FILE}.tmp"


rm -f "${TMP_MERGED}"


# ----------------------------------------------------------------------
# 如果最终要求 gzip：
#
# 先按顺序 cat，再统一压缩。
#
# 如果安装 pigz，优先 pigz；
# 否则使用 gzip。
# ----------------------------------------------------------------------

if [[ "${OUTPUT_FILE}" == *.gz ]]; then

    TMP_PLAIN="${WORK_ROOT}/merged.processed.jsonl"

    rm -f "${TMP_PLAIN}"


    for shard in \
        $(find "${SHARD_OUTPUT_DIR}" \
            -maxdepth 1 \
            -type f \
            -name '*.processed.jsonl' \
            | sort); do

        cat "${shard}" >> "${TMP_PLAIN}"

    done


    if command -v pigz >/dev/null 2>&1; then

        echo "[INFO] Compressing with pigz..."

        pigz \
            -p "${JOBS}" \
            -c "${TMP_PLAIN}" \
            > "${TMP_MERGED}"

    else

        echo "[INFO] pigz unavailable; using gzip..."

        gzip \
            -c "${TMP_PLAIN}" \
            > "${TMP_MERGED}"

    fi


    rm -f "${TMP_PLAIN}"


else

    for shard in \
        $(find "${SHARD_OUTPUT_DIR}" \
            -maxdepth 1 \
            -type f \
            -name '*.processed.jsonl' \
            | sort); do

        cat "${shard}" >> "${TMP_MERGED}"

    done

fi


mv "${TMP_MERGED}" "${OUTPUT_FILE}"


# ======================================================================
# 17. Finish
# ======================================================================

echo
echo "======================================================================"
echo "[DONE] SHARD mode completed."
echo
echo "[DONE] Shards processed : ${SHARD_COUNT}"
echo "[DONE] Output           : ${OUTPUT_FILE}"
echo
echo "[INFO] Individual logs:"
echo "       ${LOG_DIR}"
echo
echo "[INFO] Failed-read logs:"
echo "       ${SHARD_ERROR_DIR}"
echo
echo "======================================================================"
