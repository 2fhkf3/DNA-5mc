#!/usr/bin/env bash
# 从已提取的天然样本 test 缓存中筛选 label=0；只使用 CPU，不重新编码、不训练。
# 这是 WGA Step1 的并行评测分支：本步骤输出后直接进入 Step4，无需 Step2/Step3。
# 正式：bash eval/Test1/workflow/step1.1_build_native_negatives.sh
# 预览（读取清单和标签，估计输出大小；不写数据）：
#   DRY_RUN=1 bash eval/Test1/workflow/step1.1_build_native_negatives.sh
# 小测试（仅前 128 个原始窗口，输出必须另设目录）：
#   MAX_RECORDS=128 OUT_DIR=/tmp/native_bed0_smoke bash eval/Test1/workflow/step1.1_build_native_negatives.sh
set -euo pipefail

# VERSION：用于记录特征所属模型；不是把其他主干的特征转换为这个版本。
VERSION="${VERSION:-V011}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# INPUT_DIR：只读取该目录的 test.pt / test bin / metadata_test.tsv / feature_summary.json。
# 不读取 train/valid，不修改原始文件，不依据模型预测筛选阴性。
INPUT_DIR="${INPUT_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/train/data/V011-V2}"
# OUT_DIR：直接保存全阴性特征缓存；允许空目录，非空时拒绝覆盖。
# 输入已经裁剪到 5 个位置，本步骤保持原 W/H/精度，不生成原始 signal JSONL。
OUT_DIR="${OUT_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/data/step3_output/chr16-native/V011-V2}"
# LIBRARY_ID：天然实验文库 ID；同文库切换主干时保持一致，不能冒用 WGA 的 ID。
LIBRARY_ID="${LIBRARY_ID:-NATIVE_250F500490011}"
# CHUNK_SIZE：每块扫描多少个原始 test 窗口；4096 时特征缓冲为几十 MB。
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
# LOG_EVERY_CHUNKS：进度日志的块间隔；不改变样本或数值。
LOG_EVERY_CHUNKS="${LOG_EVERY_CHUNKS:-32}"
# MAX_RECORDS：仅扫描前 N 个原始窗口用于联调，输出标记 partial_test；正式运行留空。
MAX_RECORDS="${MAX_RECORDS:-}"

cmd=("${PYTHON_BIN}" -u "${TEST1_WORKFLOW_DIR}/step1.1_build_native_negatives.py"
     --input-dir "${INPUT_DIR}" --out-dir "${OUT_DIR}" --version "${VERSION}"
     --library-id "${LIBRARY_ID}" --chunk-size "${CHUNK_SIZE}"
     --log-every-chunks "${LOG_EVERY_CHUNKS}")
[[ -z "${MAX_RECORDS}" ]] || cmd+=(--max-records "${MAX_RECORDS}")
# CLI 参数最后传入，可覆盖上面的环境变量/默认配置。
cmd+=("$@")
# 本入口的 DRY_RUN 会执行 Python 只读校验，而不只是打印命令。
[[ "${DRY_RUN:-0}" != 1 ]] || cmd+=(--dry-run)
exec "${cmd[@]}"
