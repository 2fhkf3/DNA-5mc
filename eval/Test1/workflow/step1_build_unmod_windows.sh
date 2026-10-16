#!/usr/bin/env bash
# 输入已经预处理，直接构建 15 nt 全阴性窗口；不再运行原 Step1 归一化。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# INPUT_JSONL：用户提供的 WGA 原始 read JSONL，读取源文件，不修改源数据。
INPUT_JSONL="${INPUT_JSONL:-/mnt/zzbnew/rnamodel/jinyue/data/processed_data/chr19/unmod/s2_all_reads.genome_signal.preprocessed.jsonl}"
# DATA_DIR：输出 test.jsonl.gz、逐样本审计索引和 dataset_manifest.json。
DATA_DIR="${DATA_DIR:-/home/shj/ZJ-lab/Modification/DNA-5mC/eval/Test1/data/step1_out/chr19}"
# CHROM：目标染色体；当前输入为 chr19。
# MOTIF：CG 为 CpG；C 为所有 C，应与训练任务一致。
# MIN_MAPQ：read 最小 MAPQ；默认 30，可按原训练数据改成 60。
# MIN_COVERAGE：去重后的最小位点覆盖度；5× 总体深度默认保留 >=1。
# FLANK：中心左右碱基数，7 对应 15 nt，不是提取后裁剪的 5-mer。
# LIBRARY_ID：阴性实验文库 ID，不同独立文库应使用不同 ID。
cmd=("${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/step1_build_unmod_windows.py"
     --input "${INPUT_JSONL}" --out-dir "${DATA_DIR}"
     --chrom "${CHROM:-chr19}" --motif "${MOTIF:-CG}"
     --min-mapq "${MIN_MAPQ:-60}" --min-coverage "${MIN_COVERAGE:-5}"
     --flank "${FLANK:-7}" --library-id "${LIBRARY_ID:-WGA_UNMOD_chr19}")
# CANDIDATE_SITES：可选独立候选位点 TSV（无表头，chrom + 0-based position）。
[[ -z "${CANDIDATE_SITES:-}" ]] || cmd+=(--candidate-sites "${CANDIDATE_SITES}")
# MAX_RECORDS / MAX_WINDOWS：仅用于小规模联调，正式运行留空。
[[ -z "${MAX_RECORDS:-}" ]] || cmd+=(--max-records "${MAX_RECORDS}")
[[ -z "${MAX_WINDOWS:-}" ]] || cmd+=(--max-windows "${MAX_WINDOWS}")
cmd+=("$@")
run_test1_command "${cmd[@]}"
