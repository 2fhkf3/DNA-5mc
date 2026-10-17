#!/usr/bin/env bash
# Step3：每个位点和repeat生成 SHA256 排名，保存前 max(depths) 条 read 的统一观测索引。
# CONFIG/--config、PROTOCOL_DIR/--protocol-dir：与 Step1/2 一致；输出 sampling/，不覆盖。
# depths：YAML 默认[5,10,20]；取同一排名前缀，保证5⊂10⊂20，无放回且模型间共用。
# repeats：默认50；sampling_seed：抽样主种子，与模型训练seed、bootstrap seed分开。
# site_chunk_size：每块处理的位点数，控制内存；改变块大小不应改变所选 read。
# DRY_RUN=1/--dry-run：仅显示抽样计划；正式运行不复制信号或隐藏状态。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_test3_step step3_sample_site_reads.py "$@"
