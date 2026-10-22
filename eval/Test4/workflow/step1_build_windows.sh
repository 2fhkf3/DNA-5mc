#!/usr/bin/env bash
# Step1：单次顺序扫描JSONL，保留所有参考beta的合法15nt窗口；完整read边界提交续跑位置。
# MAX_RECORDS/--max-records仅限独立allow_partial_results=true协议，所有步骤必须一致。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_test4_step step1_build_windows.py "$@"
