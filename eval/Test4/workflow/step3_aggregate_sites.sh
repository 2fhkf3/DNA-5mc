#!/usr/bin/env bash
# Step3：跨主干共同read交集→共同coverage≥5→硬调用频率及概率均值；不用WGBS比例筛选。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_test4_step step3_aggregate_sites.py "$@"
