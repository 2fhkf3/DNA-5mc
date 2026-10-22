#!/usr/bin/env bash
# Step2：按旧训练摘要提取一次主干特征，复用所有分类头/seed；已完成分片校验后跳过。
# VERSION/--version只分配冻结清单中的一个主干，不改变最终共同集合。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
args=()
[[ -z "${VERSION:-}" ]] || args+=(--version "${VERSION}")
run_test4_step step2_predict_reads.py "${args[@]}" "$@"
