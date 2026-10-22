#!/usr/bin/env bash
# Step5：训练seed均值/样本方差、模型差值、HTML/CSV、可选PNG/PDF；不进行模型训练。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_test4_step step5_compare.py "$@"
