#!/usr/bin/env bash
# 所有 Bash 入口共用的项目路径；按本文件位置定位，不依赖启动时的工作目录。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE_ROOT="${BACKBONE_ROOT:-${PROJECT_ROOT}/train/backbone}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output}"

# 优先使用显式指定或已激活的环境；保留本机原有 PoreDLM 环境作为可用时的后备。
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif [[ -x /mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python ]]; then
        PYTHON_BIN="/mnt/zzbnew/rnamodel/zhoukexuan/miniconda3/envs/PoreDLM/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${OUTPUT_ROOT}/.hf_modules}"
