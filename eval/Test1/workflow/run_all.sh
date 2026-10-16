#!/usr/bin/env bash
# 默认只做隔离测试；只有显式 --run 才处理完整 WGA 文件并写正式结果。
# 测试：bash eval/Test1/workflow/run_all.sh --test
# 正式（用户准备好后自行启动）：DEVICE=cuda:0 bash eval/Test1/workflow/run_all.sh --run
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mode="${1:---test}"
if [[ $# -gt 0 ]]; then shift; fi
case "${mode}" in
    --test)
        # 测试全部写入临时目录，结束自动清理；不会运行全量 207 GB 输入。
        run_test1_command "${PYTHON_BIN}" "${TEST1_WORKFLOW_DIR}/test_step2_config.py" --real-input --backbones "$@"
        ;;
    --run)
        if [[ $# -gt 0 ]]; then echo 'Use environment variables for --run settings' >&2; exit 2; fi
        bash "${TEST1_WORKFLOW_DIR}/step1_build_unmod_windows.sh"
        read -r -a versions <<< "${VERSIONS:-V011 V543 V610}"
        for version in "${versions[@]}"; do
            VERSION="${version}" bash "${TEST1_WORKFLOW_DIR}/step2_extract_features.sh"
            VERSION="${version}" bash "${TEST1_WORKFLOW_DIR}/step3_center_crop.sh"
            VERSION="${version}" bash "${TEST1_WORKFLOW_DIR}/step4_evaluate.sh"
        done
        bash "${TEST1_WORKFLOW_DIR}/step5_compare.sh"
        ;;
    *) echo 'Usage: run_all.sh [--test|--run]' >&2; exit 2;;
esac
