#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=======================================================================
Nanopore JSONL Signal Preprocessing
=======================================================================

功能
----
对 JSONL / JSONL.GZ 文件中的每条 read：

    record["signal"]
          ↓
    np.float32
          ↓
    AutoFeatureExtractor
          ↓
    processed signal
          ↓
    替换 record["signal"]
          ↓
    写入新的 JSONL / JSONL.GZ

本脚本只执行 signal preprocessing。

不会执行：
    - AutoModel
    - encoder.encode_signal()
    - tokenizer
    - decoder
    - hidden state extraction

设计目标
--------
1. 支持百万级 / 千万级 JSONL 流式处理；
2. 不将整个文件加载到内存；
3. 尽量降低 JSON serialization 开销；
4. 支持 orjson 加速（如果环境已安装）；
5. 支持 gzip 输入输出；
6. 保留除 signal 外所有原始字段；
7. 检查 preprocessing 前后 signal 长度；
8. 检查 NaN / Inf；
9. 失败 read 可写入单独 error JSONL；
10. 输出采用临时文件 + rename，避免中途失败留下“假完成文件”；
11. 支持 --limit，用于 1k / 10k read benchmark。

推荐
----
对于正式 300 万 read 数据：

    --strict-length
    --error-file xxx.errors.jsonl
    --log-every 10000

首次运行建议先：

    --limit 10000

测速并验证。

=======================================================================
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Tuple

import numpy as np
import torch
from transformers import AutoFeatureExtractor


# =====================================================================
# Optional fast JSON backend
# =====================================================================

try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    orjson = None
    HAS_ORJSON = False


# =====================================================================
# JSON helpers
# =====================================================================

def json_loads(line: str) -> Dict[str, Any]:
    """
    JSON 反序列化。

    优先使用 orjson；
    如果环境没有安装，则自动退回 Python json。
    """
    if HAS_ORJSON:
        return orjson.loads(line)

    return json.loads(line)


def json_dumps(record: Dict[str, Any]) -> str:
    """
    JSON 序列化。

    orjson 通常比标准 json 更适合大规模 JSONL 数据。
    """
    if HAS_ORJSON:
        return orjson.dumps(
            record,
            option=orjson.OPT_APPEND_NEWLINE,
        ).decode("utf-8")

    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


# =====================================================================
# File helpers
# =====================================================================

def open_text(path: str, mode: str) -> TextIO:
    """
    自动识别普通 JSONL / gzip JSONL。

    支持：
        *.jsonl
        *.jsonl.gz
        *.gz
    """
    if path.endswith(".gz"):
        return gzip.open(
            path,
            mode + "t",
            encoding="utf-8",
        )

    return open(
        path,
        mode,
        encoding="utf-8",
        buffering=1024 * 1024,
    )


def build_tmp_path(output_path: str) -> str:
    """
    为正式输出构建临时文件路径。

    input:
        output.jsonl

    temp:
        output.jsonl.tmp.<pid>

    成功后再 os.replace()。
    """
    path = Path(output_path)

    if str(path).endswith(".gz"):
        base = str(path)[:-3]
        return f"{base}.tmp.{os.getpid()}.gz"

    return f"{path}.tmp.{os.getpid()}"


# =====================================================================
# FeatureExtractor
# =====================================================================

def load_feature_extractor(model_dir: str):
    """
    加载 encoder 配套的 AutoFeatureExtractor。

    注意：
        model_dir 应当指向 encoder 目录，例如：

        DNA-5mC/train/backbone/V003-Preprocess/encoder
    """

    print("=" * 80)
    print("[INFO] Loading AutoFeatureExtractor")
    print(f"[INFO] model_dir : {model_dir}")

    extractor = AutoFeatureExtractor.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    strategy = getattr(
        extractor,
        "strategy",
        None,
    )

    print(
        f"[INFO] class     : "
        f"{extractor.__class__.__name__}"
    )

    if strategy is not None:
        print(
            f"[INFO] strategy  : {strategy}"
        )
    else:
        print(
            "[INFO] strategy  : <not exposed>"
        )

    print("=" * 80)

    return extractor


# =====================================================================
# Signal preprocessing
# =====================================================================

def preprocess_one_signal(
    raw_signal: Any,
    feature_extractor,
) -> Tuple[list, int, int]:
    """
    对单条 read 的 signal 做预处理。

    Parameters
    ----------
    raw_signal
        原始 JSON signal，一般为 list[float]。

    feature_extractor
        AutoFeatureExtractor 对象。

    Returns
    -------
    processed_signal
        预处理后的 list[float]

    raw_len
        原始 sample 数量

    processed_len
        预处理后 sample 数量
    """

    # -----------------------------------------------------------------
    # JSON list -> contiguous float32 ndarray
    # -----------------------------------------------------------------

    signal_np = np.asarray(
        raw_signal,
        dtype=np.float32,
    )

    if signal_np.ndim != 1:
        raise ValueError(
            f"signal must be 1-D, "
            f"got shape={signal_np.shape}"
        )

    if signal_np.size == 0:
        raise ValueError(
            "signal is empty"
        )

    if not np.all(np.isfinite(signal_np)):
        raise ValueError(
            "raw signal contains NaN or Inf"
        )

    signal_np = np.ascontiguousarray(signal_np)

    raw_len = int(signal_np.size)

    # -----------------------------------------------------------------
    # Model-native preprocessing
    #
    # 与模型 README / 测试代码保持一致：
    #
    # feature_extractor(
    #     raw_signal,
    #     return_tensors="pt"
    # )["signal"]
    # -----------------------------------------------------------------

    processed_inputs = feature_extractor(
        signal_np,
        return_tensors="pt",
    )

    if "signal" not in processed_inputs:
        raise KeyError(
            "AutoFeatureExtractor output does not "
            "contain key 'signal'. "
            f"available={list(processed_inputs.keys())}"
        )

    signal_tensor = processed_inputs["signal"]

    if not isinstance(
        signal_tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "FeatureExtractor output ['signal'] "
            f"is not Tensor: {type(signal_tensor)}"
        )

    # -----------------------------------------------------------------
    # 单 read 一般是 [1,1,T]
    #
    # JSON 中重新保存为一维：
    #
    # [T]
    # -----------------------------------------------------------------

    processed_np = (
        signal_tensor
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
        .reshape(-1)
    )

    processed_len = int(
        processed_np.size
    )

    if processed_len == 0:
        raise ValueError(
            "processed signal is empty"
        )

    if not np.all(
        np.isfinite(processed_np)
    ):
        raise ValueError(
            "processed signal contains NaN or Inf"
        )

    # JSON 无法直接序列化 ndarray，
    # 因此最终必须转换为 Python list。
    return (
        processed_np.tolist(),
        raw_len,
        processed_len,
    )


# =====================================================================
# Main processing
# =====================================================================

def process_file(
    input_file: str,
    output_file: str,
    feature_extractor,
    *,
    signal_field: str,
    strict_length: bool,
    fail_fast: bool,
    error_file: Optional[str],
    limit: Optional[int],
    log_every: int,
) -> Dict[str, Any]:
    """
    流式处理整个 JSONL。

    内存复杂度：
        O(single read)

    而不是：
        O(total reads)
    """

    output_path = Path(
        output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_output = build_tmp_path(
        output_file
    )

    error_handle = None

    if error_file:
        error_path = Path(error_file)

        error_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        error_handle = open_text(
            error_file,
            "w",
        )

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    total = 0
    success = 0
    failed = 0
    empty_lines = 0
    length_changed = 0

    raw_samples_total = 0
    processed_samples_total = 0

    start_time = time.time()

    print()
    print("=" * 80)
    print("[INFO] Signal preprocessing started")
    print("=" * 80)

    print(f"[INFO] input        : {input_file}")
    print(f"[INFO] output       : {output_file}")
    print(f"[INFO] temp output  : {tmp_output}")
    print(f"[INFO] signal field : {signal_field}")
    print(f"[INFO] strict length: {strict_length}")
    print(f"[INFO] fail fast    : {fail_fast}")
    print(f"[INFO] error file   : {error_file}")
    print(f"[INFO] limit        : {limit}")
    print(f"[INFO] JSON backend : {'orjson' if HAS_ORJSON else 'json'}")

    print("=" * 80)

    try:

        with open_text(
            input_file,
            "r",
        ) as fin, open_text(
            tmp_output,
            "w",
        ) as fout:

            for line_number, line in enumerate(
                fin,
                start=1,
            ):

                if limit is not None and total >= limit:
                    break

                line = line.strip()

                if not line:
                    empty_lines += 1
                    continue

                total += 1

                record = None
                read_id = "unknown"

                try:

                    # =================================================
                    # Parse JSON
                    # =================================================

                    record = json_loads(
                        line
                    )

                    read_id = str(
                        record.get(
                            "read_id",
                            "unknown",
                        )
                    )

                    # =================================================
                    # Validate signal
                    # =================================================

                    if signal_field not in record:
                        raise KeyError(
                            f"missing field "
                            f"{signal_field!r}"
                        )

                    raw_signal = record[
                        signal_field
                    ]

                    if (
                        not isinstance(
                            raw_signal,
                            list,
                        )
                        or len(raw_signal) == 0
                    ):
                        raise ValueError(
                            f"{signal_field!r} "
                            "must be a non-empty list"
                        )

                    # =================================================
                    # Preprocess
                    # =================================================

                    (
                        processed_signal,
                        raw_len,
                        processed_len,
                    ) = preprocess_one_signal(
                        raw_signal,
                        feature_extractor,
                    )

                    raw_samples_total += raw_len
                    processed_samples_total += (
                        processed_len
                    )

                    # =================================================
                    # Length QC
                    # =================================================

                    if (
                        raw_len
                        != processed_len
                    ):

                        length_changed += 1

                        if strict_length:
                            raise ValueError(
                                "signal length changed: "
                                f"{raw_len} "
                                f"-> {processed_len}"
                            )

                    # =================================================
                    # Only replace signal.
                    #
                    # 其它 JSON 字段保持原样。
                    # =================================================

                    record[
                        signal_field
                    ] = processed_signal

                    # =================================================
                    # Write
                    # =================================================

                    fout.write(
                        json_dumps(record)
                    )

                    success += 1

                except Exception as exc:

                    failed += 1

                    error_message = (
                        f"[ERROR] "
                        f"line={line_number} "
                        f"read_id={read_id} "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    print(
                        error_message,
                        file=sys.stderr,
                        flush=True,
                    )

                    # ---------------------------------------------
                    # 将失败的原始 record 单独保存。
                    #
                    # 不静默丢弃数据。
                    # ---------------------------------------------

                    if error_handle is not None:

                        error_record = {
                            "line_number": (
                                line_number
                            ),
                            "read_id": (
                                read_id
                            ),
                            "error_type": (
                                type(exc).__name__
                            ),
                            "error": str(exc),
                        }

                        # 为避免 error 文件再次复制巨大 signal，
                        # 默认不把完整原始 signal 写进去。
                        #
                        # line_number + read_id 足以定位原始记录。

                        error_handle.write(
                            json_dumps(
                                error_record
                            )
                        )

                    if fail_fast:
                        raise

                # =====================================================
                # Progress
                # =====================================================

                if (
                    log_every > 0
                    and total % log_every == 0
                ):

                    elapsed = max(
                        time.time()
                        - start_time,
                        1e-9,
                    )

                    rate = (
                        total
                        / elapsed
                    )

                    print(
                        "[PROGRESS] "
                        f"total={total:,} "
                        f"success={success:,} "
                        f"failed={failed:,} "
                        f"len_changed="
                        f"{length_changed:,} "
                        f"rate={rate:.2f} "
                        f"reads/s "
                        f"elapsed="
                        f"{elapsed / 3600:.2f} h",
                        flush=True,
                    )

        # -----------------------------------------------------------------
        # 只有完整成功结束后，才把 tmp 文件替换为正式输出。
        # -----------------------------------------------------------------

        os.replace(
            tmp_output,
            output_file,
        )

    except Exception:

        print(
            "\n[ERROR] Processing aborted.",
            file=sys.stderr,
        )

        print(
            f"[ERROR] Partial temporary file retained: "
            f"{tmp_output}",
            file=sys.stderr,
        )

        raise

    finally:

        if error_handle is not None:
            error_handle.close()

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    rate = (
        total / elapsed
        if elapsed > 0
        else 0.0
    )

    print()
    print("=" * 80)
    print("[DONE] Signal preprocessing completed")
    print("=" * 80)

    print(
        f"[DONE] total reads        : "
        f"{total:,}"
    )

    print(
        f"[DONE] success            : "
        f"{success:,}"
    )

    print(
        f"[DONE] failed             : "
        f"{failed:,}"
    )

    print(
        f"[DONE] empty lines        : "
        f"{empty_lines:,}"
    )

    print(
        f"[DONE] length changed     : "
        f"{length_changed:,}"
    )

    print(
        f"[DONE] raw samples        : "
        f"{raw_samples_total:,}"
    )

    print(
        f"[DONE] processed samples  : "
        f"{processed_samples_total:,}"
    )

    print(
        f"[DONE] elapsed            : "
        f"{elapsed / 3600:.3f} h"
    )

    print(
        f"[DONE] average speed      : "
        f"{rate:.2f} reads/s"
    )

    print(
        f"[DONE] output             : "
        f"{output_file}"
    )

    if error_file:
        print(
            f"[DONE] error log          : "
            f"{error_file}"
        )

    print("=" * 80)

    # ---------------------------------------------------------------------
    # Consistency
    # ---------------------------------------------------------------------

    if total != success + failed:

        raise RuntimeError(
            "Internal count mismatch: "
            f"total={total}, "
            f"success={success}, "
            f"failed={failed}"
        )

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "length_changed": (
            length_changed
        ),
        "elapsed_seconds": (
            elapsed
        ),
        "reads_per_second": (
            rate
        ),
    }


# =====================================================================
# CLI
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
        description=(
            "Preprocess nanopore JSONL signal using "
            "the encoder AutoFeatureExtractor."
        ),
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help=(
            "Input .jsonl / .jsonl.gz"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "Output .jsonl / .jsonl.gz"
        ),
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Encoder directory containing "
            "AutoFeatureExtractor configuration."
        ),
    )

    parser.add_argument(
        "--signal-field",
        default="signal",
        help=(
            "JSON field containing raw signal."
        ),
    )

    parser.add_argument(
        "--strict-length",
        action="store_true",
        help=(
            "Treat preprocessing signal-length "
            "change as an error."
        ),
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help=(
            "Immediately stop when any read fails."
        ),
    )

    parser.add_argument(
        "--error-file",
        default=None,
        help=(
            "Optional JSONL/GZ file storing "
            "failed read IDs and error messages."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only first N non-empty "
            "records; useful for benchmark."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10000,
        help=(
            "Print progress every N reads."
        ),
    )

    return parser.parse_args()


# =====================================================================
# Entry
# =====================================================================

def main():

    args = parse_args()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    if not os.path.isfile(
        args.input
    ):
        raise FileNotFoundError(
            f"Input does not exist: "
            f"{args.input}"
        )

    if not os.path.isdir(
        args.model_dir
    ):
        raise FileNotFoundError(
            f"Model directory does not exist: "
            f"{args.model_dir}"
        )

    if os.path.abspath(
        args.input
    ) == os.path.abspath(
        args.output
    ):
        raise ValueError(
            "Input and output must be "
            "different files."
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be > 0"
            )

    # -----------------------------------------------------------------
    # FeatureExtractor
    # -----------------------------------------------------------------

    feature_extractor = (
        load_feature_extractor(
            args.model_dir
        )
    )

    # -----------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------

    process_file(
        input_file=args.input,
        output_file=args.output,
        feature_extractor=(
            feature_extractor
        ),
        signal_field=(
            args.signal_field
        ),
        strict_length=(
            args.strict_length
        ),
        fail_fast=(
            args.fail_fast
        ),
        error_file=(
            args.error_file
        ),
        limit=args.limit,
        log_every=(
            args.log_every
        ),
    )


if __name__ == "__main__":
    main()