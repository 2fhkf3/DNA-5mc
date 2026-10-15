#!/usr/bin/env python3
"""将 Step-4 streaming_memmap_v1 特征裁剪为中心奇数长度窗口。

分块读写，不重新运行 BERT。同步裁剪 TSV 的 seq/ref，保留标签、样本顺序、
位点与原信号窗口坐标。生成的目录可直接作为 train/train.py 的 feature_dir。
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SPLITS = ("train", "valid", "test")


def signature(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def load_plan(input_dir, window_size):
    if window_size < 5 or window_size % 2 != 1:
        raise ValueError("window_size 必须是 >= 5 的奇数，例如 5、7、9、15；现有分类头至少需要 5 个位置")
    plans = []
    for split in SPLITS:
        manifest_path = input_dir / f"{split}.pt"
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        if manifest.get("storage_format") != "streaming_memmap_v1":
            raise ValueError(f"仅支持 Step-4 streaming_memmap_v1 格式：{manifest_path}")
        shape = tuple(manifest["feature_shape"])
        if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f"非法 feature_shape：{manifest_path}: {shape}")
        if shape[1] % 2 != 1 or window_size > shape[1]:
            raise ValueError(f"源窗口须为奇数且不小于裁剪窗口：{split}: {shape[1]} -> {window_size}")
        dtype = np.dtype(manifest["feature_dtype"])
        if dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError(f"不支持的 feature_dtype：{dtype}")
        if np.dtype(manifest["label_dtype"]) != np.dtype("float32"):
            raise ValueError(f"标签必须是 float32：{manifest_path}")
        feature_path = input_dir / manifest["feature_path"]
        label_path = input_dir / manifest["label_path"]
        metadata_path = input_dir / f"metadata_{split}.tsv"
        for path, expected in ((feature_path, math.prod(shape) * dtype.itemsize),
                               (label_path, shape[0] * 4)):
            if path.stat().st_size != expected:
                raise ValueError(f"数据文件大小与清单不一致：{path}，应为 {expected} 字节")
        with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
            fields = csv.DictReader(handle, delimiter="\t").fieldnames or []
        required = {"read_id", "label", "chrom", "center_genome_pos0", "ref"}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f"元数据缺少训练必需列或有重复列：{metadata_path}")
        plans.append(dict(split=split, manifest=manifest, shape=shape, dtype=dtype,
                          feature_path=feature_path, label_path=label_path,
                          metadata_path=metadata_path, fields=fields,
                          crop_start=(shape[1] - window_size) // 2,
                          signatures=[signature(p) for p in
                                      (manifest_path, feature_path, label_path, metadata_path)]))
    if len({p["shape"][1:] for p in plans}) != 1:
        raise ValueError("三个 split 的窗口长度和特征维度必须一致")
    return plans


def crop_split(plan, stage, window_size, chunk_size, log_every_chunks):
    split, shape, dtype = plan["split"], plan["shape"], plan["dtype"]
    n, source_window, hidden = shape
    start = plan["crop_start"]
    end = start + window_size
    tag = "f16" if dtype == np.dtype("float16") else "f32"
    feature_name = f"{split}.features.{tag}.bin"
    label_name = f"{split}.labels.f32.bin"
    metadata_name = f"metadata_{split}.tsv"
    x = np.memmap(plan["feature_path"], mode="r", dtype=dtype, shape=shape)
    y = np.memmap(plan["label_path"], mode="r", dtype=np.float32, shape=(n,))
    print(f"[START] {split}: {shape} -> {(n, window_size, hidden)} dtype={dtype}", flush=True)
    started = time.perf_counter()
    with (stage / feature_name).open("wb") as features, (stage / label_name).open("wb") as labels:
        for chunk_index, lo in enumerate(range(0, n, chunk_size), 1):
            hi = min(lo + chunk_size, n)
            block = np.ascontiguousarray(x[lo:hi, start:end, :])
            if not np.isfinite(block).all():
                raise ValueError(f"裁剪特征含 NaN/Inf：{split}, rows={lo}:{hi}")
            label_block = np.ascontiguousarray(y[lo:hi])
            if not np.isin(label_block, (0, 1)).all():
                raise ValueError(f"非二分类标签：{split}, rows={lo}:{hi}")
            block.tofile(features)
            label_block.tofile(labels)
            if hi == n or (log_every_chunks and chunk_index % log_every_chunks == 0):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"[CROP] {split} rows={hi}/{n} ({hi/n:.1%}) "
                      f"rows_per_s={hi/elapsed:.1f} elapsed_s={elapsed:.1f}", flush=True)
        for handle in (features, labels):
            handle.flush()
            os.fsync(handle.fileno())

    # seq/ref 与特征使用完全相同的切片；原始值保留在带前缀的来源列中。
    fields = list(plan["fields"])
    extra_fields = ["crop_input_window_size", "crop_start0", "crop_end0"]
    sequence_fields = [key for key in ("seq", "ref") if key in fields]
    extra_fields += [f"crop_input_{key}" for key in sequence_fields]
    # 支持对已有裁剪结果再次裁剪；前缀列记录本次输入，完整历史保存在 manifest 中。
    fields += [key for key in extra_fields if key not in fields]
    with plan["metadata_path"].open(encoding="utf-8-sig", newline="") as source:
        with (stage / metadata_name).open("w", encoding="utf-8", newline="") as dest:
            reader = csv.DictReader(source, delimiter="\t")
            writer = csv.DictWriter(dest, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            count = 0
            for index, row in enumerate(reader):
                if index >= n or None in row or any(value is None for value in row.values()):
                    raise ValueError(f"元数据行数或列数不匹配：{split}, row={index}")
                if int(row["label"]) != int(y[index]) or not row["read_id"].strip():
                    raise ValueError(f"元数据标签/样本不匹配：{split}, row={index}")
                if row.get("split") and row["split"] != split:
                    raise ValueError(f"元数据 split 不匹配：{split}, row={index}")
                if "sample_index" in row and int(row["sample_index"]) != index:
                    raise ValueError(f"元数据 sample_index 不匹配：{split}, row={index}")
                for key in sequence_fields:
                    sequence = row[key].strip()
                    if (sequence or key == "ref") and len(sequence) != source_window:
                        raise ValueError(f"{split} row={index}: {key} 长度 {len(sequence)} "
                                         f"不等于源窗口 {source_window}，无法安全对齐裁剪")
                    row[f"crop_input_{key}"] = row[key]
                    row[key] = sequence[start:end] if sequence else ""
                row.update(crop_input_window_size=source_window, crop_start0=start, crop_end0=end)
                writer.writerow(row)
                count += 1
                if log_every_chunks and count % (chunk_size * log_every_chunks) == 0:
                    print(f"[METADATA] {split} rows={count}/{n}", flush=True)
            if count != n:
                raise ValueError(f"元数据行数不匹配：{split}, expected={n}, actual={count}")
            dest.flush()
            os.fsync(dest.fileno())

    assert (stage / feature_name).stat().st_size == n * window_size * hidden * dtype.itemsize
    assert (stage / label_name).stat().st_size == n * 4
    manifest = copy.deepcopy(plan["manifest"])
    manifest.update(feature_path=feature_name, label_path=label_name,
                    metadata_path=metadata_name, feature_shape=[n, window_size, hidden],
                    records=n, split=split,
                    feature_source=f"{manifest.get('feature_source', 'step4')}_center{window_size}")
    manifest["center_crop"] = dict(source_feature_shape=list(shape), start0=start, end0=end,
                                    input_signatures=plan["signatures"],
                                    previous_crop=plan["manifest"].get("center_crop"),
                                    preserves_original_bert_context=True)
    torch.save(manifest, stage / f"{split}.pt")
    del x, y
    print(f"[DONE] {split} metadata_rows={count} elapsed_s={time.perf_counter()-started:.1f}", flush=True)
    return dict(split=split, records=n, feature_shape=manifest["feature_shape"],
                feature_path=feature_name, label_path=label_name, metadata_path=metadata_name,
                crop_start0=start, crop_end0=end)


def make_training_config(template_path, output_dir, window_size, experiment):
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("训练模板必须是 YAML mapping")
    config.update(feature_dir=str(output_dir), metadata_dir=None,
                  fusion_window_size=window_size, experiment=experiment)
    return config


def run(args):
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (args.out_dir.expanduser().resolve() if args.out_dir else
                  PROJECT_ROOT / "train" / "data" / f"{input_dir.name}_center{args.window_size}")
    if output_dir == input_dir or output_dir in input_dir.parents or input_dir in output_dir.parents:
        raise ValueError("输入/输出目录不能相同，也不能互为父子目录")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，不覆盖：{output_dir}；请换一个 OUT_DIR")
    if args.chunk_size < 1 or args.log_every_chunks < 0:
        raise ValueError("chunk_size 必须 >= 1，log_every_chunks 必须 >= 0")
    plans = load_plan(input_dir, args.window_size)
    experiment = args.experiment or re.sub(r"[^A-Za-z0-9_.-]", "_", f"{input_dir.name}_center{args.window_size}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment) or experiment in (".", ".."):
        raise ValueError("experiment 只能包含字母、数字、下划线、点和连字符")
    template_path = args.train_config.expanduser().resolve()
    config = make_training_config(template_path, output_dir, args.window_size, experiment)
    output_bytes = sum(p["shape"][0] * (args.window_size*p["shape"][2]*p["dtype"].itemsize + 4)
                       + p["metadata_path"].stat().st_size*2 for p in plans)
    source_summary_path = input_dir / "feature_summary.json"
    source_summary = (json.loads(source_summary_path.read_text()) if source_summary_path.exists() else {})
    source_extra_signatures = [signature(template_path)]
    if source_summary_path.exists():
        source_extra_signatures.append(signature(source_summary_path))
    plan_report = dict(input_dir=str(input_dir), out_dir=str(output_dir), window_size=args.window_size,
                       approximate_output_gib=round(output_bytes/1024**3, 3),
                       train_config=str(output_dir / "train_config.yaml"),
                       splits={p["split"]: dict(source_shape=p["shape"],
                               cropped_shape=[p["shape"][0], args.window_size, p["shape"][2]],
                               dtype=p["dtype"].name) for p in plans})
    print(json.dumps(plan_report, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((output_dir.parent / f".{output_dir.name}.crop.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output_dir.exists():
            raise FileExistsError(f"输出目录已存在：{output_dir}")
        if shutil.disk_usage(output_dir.parent).free < output_bytes:
            raise OSError(f"输出磁盘空间不足，估计需要约 {output_bytes/1024**3:.2f} GiB")
        # 所有 split 成功后才整体发布目录；中断时不会暴露半成品训练数据。
        stage = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.partial-",
                                                                     dir=output_dir.parent)))
        summaries = [crop_split(p, stage, args.window_size, args.chunk_size, args.log_every_chunks) for p in plans]
        for sig in [s for p in plans for s in p["signatures"]] + source_extra_signatures:
            if signature(Path(sig["path"])) != sig:
                raise ValueError(f"裁剪过程中源文件发生变化：{sig['path']}")
        summary = dict(source_summary)
        summary.update(operation="center_crop", source_feature_dir=str(input_dir),
                       out_dir=str(output_dir), window_size=args.window_size, splits=summaries,
                       feature_source=f"{plans[0]['manifest'].get('feature_source', 'step4')}_center{args.window_size}",
                       source_feature_summary=source_summary,
                       preserves_original_bert_context=True,
                       metadata_coordinate_policy="Original genomic/signal coordinates preserved; seq/ref cropped; offsets in crop_start0/crop_end0")
        (stage / "feature_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        header = ("# 自动生成的裁剪特征训练配置；相对路径仍以 train/train.py 所在目录为基准。\n"
                  "# 融合头使用完整裁剪窗口；已有 center_5mer Linear/MLP/DWConv 等仍只使用中心 5 个位置。\n"
                  "# 裁剪的是原 BERT 输出，不等同于用更短信号重新运行 BERT。\n")
        (stage / "train_config.yaml").write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        os.rename(stage, output_dir)
    print(f"[SUCCESS] 输出目录：{output_dir}", flush=True)
    command = shlex.join(["bash", str(PROJECT_ROOT / "train/run/example/run.sh"),
                          str(output_dir / "train_config.yaml")])
    print(f"[TRAIN] {command}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Step-4 输出目录，含三个 split 的 .pt/bin/TSV")
    parser.add_argument("--out-dir", type=Path, help="新目录；建议使用本地磁盘，禁止覆盖现有目录")
    parser.add_argument("--window-size", type=int, default=5, help="中心裁剪长度：5、7、9 等奇数")
    parser.add_argument("--chunk-size", type=int, default=4096, help="每块样本数，控制内存占用")
    parser.add_argument("--log-every-chunks", type=int, default=16, help="每多少块打印进度，0 只打印结束进度")
    parser.add_argument("--train-config", type=Path, default=PROJECT_ROOT/"train/run/example/config.yaml",
                        help="训练 YAML 模板；保留其训练参数，替换数据目录、实验名和融合窗口")
    parser.add_argument("--experiment", help="生成的训练配置中的实验名")
    parser.add_argument("--dry-run", action="store_true", help="只检查清单、配置及文件大小，显示计划，不生成数据")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
