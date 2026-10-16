#!/usr/bin/env python3
"""Crop only the external test split, preserving original 15-nt backbone context."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import output_directory, read_json, signature, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True, help='Step2 特征目录，只需 test split')
    p.add_argument('--out-dir', type=Path, required=True, help='新的中心裁剪目录')
    p.add_argument('--window-size', type=int, default=5, help='输出的中心奇数窗口长度，v2 模型使用 5')
    p.add_argument('--chunk-size', type=int, default=4096, help='每块裁剪行数，控制内存')
    return p


def run(args):
    import numpy as np
    import torch
    root = args.input_dir.resolve()
    summary = read_json(root/'feature_summary.json')
    manifest = torch.load(root/'test.pt', map_location='cpu', weights_only=False)
    if manifest.get('storage_format') != 'streaming_memmap_v1':
        raise ValueError('Expected streaming_memmap_v1')
    n, w, h = manifest['feature_shape']
    k = args.window_size
    if k < 5 or k%2 != 1 or w%2 != 1 or k > w or args.chunk_size < 1:
        raise ValueError('Invalid crop size or chunk size')
    dtype = np.dtype(manifest['feature_dtype'])
    if dtype not in (np.dtype('float16'), np.dtype('float32')) or manifest['label_dtype'] != 'float32':
        raise ValueError('Unsupported feature/label dtype')
    fp, lp = root/manifest['feature_path'], root/manifest['label_path']
    if fp.stat().st_size != n*w*h*dtype.itemsize or lp.stat().st_size != n*4:
        raise ValueError('Incomplete input arrays')
    signatures = [signature(p) for p in (fp, lp, root/'test.pt', root/'metadata_test.tsv', root/'feature_summary.json')]
    x = np.memmap(fp, mode='r', dtype=dtype, shape=(n, w, h))
    y = np.memmap(lp, mode='r', dtype='float32', shape=(n,))
    start = (w-k)//2
    with output_directory(args.out_dir) as stage:
        fn = f'test.features.{"f16" if dtype.itemsize == 2 else "f32"}.bin'
        with (stage/fn).open('wb') as xf, (stage/'test.labels.f32.bin').open('wb') as yf:
            for lo in range(0, n, args.chunk_size):
                block = np.ascontiguousarray(x[lo:lo+args.chunk_size, start:start+k])
                labels = np.ascontiguousarray(y[lo:lo+args.chunk_size])
                if not np.isfinite(block).all() or not (labels == 0).all():
                    raise ValueError('Nonfinite features or nonnegative WGA labels')
                block.tofile(xf)
                labels.tofile(yf)
        with (root/'metadata_test.tsv').open(newline='') as src, (stage/'metadata_test.tsv').open('w', newline='') as dst:
            reader = csv.DictReader(src, delimiter='\t')
            fields = list(reader.fieldnames)
            for key in ['crop_input_seq', 'crop_input_ref', 'crop_input_window_size', 'crop_start0', 'crop_end0']:
                if key not in fields:
                    fields.append(key)
            writer = csv.DictWriter(dst, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            count = 0
            for row in reader:
                if count >= n or int(row['label']) != 0 or row.get('split', 'test') != 'test':
                    raise ValueError('Metadata/label/split mismatch')
                for key in ['seq', 'ref']:
                    sequence = row.get(key, '')
                    if len(sequence) != w:
                        raise ValueError(f'Metadata {key} length mismatch')
                    row[f'crop_input_{key}'] = sequence
                    row[key] = sequence[start:start+k]
                row.update(crop_input_window_size=w, crop_start0=start, crop_end0=start+k)
                writer.writerow(row)
                count += 1
            if count != n:
                raise ValueError('Metadata count mismatch')
        manifest.update(feature_shape=[n, k, h], feature_path=fn, label_path='test.labels.f32.bin',
                        metadata_path='metadata_test.tsv', records=n, split='test',
                        window_size=k, crop_start0=start, crop_end0=start+k)
        torch.save(manifest, stage/'test.pt')
        if [signature(Path(s['path'])) for s in signatures] != signatures:
            raise ValueError('Input changed during crop')
        cropped = dict(summary)
        cropped.update(window_size=k, records=n, operation='center_crop',
                       source_feature_dir=str(root), source_feature_summary=summary,
                       out_dir=str(args.out_dir.resolve()),
                       splits=[dict(split='test', records=n, feature_shape=[n, k, h],
                                    feature_path=fn, label_path='test.labels.f32.bin', metadata_path='metadata_test.tsv')])
        write_json(stage/'feature_summary.json', cropped)
    print(f'[DONE] test crop {w}->{k}, records={n}', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
