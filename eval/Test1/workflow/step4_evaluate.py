#!/usr/bin/env python3
"""Load completed classifiers for external all-negative inference; never train."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

from common import (PROJECT, infer_architecture, valid_model_name, install_sql_math, open_text, output_directory,
                    prediction_db, read_json, sha256, signature, summarize, tsv, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir', type=Path, help='单个已完成训练的 run，读取其中 best.pt')
    group.add_argument('--runs-root', type=Path, help='批量扫描 experiment 根目录或其 runs 子目录')
    p.add_argument('--feature-dir', type=Path, required=True, help='Test1 的 test.pt / bin / metadata 目录')
    p.add_argument('--out-dir', type=Path, required=True, help='单 run 结果目录，或批量结果的父目录')
    p.add_argument('--version', type=valid_model_name, required=True, help='用于核对特征来源和 checkpoint 版本，支持新版本名')
    p.add_argument('--strategies', nargs='+', help='批量时限制分类头；默认所有已完成的分类头')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='批量时使用的训练 seed')
    p.add_argument('--device', default='auto', help='auto / cpu / cuda:N')
    p.add_argument('--batch-size', type=int, default=2048, help='分类器推理批次大小')
    p.add_argument('--cpu-threads', type=int, default=4, help='CPU 算子线程数')
    p.add_argument('--threshold', type=float, help='read 阈值；不指定则使用 checkpoint 的训练阈值')
    p.add_argument('--site-threshold', type=float, default=.5, help='平均概率的 site 阳性阈值')
    p.add_argument('--bootstrap', type=int, default=1000, help='基因组区块 bootstrap 次数；0 为禁用')
    p.add_argument('--block-size', type=int, default=1_000_000, help='bootstrap 区块大小，单位 bp')
    p.add_argument('--bootstrap-seed', type=int, default=42, help='重采样随机种子，与训练 seed 无关')
    p.add_argument('--library-id', help='仅在特征没有 Test1 文库清单时手动指定')
    p.add_argument('--save-read-predictions', action=argparse.BooleanOptionalAction, default=True,
                   help='另存逐窗口 TSV.gz；SQLite 匹配索引始终保存')
    return p


def check_protocol(summary, config, version):
    if summary.get('eval_version') != version:
        raise ValueError('Feature version does not match --version')
    if not config.get('experiment', '').upper().startswith(version.upper()):
        raise ValueError('Checkpoint experiment does not match feature version')
    old_file = Path(config['feature_dir'])/'feature_summary.json'
    if not old_file.is_file():
        raise FileNotFoundError(f'Cannot verify training feature protocol: {old_file}')
    old = read_json(old_file)
    architecture = infer_architecture(summary)
    if architecture is None or architecture != infer_architecture(old):
        raise ValueError('Feature architecture does not match the training feature protocol')
    keys = {'diffusion': ['encoder_layer', 'hidden_source', 'ode_steps', 'ode_start_t', 'ode_self_cond_cfg_scale', 'last_t'],
            'bert': ['encoder_layer', 'hidden_layer', 'token_offset'],
            'gpt': ['encoder_layer', 'decoder_layers', 'input_layout', 'resolved_token_id_mode', 'selected_layer_name']}[architecture]
    for key in ['preprocess', *keys]:
        if key in old and summary.get(key) != old[key]:
            raise ValueError(f'Feature protocol mismatch for {key}: {summary.get(key)!r} != {old[key]!r}')
    for key in ['encoder_dir', 'decoder_dir']:
        if key in old and Path(summary[key]).resolve() != Path(old[key]).resolve():
            raise ValueError(f'Backbone path mismatch for {key}')
    return dict(training_summary=signature(old_file), compared_fields=['preprocess', *keys, 'encoder_dir', 'decoder_dir'])


def evaluate(args, run_dir, output):
    import numpy as np
    import torch
    sys.path.insert(0, str(PROJECT/'train'))
    from model import FUSION_STRATEGY, build_model
    codes = dict(zip('ACGTNWSMKRYBVDHZ', range(16)))
    checkpoint_path = run_dir/'best.pt'
    checkpoint_signature = signature(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    root = args.feature_dir.resolve()
    summary = read_json(root/'feature_summary.json')
    protocol = check_protocol(summary, config, args.version)
    meta = torch.load(root/'test.pt', map_location='cpu', weights_only=False)
    if meta.get('storage_format') != 'streaming_memmap_v1':
        raise ValueError('Only streaming_memmap_v1 features are supported')
    n, w, h = meta['feature_shape']
    window = int(config['source_window_size'])
    if n < 1 or w < window or w%2 != 1 or window%2 != 1 or h != config['hidden_dim']:
        raise ValueError('Feature shape incompatible with the saved classifier')
    dtype = np.dtype(meta['feature_dtype'])
    if dtype not in (np.dtype('float16'), np.dtype('float32')) or np.dtype(meta['label_dtype']) != np.dtype('float32'):
        raise ValueError('Invalid feature or label dtype')
    fp, lp = root/meta['feature_path'], root/meta['label_path']
    if fp.stat().st_size != n*w*h*dtype.itemsize or lp.stat().st_size != n*4:
        raise ValueError('Incomplete feature arrays')
    inputs = [signature(p) for p in [root/'test.pt', fp, lp, root/'metadata_test.tsv', root/'feature_summary.json']]
    x = np.memmap(fp, mode='r', dtype=dtype, shape=(n, w, h))
    y = np.memmap(lp, mode='r', dtype='float32', shape=(n,))
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision('highest')
    model = build_model(config['strategy'], h, window, config['model']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    fusion = config['strategy'] == FUSION_STRATEGY
    base_source = config.get('base_source', 'ref')
    fusion_width = int(config.get('fusion_window_size', 5))
    if fusion and (fusion_width > window or fusion_width < 1 or fusion_width%2 != 1):
        raise ValueError('Invalid fusion window')
    library = args.library_id or summary.get('test1_dataset', {}).get('library_id')
    if not library:
        raise ValueError('Missing WGA library identity')
    threshold = args.threshold if args.threshold is not None else float(config['training']['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Read threshold must be between 0 and 1')
    start = (w-window)//2
    fields = ['library_id', 'read_id', 'chrom', 'center_genome_pos0', 'strand', 'source',
              'label', 'prob', 'predicted_positive', 'seq', 'ref', 'mapq', 'input_signal_len']
    with output_directory(output) as stage:
        db = prediction_db(stage/'predictions.sqlite')
        install_sql_math(db)
        try:
            from contextlib import ExitStack
            with ExitStack() as stack, torch.inference_mode():
                handle = stack.enter_context((root/'metadata_test.tsv').open(newline=''))
                reader = csv.DictReader(handle, delimiter='\t')
                required = {'read_id', 'chrom', 'center_genome_pos0', 'strand', 'label', 'seq', 'ref'}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError('Missing metadata columns')
                writer = None
                if args.save_read_predictions:
                    output_file = stack.enter_context(open_text(stage/'read_predictions.tsv.gz', 'wt'))
                    writer = csv.DictWriter(output_file, fieldnames=fields, delimiter='\t', extrasaction='ignore')
                    writer.writeheader()
                for lo in range(0, n, args.batch_size):
                    hi = min(lo+args.batch_size, n)
                    rows = list(itertools.islice(reader, hi-lo))
                    if len(rows) != hi-lo or not (y[lo:hi] == 0).all():
                        raise ValueError('Metadata count mismatch or nonnegative WGA labels')
                    features = np.array(x[lo:hi, start:start+window], dtype=np.float32, copy=True)
                    if not np.isfinite(features).all():
                        raise ValueError('Nonfinite features')
                    for i, row in enumerate(rows, lo):
                        if int(row['label']) != 0 or row['strand'] != '+' or row.get('split', 'test') != 'test':
                            raise ValueError('Expected all-negative positive-strand test metadata')
                        if not row['read_id'] or int(row['center_genome_pos0']) < 0:
                            raise ValueError('Invalid read or coordinate')
                        if row.get('library_id', library) != library:
                            raise ValueError('Library identity mismatch')
                        if row.get('sample_index') and int(row['sample_index']) != i:
                            raise ValueError('Feature row/sample_index mismatch')
                        for key in ['seq', 'ref']:
                            if len(row[key]) != w:
                                raise ValueError('Sequence/feature window mismatch')
                            row[key] = row[key][start:start+window]
                    tensor = torch.from_numpy(features).to(device)
                    if fusion:
                        offset = (window-fusion_width)//2
                        sequences = [row[base_source][offset:offset+fusion_width].upper() for row in rows]
                        bases = torch.tensor([[codes[b] for b in seq] for seq in sequences], dtype=torch.long, device=device)
                        logits = model(tensor[:, offset:offset+fusion_width], bases)
                        probabilities = torch.softmax(logits, dim=-1)[:, 1]
                    else:
                        probabilities = torch.sigmoid(model(tensor))
                    probabilities = probabilities.cpu().numpy().astype(np.float64)
                    if probabilities.shape != (len(rows),) or not np.isfinite(probabilities).all():
                        raise ValueError('Invalid classifier probabilities')
                    inserts = []
                    for row, p in zip(rows, probabilities):
                        mapq = int(row['mapq']) if row.get('mapq', '') else None
                        length = int(row['input_signal_len']) if row.get('input_signal_len', '') else None
                        source = row.get('source', 'unmod')
                        inserts.append((library, row['read_id'], row['chrom'], int(row['center_genome_pos0']),
                                        row['strand'], source, row['seq'], row['ref'], float(p), mapq, length))
                        if writer:
                            writer.writerow(dict(row, library_id=library, source=source, prob=float(p),
                                                 predicted_positive=int(p>=threshold), input_signal_len=length))
                    # The primary key rejects repeated original-read/position observations.
                    db.executemany('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)', inserts)
                    db.commit()
                    if lo == 0 or hi == n or hi%100000 < args.batch_size:
                        print(f'[PREDICT] {run_dir.name} {hi}/{n}', flush=True)
                if next(reader, None) is not None:
                    raise ValueError('Extra metadata rows')
            metrics = summarize(db, read_threshold=threshold, site_threshold=args.site_threshold,
                                block_size=args.block_size, bootstrap=args.bootstrap, seed=args.bootstrap_seed,
                                site_path=stage/'site_predictions.tsv.gz', fp_path=stage/'false_positive_sites.tsv.gz')
            metrics.update(version=args.version, strategy=config['strategy'], seed=config['seed'],
                           checkpoint_epoch=checkpoint['epoch'], library_id=library,
                           partial_test=summary.get('test1_dataset', {}).get('partial_test', False))
            write_json(stage/'metrics.json', metrics)
            tsv(stage/'coverage_metrics.tsv', metrics['coverage'], ['coverage', 'n', 'fp', 'tn', 'fpr', 'specificity'])
            tsv(stage/'context_metrics.tsv', metrics['contexts'], ['ref', 'n', 'fp', 'tn', 'fpr', 'specificity'])
            thresholds = []
            for t in sorted(set([.1, .3, .5, .7, .9, threshold])):
                total, called = db.execute('SELECT COUNT(*), SUM(prob>=?) FROM predictions', (t,)).fetchone()
                thresholds.append(dict(level='read', threshold=t, n=total, fp=called, fpr=called/total))
            for t in sorted(set([.1, .3, .5, .7, .9, args.site_threshold])):
                total, called = db.execute('SELECT COUNT(*), SUM(score>=?) FROM '
                    '(SELECT AVG(prob) AS score FROM predictions GROUP BY library_id,chrom,position,strand)', (t,)).fetchone()
                thresholds.append(dict(level='site', threshold=t, n=total, fp=called, fpr=called/total))
            tsv(stage/'threshold_metrics.tsv', thresholds, ['level', 'threshold', 'n', 'fp', 'fpr'])
            write_json(stage/'run_manifest.json', dict(
                run_dir=str(run_dir.resolve()), checkpoint=checkpoint_signature,
                checkpoint_sha256=sha256(checkpoint_path), configuration=config,
                feature_summary=summary, feature_signatures=inputs, protocol_verification=protocol,
                inference=dict(device=str(device), batch_size=args.batch_size, window=window,
                               read_threshold=threshold, site_threshold=args.site_threshold),
                code_sha256={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('common.py')]}))
            if signature(checkpoint_path) != checkpoint_signature or [signature(Path(s['path'])) for s in inputs] != inputs:
                raise ValueError('Checkpoint/features changed during evaluation')
        finally:
            db.close()
    print(f'[DONE] {args.version}/{run_dir.name}', flush=True)


def run(args):
    if args.batch_size < 1 or args.cpu_threads < 1 or args.bootstrap < 0 or args.block_size < 1 or not 0 < args.site_threshold < 1:
        raise ValueError('Invalid inference/statistical parameters')
    if args.run_dir:
        selected = [args.run_dir]
    else:
        root = args.runs_root/'runs' if (args.runs_root/'runs').is_dir() else args.runs_root
        selected = []
        for path in sorted(root.glob('*/best.pt')):
            run_dir = path.parent
            if not (run_dir/'metrics.json').is_file():
                continue
            metrics = read_json(run_dir/'metrics.json')
            if metrics['seed'] not in args.seeds or (args.strategies and metrics['strategy'] not in args.strategies):
                continue
            selected.append(run_dir)
        if args.strategies:
            found = {(read_json(p/'metrics.json')['strategy'], read_json(p/'metrics.json')['seed']) for p in selected}
            missing = {(s, seed) for s in args.strategies for seed in args.seeds} - found
            if missing:
                raise ValueError(f'Requested completed runs not found: {sorted(missing)}')
    if not selected:
        raise ValueError('No completed checkpoints selected')
    for path in selected:
        if not (path/'metrics.json').is_file():
            raise ValueError(f'Checkpoint run is not completed: {path}')
        evaluate(args, path, args.out_dir if args.run_dir else args.out_dir/path.name)


if __name__ == '__main__':
    run(parser().parse_args())
