#!/usr/bin/env python3
"""Summarize completed Test1 runs and compare backbones on identical observations."""
from __future__ import annotations

import argparse
import itertools
import sqlite3
from collections import defaultdict
from pathlib import Path

from common import VERSIONS, install_sql_math, output_directory, read_json, summarize, tsv, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, required=True, help='包含 V011/V543/V610 子目录的推理结果根目录')
    p.add_argument('--out-dir', type=Path, required=True, help='新的汇总目录，禁止覆盖')
    p.add_argument('--versions', nargs='+', choices=VERSIONS, default=list(VERSIONS), help='进行共同样本比较的主干版本')
    p.add_argument('--bootstrap', type=int, default=1000, help='配对区块 bootstrap 次数；0 禁用')
    p.add_argument('--block-size', type=int, default=1_000_000, help='基因组区块长度 bp')
    p.add_argument('--bootstrap-seed', type=int, default=42, help='配对重采样随机种子')
    return p


def paired_interval(blocks, repetitions, seed):
    import numpy as np
    data = np.asarray(blocks, dtype=float)
    if len(data) < 2 or not repetitions or not np.any(data[:, 0]):
        return None
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        delta, n = data[rng.integers(len(data), size=len(data))].sum(axis=0)
        if n:
            differences.append(delta/n)
    return np.quantile(differences, [.025, .975]).tolist() if differences else None


def run(args):
    import numpy as np
    if args.bootstrap < 0 or args.block_size < 1 or len(set(args.versions)) != len(args.versions):
        raise ValueError('Invalid comparison settings')
    runs = []
    groups = defaultdict(dict)
    for path in sorted(args.results_dir.glob('*/*/metrics.json')):
        metrics = read_json(path)
        if metrics['version'] not in args.versions:
            continue
        row = dict(version=metrics['version'], strategy=metrics['strategy'], seed=metrics['seed'],
                   partial_test=metrics['partial_test'], read_threshold=metrics['thresholds']['read'],
                   site_threshold=metrics['thresholds']['site'],
                   n_read=metrics['read_level']['n'], fp_read=metrics['read_level']['fp'],
                   read_fpr=metrics['read_level']['fpr'], n_site=metrics['site_level']['n'],
                   fp_site=metrics['site_level']['fp'], site_fpr=metrics['site_level']['fpr'])
        runs.append(row)
        key = (metrics['strategy'], metrics['seed'])
        if metrics['version'] in groups[key]:
            raise ValueError('Duplicate version/strategy/seed results')
        groups[key][metrics['version']] = (path.parent, metrics)
    if not runs:
        raise ValueError('No completed Test1 runs found')
    if len({row['partial_test'] for row in runs}) != 1:
        raise ValueError('Do not mix partial smoke tests and formal results')
    fields = list(runs[0])
    with output_directory(args.out_dir) as stage:
        tsv(stage/'summary.tsv', runs, fields)
        aggregates = []
        by_head = defaultdict(list)
        for row in runs:
            by_head[row['version'], row['strategy']].append(row)
        for (version, strategy), values in sorted(by_head.items()):
            row = dict(version=version, strategy=strategy, seeds=','.join(str(v['seed']) for v in values), n_seeds=len(values))
            for metric in ['read_fpr', 'site_fpr']:
                array = [v[metric] for v in values]
                row[metric+'_mean'] = float(np.mean(array))
                row[metric+'_sd'] = float(np.std(array, ddof=1)) if len(array)>1 else None
            aggregates.append(row)
        tsv(stage/'aggregate.tsv', aggregates, list(aggregates[0]))
        common_rows, differences, incomplete = [], [], []
        for (strategy, seed), group in sorted(groups.items()):
            missing = set(args.versions)-set(group)
            if missing:
                incomplete.append(dict(strategy=strategy, seed=seed, missing=sorted(missing)))
                continue
            if len(args.versions)<2:
                continue
            db = sqlite3.connect(':memory:', uri=True)
            install_sql_math(db)
            try:
                for i, version in enumerate(args.versions):
                    path = (group[version][0]/'predictions.sqlite').resolve()
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    db.execute(f'ATTACH DATABASE ? AS m{i}', (path.as_uri()+'?mode=ro',))
                joins = ' '.join(f'JOIN m{i}.predictions p{i} ON '+ ' AND '.join(
                    f'p{i}.{key}=p0.{key}' for key in ['library_id', 'read_id', 'chrom', 'position', 'strand'])
                    for i in range(1,len(args.versions)))
                extras = ''.join(f', p{i}.prob AS prob{i}' for i in range(1,len(args.versions)))
                db.execute('CREATE TEMP VIEW common AS SELECT p0.*, p0.prob AS prob0'+extras+
                           ' FROM m0.predictions p0 '+joins)
                for i in range(1,len(args.versions)):
                    mismatch = db.execute('SELECT COUNT(*) FROM m0.predictions p0 '+joins+
                                          f' WHERE p{i}.ref != p0.ref OR p{i}.seq != p0.seq').fetchone()[0]
                    if mismatch:
                        raise ValueError('Common observations have inconsistent sequence metadata')
                count = db.execute('SELECT COUNT(*) FROM common').fetchone()[0]
                if not count:
                    raise ValueError(f'No common observations for {strategy}/seed{seed}')
                for i, version in enumerate(args.versions):
                    original = group[version][1]
                    metrics = summarize(db, table='common', prob=f'prob{i}',
                                        read_threshold=original['thresholds']['read'],
                                        site_threshold=original['thresholds']['site'],
                                        bootstrap=args.bootstrap, block_size=args.block_size, seed=args.bootstrap_seed)
                    common_rows.append(dict(version=version, strategy=strategy, seed=seed,
                                            n_read=metrics['read_level']['n'], fp_read=metrics['read_level']['fp'],
                                            read_fpr=metrics['read_level']['fpr'], n_site=metrics['site_level']['n'],
                                            fp_site=metrics['site_level']['fp'], site_fpr=metrics['site_level']['fpr']))
                    write_json(stage/f'common_{version}_{strategy}_seed{seed}.json', metrics)
                for i, j in itertools.combinations(range(len(args.versions)), 2):
                    a, b = args.versions[i], args.versions[j]
                    ta, tb = group[a][1]['thresholds'], group[b][1]['thresholds']
                    for level in ['read', 'site']:
                        if level == 'read':
                            query = (f'SELECT SUM((prob{j}>=?)-(prob{i}>=?)), COUNT(*) FROM common '
                                     'GROUP BY library_id,chrom,CAST(position/? AS INTEGER)')
                        else:
                            query = (f'SELECT SUM((b>=?)-(a>=?)), COUNT(*) FROM '
                                     f'(SELECT library_id,chrom,position,AVG(prob{i}) a,AVG(prob{j}) b FROM common '
                                     'GROUP BY library_id,chrom,position,strand) '
                                     'GROUP BY library_id,chrom,CAST(position/? AS INTEGER)')
                        blocks = list(db.execute(query, (tb[level], ta[level], args.block_size)))
                        delta = sum(v[0] for v in blocks)/sum(v[1] for v in blocks)
                        differences.append(dict(strategy=strategy, seed=seed, level=level, version_a=a, version_b=b,
                                                fpr_b_minus_a=delta, paired_block_ci95=paired_interval(
                                                    blocks,args.bootstrap,args.bootstrap_seed)))
            finally:
                db.close()
        if common_rows:
            tsv(stage/'common_summary.tsv', common_rows, list(common_rows[0]))
        write_json(stage/'comparison_manifest.json', dict(
            versions=args.versions, partial_test=runs[0]['partial_test'], incomplete_groups=incomplete,
            paired_differences=differences, common_key=['library_id','read_id','chrom','position','strand'],
            note='FPRs are proportions, not percentages. SD across seeds is not biological uncertainty. '
                 'Null paired CIs indicate disabled/degenerate/insufficient-block resampling.'))
    print(f'[DONE] summary runs={len(runs)} common rows={len(common_rows)}', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
