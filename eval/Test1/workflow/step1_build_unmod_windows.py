#!/usr/bin/env python3
"""Extract independent all-negative WGA windows using the original Step2 validator."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from collections import Counter
from pathlib import Path

from common import PROJECT, RAW_INPUT, module, open_text, output_directory, signature, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=RAW_INPUT, help='已经预处理的全 read JSONL，不再次归一化')
    p.add_argument('--out-dir', type=Path, required=True, help='新数据目录；不覆盖已有结果')
    p.add_argument('--chrom', default='chr19', help='目标染色体，与参考/比对命名一致')
    p.add_argument('--motif', choices=['CG', 'C'], default='CG', help='CG 为 CpG；C 为所有胞嘧啶')
    p.add_argument('--min-mapq', type=int, default=30, help='最小 read MAPQ；应匹配原训练数据')
    p.add_argument('--min-coverage', type=int, default=1, help='去重和质量过滤后的最小正链 read 覆盖度')
    p.add_argument('--flank', type=int, default=7, help='中心左右碱基数；7 对应 15 nt')
    p.add_argument('--library-id', default='WGA_UNMOD_chr19', help='独立阴性文库标识，进入样本匹配键')
    p.add_argument('--candidate-sites', type=Path, help='可选 chrom/0-based position TSV，不读取甲基化预测')
    p.add_argument('--max-records', type=int, help='仅扫描前 N 条输入 read，用于测试')
    p.add_argument('--max-windows', type=int, help='最多保留 N 个去重窗口，用于测试，非正式覆盖度')
    p.add_argument('--log-every', type=int, default=1000, help='扫描日志间隔，单位为输入 read')
    return p


def run(args):
    if args.min_coverage < 1 or args.flank < 2 or args.min_mapq < 0 or args.log_every < 1:
        raise ValueError('Invalid coverage/flank/MAPQ/log interval')
    if any(v is not None and v < 1 for v in (args.max_records, args.max_windows)):
        raise ValueError('Test limits must be positive')
    if not args.library_id.strip() or any(c in args.library_id for c in '\t\r\n'):
        raise ValueError('Invalid library ID')
    original = module('_test1_window_builder', PROJECT/'workflow/step2_build_chrom19_dataset.py')
    before = signature(args.input)
    candidates = None
    if args.candidate_sites:
        candidates = set()
        with open_text(args.candidate_sites) as f:
            for line in f:
                fields = line.strip().split('\t')
                if fields and fields[0] == args.chrom and len(fields) >= 2:
                    candidates.add(int(fields[1]))
    skipped = Counter()
    counts = Counter()
    unique = scanned = duplicates = 0
    partial = False
    args.center_base = 'C'
    args.test_only = True
    args.split_seed = '42'
    args.train_ratio = .9
    with output_directory(args.out_dir) as stage:
        db = sqlite3.connect(stage/'sample_audit.sqlite')
        db.execute('CREATE TABLE audit (chrom TEXT, position INTEGER, read_id TEXT, mapq INTEGER, '
                   'signal_len INTEGER, PRIMARY KEY(chrom, position, read_id)) WITHOUT ROWID')
        try:
            with open_text(args.input) as reader, gzip.open(stage/'windows.spool.jsonl.gz', 'wt', compresslevel=1) as spool:
                for line in reader:
                    if not line.strip():
                        continue
                    if args.max_records and scanned >= args.max_records:
                        partial = True
                        break
                    scanned += 1
                    record = original.loads(line)
                    if record.get('chrom') != args.chrom:
                        skipped['chrom'] += 1
                        continue
                    if record.get('strand') != '+':
                        skipped['read_strand'] += 1
                        continue
                    if int(record.get('mapq', -1)) < args.min_mapq:
                        skipped['mapq'] += 1
                        continue
                    by_pos = {}
                    for item in record.get('per_base', []):
                        if item.get('chrom') == args.chrom and 'genome_pos0' in item:
                            pos = int(item['genome_pos0'])
                            if pos in by_pos and by_pos[pos] != item:
                                raise ValueError(f'Ambiguous per-base mapping at {args.chrom}:{pos}')
                            by_pos[pos] = item
                    for pos, item in sorted(by_pos.items()):
                        if item.get('ref_base', '').upper() != 'C' or (candidates is not None and pos not in candidates):
                            continue
                        if args.motif == 'CG' and by_pos.get(pos+1, {}).get('ref_base', '').upper() != 'G':
                            continue
                        sample, reason = original.extract_window(record, pos, by_pos, original.BedSite(0, 0, 0.), args)
                        if sample is None:
                            skipped[reason] += 1
                            continue
                        # WGA identity supplies the truth. Do not invent measured bedMethyl statistics.
                        sample.pop('bed_coverage', None)
                        sample.pop('bed_percent', None)
                        sample.update(label=0, source='unmod', library_id=args.library_id,
                                      truth_source='WGA_negative_control', split='test')
                        inserted = db.execute('INSERT OR IGNORE INTO audit VALUES (?,?,?,?,?)',
                                              (args.chrom, pos, sample['read_id'], sample['mapq'], sample['signal_len'])).rowcount
                        if not inserted:
                            duplicates += 1
                            continue
                        counts[pos] += 1  # Coverage counts only unique original reads.
                        spool.write(original.dumps(sample).decode()+'\n')
                        unique += 1
                        if args.max_windows and unique >= args.max_windows:
                            partial = True
                            break
                    if scanned % args.log_every == 0:
                        db.commit()
                        print(f'[SCAN] reads={scanned} windows={unique}', flush=True)
                    if args.max_windows and unique >= args.max_windows:
                        break
            db.commit()
            retained = {p for p, n in counts.items() if n >= args.min_coverage}
            if not retained:
                raise ValueError('No valid positive-strand windows; inspect motif, MAPQ, spans and test limits')
            emitted = 0
            with gzip.open(stage/'windows.spool.jsonl.gz', 'rt') as spool, gzip.open(stage/'test.jsonl.gz', 'wt', compresslevel=3) as out:
                for line in spool:
                    sample = original.loads(line)
                    pos = sample['center_genome_pos0']
                    if pos in retained:
                        sample['actual_coverage'] = counts[pos]
                        out.write(original.dumps(sample).decode()+'\n')
                        emitted += 1
            (stage/'windows.spool.jsonl.gz').unlink()
            with (stage/'sites.tsv').open('w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['chrom', 'position', 'label', 'n_reads', 'retained'])
                writer.writerows((args.chrom, p, 0, n, int(p in retained)) for p, n in sorted(counts.items()))
            if signature(args.input) != before:
                raise ValueError('Input changed during extraction')
            write_json(stage/'dataset_manifest.json', dict(
                truth_source='WGA_negative_control', library_id=args.library_id, chrom=args.chrom,
                strand='+', motif=args.motif, window_size=2*args.flank+1,
                min_mapq=args.min_mapq, min_coverage=args.min_coverage,
                label_counts={'0': emitted, '1': 0}, candidate_policy='reference_bases_in_read_alignment',
                candidate_sites=signature(args.candidate_sites) if args.candidate_sites else None,
                input=before, reads_scanned=scanned, unique_windows_before_coverage=unique,
                records=emitted, sites=len(retained), duplicate_windows_removed=duplicates,
                skipped=dict(skipped), partial_test=partial, input_scan_complete=not partial,
                max_records=args.max_records, max_windows=args.max_windows,
                preprocessing='Input is already preprocessed; no normalization performed'))
        finally:
            db.close()
    print(f'[DONE] windows={emitted} sites={len(retained)} partial_test={partial}', flush=True)


if __name__ == '__main__':
    run(parser().parse_args())
