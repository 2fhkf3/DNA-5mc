#!/usr/bin/env python3
"""Balance Step-2 train/valid/test windows while keeping a depth floor.

Accepts final Step-2 outputs, verified exported intermediate snapshots, or
checksum-verified MAPQ-filtered Step-2 outputs.
Uses candidate_sites.tsv.gz or retained rows of site_coverage.tsv.gz. Source
JSONL hashes and every site's record count are checked during selection. Signal
and sequence bytes are copied unchanged; only actual_coverage is updated and
input_actual_coverage is added. Gzip members are compressed in parallel, emitted
in order, and fully decompressed again to verify counts, metadata, CRC and SHA256.
The complete staging directory is published atomically, with no overwriting.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import gzip
import hashlib
import heapq
import io
import json
from pathlib import Path
import random
import re
import sys
import time
import zlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / 'output/step2_output/chr19_bed01_cov10_15nt_site_split_9_1'
DEFAULT_OUTPUT = PROJECT_ROOT / 'output/step3_output/chr19'
SPLITS = ('train', 'valid')
PATTERNS = {name: re.compile(rb'"' + name.encode() + rb'"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+)')
            for name in ('chrom', 'center_genome_pos0', 'label', 'split', 'actual_coverage')}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--splits', nargs='+', choices=('train','valid','test'),
                   help='Default: all splits listed in the source build summary.')
    p.add_argument('--site-policy', choices=('auto','preserve_all'), default='auto',
                   help='auto may subset majority-class sites when preserving all sites would violate the depth floor.')
    p.add_argument('--min-coverage', type=int, default=10)
    p.add_argument('--workers', type=int, default=4, help='Parallel gzip compression threads.')
    p.add_argument('--chunk-mb', type=int, default=16)
    p.add_argument('--gzip-level', type=int, default=3)
    p.add_argument('--log-every', type=int, default=100000)
    return p.parse_args()


def field(line, name):
    # These fields occur near the ends of Step-2 records, away from large arrays.
    tail = name in ('label', 'split', 'actual_coverage')
    offset = max(0, len(line)-1024) if tail else 0
    small = line[offset:] if tail else line[:1024]
    match = PATTERNS[name].search(small)
    if match is None:
        offset = 0
        match = PATTERNS[name].search(line)
    if match is None:
        raise ValueError(f'Missing JSON field {name}')
    value = match.group(1)
    if value.startswith(b'"'):
        result = json.loads(value) if b'\\' in value else value[1:-1].decode('utf-8')
    else:
        result = int(value)
    return result, offset + match.start(1), offset + match.end(1)


def row_metadata(line):
    chrom = field(line, 'chrom')[0]
    position = field(line, 'center_genome_pos0')[0]
    label = field(line, 'label')[0]
    split = field(line, 'split')[0]
    coverage, start, end = field(line, 'actual_coverage')
    if label not in (0, 1) or not line.rstrip().endswith(b'}'):
        raise ValueError('Invalid binary-label JSONL record')
    return (chrom, position), label, split, coverage, start, end


def update_coverage(line, coverage, original, start, end):
    if b'"input_actual_coverage"' in line[-1024:]:
        raise ValueError('Input is already balanced or contains input_actual_coverage')
    changed = line[:start] + str(coverage).encode() + line[end:]
    close = changed.rfind(b'}')
    return (changed[:close] + b',"input_actual_coverage":' + str(original).encode()
            + changed[close:])


@contextmanager
def read_lines(path):
    with path.open('rb', buffering=1024*1024) as raw:
        with gzip.GzipFile(fileobj=raw, mode='rb') as gz:
            with io.BufferedReader(gz, buffer_size=4*1024*1024) as reader:
                yield reader


class ParallelGzipWriter:
    def __init__(self, path, workers, chunk_bytes, level):
        self.handle = path.open('wb')
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.pending = deque()
        self.buffer = bytearray()
        self.chunk_bytes, self.level = chunk_bytes, level
        self.maximum_pending = max(1, workers*2)
        self.closed = False

    def append(self, line):
        self.buffer.extend(line)
        if len(self.buffer) >= self.chunk_bytes:
            self.submit()

    def submit(self):
        if not self.buffer:
            return
        payload = bytes(self.buffer)
        self.buffer.clear()
        self.pending.append(self.pool.submit(gzip.compress, payload,
                                             compresslevel=self.level, mtime=0))
        if len(self.pending) >= self.maximum_pending:
            self.handle.write(self.pending.popleft().result())

    def close(self):
        if self.closed:
            return
        try:
            self.submit()
            while self.pending:
                self.handle.write(self.pending.popleft().result())
        finally:
            self.pool.shutdown(wait=True)
            self.handle.close()
            self.closed = True

    def abort(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.handle.close()
        self.closed = True


def allocate_quotas(depths, labels, floor, seed, split):
    """Reserve floor/site, allocate remaining slots proportionally to capacity."""
    totals = Counter()
    for key, depth in depths.items():
        if depth < floor:
            raise ValueError(f'Input site below coverage floor: {key}: {depth} < {floor}')
        totals[labels[key]] += depth
    target = min(totals[0], totals[1])
    if target == 0:
        raise ValueError(f'{split}: both labels must contain samples')
    quotas = {}
    for label in (0, 1):
        keys = sorted(key for key in depths if labels[key] == label)
        if len(keys)*floor > target:
            raise ValueError(f'{split}: cannot keep all label-{label} sites at depth >= {floor} '
                             f'and balance to {target} rows; need {len(keys)*floor}')
        if totals[label] == target:
            quotas.update((key, depths[key]) for key in keys)
            continue
        capacity = sum(depths[key]-floor for key in keys)
        wanted = target-len(keys)*floor
        remainders = []
        for key in keys:
            quotient, remainder = divmod((depths[key]-floor)*wanted, capacity)
            quotas[key] = floor+quotient
            tie = hashlib.sha256(f'{seed}:{split}:{key[0]}:{key[1]}'.encode()).digest()
            remainders.append((remainder, tie, key))
        extra = target-sum(quotas[key] for key in keys)
        for _, _, key in sorted(remainders, reverse=True)[:extra]:
            quotas[key] += 1
    if sum(quotas.values()) != 2*target:
        raise RuntimeError('Quota allocation is not balanced')
    return quotas


def choose_site_quotas(depths, labels, targets, floor, seed):
    if floor < 1:
        raise ValueError('--min-coverage must be >= 1')
    quotas = {}
    for label, target in targets.items():
        keys = sorted(key for key in depths if labels[key] == label and depths[key] >= floor)
        if target < floor or sum(depths[key] for key in keys) < target:
            raise ValueError(f'label={label}: insufficient eligible coverage for target={target}, floor={floor}')
        rng = random.Random(f'{seed}:label:{label}:sites')
        rng.shuffle(keys)
        chosen, capacity = [], 0
        for key in keys:
            chosen.append(key)
            capacity += depths[key]
            if capacity >= target:
                break
        if len(chosen)*floor > target:
            # With tiny sites, replacing low-capacity sites can make exact quotas
            # feasible without selecting more sites than the depth floor permits.
            max_sites = target//floor
            chosen = keys[:max_sites]
            capacity = sum(depths[key] for key in chosen)
            heap = [(depths[key], rank, key) for rank, key in enumerate(chosen)]
            heapq.heapify(heap)
            for rank, key in enumerate(keys[max_sites:], max_sites):
                if capacity >= target:
                    break
                if depths[key] > heap[0][0]:
                    old, _, _ = heapq.heapreplace(heap, (depths[key], rank, key))
                    capacity += depths[key]-old
            if capacity < target:
                raise ValueError(f'label={label}: exact target cannot satisfy the coverage floor')
            chosen = [key for _, _, key in sorted(heap, key=lambda entry:entry[1])]
        local = {key:depths[key] for key in chosen}
        excess = capacity-target
        for key in reversed(chosen):
            removed = min(excess, local[key]-floor)
            local[key] -= removed
            excess -= removed
            if not excess:
                break
        if excess or sum(local.values()) != target or min(local.values()) < floor:
            raise RuntimeError('Site quota plan does not meet exact count/coverage constraints')
        quotas.update(local)
    return quotas



def stats(depths, labels):
    rows, sites, histogram = Counter(), Counter(), Counter()
    by_label = {0: Counter(), 1: Counter()}
    for key, depth in depths.items():
        if depth:
            label = labels[key]
            rows[label] += depth
            sites[label] += 1
            histogram[depth] += 1
            by_label[label][depth] += 1
    return {'records': sum(rows.values()),
            'written_by_label': {str(k): rows[k] for k in (0, 1)},
            'unique_sites': sum(sites.values()),
            'sites_by_label': {str(k): sites[k] for k in (0, 1)},
            'minimum_actual_coverage': min(depths.values()),
            'actual_coverage_histogram': {str(k): v for k, v in sorted(histogram.items())},
            'actual_coverage_histogram_by_label': {
                str(label): {str(k): v for k, v in sorted(by_label[label].items())} for label in (0, 1)}}


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def select_split(split, source, output, final_path, depths, quotas, labels, source_stats, args, stage):
    remaining, wanted = depths.copy(), quotas.copy()
    actual = Counter()
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    rng = random.Random(f'{args.seed}:{split}:within_site')
    writer = ParallelGzipWriter(output, args.workers, args.chunk_mb*1024*1024, args.gzip_level)
    count = kept = 0
    start_stat = source.stat()
    started = time.monotonic()
    try:
        with read_lines(source) as reader:
            for count, line in enumerate(reader, 1):
                source_hash.update(line)
                key, label, row_split, original, a, b = row_metadata(line)
                if key not in depths or row_split != split or label != labels[key] or original != depths[key]:
                    raise ValueError(f'{source}:{count}: metadata disagrees with source site table')
                available, needed = remaining[key], wanted.get(key,0)
                if available <= 0:
                    raise ValueError(f'{source}:{count}: site has more rows than declared coverage')
                keep = needed == available or (needed > 0 and rng.randrange(available) < needed)
                if keep:
                    updated = update_coverage(line, quotas[key], original, a, b)
                    writer.append(updated)
                    output_hash.update(updated)
                    wanted[key] -= 1
                    actual[key] += 1
                    kept += 1
                remaining[key] -= 1
                if args.log_every and count % args.log_every == 0:
                    progress = {'phase': 'write', 'split': split, 'scanned': count,
                                'input_records': source_stats['records'], 'kept': kept,
                                'percent': round(100*count/source_stats['records'], 2),
                                'seconds': round(time.monotonic()-started, 1)}
                    atomic_json(progress, stage/'progress.json')
                    print('WRITE '+json.dumps(progress), flush=True)
        writer.close()
    except BaseException:
        if not writer.closed:
            writer.abort()
        raise
    end_stat = source.stat()
    if (start_stat.st_size, start_stat.st_mtime_ns) != (end_stat.st_size, end_stat.st_mtime_ns):
        raise RuntimeError(f'Input changed while reading: {source}')
    if count != source_stats['records'] or source_hash.hexdigest() != source_stats['uncompressed_sha256']:
        raise ValueError(f'Input count/SHA256 differs from verified Step-2 summary: {source}')
    if any(remaining.values()) or any(wanted.values()) or actual != quotas:
        raise RuntimeError(f'{split}: selected counts do not match quotas')
    result = stats(actual, labels)
    result.update(output_path=str(final_path), uncompressed_sha256=output_hash.hexdigest(),
                  compressed_bytes=output.stat().st_size,
                  removed_by_label={str(label): source_stats['written_by_label'][str(label)]
                                    -result['written_by_label'][str(label)] for label in (0, 1)})
    return result


def verify_split(split, path, quotas, depths, labels, expected, args, stage):
    counted = Counter()
    digest = hashlib.sha256()
    with read_lines(path) as reader:
        for count, line in enumerate(reader, 1):
            digest.update(line)
            key, label, row_split, coverage, _, _ = row_metadata(line)
            if key not in quotas or label != labels[key] or row_split != split or coverage != quotas[key]:
                raise ValueError(f'Output metadata disagrees with quotas: {path}:{count}')
            marker = re.search(rb'"input_actual_coverage"\s*:\s*(\d+)', line[-1024:])
            if not marker or int(marker.group(1)) != depths[key]:
                raise ValueError(f'Output lost original coverage: {path}:{count}')
            counted[key] += 1
            if args.log_every and count % args.log_every == 0:
                progress = {'phase': 'verify', 'split': split, 'records': count,
                            'expected_records': expected['records'],
                            'percent': round(100*count/expected['records'], 2)}
                atomic_json(progress, stage/'progress.json')
                print('VERIFY '+json.dumps(progress), flush=True)
    if counted != quotas or digest.hexdigest() != expected['uncompressed_sha256']:
        raise RuntimeError(f'Output SHA256/site counts mismatch: {path}')
    if expected['written_by_label']['0'] != expected['written_by_label']['1']:
        raise RuntimeError(f'{split}: output classes are not balanced')
    print(f'VERIFIED {split}: rows={sum(counted.values()):,}, sites={len(counted):,}, '
          f'minimum coverage={min(counted.values())}', flush=True)


def source_verification_mode(summary):
    """Recognize producer-specific verification schemas; keep integrity checks."""
    required = summary.get('verification', {})
    final_verified = required.get('all_output_rows_reparsed') and required.get('cross_split_sites') == 0
    snapshot_verified = (summary.get('is_intermediate') and required.get('gzip_crc_checked')
                         and (required.get('sha256_matched') or required.get('output_sha256_matched'))
                         and required.get('output_rows_equal_snapshot_distinct_coverage'))
    filtered_verified = (
        summary.get('format') == 'step3_mapq_actual_coverage_filtered_jsonl_v1'
        and str(summary.get('source_format', '')).startswith('step2_')
        and required.get('cross_split_sites') == 0
        and all(required.get(name) is True for name in (
            'gzip_crc_checked', 'output_sha256_matched',
            'all_input_counts_and_sha256_matched',
            'all_output_metadata_and_site_counts_checked', 'mapq_at_least_threshold'))
    )
    if required.get('original_read_site_duplicates') == 0:
        if filtered_verified:
            return 'mapq_filtered_step2'
        if snapshot_verified:
            return 'exported_snapshot'
        if final_verified:
            return 'final_step2'
    raise ValueError(
        'Input verification schema is unsupported or incomplete. Requires a verified '
        'final Step-2 dataset, exported snapshot, or MAPQ-filtered Step-2 dataset '
        'with CRC/SHA256, unique read/site pairs and site-depth verification. '
        f"Input format: {summary.get('format')!r}; source_format: {summary.get('source_format')!r}"
    )


def run(args):
    if args.min_coverage < 1 or args.workers < 1 or args.chunk_mb < 1 or args.log_every < 0:
        raise ValueError('Invalid minimum coverage/workers/chunk size/log interval')
    if not 0 <= args.gzip_level <= 9:
        raise ValueError('gzip level must be in [0,9]')
    source_root, destination = args.input_dir.resolve(), args.out_dir.resolve()
    stage = destination.with_name(destination.name+'.inprogress')
    if destination.exists() or stage.exists():
        raise FileExistsError(f'Destination or staging directory exists: {destination}')
    source_summary_path = source_root/'build_summary.json'
    source_summary = json.loads(source_summary_path.read_text())
    verification_mode = source_verification_mode(source_summary)
    print(f'[INFO] input verification mode: {verification_mode}', flush=True)
    available_splits = source_summary.get('write_stats',{})
    splits = tuple(getattr(args,'splits',None) or
                   [name for name in ('train','valid','test') if name in available_splits])
    if not splits or len(set(splits)) != len(splits):
        raise ValueError('No input splits found, or duplicate requested splits')
    for split in splits:
        if split not in available_splits or not (source_root/f'{split}.jsonl.gz').is_file():
            raise FileNotFoundError(f'Missing source split/statistics: {split}')
    table_path = source_root/'candidate_sites.tsv.gz'
    if not table_path.is_file():
        table_path = source_root/'site_coverage.tsv.gz'
    if not table_path.is_file():
        raise FileNotFoundError('No candidate_sites.tsv.gz or site_coverage.tsv.gz found')
    window_nt = source_summary.get('window_nt')
    if window_nt is None:
        with read_lines(source_root/f'{splits[0]}.jsonl.gz') as reader:
            first_row = json.loads(next(reader))
        window_nt = first_row.get('window_size') or len(first_row.get('ref','')) or None
    tables = {split:{} for split in splits}
    labels, site_split, bed_info = {}, {}, {}
    with gzip.open(table_path,'rt') as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            if row['split'] not in tables or int(row.get('retained','1')) == 0:
                continue
            key = (row['chrom'],int(row['genome_pos0']))
            label, split, depth = int(row['label']),row['split'],int(row['actual_coverage'])
            if label not in (0,1) or key in site_split:
                raise ValueError(f'Invalid/duplicate/cross-split genomic site: {key}')
            tables[split][key] = depth
            labels[key],site_split[key] = label,split
            bed_info[key] = (row.get('bed_coverage',''),row.get('bed_percent',''))
    quotas,plans = {},{}
    for split in splits:
        original = stats(tables[split],labels)
        expected = source_summary['write_stats'][split]
        for name in ('records','written_by_label','unique_sites','sites_by_label'):
            if name in expected and original[name] != expected[name]:
                raise ValueError(f'{split}: source site table differs from source summary: {name}')
        eligible = {key:depth for key,depth in tables[split].items() if depth >= args.min_coverage}
        if not eligible:
            raise ValueError(f'{split}: no sites reach coverage >= {args.min_coverage}')
        eligible_stats = stats(eligible,labels)
        target = min(eligible_stats['written_by_label'].values())
        infeasible_all_sites = any(eligible_stats['sites_by_label'][str(label)]*args.min_coverage > target
                                   for label in (0,1))
        policy = getattr(args,'site_policy','auto')
        if infeasible_all_sites and policy == 'auto':
            print(f'[INFO] {split}: retaining all sites at coverage >= {args.min_coverage} cannot '
                  f'meet 1:1; selecting a majority-class site subset to {target:,} rows/class',flush=True)
            quotas[split] = choose_site_quotas(eligible,labels,{0:target,1:target},args.min_coverage,args.seed)
            sampling_mode = 'random_site_subset_with_depth_floor'
        else:
            quotas[split] = allocate_quotas(eligible,labels,args.min_coverage,args.seed,split)
            sampling_mode = 'all_eligible_sites_preserved'
        plans[split] = {'input':original,'after_coverage_filter':eligible_stats,
                       'planned_output':stats(quotas[split],labels),'sampling_mode':sampling_mode,
                       'dropped_sites':len(tables[split])-len(quotas[split])}
        print('PLAN '+json.dumps({'split':split,'input_rows':original['records'],
                                  'output_rows':plans[split]['planned_output']['records'],
                                  'output_by_label':plans[split]['planned_output']['written_by_label'],
                                  'input_sites':len(tables[split]),'retained_sites':len(quotas[split])}),flush=True)
    stage.mkdir(parents=True)
    started = time.monotonic()
    atomic_json({'seed': args.seed, 'min_coverage': args.min_coverage, 'splits': plans}, stage/'balance_plan.json')
    written = {}
    for split in splits:
        written[split] = select_split(split, source_root/f'{split}.jsonl.gz', stage/f'{split}.jsonl.gz',
                                      destination/f'{split}.jsonl.gz', tables[split], quotas[split], labels,
                                      source_summary['write_stats'][split], args, stage)
    for split in splits:
        verify_split(split, stage/f'{split}.jsonl.gz', quotas[split], tables[split], labels,
                     written[split], args, stage)
    with gzip.open(stage/'site_coverage.tsv.gz','wt',compresslevel=3) as writer:
        writer.write('chrom\tgenome_pos0\tlabel\tsplit\tinput_actual_coverage\tactual_coverage\tremoved_records\tretained\n')
        for key in sorted(labels):
            split = site_split[key]; original = tables[split][key]; kept = quotas[split].get(key,0)
            writer.write(f'{key[0]}\t{key[1]}\t{labels[key]}\t{split}\t{original}\t{kept}\t{original-kept}\t{int(kept>0)}\n')
    with gzip.open(stage/'candidate_sites.tsv.gz','wt',compresslevel=3) as writer:
        writer.write('chrom\tgenome_pos0\tlabel\tbed_coverage\tbed_percent\tactual_coverage\tsplit\tinput_actual_coverage\n')
        for key in sorted(labels):
            split = site_split[key]; kept = quotas[split].get(key,0)
            if not kept:
                continue
            bed,pct = bed_info[key]
            writer.write(f'{key[0]}\t{key[1]}\t{labels[key]}\t{bed}\t{pct}\t{kept}\t{split}\t{tables[split][key]}\n')
    total = sum(x['records'] for x in written.values())
    summary = {
        'format': 'step3_depth_preserving_balanced_windows_jsonl_v2',
        'input_dir': str(source_root), 'out_dir': str(destination),
        'source_summary': str(source_summary_path), 'seed': args.seed,
        'splits':list(splits),'source_site_table':str(table_path),
        'is_intermediate':bool(source_summary.get('is_intermediate',False)),
        'input_scan_complete':source_summary.get('input_scan_complete',source_summary.get('input',{}).get('input_scan_complete',True)),
        'source_snapshot':source_summary.get('source_snapshot'),
        'site_policy':getattr(args,'site_policy','auto'),
        'minimum_actual_coverage_after_balancing': args.min_coverage,
        'balance_definition': 'Equal label-0/label-1 window counts in each split; no oversampling',
        'sampling': 'Retain minority class; preserve all eligible sites if feasible, otherwise sample a majority-class site subset; '
                    'allocate surplus by original remaining capacity with largest remainders and seeded hash ties; '
                    'uniform sampling without replacement within sites',
        'split_strategy': 'Preserve original train/valid/test assignment; no reshuffling',
        'coverage_definition': 'Retained distinct original read windows per genomic site',
        'coverage_fields': {'actual_coverage': 'post-balancing retained depth',
                            'input_actual_coverage': 'direct input retained depth before balancing',
                            'bed_coverage': 'unchanged original BED coverage'},
        'signal_processing': 'Signal/sequence/span bytes preserved; no normalization or re-extraction',
        'window_nt':window_nt, 'label_definition':source_summary.get('label_definition') or 'Input label 0/1 preserved without relabeling',
        'input': plans, 'write_stats': written,
        'overall': {'total_written': total,
                    'written_by_label': {str(k): sum(x['written_by_label'][str(k)] for x in written.values()) for k in (0, 1)},
                    'unique_sites': sum(len(q) for q in quotas.values()),
                    'removed_records': sum(source_summary['write_stats'][s]['records'] for s in splits)-total},
        'verification': {'balanced_in_each_split': True, 'cross_split_sites': 0,
                         'all_input_counts_and_sha256_matched': True,
                         'all_output_metadata_and_site_counts_checked': True,
                         'gzip_crc_checked': True, 'output_sha256_matched': True,
                         'all_source_sites_retained':all(len(quotas[s])==len(tables[s]) for s in splits),
                         'minimum_actual_coverage': min(min(q.values()) for q in quotas.values()),
                         'original_read_site_duplicates': 0,
                         'duplicate_verification': 'inherited from checksum-verified source; output is a subset',
                         'source_verification_mode': verification_mode},
        'compression': {'workers': args.workers, 'chunk_mb': args.chunk_mb,
                        'gzip_level': args.gzip_level, 'ordered_concatenated_gzip_members': True,
                        'gzip_member_mtime': 0, 'zlib_version': zlib.ZLIB_VERSION},
        'python_version': sys.version.split()[0], 'seconds': round(time.monotonic()-started, 2),
    }
    atomic_json(summary, stage/'build_summary.json')
    (stage/'progress.json').unlink(missing_ok=True)
    stage.rename(destination)
    print('COMPLETE '+json.dumps(summary['overall']), flush=True)
    return summary


if __name__ == '__main__':
    run(parse_args())
