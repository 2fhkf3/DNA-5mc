#!/usr/bin/env python3
"""Build BED-labelled 15-nt windows with distinct-read coverage filtering.

The input signal is reused without further normalization. Scan the large input
once, validate and deduplicate windows, then filter the compressed window spool
by actual coverage and assign train/valid by a deterministic genomic-site hash,
or assign all retained windows to test with --test-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

try:
    import orjson
except ImportError:
    orjson = None


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_JSONL = PROJECT / 'output/step1_output/s2_all_reads.genome_signal.preprocessed-2.jsonl'
DEFAULT_BED = PROJECT / 'output/step2_input/bed/HG002_250F500490011.bed.gz'
DEFAULT_OUTPUT = PROJECT / 'output/step2_output/chr19_bed01_cov10_15nt_site_split_9_1'
SPLITS = ('train', 'valid')
CHUNK_SUFFIX = re.compile(r'^(.*):[1-9]\d*:[1-9]\d*$')


@dataclass(frozen=True)
class BedSite:
    label: int
    coverage: int
    percent: float


def dumps(record: dict) -> bytes:
    if orjson is not None:
        return orjson.dumps(record)
    return json.dumps(record, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def loads(line: bytes) -> dict:
    return orjson.loads(line) if orjson is not None else json.loads(line)


def open_input(path: Path):
    if path.suffix == '.gz':
        return gzip.open(path, 'rb')
    return path.open('rb', buffering=8 * 1024 * 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jsonl', '--mod-jsonl', dest='jsonl', type=Path, default=DEFAULT_JSONL,
                   help='Single MOD input; both labels come from the BED, not MOD/UNMOD sources.')
    p.add_argument('--bed-gz', type=Path, default=DEFAULT_BED)
    p.add_argument('--candidate-sites', type=Path,
                   help='Optional chrom/position TSV[.gz] to intersect with BED-filtered sites; never bypasses BED filters.')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--chrom', default='chr19')
    p.add_argument('--mod-type', default='m')
    p.add_argument('--bed-strand', choices=['+'], default='+')
    p.add_argument('--read-strand', choices=['+'], default='+')
    p.add_argument('--center-base', choices=['C'], default='C')
    p.add_argument('--flank', type=int, default=7)
    p.add_argument('--min-coverage', '--min-bed-coverage', dest='min_coverage', type=int, default=10)
    p.add_argument('--coverage-op', choices=['ge'], default='ge')
    p.add_argument('--min-actual-coverage', type=int, default=10)
    p.add_argument('--min-mapq', type=int, default=30)
    p.add_argument('--test-only', action='store_true',
                   help='Write every retained window to test only; train/valid ratios and split seed are ignored.')
    p.add_argument('--train-ratio', type=float, default=0.9)
    p.add_argument('--valid-ratio', type=float, default=0.1)
    p.add_argument('--split-seed', default='42')
    p.add_argument('--max-records', type=int,
                   help='Smoke-run limit; actual coverage is computed only in the scanned subset.')
    p.add_argument('--plain-jsonl', action='store_true')
    p.add_argument('--gzip-level', type=int, choices=range(1, 10), default=3)
    p.add_argument('--log-every', type=int, default=1000)
    p.add_argument('--force', action='store_true',
                   help='Publish over an existing result after validation, retaining the old directory as a backup.')
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.flank < 0 or args.min_coverage < 1 or args.min_actual_coverage < 1:
        raise SystemExit('flank must be non-negative and coverage thresholds must be positive')
    if not args.test_only:
        if not (0 < args.train_ratio < 1 and 0 < args.valid_ratio < 1):
            raise SystemExit('Both train and valid ratios must be between 0 and 1')
        if not math.isclose(args.train_ratio + args.valid_ratio, 1.0, abs_tol=1e-9):
            raise SystemExit('--train-ratio + --valid-ratio must equal 1; use --test-only for an independent test set')
    if args.min_mapq < 0 or args.log_every < 1 or (args.max_records is not None and args.max_records < 1):
        raise SystemExit('Invalid MAPQ, logging interval or record limit')
    for path in [args.jsonl, args.bed_gz, args.candidate_sites]:
        if path is not None and not path.is_file():
            raise SystemExit(f'Input file does not exist: {path}')
    if args.out_dir.exists() and (not args.out_dir.is_dir() or not args.force):
        raise SystemExit(f'Output exists: {args.out_dir}; choose a new path or use --force')
    if any(args.out_dir.resolve() in path.resolve().parents
           for path in [args.jsonl, args.bed_gz, args.candidate_sites] if path is not None):
        raise SystemExit('Output directory cannot contain an input file')


def load_candidates(args: argparse.Namespace) -> tuple[dict[int, BedSite], dict]:
    sites = {}
    skipped = Counter()
    lines = 0
    with open_input(args.bed_gz) as reader:
        for line in reader:
            lines += 1
            if not line.strip() or line.startswith(b'#'):
                continue
            fields = line.decode('utf-8').rstrip('\r\n').split('\t')
            if len(fields) < 11:
                skipped['too_few_columns'] += 1
                continue
            if fields[0] != args.chrom or fields[5] != args.bed_strand or fields[3] != args.mod_type:
                skipped['chrom_strand_or_mod_type'] += 1
                continue
            try:
                pos = int(fields[1])
                coverage_value = float(fields[9])
                coverage = int(coverage_value)
                percent = float(fields[10])
            except (ValueError, OverflowError):
                skipped['parse_error'] += 1
                continue
            if coverage_value != coverage or coverage < args.min_coverage or pos < 0:
                skipped['coverage_or_coordinate'] += 1
                continue
            if percent not in (0.0, 100.0):
                skipped['percent_not_0_or_100'] += 1
                continue
            site = BedSite(int(percent == 100.0), coverage, percent)
            if pos in sites and sites[pos] != site:
                raise ValueError(f'Conflicting BED entries at {args.chrom}:{pos}')
            sites[pos] = site
    if args.candidate_sites is not None:
        requested = set()
        with open_input(args.candidate_sites) as reader:
            for line in reader:
                fields = line.decode().strip().split('\t')
                if len(fields) >= 2 and fields[0] == args.chrom:
                    requested.add(int(fields[1]))
        sites = {pos: site for pos, site in sites.items() if pos in requested}
    if not sites:
        raise ValueError('No BED sites passed chromosome/strand/type/coverage/0%-100% filters')
    stats = {'bed_path': str(args.bed_gz.resolve()), 'lines_seen': lines,
             'candidate_sites': len(sites), 'skipped': dict(skipped),
             'sites_by_label': {str(k): sum(s.label == k for s in sites.values()) for k in [0, 1]}}
    print('BED ' + json.dumps(stats), flush=True)
    return sites, stats


def split_for_site(chrom: str, pos: int, seed: str, train_ratio: float,
                   test_only: bool = False) -> str:
    if test_only:
        return 'test'
    digest = hashlib.md5(f'{seed}:{chrom}:{pos}'.encode()).digest()
    value = int.from_bytes(digest[:6], 'big') / float(1 << 48)
    return 'train' if value < train_ratio else 'valid'


def original_read_id(record: dict) -> str:
    explicit = record.get('fast5_raw_read_id')
    if explicit:
        return str(explicit)
    identifier = str(record.get('read_id') or '')
    match = CHUNK_SUFFIX.fullmatch(identifier)
    return match.group(1) if match else identifier


def normalize_span(value: Any, signal_len: int) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        left, right = int(value[0]), int(value[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if left != value[0] or right != value[1]:
        return None
    if left < 0 or right <= left or right > signal_len:
        return None
    return left, right


def extract_window(record: dict, pos: int, by_pos: dict, site: BedSite,
                   args: argparse.Namespace) -> tuple[dict | None, str]:
    start, end = pos - args.flank, pos + args.flank + 1
    items = [by_pos.get(p) for p in range(start, end)]
    if start < 0 or any(item is None for item in items):
        return None, 'incomplete_genomic_window'
    signal = record.get('signal')
    all_seq_spans = record.get('base_sample_span_seq')
    if not isinstance(signal, list) or not signal or not isinstance(all_seq_spans, list):
        return None, 'missing_signal_or_seq_spans'
    seq_chars, ref_chars, query_positions, seq_spans, ref_spans = [], [], [], [], []
    full_seq = record.get('seq')
    for item in items:
        try:
            query_pos = int(item['query_pos_signal'])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None, 'invalid_query_position'
        if not 0 <= query_pos < len(all_seq_spans):
            return None, 'invalid_query_position'
        seq_span = normalize_span(all_seq_spans[query_pos], len(signal))
        ref_span = normalize_span(item.get('signal_span'), len(signal))
        if seq_span is None or ref_span is None:
            return None, 'invalid_or_out_of_bounds_signal_span'
        query = str(item.get('query_base', '')).upper()
        ref = str(item.get('ref_base', '')).upper()
        if len(query) != 1 or len(ref) != 1:
            return None, 'invalid_base'
        if isinstance(full_seq, str) and (query_pos >= len(full_seq) or full_seq[query_pos].upper() != query):
            return None, 'query_base_inconsistent_with_read_sequence'
        seq_chars.append(query)
        ref_chars.append(ref)
        query_positions.append(query_pos)
        seq_spans.append(seq_span)
        ref_spans.append(ref_span)
    seq, ref = ''.join(seq_chars), ''.join(ref_chars)
    if ref[args.flank] != args.center_base:
        return None, 'center_not_C'
    if seq != ref:
        return None, 'query_reference_mismatch'
    signal_start = min(a for a, _ in ref_spans)
    signal_end = max(b for _, b in ref_spans)
    # Both seq/ref spans must be valid in the exported local signal as well.
    if any(a < signal_start or b > signal_end for a, b in seq_spans):
        return None, 'seq_span_outside_exported_window'
    window_signal = signal[signal_start:signal_end]
    try:
        finite = all(math.isfinite(value) for value in window_signal)
    except TypeError:
        finite = False
    if not window_signal or not finite:
        return None, 'empty_or_nonfinite_window_signal'
    raw_read = original_read_id(record)
    if not raw_read:
        return None, 'missing_original_read_id'
    relative = lambda spans: [[a - signal_start, b - signal_start] for a, b in spans]
    return {
        'read_id': raw_read, 'fast5_raw_read_id': raw_read,
        'segment_read_id': record.get('read_id'),
        'window_id': f'{pos}:{raw_read}', 'chrom': args.chrom,
        'genome_start0': start, 'genome_end0': end, 'center_genome_pos0': pos,
        'strand': '+', 'seq': seq, 'ref': ref, 'window_size': 2 * args.flank + 1,
        'mv_stride': record.get('mv_stride'), 'signal': window_signal,
        'signal_len': len(window_signal), 'signal_offset': signal_start,
        'signal_offset_scope': 'input_record',
        'base_sample_span_seq': relative(seq_spans),
        'base_sample_span_ref': relative(ref_spans), 'query_pos_signal': query_positions,
        'query_pos_signal_scope': 'input_record',
        'dwell': [b - a for a, b in ref_spans],
        'label': site.label, 'bed_coverage': site.coverage, 'bed_percent': site.percent,
        'source': 'mod', 'source_part': record.get('source_part'),
        'mapq': int(record['mapq']), 'split_key': f'{args.chrom}:{pos}',
        'split': split_for_site(args.chrom, pos, args.split_seed, args.train_ratio, args.test_only),
    }, ''


def scan_input(args: argparse.Namespace, sites: dict[int, BedSite], stage: Path,
               db: sqlite3.Connection) -> tuple[dict, Counter]:
    skipped = Counter()
    valid_rows_by_site = Counter()
    records = reads_passed = valid_windows = unique_windows = duplicates = 0
    limit_reached = False
    last_log = time.monotonic()
    with open_input(args.jsonl) as reader, gzip.open(stage / 'windows.spool.tsv.gz', 'wb', compresslevel=1) as spool:
        for line in reader:
            if not line.strip():
                continue
            if args.max_records is not None and records >= args.max_records:
                limit_reached = True
                break
            records += 1
            try:
                record = loads(line)
            except (json.JSONDecodeError, ValueError):
                skipped['json_decode'] += 1
                continue
            if not isinstance(record, dict) or record.get('chrom') != args.chrom:
                skipped['chrom'] += 1
                continue
            if record.get('strand') != '+':
                skipped['read_strand'] += 1
                continue
            try:
                mapq = int(record.get('mapq', -1))
            except (TypeError, ValueError, OverflowError):
                mapq = -1
            if mapq < args.min_mapq:
                skipped['mapq'] += 1
                continue
            per_base = record.get('per_base')
            if not isinstance(per_base, list) or not per_base:
                skipped['empty_per_base'] += 1
                continue
            by_pos = {}
            for item in per_base:
                if not isinstance(item, dict) or item.get('chrom') != args.chrom:
                    continue
                try:
                    position = int(item['genome_pos0'])
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                by_pos.setdefault(position, item)
            centers = sorted(sites.keys() & by_pos.keys())
            if not centers:
                skipped['no_candidate_center'] += 1
                continue
            reads_passed += 1
            for pos in centers:
                sample, reason = extract_window(record, pos, by_pos, sites[pos], args)
                if sample is None:
                    skipped[reason] += 1
                    continue
                valid_windows += 1
                valid_rows_by_site[pos] += 1
                inserted = db.execute('INSERT OR IGNORE INTO seen VALUES (?, ?)',
                                      (pos, sample['fast5_raw_read_id'])).rowcount
                if not inserted:
                    duplicates += 1
                    continue
                spool.write(str(pos).encode() + b'\t' + dumps(sample) + b'\n')
                unique_windows += 1
            if records % 1000 == 0:
                db.commit()
            now = time.monotonic()
            if records % args.log_every == 0 or now - last_log >= 30:
                print(f'SCAN records={records:,} unique_windows={unique_windows:,} duplicates={duplicates:,}', flush=True)
                last_log = now
    db.commit()
    return {'records_seen': records, 'records_with_candidate_centers': reads_passed,
            'valid_windows_before_dedup': valid_windows, 'unique_windows': unique_windows,
            'duplicate_windows_removed': duplicates, 'skipped': dict(skipped),
            'input_scan_complete': not limit_reached}, valid_rows_by_site


def emit_outputs(args: argparse.Namespace, sites: dict[int, BedSite], coverage: dict[int, int],
                 stage: Path, destination: Path) -> tuple[dict, dict]:
    retained = {pos for pos, depth in coverage.items() if depth >= args.min_actual_coverage}
    splits = ('test',) if args.test_only else SPLITS
    paths, handles = {}, {}
    counters = {split: Counter() for split in splits}
    site_sets = {split: set() for split in splits}
    written_by_site = Counter()
    digests = {split: hashlib.sha256() for split in splits}
    suffix = '.jsonl' if args.plain_jsonl else '.jsonl.gz'
    try:
        for split in splits:
            path = stage / f'{split}{suffix}'
            handles[split] = path.open('wb') if args.plain_jsonl else gzip.open(path, 'wb', compresslevel=args.gzip_level)
            paths[split] = str(destination / path.name)
        with gzip.open(stage / 'windows.spool.tsv.gz', 'rb') as reader:
            for index, line in enumerate(reader, 1):
                position, _, payload = line.partition(b'\t')
                pos = int(position)
                if pos not in retained:
                    continue
                split = split_for_site(args.chrom, pos, args.split_seed, args.train_ratio, args.test_only)
                payload = payload.rstrip(b'\r\n')
                if not payload.endswith(b'}'):
                    raise ValueError('Invalid cached JSON record')
                payload = payload[:-1] + b',"actual_coverage":' + str(coverage[pos]).encode() + b'}\n'
                handles[split].write(payload)
                digests[split].update(payload)
                counters[split][sites[pos].label] += 1
                site_sets[split].add(pos)
                written_by_site[pos] += 1
                if index % 500000 == 0:
                    print(f'WRITE scanned_windows={index:,}', flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    assert sum(len(s) for s in site_sets.values()) == len(set().union(*site_sets.values()))
    assert set(written_by_site) == retained
    assert all(written_by_site[pos] == coverage[pos] for pos in retained)
    stats = {}
    # Read final files and their JSON records completely, including gzip CRC.
    for split in splits:
        verified_by_site = Counter()
        verified_labels = Counter()
        verified_digest = hashlib.sha256()
        path = stage / Path(paths[split]).name
        with open_input(path) as reader:
            for index, line in enumerate(reader, 1):
                sample = loads(line)
                pos = sample['center_genome_pos0']
                assert sample['chrom'] == args.chrom and sample['strand'] == '+'
                assert sample['split'] == split and sample['label'] == sites[pos].label
                assert sample['seq'] == sample['ref'] and len(sample['ref']) == 2 * args.flank + 1
                assert sample['ref'][args.flank] == 'C'
                assert sample['actual_coverage'] == coverage[pos] >= args.min_actual_coverage
                assert sample['signal_len'] == len(sample['signal'])
                verified_by_site[pos] += 1
                verified_labels[sample['label']] += 1
                verified_digest.update(line)
                if index % 500000 == 0:
                    print(f'VERIFY {split} rows={index:,}', flush=True)
        assert verified_labels == counters[split]
        assert set(verified_by_site) == site_sets[split]
        assert all(count == coverage[pos] for pos, count in verified_by_site.items())
        assert verified_digest.hexdigest() == digests[split].hexdigest()
        depth_histogram = Counter(coverage[pos] for pos in site_sets[split])
        stats[split] = {
            'output_path': paths[split], 'records': sum(counters[split].values()),
            'written_by_label': {str(k): counters[split][k] for k in [0, 1]},
            'unique_sites': len(site_sets[split]),
            'sites_by_label': {str(k): sum(sites[pos].label == k for pos in site_sets[split]) for k in [0, 1]},
            'actual_coverage_histogram': {str(k): v for k, v in sorted(depth_histogram.items())},
            'uncompressed_sha256': digests[split].hexdigest(), 'compressed_bytes': path.stat().st_size}
    return paths, stats


def main() -> int:
    args = parse_args()
    validate_args(args)
    destination = args.out_dir.resolve()
    stage = destination.with_name(destination.name + '.inprogress')
    if stage.exists():
        raise SystemExit(f'Incomplete staging directory already exists: {stage}; inspect it or choose a new output path')
    started = time.monotonic()
    sites, bed_stats = load_candidates(args)
    stage.mkdir(parents=True)
    db_path = stage / 'distinct_read_sites.sqlite'
    with sqlite3.connect(db_path) as db:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('PRAGMA cache_size=-131072')
        db.execute('CREATE TABLE seen (site_pos0 INTEGER NOT NULL, raw_read_id TEXT NOT NULL, '
                   'PRIMARY KEY (site_pos0, raw_read_id)) WITHOUT ROWID')
        input_stats, valid_rows = scan_input(args, sites, stage, db)
        coverage = dict(db.execute('SELECT site_pos0, COUNT(*) FROM seen GROUP BY site_pos0'))
    retained = {pos for pos, count in coverage.items() if count >= args.min_actual_coverage}
    (stage / 'scan_summary.json').write_text(json.dumps({
        'input': input_stats, 'bed': bed_stats,
        'actual_coverage_histogram': {str(k): v for k, v in sorted(Counter(coverage.values()).items())},
        'retained_sites': len(retained), 'min_actual_coverage': args.min_actual_coverage,
    }, indent=2) + '\n')
    if not retained:
        raise SystemExit('No sites reached distinct-read coverage after all quality filters; '
                         f'input/staging evidence retained in {stage}')
    print(f'COVERAGE retained_sites={len(retained):,} removed_sites={len(sites) - len(retained):,}', flush=True)
    paths, write_stats = emit_outputs(args, sites, coverage, stage, destination)
    with gzip.open(stage / 'site_coverage.tsv.gz', 'wt') as writer, gzip.open(stage / 'candidate_sites.tsv.gz', 'wt') as candidates:
        writer.write('chrom\tgenome_pos0\tlabel\tbed_coverage\tbed_percent\tvalid_window_records\t'
                     'actual_coverage\textra_duplicate_records\tretained\tsplit\n')
        candidates.write('chrom\tgenome_pos0\tlabel\tbed_coverage\tbed_percent\tactual_coverage\tsplit\n')
        for pos in sorted(sites):
            site = sites[pos]
            depth = coverage.get(pos, 0)
            split = split_for_site(args.chrom, pos, args.split_seed, args.train_ratio, args.test_only)
            writer.write(f'{args.chrom}\t{pos}\t{site.label}\t{site.coverage}\t{site.percent}\t'
                         f'{valid_rows[pos]}\t{depth}\t{valid_rows[pos] - depth}\t{int(pos in retained)}\t{split}\n')
            if pos in retained:
                candidates.write(f'{args.chrom}\t{pos}\t{site.label}\t{site.coverage}\t{site.percent}\t{depth}\t{split}\n')
    summary = {
        'format': 'step2_bed_label_signal_window_jsonl_v2',
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'outputs': paths, 'window_nt': 2 * args.flank + 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'label_definition': 'Same MOD input: BED 0% -> label 0; BED 100% -> label 1',
        'signal_processing': 'Input signal reused unchanged; no additional normalization',
        'coverage_definition': 'Distinct original reads per genomic site AFTER MAPQ/strand/'
                               'complete-window/sequence-match/signal validity filters',
        'deduplication': 'One valid window per genomic site and original read; first valid occurrence retained',
        'split_mode': 'test_only' if args.test_only else 'site_hash',
        'split_strategy': ('All retained sites assigned to test; no class downsampling'
                           if args.test_only else
                           'MD5(seed:chrom:site_pos0), train/valid only; no class downsampling'),
        'bed': bed_stats, 'input': input_stats, 'write_stats': write_stats,
        'coverage': {'bed_candidate_sites': len(sites), 'sites_with_valid_windows': len(coverage),
                     'retained_sites': len(retained), 'dropped_sites': len(sites) - len(retained),
                     'windows_removed_low_coverage': sum(count for pos, count in coverage.items() if pos not in retained)},
        'overall': {'total_written': sum(s['records'] for s in write_stats.values()),
                    'written_by_label': {str(k): sum(s['written_by_label'][str(k)] for s in write_stats.values()) for k in [0, 1]}},
        'verification': {'cross_split_sites': 0, 'original_read_site_duplicates': 0,
                         'minimum_actual_coverage': min(coverage[pos] for pos in retained),
                         'all_output_rows_reparsed': True, 'gzip_crc_checked': not args.plain_jsonl,
                         'output_sha256_matched': True},
        'seconds': round(time.monotonic() - started, 2)}
    (stage / 'build_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (stage / 'windows.spool.tsv.gz').unlink()
    db_path.unlink()
    if destination.exists():
        backup = destination.with_name(destination.name + f'.previous-{time.time_ns()}')
        destination.rename(backup)
        print(f'PREVIOUS_OUTPUT {backup}', flush=True)
    stage.rename(destination)
    print('COMPLETE ' + json.dumps({'output_dir': str(destination),
                                    'overall': summary['overall'],
                                    'splits': {k: v['records'] for k, v in write_stats.items()},
                                    'minimum_actual_coverage': summary['verification']['minimum_actual_coverage']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
