#!/usr/bin/env python3
"""Sample a train/valid/test subset with exact class counts and coverage >=5.

Randomly select genomic sites per class, retaining complete site coverage where
possible. Reduce boundary-site quotas to meet exact class targets, while keeping
at least min_coverage distinct reads at every retained site. Sample uniformly
without replacement within partially retained sites. Source SHA256, output JSON,
site counts and duplicates are verified before publishing. Existing owned outputs
may be replaced only with --replace-existing; the previous directory is backed up.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / 'output/step3_output/chr19/train.jsonl.gz'
DEFAULT_OUTPUT = PROJECT_ROOT / 'output/step3_output/chr19-small'
_spec = importlib.util.spec_from_file_location('step31_balance_helpers', SCRIPT_DIR/'step3_balance_chr19.py')
COMMON = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(COMMON)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--split', choices=('train','valid','test'),
                   help='Defaults to the split inferred from the input filename.')
    p.add_argument('--positive-count', type=int, default=40000)
    p.add_argument('--negative-count', type=int, default=40000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-coverage', type=int, default=5)
    p.add_argument('--replace-existing', action='store_true',
                   help='Back up and replace an existing Step-3.1 dataset after verification.')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--gzip-level', type=int, default=3)
    p.add_argument('--log-every', type=int, default=100000)
    return p.parse_args()


def resolve_input_split(path, explicit=None):
    name = path.name
    if name.endswith('.gz'):
        name = name[:-3]
    if name.endswith('.jsonl'):
        name = name[:-6]
    inferred = {'train':'train','valid':'valid','val':'valid','test':'test'}.get(name)
    if explicit is not None:
        if inferred is not None and inferred != explicit:
            raise ValueError(f'--split={explicit} conflicts with input filename {path.name}')
        return explicit
    if inferred is None:
        raise ValueError(f'Cannot infer split from {path.name}; specify --split train/valid/test')
    return inferred


def choose_site_quotas(depths, labels, targets, floor, seed):
    return COMMON.choose_site_quotas(depths,labels,targets,floor,seed)


def destination_snapshot(destination):
    if not destination.exists():
        return None
    if not destination.is_dir():
        raise FileExistsError(f'Output is not a directory: {destination}')
    return {p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in destination.iterdir()}


def prepare_destination(destination, replace_existing=False):
    stage = destination.with_name(destination.name+'.inprogress')
    if stage.exists():
        raise FileExistsError(f'Staging directory already exists: {stage}')
    snapshot = destination_snapshot(destination)
    if snapshot:
        if not replace_existing:
            raise FileExistsError(f'Existing output: {destination}; use --replace-existing to back up and replace')
        manifest = destination/'build_summary.json'
        if not manifest.is_file() or not json.loads(manifest.read_text()).get('format','').startswith('step31_'):
            raise FileExistsError(f'Replacement is restricted to an existing Step-3.1 dataset: {destination}')
    stage.mkdir(parents=True)
    return stage, snapshot


def publish(stage, destination, snapshot, summary, floor):
    if destination_snapshot(destination) != snapshot:
        raise RuntimeError('Existing destination changed during extraction; refusing replacement')
    backup = None
    if snapshot:
        base = destination.with_name(destination.name+f'.backup-before-cov{floor}')
        backup = base
        suffix = 1
        while backup.exists():
            backup = base.with_name(base.name+f'.{suffix}')
            suffix += 1
        summary['previous_output_backup'] = str(backup)
        COMMON.atomic_json(summary, stage/'build_summary.json')
        destination.rename(backup)
    elif destination.exists():
        destination.rmdir()
    try:
        stage.rename(destination)
    except BaseException:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    return backup


def update_small_coverage(line, coverage):
    parent, a, b = COMMON.field(line, 'actual_coverage')
    if b'"parent_actual_coverage"' in line[-1024:]:
        raise ValueError('Input already contains parent_actual_coverage')
    changed = line[:a]+str(coverage).encode()+line[b:]
    close = changed.rfind(b'}')
    return changed[:close]+b',"parent_actual_coverage":'+str(parent).encode()+changed[close:]


def run(args):
    if args.workers < 1 or args.log_every < 0 or not 0 <= args.gzip_level <= 9:
        raise ValueError('Invalid workers/log interval/gzip level')
    split = resolve_input_split(args.input,getattr(args,'split',None))
    source, destination = args.input.resolve(), args.out_dir.resolve()
    summary_path = source.parent/'build_summary.json'
    source_summary = json.loads(summary_path.read_text())
    verified = source_summary.get('verification', {})
    if not verified.get('output_sha256_matched') or verified.get('original_read_site_duplicates') != 0:
        raise ValueError('Requires checksum-verified source with unique original read/site pairs')
    if split not in source_summary.get('write_stats',{}):
        raise ValueError(f'Source summary has no statistics for split={split}')
    expected = source_summary['write_stats'][split]
    targets = {0: args.negative_count, 1: args.positive_count}
    parent_depths, parent_labels = {}, {}
    with gzip.open(source.parent/'candidate_sites.tsv.gz','rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            if row['split'] != split:
                continue
            key = (row['chrom'], int(row['genome_pos0']))
            if key in parent_depths or int(row['label']) not in (0,1):
                raise ValueError(f'Duplicate/invalid parent site: {key}')
            parent_depths[key] = int(row['actual_coverage'])
            parent_labels[key] = int(row['label'])
    original_stats = COMMON.stats(parent_depths,parent_labels)
    if original_stats['records'] != expected['records'] or original_stats['written_by_label'] != expected['written_by_label']:
        raise ValueError('Parent site table does not match the source summary')
    quotas = choose_site_quotas(parent_depths,parent_labels,targets,args.min_coverage,args.seed)
    remaining = {key:parent_depths[key] for key in quotas}
    wanted = quotas.copy()
    read_rng = random.Random(f'{args.seed}:within_site')
    stage, snapshot = prepare_destination(destination,args.replace_existing)
    started = time.monotonic()
    source_stat = source.stat()
    source_hash, selected_hash = hashlib.sha256(), hashlib.sha256()
    seen, selected, depths = Counter(), Counter(), Counter()
    site_info, pairs = {}, set()
    temporary = stage/'selected_rows.jsonl.tmp'
    print('PLAN '+json.dumps({'input_records': expected['records'], 'target_by_label': targets,
                              'seed': args.seed, 'split':split, 'min_coverage':args.min_coverage,
                              'selected_sites':len(quotas), 'out_dir': str(destination)}), flush=True)
    with COMMON.read_lines(source) as reader, temporary.open('wb', buffering=1024*1024) as writer:
        for scanned, line in enumerate(reader, 1):
            source_hash.update(line)
            label = COMMON.field(line, 'label')[0]
            if label not in targets:
                raise ValueError(f'Unexpected label at {source}:{scanned}')
            seen[label] += 1
            key = (COMMON.field(line,'chrom')[0], COMMON.field(line,'center_genome_pos0')[0])
            keep = False
            if key in quotas:
                available, needed = remaining[key], wanted[key]
                if available <= 0 or label != parent_labels[key]:
                    raise ValueError(f'Parent site count/label mismatch: {key}')
                keep = needed == available or (needed > 0 and read_rng.randrange(available) < needed)
                remaining[key] -= 1
                if keep:
                    wanted[key] -= 1
            if keep:
                row = json.loads(line)
                if int(row['actual_coverage']) != parent_depths[key]:
                    raise ValueError(f'Parent depth mismatch: {key}')
                if row['split'] != split:
                    raise ValueError(f'Expected split={split} at {source}:{scanned}, got {row["split"]}')
                read = row.get('fast5_raw_read_id') or row['read_id']
                pair = (*key, read)
                if pair in pairs:
                    raise ValueError(f'Duplicate original read/site pair: {pair}')
                pairs.add(pair)
                info = (label, int(row['actual_coverage']), row.get('input_actual_coverage'),
                        row.get('bed_coverage'), row.get('bed_percent'))
                if key in site_info and site_info[key] != info:
                    raise ValueError(f'Conflicting metadata at genomic site: {key}')
                site_info[key] = info
                depths[key] += 1
                selected[label] += 1
                writer.write(line)
                selected_hash.update(line)
            if args.log_every and scanned % args.log_every == 0:
                progress = {'phase':'sample', 'split':split, 'scanned':scanned, 'input_records':expected['records'],
                            'percent':round(100*scanned/expected['records'],2),
                            'selected_by_label':dict(selected), 'seconds':round(time.monotonic()-started,1)}
                COMMON.atomic_json(progress, stage/'progress.json')
                print('SCAN '+json.dumps(progress), flush=True)
    end_stat = source.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (end_stat.st_size, end_stat.st_mtime_ns):
        raise RuntimeError('Input changed during sampling')
    if sum(seen.values()) != expected['records']:
        raise ValueError(f'{source}: split={split} expected {expected["records"]} records, '
                         f'found {sum(seen.values())}; check input and summary paths')
    if source_hash.hexdigest() != expected['uncompressed_sha256']:
        raise ValueError(f'{source}: SHA256 mismatch for split={split}; '
                         f'expected {expected["uncompressed_sha256"]}, got {source_hash.hexdigest()}')
    if {str(k):seen[k] for k in (0,1)} != expected['written_by_label'] or dict(selected) != targets:
        raise RuntimeError('Source or sampled class counts differ from plan')
    if any(remaining.values()) or any(wanted.values()) or depths != quotas:
        raise RuntimeError('Selected site depths do not match planned quotas')
    output = stage/f'{split}.jsonl.gz'
    selected_hash_second, output_hash = hashlib.sha256(), hashlib.sha256()
    compressor = COMMON.ParallelGzipWriter(output, args.workers, 16*1024*1024, args.gzip_level)
    try:
        with temporary.open('rb') as reader:
            for line in reader:
                selected_hash_second.update(line)
                key = (COMMON.field(line,'chrom')[0], COMMON.field(line,'center_genome_pos0')[0])
                updated = update_small_coverage(line, depths[key])
                output_hash.update(updated)
                compressor.append(updated)
        compressor.close()
    except BaseException:
        if not compressor.closed:
            compressor.abort()
        raise
    if selected_hash_second.hexdigest() != selected_hash.hexdigest():
        raise RuntimeError('Selected temporary records changed during export')
    checked, checked_labels, checked_pairs = Counter(), Counter(), set()
    checked_hash = hashlib.sha256()
    print('VERIFY: fully parse sampled output and check coverage, labels, duplicates, CRC and SHA256', flush=True)
    with COMMON.read_lines(output) as reader:
        for line in reader:
            checked_hash.update(line)
            row = json.loads(line)
            key = (row['chrom'], int(row['center_genome_pos0']))
            info = site_info[key]
            if row['label'] != info[0] or row['split'] != split or row['actual_coverage'] != depths[key]:
                raise ValueError('Output label/split/coverage does not match the sample plan')
            if row['parent_actual_coverage'] != info[1] or row.get('input_actual_coverage') != info[2]:
                raise ValueError('Output coverage provenance does not match source')
            pair = (*key, row.get('fast5_raw_read_id') or row['read_id'])
            if pair in checked_pairs:
                raise ValueError(f'Duplicate output read/site pair: {pair}')
            checked_pairs.add(pair)
            checked[key] += 1
            checked_labels[row['label']] += 1
    if checked != depths or dict(checked_labels) != targets or checked_pairs != pairs:
        raise RuntimeError('Verified sampled output differs from selection')
    if min(checked.values()) < args.min_coverage:
        raise RuntimeError('Output actual coverage is below the requested floor')
    if checked_hash.hexdigest() != output_hash.hexdigest():
        raise RuntimeError('Output SHA256 mismatch')
    labels = {key:info[0] for key,info in site_info.items()}
    result = COMMON.stats(depths, labels)
    result.update(output_path=str(destination/f'{split}.jsonl.gz'),
                  uncompressed_sha256=output_hash.hexdigest(), compressed_bytes=output.stat().st_size)
    with gzip.open(stage/'site_coverage.tsv.gz','wt',compresslevel=3) as handle:
        handle.write('chrom\tgenome_pos0\tlabel\tsplit\tparent_actual_coverage\tinput_actual_coverage\tactual_coverage\n')
        for key in sorted(depths):
            label,parent,original,_,_ = site_info[key]
            handle.write(f'{key[0]}\t{key[1]}\t{label}\t{split}\t{parent}\t{original}\t{depths[key]}\n')
    with gzip.open(stage/'candidate_sites.tsv.gz','wt',compresslevel=3) as handle:
        handle.write('chrom\tgenome_pos0\tlabel\tbed_coverage\tbed_percent\tactual_coverage\tsplit\tparent_actual_coverage\tinput_actual_coverage\n')
        for key in sorted(depths):
            label,parent,original,bed,pct = site_info[key]
            handle.write(f'{key[0]}\t{key[1]}\t{label}\t{bed}\t{pct}\t{depths[key]}\t{split}\t{parent}\t{original}\n')
    summary = {
        'format':'step31_site_sample_split_subset_jsonl_v3', 'split':split, 'input':str(source),
        'out_dir':str(destination), 'source_summary':str(summary_path), 'seed':args.seed,
        'requested_by_label':{str(k):v for k,v in targets.items()},
        'source_records':sum(seen.values()), 'source_records_by_label':{str(k):seen[k] for k in (0,1)},
        'sampling':'Seeded random sites per class, complete site reads where possible; boundary quotas adjusted to exact class counts with a depth floor; uniform read sampling without replacement within partial sites; preserve source record order',
        'split_strategy':f'Subset of {split} only; preserve original split assignment',
        'coverage_definition':'Distinct retained original read windows per site in this small subset',
        'coverage_fields':{'actual_coverage':'depth in chr19-small',
                           'parent_actual_coverage':'depth in the immediate chr19 input',
                           'input_actual_coverage':'original Step-2 depth, retained unchanged'},
        'minimum_coverage_constraint':args.min_coverage,
        'signal_processing':'Signal/sequence/span bytes retained unchanged; coverage metadata only is updated',
        'window_nt':source_summary.get('window_nt'), 'label_definition':source_summary.get('label_definition'),
        'write_stats':{split:result},
        'overall':{'total_written':sum(selected.values()), 'written_by_label':{str(k):selected[k] for k in (0,1)},
                   'unique_sites':len(depths)},
        'verification':{'exact_requested_class_counts':True,'all_output_rows_reparsed':True,
                        'source_sha256_matched':True,'output_sha256_matched':True,'gzip_crc_checked':True,
                        'original_read_site_duplicates':0,'all_rows_from_requested_split':True,
                        'all_site_depths_recounted':True,
                        'minimum_actual_coverage':min(checked.values()),'coverage_floor_satisfied':True},
        'source_uncompressed_sha256':source_hash.hexdigest(),
        'selected_original_lines_sha256':selected_hash.hexdigest(),
        'python_version':sys.version.split()[0], 'seconds':round(time.monotonic()-started,2),
    }
    COMMON.atomic_json(summary, stage/'build_summary.json')
    temporary.unlink()
    (stage/'progress.json').unlink(missing_ok=True)
    backup = publish(stage,destination,snapshot,summary,args.min_coverage)
    if backup is not None:
        print(f'BACKUP previous output: {backup}',flush=True)
    print('COMPLETE '+json.dumps({'overall':summary['overall'],
                                 'minimum_actual_coverage':result['minimum_actual_coverage']}),flush=True)
    return summary


if __name__=='__main__':
    run(parse_args())
