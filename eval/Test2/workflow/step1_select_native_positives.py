#!/usr/bin/env python3
"""Select native BED=100 test windows and their cached features, without re-encoding.

This supplies cached features to Step2's optional inference route.
Labels are inherited from the original test reference; their experimental
accuracy is not established by this selection operation.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from common import PROJECT, infer_architecture, output_directory, read_json, sha256, signature, valid_model_name, write_json

KIND = 'native_bed100'
TRUTH = 'native_test_reference_label_one'
REQUIRED = {'read_id', 'source', 'chrom', 'center_genome_pos0', 'strand', 'label', 'seq', 'ref'}


def parser():
    from common import cli
    p = cli('Step1: select native label=1 cached test features (optional for prediction reuse).')
    p.add_argument('--input-dir', type=Path, help='只选择一个模型时覆盖原缓存目录')
    return p


def load_source(args):
    import numpy as np
    import torch
    if args.chunk_size < 1 or args.log_every_chunks < 1 or (args.max_records is not None and args.max_records < 1):
        raise ValueError('Chunk/log/test limits must be positive')
    if not args.library_id.strip() or any(c in args.library_id for c in '\t\r\n'):
        raise ValueError('Invalid native library ID')
    root, out = args.input_dir.resolve(), args.out_dir.resolve()
    if out == root or out in root.parents or root in out.parents:
        raise ValueError('Input and output directories must be separate')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Output must be a new or empty directory: {out}')
    meta = torch.load(root/'test.pt', map_location='cpu', weights_only=False)
    if meta.get('storage_format') != 'streaming_memmap_v1' or meta.get('split', 'test') != 'test':
        raise ValueError('Expected a streaming_memmap_v1 test split')
    shape = meta.get('feature_shape', [])
    if len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape) or shape[1] % 2 != 1:
        raise ValueError('Invalid feature shape')
    n, w, h = shape
    if meta.get('records', n) != n:
        raise ValueError('Manifest record count does not match feature shape')
    dtype = np.dtype(meta['feature_dtype'])
    if dtype not in (np.dtype('float16'), np.dtype('float32')) or np.dtype(meta['label_dtype']) != np.dtype('float32'):
        raise ValueError('Only float16/float32 features and float32 labels are supported')
    paths = [root/'test.pt', root/meta['feature_path'], root/meta['label_path'],
             root/'metadata_test.tsv', root/'feature_summary.json']
    inputs = [signature(p) for p in paths]
    if paths[1].stat().st_size != n*w*h*dtype.itemsize or paths[2].stat().st_size != n*4:
        raise ValueError('Incomplete feature/label arrays')
    summary = read_json(paths[4])
    if summary.get('test1_dataset', {}).get('truth_source') == 'WGA_negative_control':
        raise ValueError('This step expects native test features; WGA controls must retain their WGA identity')
    architecture = infer_architecture(summary)
    if architecture is None:
        raise ValueError('Cannot infer source backbone architecture')
    with paths[3].open(newline='', encoding='utf-8-sig') as f:
        fields = next(csv.reader(f, delimiter='\t'), [])
    if len(fields) != len(set(fields)) or not REQUIRED.issubset(fields):
        raise ValueError('Missing or duplicate source metadata columns')
    y = np.memmap(paths[2], mode='r', dtype='float32', shape=(n,))
    stop = min(n, args.max_records) if args.max_records is not None else n
    labels = Counter()
    selected = 0
    for lo in range(0, n, args.chunk_size):
        block = y[lo:lo+args.chunk_size]
        if not ((block == 0) | (block == 1)).all():
            raise ValueError(f'Nonbinary or nonfinite source labels at row {lo}')
        zeros = int(np.count_nonzero(block == 0))
        labels['0'] += zeros
        labels['1'] += len(block)-zeros
        if lo < stop:
            selected += int(np.count_nonzero(block[:stop-lo] == 1))
    if not selected:
        raise ValueError('No label=1 windows in the selected test range')
    array_bytes = selected*(w*h*dtype.itemsize+4)
    # Conservative working estimate: copied TSV plus SQLite rows/indexes and arrays.
    estimated = array_bytes+paths[3].stat().st_size+selected*512+16*1024**2
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    partial_source = summary
    source_partial = False
    while isinstance(partial_source, dict):
        source_partial |= partial_source.get('limit') is not None
        source_partial |= bool(partial_source.get('test2_dataset', {}).get('partial_test', False))
        source_partial |= bool(partial_source.get('test1_dataset', {}).get('partial_test', False))
        partial_source = partial_source.get('source_feature_summary')
    plan = dict(input_dir=str(root), out_dir=str(out), version=args.version, model_architecture=architecture,
                dataset_kind=KIND, truth_source=TRUTH, library_id=args.library_id,
                source_shape=shape, source_label_counts=dict(labels), scanned_windows=stop,
                selected_positive_windows=selected, output_shape=[selected, w, h],
                feature_dtype=dtype.name, feature_and_label_bytes=array_bytes,
                estimated_required_bytes=estimated, available_bytes=shutil.disk_usage(parent).free,
                partial_test=source_partial or stop < n, max_records=args.max_records,
                source_signatures=inputs,
                note='Select only existing test label=1. Preserve features, row order, source and reference labels; no WGA assumption.')
    return plan, meta, summary, paths, y


@contextlib.contextmanager
def native_output(path):
    """Reuse atomic publication, also allowing a pre-created empty destination."""
    previous_mode = None
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f'Refusing nonempty output: {path}')
        previous_mode = path.stat().st_mode & 0o777
        path.rmdir()  # rmdir itself rejects a concurrent writer's new files.
    try:
        with output_directory(path) as stage:
            if previous_mode is not None:
                stage.chmod(previous_mode)
            yield stage
    except BaseException:
        if previous_mode is not None and not path.exists():
            path.mkdir(mode=previous_mode, parents=True)
        raise


def run(args):
    import numpy as np
    import torch
    plan, meta, summary, paths, y = load_source(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan
    if plan['available_bytes'] < plan['estimated_required_bytes']:
        raise OSError('Insufficient output disk space for the estimated native positive cache')
    out = Path(plan['out_dir'])
    n, w, h = meta['feature_shape']
    x = np.memmap(paths[1], mode='r', dtype=meta['feature_dtype'], shape=(n, w, h))
    selected = seen = 0
    chromosomes = Counter()
    original_widths = set()
    extension = 'f16' if x.dtype.itemsize == 2 else 'f32'
    fn = f'test.features.{extension}.bin'
    with native_output(out) as stage:
        db = sqlite3.connect(stage/'sample_audit.sqlite')
        db.execute('''CREATE TABLE audit (
            chrom TEXT, position INTEGER, strand TEXT, read_id TEXT, source TEXT,
            sample_index INTEGER, source_sample_index INTEGER, mapq INTEGER, signal_len INTEGER,
            PRIMARY KEY(chrom,position,strand,read_id)) WITHOUT ROWID''')
        try:
            with paths[3].open(newline='', encoding='utf-8-sig') as source, \
                    (stage/'metadata_test.tsv').open('w', newline='', encoding='utf-8') as dest, \
                    (stage/fn).open('wb') as xf, (stage/'test.labels.f32.bin').open('wb') as yf:
                reader = csv.DictReader(source, delimiter='\t')
                fields = list(reader.fieldnames)
                for field in ['library_id', 'truth_source', 'dataset_kind', 'sample_index',
                              'source_sample_index', 'mapq', 'input_signal_len']:
                    if field not in fields:
                        fields.append(field)
                writer = csv.DictWriter(dest, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                for chunk, lo in enumerate(range(0, plan['scanned_windows'], args.chunk_size), 1):
                    hi = min(lo+args.chunk_size, plan['scanned_windows'])
                    rows = list(itertools.islice(reader, hi-lo))
                    if len(rows) != hi-lo:
                        raise ValueError('Metadata has fewer rows than the feature/label arrays')
                    indices, audit_rows = [], []
                    for index, row in enumerate(rows, lo):
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(f'Malformed metadata at source row {index}')
                        label = int(row['label'])
                        if label not in (0, 1) or label != y[index]:
                            raise ValueError(f'Metadata/binary label mismatch at source row {index}')
                        if row['source'] != 'mod' or row['chrom'] != args.chromosome:
                            raise ValueError('Expected original native chr16 source=mod')
                        if row.get('split', 'test') != 'test':
                            raise ValueError('Only original test metadata may be selected')
                        if row.get('sample_index') and int(row['sample_index']) != index:
                            raise ValueError(f'Source sample_index mismatch at row {index}')
                        seen += 1
                        if label != 1:
                            continue
                        if not row['read_id'] or not row['chrom'] or row['strand'] != '+' or int(row['center_genome_pos0']) < 0:
                            raise ValueError('Expected positive-strand native windows with valid coordinates')
                        for key in ['seq', 'ref']:
                            if len(row[key]) != w or set(row[key].upper())-set('ACGTNWSMKRYBVDHZ'):
                                raise ValueError(f'Invalid {key} at source row {index}')
                        if row.get('library_id') and row['library_id'] != args.library_id:
                            raise ValueError('Source library ID differs from --library-id')
                        row['source'] = row.get('source') or 'mod'
                        row.update(library_id=args.library_id, truth_source=TRUTH, dataset_kind=KIND,
                                   source_sample_index=index, sample_index=selected, label=1)
                        # signal_len refers to the original signal window, even after feature cropping.
                        row['input_signal_len'] = row.get('input_signal_len') or row.get('signal_len', '')
                        mapq = int(row['mapq']) if row.get('mapq') else None
                        length = int(row['input_signal_len']) if row['input_signal_len'] else None
                        if (mapq is not None and mapq < 0) or (length is not None and length < 1):
                            raise ValueError('Invalid inherited QC fields')
                        row['mapq'] = '' if mapq is None else mapq
                        indices.append(index)
                        audit_rows.append((row['chrom'], int(row['center_genome_pos0']), row['strand'], row['read_id'],
                                           row['source'], selected, index, mapq, length))
                        chromosomes[row['chrom']] += 1
                        if row.get('crop_input_window_size'):
                            original_widths.add(int(row['crop_input_window_size']))
                        writer.writerow(row)
                        selected += 1
                    if indices:
                        block = np.ascontiguousarray(x[indices])
                        if not np.isfinite(block).all():
                            raise ValueError('Nonfinite selected positive features')
                        block.tofile(xf)
                        np.ascontiguousarray(y[indices]).tofile(yf)
                        try:
                            db.executemany('INSERT INTO audit VALUES (?,?,?,?,?,?,?,?,?)', audit_rows)
                        except sqlite3.IntegrityError as e:
                            raise ValueError('Duplicate original read/position in the selected native test') from e
                        db.commit()
                    if chunk == 1 or chunk % args.log_every_chunks == 0 or hi == plan['scanned_windows']:
                        print(f'[SELECT] scanned={seen}/{plan["scanned_windows"]} positive={selected}', flush=True)
                if seen == n and next(reader, None) is not None:
                    raise ValueError('Metadata has extra rows beyond the feature/label arrays')
            if selected != plan['selected_positive_windows']:
                raise ValueError('Selected count changed during extraction')
            site_count = 0
            min_coverage = None
            with (stage/'sites.tsv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['chrom', 'position', 'strand', 'label', 'n_reads', 'retained'])
                for chrom, pos, strand, depth in db.execute(
                        'SELECT chrom,position,strand,COUNT(*) FROM audit GROUP BY chrom,position,strand '
                        'ORDER BY chrom,position,strand'):
                    writer.writerow([chrom, pos, strand, 1, depth, 1])
                    site_count += 1
                    min_coverage = depth if min_coverage is None else min(min_coverage, depth)
            dataset = dict(dataset_kind=KIND, truth_source=TRUTH, library_id=args.library_id,
                           source_dataset=plan['input_dir'], source_split='test', selection='source label == 1',
                           source_label_counts=plan['source_label_counts'], source_records=n,
                           scanned_records=seen, records=selected, sites=site_count,
                           label_counts={'0': 0, '1': selected}, chrom=next(iter(chromosomes)) if len(chromosomes) == 1 else None,
                           chromosomes=dict(chromosomes), strand='+', window_size=w,
                           original_signal_window_sizes=sorted(original_widths), min_actual_coverage=min_coverage,
                           input_scan_complete=seen == n, partial_test=plan['partial_test'], max_records=args.max_records,
                           reference_validation_status=args.reference_validation_status,
                           reference_label_rule='original_test_label_equals_1',
                           reference_label_note='Inherited native test BED=100 labels; BED assay/source and residual modifications are not verified here.',
                           source_signatures=plan['source_signatures'])
            selection = dict(predicate='label == 1', input_dir=plan['input_dir'], source_records=n,
                             scanned_records=seen, selected_records=selected, preserves_order=True,
                             preserves_feature_dtype=True, feature_values_unchanged=True,
                             source_row_mapping='metadata_test.tsv: source_sample_index')
            manifest = dict(meta)
            manifest.update(feature_shape=[selected, w, h], records=selected, split='test',
                            feature_path=fn, label_path='test.labels.f32.bin', metadata_path='metadata_test.tsv',
                            eval_version=args.version, model_architecture=plan['model_architecture'],
                            dataset_kind=KIND, truth_source=TRUTH, label_counts={'0': 0, '1': selected},
                            selection=selection)
            torch.save(manifest, stage/'test.pt')
            filtered = dict(summary)
            filtered.update(operation='select_native_test_label_one', source_feature_dir=plan['input_dir'],
                            source_feature_summary=summary, source_dataset=plan['input_dir'], out_dir=str(out),
                            eval_version=args.version, model_architecture=plan['model_architecture'],
                            dataset_kind=KIND, truth_source=TRUTH, window_size=w, records=selected,
                            test2_dataset=dataset, selection=selection,
                            splits=[dict(split='test', records=selected, feature_shape=[selected, w, h],
                                         storage_format='streaming_memmap_v1', pt_path='test.pt', feature_path=fn,
                                         label_path='test.labels.f32.bin', metadata_path='metadata_test.tsv')])
            write_json(stage/'feature_summary.json', filtered)
            write_json(stage/'dataset_manifest.json', dataset)
            write_json(stage/'selection_manifest.json', dict(plan=plan, selection=selection,
                       script_sha256=sha256(Path(__file__)), records=selected, sites=site_count))
            if [signature(Path(s['path'])) for s in plan['source_signatures']] != plan['source_signatures']:
                raise ValueError('Source test cache changed during selection')
        finally:
            db.close()
    print(f'[DONE] {args.version} native BED=100 windows={selected} sites={site_count} partial_test={plan["partial_test"]} out={out}', flush=True)
    return dataset


def main():
    from common import load_config
    from types import SimpleNamespace
    args = parser().parse_args()
    config, versions = load_config(args)
    if (args.input_dir or args.out_dir) and len(versions) != 1:
        raise ValueError('Specify --version when overriding input/output directory')
    for version in versions:
        settings = config['models'][version]
        native = SimpleNamespace(input_dir=args.input_dir or Path(settings['feature_dir']),
            out_dir=args.out_dir or Path(config['data_root'])/version, version=version,
            library_id=config['library_id'], chromosome=config['chromosome'],
            reference_validation_status=config.get('reference_validation_status', 'unverified'),
            chunk_size=config['selection']['chunk_size'],
            log_every_chunks=config['selection']['log_every_chunks'],
            max_records=args.max_records, dry_run=args.dry_run)
        summary = read_json(native.input_dir/'feature_summary.json')
        if infer_architecture(summary) != settings['architecture']:
            raise ValueError('Configured architecture differs from source feature architecture')
        run(native)


if __name__ == '__main__':
    main()
