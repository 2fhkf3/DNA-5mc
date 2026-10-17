"""Test2 configuration, positive-only statistics and audited prediction storage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sqlite3
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / 'eval/Test2'
# Reuse only Test1's generic I/O helpers. Its FP/TN statistics are never used.
_spec = importlib.util.spec_from_file_location('_test2_shared_io', PROJECT/'eval/Test1/workflow/common.py')
_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_io)
for _name in ('infer_architecture', 'output_directory', 'read_json', 'write_json',
              'sha256', 'signature', 'valid_model_name', 'open_text', 'tsv', 'readonly_db', 'wilson'):
    globals()[_name] = getattr(_io, _name)

KIND = 'native_bed100'
TRUTH = 'native_test_reference_label_one'
KEYS = 'library_id,read_id,chrom,position,strand'
SITE_KEYS = 'library_id,chrom,position,strand'
FIELDS = ['library_id', 'read_id', 'chrom', 'position', 'strand', 'source', 'seq', 'ref',
          'prob', 'source_sample_index', 'mapq', 'signal_len']
REQUIRED = {'read_id', 'chrom', 'center_genome_pos0', 'strand', 'source', 'label', 'seq', 'ref'}


def cli(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    g = p.add_mutually_exclusive_group()
    g.add_argument('--version', help='只处理指定模型，例如 V011')
    g.add_argument('--versions', nargs='+', help='选择多个模型；默认配置中的 enabled_versions')
    p.add_argument('--out-dir', type=Path, help='覆盖本步骤的输出目录；Step2 为结果根目录')
    p.add_argument('--max-records', type=int, help='仅扫描输入前 N 行，自动标记 partial_test；只用于联调')
    p.add_argument('--dry-run', action='store_true', help='只审计输入及显示计划，不生成结果')
    return p


def load_config(args):
    import yaml
    c = yaml.safe_load(args.config.read_text())
    if not isinstance(c, dict) or not isinstance(c.get('models'), dict):
        raise ValueError('Expected a YAML mapping with models')
    versions = [args.version] if args.version else (args.versions or c['enabled_versions'])
    if not versions or len(set(versions)) != len(versions):
        raise ValueError('Model selection is empty or duplicated')
    for v in versions:
        valid_model_name(v)
        if v not in c['models']:
            raise ValueError(f'Model not configured: {v}')
    for key in ('data_root', 'results_root', 'comparison_dir'):
        if not Path(c[key]).is_absolute():
            raise ValueError(f'{key} must be an absolute path')
    for v, m in c['models'].items():
        for key in ('feature_dir', 'runs_root'):
            if not Path(m[key]).is_absolute():
                raise ValueError(f'models.{v}.{key} must be absolute')
    library = c['library_id']
    if not isinstance(library, str) or not library.strip() or any(x in library for x in '\r\n\t'):
        raise ValueError('Invalid library_id')
    if args.max_records is not None and args.max_records < 1:
        raise ValueError('max-records must be positive')
    stat = c['statistics']
    if (stat['bootstrap'] < 0 or stat['block_size'] < 1 or
            not stat['coverage_minima'] or any(type(x) is not int or x < 1 for x in stat['coverage_minima'])):
        raise ValueError('Invalid statistical settings')
    for t in [c['evaluation'].get('read_threshold'), c['evaluation']['site_threshold'], *stat['thresholds']]:
        if t is not None and (not isinstance(t, (float, int)) or not 0 < t < 1):
            raise ValueError('Thresholds must be strictly between zero and one')
    return c, versions


def identity(c, partial=False):
    return dict(dataset_kind=KIND, truth_source=TRUTH, library_id=c['library_id'],
                reference_label_rule='original_test_label_equals_1', source_split='test',
                reference_validation_status=c.get('reference_validation_status', 'unverified'),
                partial_test=bool(partial))


def prediction_db(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA temp_store=FILE')
    db.execute('''CREATE TABLE predictions (
        library_id TEXT NOT NULL, read_id TEXT NOT NULL, chrom TEXT NOT NULL,
        position INTEGER NOT NULL, strand TEXT NOT NULL, source TEXT NOT NULL,
        seq TEXT NOT NULL, ref TEXT NOT NULL, prob REAL NOT NULL,
        source_sample_index INTEGER NOT NULL, mapq INTEGER, signal_len INTEGER,
        PRIMARY KEY(library_id,read_id,chrom,position,strand)) WITHOUT ROWID''')
    return db


def reader(f, required=REQUIRED):
    r = csv.DictReader(f, delimiter='\t')
    if (not required.issubset(r.fieldnames or []) or len(r.fieldnames) != len(set(r.fieldnames))):
        raise ValueError('Missing or duplicate TSV columns')
    return r


def validate_row(row, index, library, chrom='chr16', window=None):
    if None in row or any(v is None for v in row.values()):
        raise ValueError(f'Malformed metadata/prediction at row {index}')
    if row.get('split', 'test') != 'test' or row.get('library_id', library) != library:
        raise ValueError('Only matching native test library may be evaluated')
    if row.get('sample_index') and int(row['sample_index']) != index:
        raise ValueError(f'Nonaligned sample_index at row {index}')
    label = float(row['label'])
    if label not in (0., 1.):
        raise ValueError('Nonbinary reference label')
    if not row['read_id'] or row['chrom'] != chrom or row['strand'] != '+' or int(row['center_genome_pos0']) < 0:
        raise ValueError('Expected native chr16 positive-strand test observations')
    if row['source'] != 'mod':
        raise ValueError('Expected original native source=mod, not WGA/UNMOD')
    for key in ('seq', 'ref'):
        if not row[key] or (window is not None and len(row[key]) != window) or set(row[key].upper())-set('ACGTNWSMKRYBVDHZ'):
            raise ValueError(f'Invalid {key} at row {index}')
    if len(row['seq']) != len(row['ref']) or len(row['ref']) % 2 != 1:
        raise ValueError('Expected matching odd-width seq/ref')
    return int(label)


def observation(row, p, index, library):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(f'Nonfinite or out-of-range probability at row {index}')
    mq = int(row['mapq']) if row.get('mapq') else None
    sl = row.get('input_signal_len') or row.get('signal_len')
    sl = int(sl) if sl else None
    if (mq is not None and mq < 0) or (sl is not None and sl < 1):
        raise ValueError('Invalid quality metadata')
    return (library, row['read_id'], row['chrom'], int(row['center_genome_pos0']), row['strand'],
            row['source'], row['seq'].upper(), row['ref'].upper(), p,
            int(row.get('source_sample_index', index)), mq, sl)


def insert(db, rows):
    try:
        db.executemany('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows)
    except sqlite3.IntegrityError as e:
        raise ValueError('Duplicate original read/position observation') from e


def proportion(fn, n):
    n, fn = int(n), int(fn or 0)
    return dict(n=n, n_positive=n, fn=fn, tp=n-fn, fnr=fn/n if n else None,
                recall=1-fn/n if n else None, wilson_ci95_independent_approx=wilson(fn, n),
                zero_fn_upper95_independent_approx=-math.expm1(math.log(.05)/n) if n and not fn else None)


def block_ci(blocks, repetitions, seed, paired=False):
    import numpy as np
    data = np.asarray(list(blocks), dtype=float)
    if len(data) < 2 or repetitions < 1:
        return None
    if not paired and data[:, 0].sum() in (0, data[:, 1].sum()):
        return None
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repetitions):
        counts = data[rng.integers(len(data), size=len(data))].sum(axis=0)
        scores.append((counts[0]/counts[1]-counts[2]/counts[3]) if paired else counts[0]/counts[1])
    lo, hi = np.quantile(scores, [.025, .975]).tolist()
    return [lo, hi] if hi > lo else None


def prepare_sites(db, threshold):
    db.create_function('ln', 1, math.log)
    db.execute('DROP TABLE IF EXISTS temp.sites')
    db.execute(f'''CREATE TEMP TABLE sites AS SELECT {SITE_KEYS}, COUNT(*) AS coverage,
        AVG(prob) AS score, SUM(prob < ?) AS read_fn,
        MIN(ref) AS ref, MIN(seq) AS seq, COUNT(DISTINCT ref) AS ref_count
        FROM predictions GROUP BY {SITE_KEYS}''', (threshold,))
    if db.execute('SELECT 1 FROM sites WHERE ref_count != 1 LIMIT 1').fetchone():
        raise ValueError('Conflicting reference contexts within a site')
    db.execute(f'CREATE UNIQUE INDEX temp.site_index ON sites ({SITE_KEYS})')


def summarize(db, c, read_threshold, site_threshold, minimum=1):
    """Coverage is always computed from the final prediction table."""
    import numpy as np
    stat = c['statistics']
    join = f'predictions JOIN sites USING ({SITE_KEYS})'
    where = 'coverage >= ?'
    n, fn, brier, bce = db.execute(f'''SELECT COUNT(*), SUM(prob < ?),
        AVG((1.-prob)*(1.-prob)), AVG(-ln(MAX(prob,1e-15))) FROM {join} WHERE {where}''',
                                 (read_threshold, minimum)).fetchone()
    read = proportion(fn, n)
    read.update(brier=brier, bce=bce)
    nsite, fnsite = db.execute('SELECT COUNT(*), SUM(score < ?) FROM sites WHERE '+where,
                             (site_threshold, minimum)).fetchone()
    site = proportion(fnsite, nsite)
    for level, score, threshold, table in [('read', 'prob', read_threshold, join), ('site', 'score', site_threshold, 'sites')]:
        blocks = [tuple(r) for r in db.execute(f'''SELECT SUM({score} < ?), COUNT(*) FROM {table}
            WHERE {where} GROUP BY library_id,chrom,CAST(position / ? AS INTEGER)''',
                                              (threshold, minimum, stat['block_size']))]
        value = read if level == 'read' else site
        value.update(blocks=len(blocks), fnr_block_bootstrap_ci95=block_ci(blocks, stat['bootstrap'], stat['bootstrap_seed']))
        value['ci_note'] = 'Within-library genomic blocks; null for disabled, insufficient or degenerate bootstrap.'
    rates = np.asarray([r[0] for r in db.execute('SELECT 1.*read_fn/coverage FROM sites WHERE '+where, (minimum,))])
    site.update(read_fn_fraction_mean=float(rates.mean()) if len(rates) else None,
                read_fn_fraction_quantiles=dict(zip(['p50', 'p95', 'p99'], np.quantile(rates, [.5, .95, .99]).tolist())) if len(rates) else None,
                fraction_sites_with_missed_reads=float(np.mean(rates > 0)) if len(rates) else None)
    # One float64 vector, ~10 MB for the current positive subset; feature arrays remain memory mapped.
    scores = np.asarray([r[0] for r in db.execute(f'SELECT prob FROM {join} WHERE {where}', (minimum,))])
    read['probability_quantiles'] = dict(zip(['p05', 'p50', 'p95'], np.quantile(scores, [.05, .5, .95]).tolist())) if len(scores) else None
    read['fraction_near_threshold_002'] = float(np.mean(np.abs(scores-read_threshold) <= .02)) if len(scores) else None
    return dict(min_coverage=minimum, read_level=read, site_level=site)


def observation_hash(db):
    h = hashlib.sha256()
    for row in db.execute(f'SELECT {KEYS},seq,ref FROM predictions ORDER BY {KEYS}'):
        h.update((json.dumps(list(row), ensure_ascii=False, separators=(',', ':'))+'\n').encode())
    return h.hexdigest()


def export_results(db, stage, c, rt, st):
    prepare_sites(db, rt)
    slices = {str(k): summarize(db, c, rt, st, k) for k in sorted(set([1, *c['statistics']['coverage_minima']]))}
    contexts, coverage = [], []
    for minimum in sorted(set([1, *c['statistics']['coverage_minima']])):
        for ref, n, fn in db.execute(f'''SELECT predictions.ref,COUNT(*),SUM(prob < ?) FROM predictions
            JOIN sites USING ({SITE_KEYS}) WHERE coverage >= ? GROUP BY predictions.ref''', (rt, minimum)):
            contexts.append(dict(level='read', ref=ref, min_coverage=minimum, **proportion(fn, n)))
        for ref, n, fn in db.execute('SELECT ref,COUNT(*),SUM(score < ?) FROM sites WHERE coverage >= ? GROUP BY ref', (st, minimum)):
            contexts.append(dict(level='site', ref=ref, min_coverage=minimum, **proportion(fn, n)))
    for lo, hi, name in [(1,1,'1'),(2,2,'2'),(3,3,'3'),(4,4,'4'),(5,9,'5-9'),(10,19,'10-19'),(20,2**63-1,'>=20')]:
        for level, table, score, threshold in [('read', f'predictions JOIN sites USING ({SITE_KEYS})', 'prob', rt), ('site', 'sites', 'score', st)]:
            n, fn = db.execute(f'SELECT COUNT(*),SUM({score} < ?) FROM {table} WHERE coverage BETWEEN ? AND ?', (threshold, lo, hi)).fetchone()
            coverage.append(dict(level=level, coverage=name, **proportion(fn, n)))
    thresholds = []
    for minimum in sorted(set([1, *c['statistics']['coverage_minima']])):
        for level, table, score, main in [('read', f'predictions JOIN sites USING ({SITE_KEYS})', 'prob', rt), ('site', 'sites', 'score', st)]:
            for t in sorted(set([*c['statistics']['thresholds'], main])):
                n, fn = db.execute(f'SELECT COUNT(*),SUM({score} < ?) FROM {table} WHERE coverage >= ?', (t, minimum)).fetchone()
                thresholds.append(dict(level=level, min_coverage=minimum, threshold=t, **proportion(fn, n)))
    counts = ['n', 'fn', 'tp', 'fnr', 'recall']
    tsv(stage/'context_metrics.tsv', contexts, ['level','ref','min_coverage',*counts])
    tsv(stage/'coverage_metrics.tsv', coverage, ['level','coverage',*counts])
    tsv(stage/'threshold_metrics.tsv', thresholds, ['level','min_coverage','threshold',*counts])
    def rows(query, params=()):
        for r in db.execute(query, params):
            d = dict(r)
            d.update(label=1, predicted_positive=int(d.get('prob', d.get('score', 0)) >= (rt if 'prob' in d else st)))
            if 'coverage' in d:
                d['read_fn_fraction'] = d['read_fn']/d['coverage']
            yield d
    fields = [*FIELDS, 'label', 'predicted_positive']
    if c['evaluation']['save_read_predictions']:
        tsv(stage/'read_predictions.tsv.gz', rows('SELECT * FROM predictions'), fields)
    tsv(stage/'false_negative_windows.tsv.gz', rows('SELECT * FROM predictions WHERE prob < ?', (rt,)), fields)
    fields = [*SITE_KEYS.split(','), 'coverage', 'score', 'read_fn', 'read_fn_fraction', 'ref', 'label', 'predicted_positive']
    tsv(stage/'site_predictions.tsv.gz', rows('SELECT * FROM sites'), fields)
    tsv(stage/'false_negative_sites.tsv.gz', rows('SELECT * FROM sites WHERE score < ?', (st,)), fields)
    return dict(**slices['1'], coverage_slices=slices, read_threshold=rt, site_threshold=st,
                observation_sha256=observation_hash(db), bce_epsilon=1e-15,
                upstream_reference_positive_count=None, upstream_call_rate=None,
                no_call_note='Original BED/JSONL candidate denominator unavailable; all cached rows must predict successfully.')
