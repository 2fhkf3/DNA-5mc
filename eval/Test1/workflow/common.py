"""Shared I/O, provenance and negative-control statistics for Test1."""
from __future__ import annotations

import contextlib
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
TEST_ROOT = Path(__file__).resolve().parents[1]
RAW_INPUT = Path('/mnt/zzbnew/rnamodel/jinyue/data/processed_data/chr19/unmod/'
                 's2_all_reads.genome_signal.preprocessed.jsonl')
VERSIONS = ('V011', 'V543', 'V610')
KEYS = ('library_id', 'read_id', 'chrom', 'position', 'strand')


def valid_model_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Model version must be a nonempty name containing letters, digits, _, - or .')
    return value


def infer_architecture(config):
    explicit = config.get('model_architecture')
    if explicit in ('diffusion', 'bert', 'gpt'):
        return explicit
    names = ' '.join(str(config.get(k, '')) for k in ['model_type', 'model_family', 'architectures']).lower()
    if 'diffusion' in names or 'poredlm' in names:
        return 'diffusion'
    if 'bert' in names:
        return 'bert'
    if 'gpt' in names or 'olmo' in names:
        return 'gpt'
    if config.get('hidden_source') is not None:
        return 'diffusion'
    if config.get('hidden_layer') is not None:
        return 'bert'
    if config.get('decoder_layers') is not None:
        return 'gpt'
    return None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def signature(path):
    path = Path(path)
    st = path.stat()
    return dict(path=str(path.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def open_text(path, mode='rt'):
    path = Path(path)
    return gzip.open(path, mode, encoding='utf-8', newline='') if path.suffix == '.gz' else path.open(mode.replace('t', ''), encoding='utf-8', newline='')


@contextlib.contextmanager
def output_directory(path):
    """Publish only a complete new result; never overwrite an old experiment."""
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}; choose a new directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{path.name}.partial-', dir=path.parent))
    try:
        yield stage
        if path.exists():
            raise FileExistsError(path)
        stage.rename(path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def tsv(path, rows, fields):
    with open_text(path, 'wt') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def readonly_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def prediction_db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA temp_store=FILE')
    db.execute('''CREATE TABLE predictions (
        library_id TEXT NOT NULL, read_id TEXT NOT NULL, chrom TEXT NOT NULL,
        position INTEGER NOT NULL, strand TEXT NOT NULL, source TEXT NOT NULL,
        seq TEXT NOT NULL, ref TEXT NOT NULL, prob REAL NOT NULL,
        mapq INTEGER, signal_len INTEGER,
        PRIMARY KEY (library_id, read_id, chrom, position, strand)) WITHOUT ROWID''')
    return db


def wilson(fp, n):
    if not n:
        return None
    z = 1.959963984540054
    p = fp / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., center-half), min(1., center+half)]


def proportion(fp, n):
    return dict(n=int(n), fp=int(fp), tn=int(n-fp), fpr=fp/n if n else None,
                specificity=1-fp/n if n else None,
                wilson_ci95_independent_approx=wilson(fp, n),
                zero_fp_upper95_independent_approx=-math.expm1(math.log(.05)/n) if n and not fp else None)


def block_interval(blocks, repetitions, seed):
    import numpy as np
    data = np.asarray(list(blocks.values()), dtype=np.int64)
    if len(data) < 2 or not repetitions or data[:, 0].sum() in (0, data[:, 1].sum()):
        return None
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repetitions):
        counts = data[rng.integers(len(data), size=len(data))].sum(axis=0)
        if counts[1]:
            scores.append(counts[0]/counts[1])
    return np.quantile(scores, [.025, .975]).tolist() if scores else None


def coverage_name(n):
    return str(n) if n < 5 else ('5-9' if n < 10 else '>=10')


def summarize(db, table='predictions', prob='prob', read_threshold=.5, site_threshold=.5,
              block_size=1_000_000, bootstrap=1000, seed=42, site_path=None, fp_path=None):
    """SQL aggregation is bounded in Python memory by number of genomic blocks."""
    import numpy as np
    from collections import defaultdict
    n, fp, brier, bce = db.execute(
        f'SELECT COUNT(*), SUM({prob}>=?), SUM({prob}*{prob}), '
        f'SUM(-ln(MAX(1.-{prob}, 1e-15))) FROM {table}', (read_threshold,)).fetchone()
    if not n:
        raise ValueError('No callable negative-control observations')
    read = proportion(fp, n)
    read.update(brier=brier/n, bce=bce/n, fp_per_million=fp/n*1e6)
    read_blocks = {
        (lib, chrom, block): [k, count]
        for lib, chrom, block, k, count in db.execute(
            f'SELECT library_id, chrom, CAST(position/? AS INTEGER), SUM({prob}>=?), COUNT(*) '
            f'FROM {table} GROUP BY library_id, chrom, CAST(position/? AS INTEGER)',
            (block_size, read_threshold, block_size))}
    read['block_bootstrap_ci95'] = block_interval(read_blocks, bootstrap, seed)
    bins = {k: [0, 0] for k in ['1', '2', '3', '4', '5-9', '>=10']}
    cumulative = {k: [0, 0] for k in [1, 3, 5]}
    site_blocks = defaultdict(lambda: [0, 0])
    context = defaultdict(lambda: [0, 0])
    count = site_fp = 0
    sum_q = sum_q2 = sum_score = 0.
    query = (f'SELECT library_id, chrom, position, strand, COUNT(*), AVG({prob}), '
             f'SUM({prob}>=?), MIN(source) FROM {table} '
             'GROUP BY library_id, chrom, position, strand ORDER BY library_id, chrom, position, strand')
    fields = ['library_id', 'chrom', 'center_genome_pos0', 'strand', 'source', 'label',
              'n_reads', 'prob_mean', 'positive_reads', 'positive_read_fraction', 'predicted_positive']
    with contextlib.ExitStack() as stack:
        writers = []
        for path in [site_path, fp_path]:
            if path:
                f = stack.enter_context(open_text(path, 'wt'))
                writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                writers.append(writer)
            else:
                writers.append(None)
        for lib, chrom, pos, strand, depth, score, positives, source in db.execute(query, (read_threshold,)):
            called = int(score >= site_threshold)
            q = positives/depth
            row = dict(zip(fields, [lib, chrom, pos, strand, source, 0, depth, score, positives, q, called]))
            if writers[0]:
                writers[0].writerow(row)
            if called and writers[1]:
                writers[1].writerow(row)
            count += 1
            site_fp += called
            sum_q += q
            sum_q2 += q*q
            sum_score += score
            bins[coverage_name(depth)][0] += called
            bins[coverage_name(depth)][1] += 1
            for minimum, values in cumulative.items():
                if depth >= minimum:
                    values[0] += called
                    values[1] += 1
            values = site_blocks[(lib, chrom, pos//block_size)]
            values[0] += called
            values[1] += 1
    site = proportion(site_fp, count)
    site.update(mean_reads_per_site=n/count, mean_score=sum_score/count,
                positive_read_fraction_mean=sum_q/count, positive_read_fraction_mae=sum_q/count,
                positive_read_fraction_rmse=math.sqrt(sum_q2/count),
                block_bootstrap_ci95=block_interval(site_blocks, bootstrap, seed))
    # Group by ref k-mer after normalizing windows in the evaluator.
    for kmer, k, total in db.execute(f'SELECT ref, SUM({prob}>=?), COUNT(*) FROM {table} GROUP BY ref', (read_threshold,)):
        context[kmer] = [k, total]
    metrics = dict(read_level=read, site_level=site,
                   coverage=[dict(coverage=k, **proportion(*values)) for k, values in bins.items()],
                   min_coverage=[dict(min_reads=k, **proportion(*values)) for k, values in cumulative.items()],
                   contexts=[dict(ref=k, **proportion(*values)) for k, values in sorted(context.items())],
                   thresholds=dict(read=read_threshold, site=site_threshold),
                   inference_only_metrics=dict(auroc=None, auprc=None, recall=None, precision=None, f1=None, mcc=None),
                   ci_note='Wilson/zero-FP upper bound assume independence. Block percentile CI is null '
                           'with <2 blocks, no resampling, or all-zero/all-one outcomes.',
                   bootstrap=dict(repetitions=bootstrap, seed=seed, block_size=block_size,
                                  n_blocks=len(read_blocks)))
    return metrics


def install_sql_math(db):
    # Python SQLite builds need not include ENABLE_MATH_FUNCTIONS.
    db.create_function('ln', 1, math.log, deterministic=True)
