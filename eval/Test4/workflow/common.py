"""Test4 contracts, transactional stages and continuous methylation statistics."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ESTIMATORS = ('beta_call', 'beta_prob')
METRICS = ('pearson_r', 'spearman_rho', 'mae', 'rmse', 'bias', 'slope', 'intercept')
BOOT_METRICS = tuple(m for m in METRICS if m != 'spearman_rho')
STRATEGIES = ('center_base_linear_raw768', 'center_5mer_linear_raw768',
              'center_5mer_mlp_raw768', 'center_5mer_residual_dwconv_raw768', 'center_5mer_emb8_none')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def signature(path, content=False):
    p = Path(path).resolve(); s = p.stat()
    result = dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)
    if content:
        result['sha256'] = sha(p)
    return result


def check_signatures(records):
    for saved in records:
        if signature(saved['path'], 'sha256' in saved) != saved:
            raise ValueError(f'Input changed: {saved["path"]}')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, p)


def open_text(path, mode='rt'):
    p = Path(path)
    return gzip.open(p, mode, encoding='utf-8', newline='') if p.suffix == '.gz' else p.open(mode.replace('t', ''), encoding='utf-8', newline='')


def tsv(path, rows, fields):
    with open_text(path, 'wt') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def read_tsv(path):
    with open_text(path) as f:
        yield from csv.DictReader(f, delimiter='\t')


def module(name, path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(name, p,
            submodule_search_locations=[str(p.parent)] if p.name == '__init__.py' else None)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


@contextlib.contextmanager
def lock(path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'Another process owns {p}') from e
        yield


@contextlib.contextmanager
def stage_directory(path):
    p = Path(path)
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.' + p.name + '.partial-', dir=p.parent))
    try:
        yield tmp
        if p.exists():
            raise FileExistsError(p)
        tmp.rename(p)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def code_hashes():
    paths = [*HERE.glob('*.py'), *PROJECT.glob('train/model/*.py'),
             *PROJECT.glob('workflow/step4_extract_window_hidden_states_*.py'),
             PROJECT/'eval/Test1/workflow/common.py']
    return {str(p.relative_to(PROJECT)): sha(p) for p in sorted(paths)}


def seal(path, c, **extra):
    p = Path(path)
    m = dict(config_sha256=digest(c), files_sha256={str(f.relative_to(p)): sha(f)
             for f in sorted(p.rglob('*')) if f.is_file() and f != p/'manifest.json'}, **extra)
    write_json(p/'manifest.json', m)
    return m


def verify(path, c):
    p = Path(path); m = read_json(p/'manifest.json')
    if m['config_sha256'] != digest(c):
        raise ValueError(f'Configuration mismatch: {p}')
    for name, value in m['files_sha256'].items():
        if sha(p/name) != value:
            raise ValueError(f'Artifact hash mismatch: {p/name}')
    return m


def frozen(c, root):
    m = verify(root/'audit', c)
    if yaml.safe_load((root/'protocol.yaml').read_text()) != c:
        raise ValueError('Frozen protocol differs; choose a new protocol directory')
    if m['code_sha256'] != code_hashes():
        raise ValueError('Pipeline/model implementation changed since Step0')
    if m['partial_test'] and not c['allow_partial_results']:
        raise ValueError('Partial inputs require isolated allow_partial_results=true')
    check_signatures(m['inputs'])
    return m


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=HERE/'config.yaml')
    p.add_argument('--protocol-dir', type=Path)
    p.add_argument('--max-records', type=int, help='Prefix smoke only; must be identical at every step')
    p.add_argument('--dry-run', action='store_true')
    return p


def load_config(args):
    c = yaml.safe_load(args.config.read_text())
    expected = {'protocol_id', 'output_dir', 'library_id', 'chromosome', 'strand', 'inputs',
                'reference', 'preprocessing', 'filters', 'scan', 'runtime', 'bootstrap',
                'plots', 'enabled_versions', 'strategies', 'training_seeds', 'models', 'allow_partial_results'}
    if not isinstance(c, dict) or set(c) != expected:
        raise ValueError(f'Configuration keys must be exactly {sorted(expected)}')
    if args.protocol_dir:
        c['output_dir'] = str(args.protocol_dir.resolve())
    if args.max_records is not None:
        c['scan']['max_records'] = args.max_records
    for k in ('protocol_id', 'library_id'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', str(c[k])):
            raise ValueError(f'Invalid {k}')
    if c['chromosome'] != 'chr16' or c['strand'] != '+':
        raise ValueError('This implementation supports chr16 positive-strand CpG only')
    for k in ('wgbs_bed', 'signal_jsonl'):
        if not Path(c['inputs'][k]).is_absolute():
            raise ValueError(f'Input path must be absolute: {k}')
    if not Path(c['output_dir']).is_absolute():
        raise ValueError('output_dir must be absolute')
    for k in ('min_mapq', 'min_wgbs_coverage', 'min_common_reads'):
        x = c['filters'][k]
        if type(x) is not int or x < (0 if k == 'min_mapq' else 1):
            raise ValueError(f'Invalid {k}')
    if not 0 < c['filters']['read_call_threshold'] < 1:
        raise ValueError('read_call_threshold must be in (0,1)')
    if type(c['filters']['require_query_reference_match']) is not bool:
        raise ValueError('require_query_reference_match must be boolean')
    for k in ('shard_windows', 'checkpoint_records', 'log_every_records'):
        if type(c['scan'][k]) is not int or c['scan'][k] < 1:
            raise ValueError(f'Invalid scan.{k}')
    maximum = c['scan']['max_records']
    if maximum is not None and (type(maximum) is not int or maximum < 1 or not c['allow_partial_results']):
        raise ValueError('max_records requires positive integer and allow_partial_results=true')
    region = c['scan']['region']
    if region is not None and (len(region) != 2 or any(type(x) is not int for x in region)
                               or not 0 <= region[0] < region[1]):
        raise ValueError('region must be null or [start0,end0]')
    for k in ('cpu_threads', 'head_batch_size'):
        if type(c['runtime'][k]) is not int or c['runtime'][k] < 1:
            raise ValueError(f'Invalid runtime.{k}')
    for k in ('repeats', 'seed', 'block_size_bp', 'minimum_blocks'):
        if type(c['bootstrap'][k]) is not int or c['bootstrap'][k] < (0 if k in ('repeats', 'seed') else 1):
            raise ValueError(f'Invalid bootstrap.{k}')
    if type(c['allow_partial_results']) is not bool or type(c['plots']['enabled']) is not bool:
        raise ValueError('Boolean settings must be YAML booleans')
    if type(c['plots']['max_points']) is not int or c['plots']['max_points'] < 1:
        raise ValueError('plots.max_points must be positive')
    vs = c['enabled_versions']
    if not vs or len(vs) != len(set(vs)) or any(v not in c['models'] for v in vs):
        raise ValueError('Invalid enabled_versions')
    if len(c['strategies']) != len(set(c['strategies'])) or set(c['strategies']) - set(STRATEGIES):
        raise ValueError('Unknown/duplicate classifier strategies')
    if len(c['training_seeds']) != len(set(c['training_seeds'])) or any(type(s) is not int or s < 0 for s in c['training_seeds']):
        raise ValueError('Invalid training_seeds')
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=c['runtime']['cpu_threads'])
    except ImportError:
        pass
    return c, Path(c['output_dir'])


def readiness(c):
    issues = []
    for k in ('wgbs_bed', 'signal_jsonl'):
        if not Path(c['inputs'][k]).is_file():
            issues.append(f'Missing inputs.{k}: {c["inputs"][k]}')
    r = c['reference']
    if r['source'] != 'independent_wgbs' or not r['source_evidence'].strip():
        issues.append('reference.source/source_evidence must document independent WGBS')
    if not r['build'] or str(r['build']).startswith('pending'):
        issues.append('Set reference.build from WGBS/BAM provenance, not a guess')
    if not r['fasta'] or not Path(r['fasta']).is_file():
        issues.append('Set reference.fasta to the matching reference FASTA')
    if r['coordinate_system'] != 'bed0' or r['strand_mode'] != 'strand_specific':
        issues.append('Only verified bed0, strand-specific WGBS is implemented; convert other formats explicitly')
    if not r['coordinate_evidence'].strip():
        issues.append('Document reference.coordinate_evidence for the BED conversion')
    if c['preprocessing']['status'] != 'training_matched' or not c['preprocessing']['evidence'].strip():
        issues.append('Verify preprocessing matches the existing classifier training inputs; set status=training_matched and evidence')
    return issues


def load_chromosome(path, chrom):
    """Read one contig without requiring pysam or a pre-existing FASTA index."""
    pieces = []; active = False
    with open_text(path) as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].split()[0]
                if active:
                    break
                active = name == chrom
            elif active:
                pieces.append(line.strip().upper())
    sequence = ''.join(pieces)
    if not sequence or not set(sequence) <= set('ACGTNRYWSMKBDHV'):
        raise ValueError(f'Missing/invalid reference contig {chrom}')
    return sequence


def ranks(a):
    a = np.asarray(a); order = np.argsort(a, kind='stable'); sorted_a = a[order]
    if not len(a):
        return np.empty(0, dtype=float)
    ends = np.r_[np.flatnonzero(sorted_a[1:] != sorted_a[:-1]) + 1, len(a)]
    starts = np.r_[0, ends[:-1]]; result = np.empty(len(a), dtype=float)
    result[order] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
    return result


def moment_metrics(s):
    """Mergeable exact sums: n,sumx,sumy,sumxx,sumyy,sumxy,sumabs,sumerr2."""
    n, sx, sy, xx, yy, xy, ae, se = map(float, s)
    out = dict.fromkeys(BOOT_METRICS)
    if n < 1:
        return out
    vx = max(0., xx-sx*sx/n); vy = max(0., yy-sy*sy/n); cov = xy-sx*sy/n
    out.update(mae=ae/n, rmse=math.sqrt(max(0., se/n)), bias=(sy-sx)/n)
    if n >= 2 and vx > 1e-14 and vy > 1e-14:
        out['pearson_r'] = max(-1., min(1., cov/math.sqrt(vx*vy)))
    if n >= 2 and vx > 1e-14:
        out['slope'] = cov/vx; out['intercept'] = (sy-out['slope']*sx)/n
    return out


def moments(x, y):
    return np.array([len(x), x.sum(), y.sum(), x@x, y@y, x@y,
                     np.abs(y-x).sum(), (y-x)@(y-x)], dtype=float)


def correlation_metrics(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid paired beta arrays')
    if np.any((x < 0) | (x > 1) | (y < 0) | (y > 1)):
        raise ValueError('Beta values must be in [0,1]')
    out = moment_metrics(moments(x, y)); rho = None
    if len(x) >= 2:
        rx, ry = ranks(x), ranks(y)
        rho = moment_metrics(moments(rx, ry))['pearson_r']
    return dict(n_sites=len(x), **out, spearman_rho=rho,
                correlation_status='ok' if out['pearson_r'] is not None else 'insufficient_or_constant')


def sample_summary(values):
    a = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    return dict(mean=float(a.mean()) if len(a) else None,
                variance=float(a.var(ddof=1)) if len(a) > 1 else None, n_valid=len(a))


def interval(values):
    a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
    if not len(a):
        return dict(ci_low=None, ci_high=None, ci_status='insufficient_or_disabled', bootstrap_valid=0)
    return dict(ci_low=float(np.quantile(a, .025)), ci_high=float(np.quantile(a, .975)),
                ci_status='degenerate' if np.ptp(a) == 0 else 'ok', bootstrap_valid=len(a))
