"""Frozen Test3 protocols, generic I/O, binary metrics and paired statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT/'eval/Test3'
spec = importlib.util.spec_from_file_location('_test3_io', PROJECT/'eval/Test1/workflow/common.py')
_io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_io)
for name in ('read_json','write_json','signature','sha256','output_directory','open_text','tsv','readonly_db','infer_architecture'):
    globals()[name] = getattr(_io,name)

KEYS = ['library_id','read_id','chrom','position','strand']
SITE = ['library_id','chrom','position','strand']
META = ['read_id','chrom','center_genome_pos0','strand','source','label','seq','ref']
METRICS = ['auroc','average_precision','balanced_accuracy','accuracy','precision','recall','f1','mcc',
           'fpr','fnr','specificity','brier','bce']
LOWER = {'fpr','fnr','brier','bce'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def ranking(*parts):
    return hashlib.sha256(json.dumps(list(parts),ensure_ascii=False,separators=(',',':')).encode()).digest()


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'),help='共享 YAML 配置')
    p.add_argument('--protocol-dir',type=Path,help='覆盖本次协议的输出根目录；所有步骤须一致')
    p.add_argument('--dry-run',action='store_true',help='只读检查/显示计划，不创建结果')
    return p


def load_config(args):
    c = yaml.safe_load(args.config.read_text())
    if not isinstance(c,dict): raise ValueError('Expected YAML mapping')
    if args.protocol_dir: c['output_dir'] = str(args.protocol_dir.resolve())
    if not Path(c['output_dir']).is_absolute(): raise ValueError('output_dir must be absolute')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',c['protocol_id']): raise ValueError('Invalid protocol_id')
    for field in ('library_id','chromosome','strand','reference_validation_status'):
        if not isinstance(c[field],str) or not c[field].strip() or any(x in c[field] for x in '\t\r\n'):
            raise ValueError(f'Invalid {field}')
    if c['strand'] != '+': raise ValueError('Current native test protocol supports positive strand only')
    versions=c['enabled_versions']
    if not versions or len(set(versions))!=len(versions): raise ValueError('Empty/duplicate model list')
    for v in versions:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',v) or v not in c['models']:
            raise ValueError(f'Invalid/unconfigured model: {v}')
        for field in ('feature_dir','runs_root'):
            if not Path(c['models'][v][field]).is_absolute(): raise ValueError('Model paths must be absolute')
    depths=c['depths']
    if not depths or any(type(d) is not int or d<1 for d in depths) or depths!=sorted(set(depths)):
        raise ValueError('depths must be increasing unique positive integers')
    if type(c['minimum_common_reads']) is not int or c['minimum_common_reads']<max(depths):
        raise ValueError('minimum_common_reads must support every depth')
    for k in ('repeats','cpu_threads','batch_size','site_chunk_size'):
        if type(c[k]) is not int or c[k]<1: raise ValueError(f'{k} must be positive integer')
    if not isinstance(c['site_threshold'],(int,float)) or not 0<c['site_threshold']<1: raise ValueError('Invalid site threshold')
    if c['site_balance'] not in ('none','equal'): raise ValueError('site_balance must be none or equal')
    if c['prediction_source'] not in ('saved','inference'): raise ValueError('prediction_source must be saved/inference')
    if (type(c['bootstrap']['repeats']) is not int or c['bootstrap']['repeats']<0 or
            type(c['bootstrap']['block_size_bp']) is not int or c['bootstrap']['block_size_bp']<1):
        raise ValueError('Invalid bootstrap settings')
    for value in (c['sampling_seed'],c['balance_seed'],c['bootstrap']['seed']):
        if type(value) is not int or value<0: raise ValueError('Random seeds must be nonnegative integers')
    for value in (c['save_all_site_scores'],c['save_site_stability'],c['allow_partial_results'],c['plots']['enabled']):
        if type(value) is not bool: raise ValueError('Boolean settings must be YAML true/false')
    if set(c['bootstrap']['metrics'])-{'balanced_accuracy','fpr','fnr'}:
        raise ValueError('Block confusion bootstrap supports BA/FPR/FNR; it cannot approximate AUROC/AP')
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=c['cpu_threads'])
    except ImportError:
        pass  # Shell entrypoints also limit BLAS threads before NumPy imports.
    return c,Path(c['output_dir'])


def frozen(c,root):
    m=read_json(root/'manifests/audit_manifest.json')
    saved=yaml.safe_load((root/'protocol.yaml').read_text())
    if digest(c)!=m['config_sha256'] or saved!=c: raise ValueError('Configuration changed since Step1; create a new protocol')
    if m['partial_test'] and not c['allow_partial_results']:
        raise ValueError('Partial smoke data forbidden; use isolated config allow_partial_results=true')
    return m


def manifest(c,stage,**extra):
    files={str(p.relative_to(stage)):sha256(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    result=dict(config_sha256=digest(c),files_sha256=files,**extra)
    write_json(stage/'manifest.json',result)
    return result


def verify(stage,c):
    m=read_json(stage/'manifest.json')
    if m['config_sha256']!=digest(c): raise ValueError('Stage configuration differs from frozen protocol')
    for name,h in m['files_sha256'].items():
        if sha256(stage/name)!=h: raise ValueError(f'Stage file changed: {stage/name}')
    return m


def stage_signatures(stage,m):
    return [signature(stage/'manifest.json'),*[signature(stage/name) for name in m['files_sha256']]]


def unchanged(snapshots):
    if [signature(Path(s['path'])) for s in snapshots]!=snapshots:
        raise ValueError('Input files changed while processing')


def code_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def read_tsv(path,required=()):
    with open_text(path) as f:
        r=csv.DictReader(f,delimiter='\t')
        if not set(required).issubset(r.fieldnames or []) or len(r.fieldnames or [])!=len(set(r.fieldnames or [])):
            raise ValueError(f'Missing or duplicate TSV columns: {path}')
        for row in r:
            if None in row or any(v is None for v in row.values()): raise ValueError(f'Malformed TSV: {path}')
            yield row


def discover(c):
    result=[]
    for v in c['enabled_versions']:
        root=Path(c['models'][v]['runs_root'])
        if (root/'runs').is_dir(): root/='runs'
        selected=[]
        for file in sorted(root.glob('*/metrics.json')):
            m=read_json(file); cfg=read_json(file.parent/'config.json')
            if c['strategies'] and m['strategy'] not in c['strategies']: continue
            if c['training_seeds'] and m['seed'] not in c['training_seeds']: continue
            if any(m.get(k)!=cfg.get(k) for k in ('strategy','seed','fingerprint')): raise ValueError('Run/config identity mismatch')
            if not re.fullmatch(re.escape(v)+r'(?:[-_].*)?',cfg['experiment'],re.I): raise ValueError('Run version mismatch')
            if file.parent.name!=f'{m["strategy"]}_seed{m["seed"]}': raise ValueError('Noncanonical run name')
            if Path(cfg['feature_dir']).resolve()!=Path(c['models'][v]['feature_dir']).resolve(): raise ValueError('Run uses another feature cache')
            for name in ['best.pt',*(['test_predictions.tsv'] if c['prediction_source']=='saved' else [])]:
                if not (file.parent/name).is_file(): raise FileNotFoundError(file.parent/name)
            selected.append(dict(version=v,strategy=m['strategy'],training_seed=m['seed'],
                                 run_name=file.parent.name,run_dir=str(file.parent.resolve())))
        if not selected: raise ValueError(f'No completed runs selected for {v}')
        if len({(r['strategy'],r['training_seed']) for r in selected})!=len(selected): raise ValueError('Duplicate run identity')
        if c['strategies'] and {r['strategy'] for r in selected}!=set(c['strategies']): raise ValueError(f'Requested strategy missing: {v}')
        result.extend(selected)
    strategies={r['strategy'] for r in result}
    if any({r['strategy'] for r in result if r['version']==v}!=strategies for v in c['enabled_versions']):
        raise ValueError('All selected versions must have the same strategy set; freeze a smaller scope explicitly')
    return result


def confusion(y,p,t):
    y=np.asarray(y,dtype=np.int8); pred=np.asarray(p)>=t
    return np.asarray([np.sum(pred&(y==1)),np.sum(~pred&(y==0)),np.sum(pred&(y==0)),np.sum(~pred&(y==1))],dtype=np.int64)


def count_metrics(counts):
    x=np.asarray(counts,dtype=float)
    tp,tn,fp,fn=np.moveaxis(x,-1,0)
    def divide(a,b): return np.divide(a,b,out=np.full(np.broadcast(a,b).shape,np.nan),where=b!=0)
    recall=divide(tp,tp+fn); specificity=divide(tn,tn+fp)
    return dict(balanced_accuracy=(recall+specificity)/2,fpr=divide(fp,fp+tn),fnr=divide(fn,fn+tp))


def binary_metrics(y,p,threshold):
    y,p=np.asarray(y),np.asarray(p,dtype=np.float64)
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid labels or probabilities for binary metrics')
    tp,tn,fp,fn=map(int,confusion(y,p,threshold)); n=len(y); n1=tp+fn; n0=tn+fp
    recall=tp/n1 if n1 else None; specificity=tn/n0 if n0 else None
    precision=tp/(tp+fp) if tp+fp else 0.
    auroc=ap=None
    if n1 and n0:
        order=np.argsort(-p,kind='stable'); scores=p[order]
        ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),n-1]
        tps=np.cumsum(y[order]==1)[ends]; fps=ends+1-tps
        tpr=np.r_[0,tps/n1]; fpr=np.r_[0,fps/n0]
        auroc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
        ap=float(np.sum(np.diff(np.r_[0,tps])*tps/(ends+1))/n1)
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(n_sites=n,n_label0=n0,n_label1=n1,tp=tp,tn=tn,fp=fp,fn=fn,
        auroc=auroc,average_precision=ap,balanced_accuracy=(recall+specificity)/2 if n0 and n1 else None,
        accuracy=(tp+tn)/n,precision=precision,recall=recall,specificity=specificity,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,mcc=(tp*tn-fp*fn)/den if den else 0.,
        fpr=fp/n0 if n0 else None,fnr=fn/n1 if n1 else None,
        brier=float(np.mean((y-p)**2)),bce=float(np.mean(-y*np.log(np.clip(p,1e-15,1))-(1-y)*np.log(np.clip(1-p,1e-15,1)))))


def statistics(values):
    a=np.asarray([v for v in values if v is not None and math.isfinite(v)],dtype=float)
    return dict(mean=float(a.mean()) if len(a) else None,sd=float(a.std(ddof=1)) if len(a)>1 else None,
                q025=float(np.quantile(a,.025)) if len(a) else None,q975=float(np.quantile(a,.975)) if len(a) else None,n_valid=len(a))


def interval(draws):
    a=np.asarray(draws,dtype=float); a=a[np.isfinite(a)]
    if len(a)<2: return dict(ci_low=None,ci_high=None,ci_status='insufficient_or_disabled',bootstrap_valid=len(a))
    lo,hi=map(float,np.quantile(a,[.025,.975]))
    return dict(ci_low=lo if hi>lo else None,ci_high=hi if hi>lo else None,
                ci_status='ok' if hi>lo else 'degenerate',bootstrap_valid=len(a))
