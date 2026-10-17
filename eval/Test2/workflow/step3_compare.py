#!/usr/bin/env python3
"""Compare raw and common native-positive observations without requiring equal seed lists."""
from __future__ import annotations

import itertools
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from common import (KEYS, SITE_KEYS, KIND, TRUTH, block_ci, cli, load_config, observation_hash,
                    output_directory, prepare_sites, read_json, readonly_db, sha256,
                    signature, summarize, tsv, write_json)
from step2_evaluate import discover

FIELDS = ['version','strategy','seed','collection','min_coverage','read_threshold','site_threshold',
          'read_n','read_fn','read_tp','read_fnr','read_recall','read_fnr_ci95',
          'site_n','site_fn','site_tp','site_fnr','site_recall','site_fnr_ci95','observation_sha256']


def flat(m, s, collection, digest):
    row = {k:m[k] for k in ('version','strategy','seed','read_threshold','site_threshold')}
    row.update(collection=collection,min_coverage=s['min_coverage'],observation_sha256=digest)
    for level in ('read','site'):
        for k in ('n','fn','tp','fnr','recall'):
            row[f'{level}_{k}'] = s[level+'_level'][k]
        row[f'{level}_fnr_ci95'] = json.dumps(s[level+'_level']['fnr_block_bootstrap_ci95'])
    return row


def connect_mask(path, runs):
    """Intersect all selected runs in this strategy, independent of their seed IDs."""
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute(f'''CREATE TABLE predictions (
        library_id TEXT,read_id TEXT,chrom TEXT,position INTEGER,strand TEXT,seq TEXT,ref TEXT,
        PRIMARY KEY({KEYS})) WITHOUT ROWID''')
    for i,r in enumerate(runs):
        db.execute('ATTACH DATABASE ? AS r', ((r/'predictions.sqlite').resolve().as_uri()+'?mode=ro',))
        if i == 0:
            db.execute(f'INSERT INTO main.predictions SELECT {KEYS},seq,ref FROM r.predictions')
        else:
            join = ' AND '.join(f'a.{k}=b.{k}' for k in KEYS.split(','))
            if db.execute(f'''SELECT 1 FROM main.predictions a JOIN r.predictions b ON {join}
                WHERE a.ref != b.ref OR a.seq != b.seq LIMIT 1''').fetchone():
                raise ValueError('Common observation has conflicting seq/ref between models or seeds')
            db.execute(f'DELETE FROM main.predictions AS a WHERE NOT EXISTS (SELECT 1 FROM r.predictions b WHERE {join})')
        db.commit()
        db.execute('DETACH DATABASE r')
    if not db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]:
        db.close()
        raise ValueError('No common positive observations')
    return db


def common_stats(path, mask, c, rt, st):
    db = readonly_db(path/'predictions.sqlite')
    db.row_factory = sqlite3.Row
    try:
        db.execute('ATTACH DATABASE ? AS mask', (mask.resolve().as_uri()+'?mode=ro',))
        db.execute(f'''CREATE TEMP TABLE predictions AS SELECT a.* FROM main.predictions a
            JOIN mask.predictions b USING ({KEYS})''')
        prepare_sites(db,rt)
        return [summarize(db,c,rt,st,k) for k in sorted(set([1,*c['statistics']['coverage_minima']]))]
    finally:
        db.close()


def paired(path_a,path_b,mask,c,rt,st,minimum):
    db = sqlite3.connect(':memory:')
    try:
        for name,path in [('a',path_a/'predictions.sqlite'),('b',path_b/'predictions.sqlite'),('mask',mask)]:
            db.execute(f'ATTACH DATABASE ? AS {name}', (path.resolve().as_uri()+'?mode=ro',))
        # Pair at the same observed windows, then recompute site coverage and probabilities.
        db.execute(f'''CREATE TEMP TABLE joint AS SELECT {','.join('a.'+k for k in KEYS.split(','))},
            a.prob AS pa,b.prob AS pb FROM a.predictions a JOIN b.predictions b USING ({KEYS})
            JOIN mask.predictions m USING ({KEYS})''')
        db.execute(f'''CREATE TEMP TABLE site_pairs AS SELECT {SITE_KEYS},COUNT(*) AS coverage,
            AVG(pa) AS pa,AVG(pb) AS pb FROM joint GROUP BY {SITE_KEYS}''')
        result=[]
        for level,table,t in [('read',f'joint JOIN site_pairs USING ({SITE_KEYS})',rt),('site','site_pairs',st)]:
            pa,pb = ('joint.pa','joint.pb') if level == 'read' else ('pa','pb')
            blocks = db.execute(f'''SELECT SUM({pb} < ?),COUNT(*),SUM({pa} < ?),COUNT(*)
                FROM {table} WHERE coverage >= ? GROUP BY library_id,chrom,CAST(position / ? AS INTEGER)''',
                (t,t,minimum,c['statistics']['block_size'])).fetchall()
            n=sum(r[1] for r in blocks)
            afn,bfn=sum(r[2] for r in blocks),sum(r[0] for r in blocks)
            result.append(dict(level=level,min_coverage=minimum,n=n,fn_a=afn,fn_b=bfn,
                difference_fnr_b_minus_a=(bfn-afn)/n if n else None,blocks=len(blocks),
                block_bootstrap_ci95=json.dumps(block_ci(blocks,c['statistics']['bootstrap'],c['statistics']['bootstrap_seed'],paired=True))))
        return result
    finally:
        db.close()


def compare(c,versions,output,results_root=None,dry_run=False):
    root=Path(results_root or c['results_root'])
    expected=[(v,r.name) for v in versions for r in discover(c,v)]
    if dry_run:
        print(json.dumps(dict(results_root=str(root),out_dir=str(output),versions=versions,
            expected_results=len(expected),missing_results=[str(root/v/r) for v,r in expected if not (root/v/r/'metrics.json').is_file()],
            common_mask='intersection per strategy across all selected versions and seeds, then coverage filter'),indent=2))
        return
    runs=[]
    signatures=[]
    for v,name in expected:
        p=root/v/name
        m=read_json(p/'metrics.json')
        provenance=read_json(p/'run_manifest.json')
        if m['version'] != v or name != f'{m["strategy"]}_seed{m["seed"]}':
            raise ValueError('Result path and model identity differ')
        if (m['dataset_kind'],m['truth_source'],m['library_id']) != (KIND,TRUTH,c['library_id']):
            raise ValueError('Only matching Test2 native-positive results can be compared')
        if m['reference_validation_status'] != c.get('reference_validation_status','unverified'):
            raise ValueError('Reference provenance differs')
        if m['partial_test'] and not c['comparison']['allow_partial']:
            raise ValueError('Partial smoke results cannot enter a formal comparison')
        if provenance['statistics'] != c['statistics']:
            raise ValueError('Statistical settings changed; regenerate results or use their frozen config')
        signatures.extend(signature(p/f) for f in ('metrics.json','run_manifest.json','predictions.sqlite'))
        runs.append((p,m))
    raw=[]
    for p,m in runs:
        for s in m['coverage_slices'].values():
            raw.append(flat(m,s,'raw_callable_positives',m['observation_sha256']))
    grouped=defaultdict(list)
    for p,m in runs:
        grouped[m['strategy']].append((p,m))
    common,aggregates,differences,masks,omitted=[],[],[],[],[]
    with output_directory(output) as stage:
        for strategy,items in sorted(grouped.items()):
            present={m['version'] for _,m in items}
            if present != set(versions):
                omitted.append(dict(strategy=strategy,missing_versions=sorted(set(versions)-present)))
                continue
            thresholds={(m['read_threshold'],m['site_threshold']) for _,m in items}
            if len(thresholds) != 1:
                raise ValueError(f'Cannot compare unequal operating thresholds: {strategy}')
            rt,st=next(iter(thresholds))
            mask=stage/'working_common.sqlite'
            db=connect_mask(mask,[p for p,_ in items])
            try:
                digest=observation_hash(db)
                n=db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
                tsv(stage/f'common_samples_{strategy}.tsv.gz',
                    (dict(r) for r in db.execute(f'SELECT * FROM predictions ORDER BY {KEYS}')),
                    [*KEYS.split(','),'seq','ref'])
            finally:
                db.close()
            masks.append(dict(strategy=strategy,n=n,sha256=digest,versions=versions,
                              runs=[str(p) for p,_ in items]))
            start=len(common)
            for p,m in items:
                for s in common_stats(p,mask,c,rt,st):
                    common.append(flat(m,s,'common_callable_positives',digest))
            groups=defaultdict(list)
            for row in common[start:]:
                groups[(row['version'],row['min_coverage'])].append(row)
            for (version,minimum),rows in groups.items():
                a=dict(version=version,strategy=strategy,min_coverage=minimum,n_seeds=len(rows),
                       seeds=','.join(str(r['seed']) for r in rows),read_n=rows[0]['read_n'],site_n=rows[0]['site_n'],
                       collection='common_callable_positives',observation_sha256=digest,
                       read_threshold=rt,site_threshold=st)
                for k in ('read_fnr','read_recall','site_fnr','site_recall'):
                    values=[r[k] for r in rows if r[k] is not None]
                    mean=sum(values)/len(values) if values else None
                    a[k+'_mean']=mean
                    a[k+'_sd']=math.sqrt(sum((x-mean)**2 for x in values)/(len(values)-1)) if len(values)>1 else None
                aggregates.append(a)
            for va,vb in itertools.combinations(versions,2):
                aa={m['seed']:p for p,m in items if m['version']==va}
                bb={m['seed']:p for p,m in items if m['version']==vb}
                shared=sorted(aa.keys() & bb.keys())
                if not shared:
                    differences.append(dict(strategy=strategy,version_a=va,version_b=vb,status='no_matching_training_seed'))
                for seed in shared:
                    for minimum in sorted(set([1,*c['statistics']['coverage_minima']])):
                        for diff in paired(aa[seed],bb[seed],mask,c,rt,st,minimum):
                            differences.append(dict(strategy=strategy,version_a=va,version_b=vb,seed=seed,status='paired_same_seed',**diff))
            mask.unlink()
            print(f'[COMMON] {strategy}: {n} matched positive windows; all selected seeds retained',flush=True)
        tsv(stage/'raw_summary.tsv',raw,FIELDS)
        tsv(stage/'common_summary.tsv',common,FIELDS)
        agg_fields=['version','strategy','min_coverage','n_seeds','seeds','collection','read_n','site_n','read_threshold','site_threshold',
            *[k+s for k in ('read_fnr','read_recall','site_fnr','site_recall') for s in ('_mean','_sd')],'observation_sha256']
        tsv(stage/'common_aggregate.tsv',aggregates,agg_fields)
        tsv(stage/'paired_differences.tsv',differences,['strategy','version_a','version_b','seed','status','level','min_coverage','n','fn_a','fn_b',
              'difference_fnr_b_minus_a','block_bootstrap_ci95','blocks'])
        write_json(stage/'comparison_manifest.json',dict(config=c,versions=versions,input_signatures=signatures,
            common_masks=masks,omitted_strategies=omitted,seed_sd='sample standard deviation, ddof=1; training randomness only',
            paired_difference='FNR_B-FNR_A; only equal seed IDs; genomic-block intervals within the current library',
            code_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('common.py')]}))
        if [signature(Path(s['path'])) for s in signatures] != signatures:
            raise ValueError('Result inputs changed during comparison')
    return dict(raw=raw,common=common,aggregate=aggregates,paired=differences)


def main():
    p=cli(__doc__)
    p.add_argument('--results-root',type=Path,help='读取 Step2 结果的根目录')
    args=p.parse_args()
    c,versions=load_config(args)
    if args.max_records is not None:
        raise ValueError('Step3 matches complete input results; use comparison.allow_partial only for isolated tests')
    compare(c,versions,args.out_dir or Path(c['comparison_dir']),args.results_root,args.dry_run)


if __name__ == '__main__':
    main()
