#!/usr/bin/env python3
"""Intersect verified observations globally, then fix sites with sufficient common reads."""
from __future__ import annotations

from collections import Counter
from common import *


def run(c,root,dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=2,input=str(root/'audit'),output=str(root/'common'),minimum_common_reads=c['minimum_common_reads'],site_balance=c['site_balance']))); return
    audit=frozen(c,root); audited=verify(root/'audit',c)
    before=[signature(root/'manifests/audit_manifest.json'),*stage_signatures(root/'audit',audited)]
    with output_directory(root/'common') as stage:
        work=sqlite3.connect(stage/'working.sqlite'); work.row_factory=sqlite3.Row
        fields=[*KEYS,'label','seq','ref']
        work.execute(f'''CREATE TABLE observations(library_id TEXT,read_id TEXT,chrom TEXT,position INTEGER,strand TEXT,
            label INTEGER,seq TEXT,ref TEXT,PRIMARY KEY({','.join(KEYS)})) WITHOUT ROWID''')
        union={}
        try:
            for i,v in enumerate(c['enabled_versions']):
                version=root/'audit/versions'/v
                for r in read_tsv(version/'sites.tsv'):
                    key=(r['library_id'],r['chrom'],int(r['position']),r['strand'])
                    value=(int(r['label']),r['ref'])
                    if key in union and union[key]!=value: raise ValueError('Cross-model site reference label/context conflict')
                    union[key]=value
                work.execute('ATTACH DATABASE ? AS v',(str((version/'observations.sqlite').resolve()),))
                if i==0:
                    work.execute(f'INSERT INTO main.observations SELECT {",".join(fields)} FROM v.observations')
                else:
                    join=' AND '.join(f'a.{k}=b.{k}' for k in KEYS)
                    if work.execute(f'''SELECT 1 FROM main.observations a JOIN v.observations b ON {join}
                        WHERE a.label!=b.label OR a.seq!=b.seq OR a.ref!=b.ref LIMIT 1''').fetchone():
                        raise ValueError('Common read has different label/seq/ref')
                    work.execute(f'DELETE FROM main.observations AS a WHERE NOT EXISTS(SELECT 1 FROM v.observations b WHERE {join})')
                work.commit(); work.execute('DETACH DATABASE v')
            cov={tuple(r[k] for k in SITE):r['n'] for r in work.execute(f'SELECT {",".join(SITE)},COUNT(*) AS n FROM observations GROUP BY {",".join(SITE)}')}
            eligible=[k for k in sorted(union) if cov.get(k,0)>=c['minimum_common_reads']]
            if not eligible: raise ValueError('No common sites can support the largest depth; increase smoke range or supply full test')
            class_counts=Counter(union[k][0] for k in eligible)
            if c['site_balance']=='equal':
                n=min(class_counts.get(0,0),class_counts.get(1,0))
                if not n: raise ValueError('Equal balance requires both classes')
                chosen=[]
                for label in (0,1):
                    group=[k for k in eligible if union[k][0]==label]
                    chosen+=sorted(group,key=lambda k:(ranking(c['balance_seed'],*k),k))[:n]
                eligible=sorted(chosen)
            if not audit['partial_test'] and len({union[k][0] for k in eligible})!=2:
                raise ValueError('Formal Test3 needs both reference classes')
            keep=set(eligible)
            excluded=[dict(zip(SITE,k),label=union[k][0],n_common_reads=cov.get(k,0),
                reason='no_common_read' if not cov.get(k,0) else ('below_minimum_common_reads' if cov[k]<c['minimum_common_reads'] else 'site_balance_not_selected'))
                for k in sorted(union) if k not in keep]
            sites=[dict(site_index=i,**dict(zip(SITE,k)),label=union[k][0],ref=union[k][1],
                        ref5=union[k][1][len(union[k][1])//2-2:len(union[k][1])//2+3],n_common_reads=cov[k]) for i,k in enumerate(eligible)]
            tsv(stage/'fixed_sites.tsv',sites,['site_index',*SITE,'label','ref','ref5','n_common_reads'])
            tsv(stage/'excluded_sites.tsv',excluded,[*SITE,'label','n_common_reads','reason'])
            work.execute(f'CREATE TABLE selected(site_index INTEGER,{",".join(k+" "+("INTEGER" if k=="position" else "TEXT") for k in SITE)},PRIMARY KEY({",".join(SITE)}))')
            work.executemany('INSERT INTO selected VALUES (?,?,?,?,?)',[(i,*k) for i,k in enumerate(eligible)]); work.commit()
            counts=np.asarray([s['n_common_reads'] for s in sites],dtype='int64'); offsets=np.r_[0,np.cumsum(counts)]
            nobs=int(offsets[-1]); nsite=len(sites)
            np.save(stage/'offsets.npy',offsets); np.save(stage/'labels.npy',np.asarray([s['label'] for s in sites],dtype='int8'))
            ids=np.lib.format.open_memmap(stage/'site_indices.npy',mode='w+',dtype='int32',shape=(nobs,))
            pool=sqlite3.connect(stage/'pool.sqlite')
            pool.execute('''CREATE TABLE observations(obs_index INTEGER PRIMARY KEY,site_index INTEGER,
                library_id TEXT,read_id TEXT,chrom TEXT,position INTEGER,strand TEXT,label INTEGER,seq TEXT,ref TEXT,
                UNIQUE(library_id,read_id,chrom,position,strand))''')
            batch=[]; total=0
            try:
                cursor=work.execute(f'''SELECT b.site_index,{','.join('a.'+k for k in fields)}
                    FROM observations a JOIN selected b USING({','.join(SITE)}) ORDER BY b.site_index,a.read_id''')
                for i,row in enumerate(cursor):
                    batch.append((i,*tuple(row))); ids[i]=row['site_index']; total+=1
                    if len(batch)==8192:
                        pool.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)',batch); batch.clear()
                pool.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)',batch); pool.commit(); ids.flush()
                if total!=nobs: raise ValueError('Common index count mismatch')
                mapping_signatures={}
                for v in c['enabled_versions']:
                    pool.execute('ATTACH DATABASE ? AS v',(str((root/'audit/versions'/v/'observations.sqlite').resolve()),))
                    mapping=np.lib.format.open_memmap(stage/f'mapping_{v}.npy',mode='w+',dtype='int64',shape=(nobs,))
                    count=0
                    for index,original in pool.execute(f'''SELECT a.obs_index,b.source_sample_index
                        FROM main.observations a JOIN v.observations b USING({','.join(KEYS)}) ORDER BY a.obs_index'''):
                        mapping[index]=original; count+=1
                    if count!=nobs: raise ValueError('Missing common observation in model mapping')
                    mapping.flush(); del mapping
                    pool.execute('DETACH DATABASE v'); mapping_signatures[v]=sha256(stage/f'mapping_{v}.npy')
            finally: pool.close()
        finally: work.close()
        (stage/'working.sqlite').unlink()
        unchanged(before)
        result=manifest(c,stage,n_sites=nsite,n_observations=nobs,
            n_label0=sum(s['label']==0 for s in sites),n_label1=sum(s['label']==1 for s in sites),
            union_sites=len(union),sites_with_common_reads=len(cov),sites_before_balance=len([k for k in union if cov.get(k,0)>=c['minimum_common_reads']]),
            exclusion_counts=dict(Counter(r['reason'] for r in excluded)),partial_test=audit['partial_test'],
            site_balance=c['site_balance'],audit_manifest_sha256=sha256(root/'manifests/audit_manifest.json'),code_sha256=code_hashes())
    print(f'[COMMON] sites={nsite} observations={nobs} negative={result["n_label0"]} positive={result["n_label1"]}',flush=True)
    return result


if __name__=='__main__':
    args=parser(__doc__).parse_args(); c,root=load_config(args); run(c,root,args.dry_run)
