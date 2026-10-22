#!/usr/bin/env python3
"""Freeze runs; audit complete native test observations and saved/inferred probabilities."""
from __future__ import annotations

import csv
import itertools
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from common import *


def source_cache(c,v):
    import torch
    root=Path(c['models'][v]['feature_dir'])
    meta=torch.load(root/'test.pt',map_location='cpu',weights_only=False)
    if meta.get('storage_format')!='streaming_memmap_v1' or meta.get('split','test')!='test': raise ValueError('Expected original streaming test cache')
    shape=meta['feature_shape']
    if len(shape)!=3 or any(type(x) is not int or x<1 for x in shape) or shape[1]<5 or shape[1]%2!=1 or meta.get('records',shape[0])!=shape[0]:
        raise ValueError('Invalid original feature shape')
    dtype=np.dtype(meta['feature_dtype'])
    if dtype.name not in ('float16','float32') or np.dtype(meta['label_dtype']).name!='float32': raise ValueError('Invalid cache dtype')
    paths=[root/'test.pt',root/meta['feature_path'],root/meta['label_path'],root/'metadata_test.tsv',root/'feature_summary.json']
    if paths[1].stat().st_size!=math.prod(shape)*dtype.itemsize or paths[2].stat().st_size!=shape[0]*4: raise ValueError('Truncated feature/label arrays')
    summary=read_json(paths[-1])
    if infer_architecture(summary)!=c['models'][v]['architecture']: raise ValueError('Configured/source architecture mismatch')
    partial=False; s=summary
    while isinstance(s,dict):
        partial |= s.get('limit') is not None or any(s.get(k,{}).get('partial_test',False) for k in ('test1_dataset','test2_dataset'))
        s=s.get('source_feature_summary')
    if 'test1_dataset' in summary or 'test2_dataset' in summary:
        raise ValueError('Use original complete positive+negative test, not Test1/Test2 subsets')
    return meta,paths,summary,partial


def normalize(row,c,width,index):
    if row.get('split','test')!='test' or row.get('source')!='mod': raise ValueError('Only original native source=mod test is allowed')
    if row.get('library_id',c['library_id'])!=c['library_id']: raise ValueError('Native library mismatch')
    if row.get('sample_index') and int(row['sample_index'])!=index: raise ValueError('Source row index mismatch')
    label=float(row['label'])
    if label not in (0,1): raise ValueError('Invalid reference label')
    if row['chrom']!=c['chromosome'] or row['strand']!=c['strand'] or int(row['center_genome_pos0'])<0 or not row['read_id']:
        raise ValueError('Wrong chromosome/strand or invalid identity')
    for key in ('seq','ref'):
        if len(row[key])!=width or set(row[key].upper())-set('ACGTNWSMKRYBVDHZ'): raise ValueError('Invalid seq/ref')
    return (row['read_id'],row['chrom'],int(row['center_genome_pos0']),row['strand'],row['source'],int(label),row['seq'].upper(),row['ref'].upper())


def index_source(c,v,output,maximum):
    meta,paths,summary,partial=source_cache(c,v)
    n,w,h=meta['feature_shape']; stop=min(n,maximum) if maximum else n
    before=[signature(p) for p in paths]
    labels=np.memmap(paths[2],dtype='float32',mode='r',shape=(n,))
    ids=np.lib.format.open_memmap(output/'site_indices.npy',mode='w+',dtype='int32',shape=(stop,))
    db=sqlite3.connect(output/'observations.sqlite')
    db.execute('''CREATE TABLE observations(source_sample_index INTEGER PRIMARY KEY,library_id TEXT,
        read_id TEXT,chrom TEXT,position INTEGER,strand TEXT,source TEXT,label INTEGER,seq TEXT,ref TEXT,site_index INTEGER,
        UNIQUE(library_id,read_id,chrom,position,strand))''')
    sites={}; pending=[]
    try:
        rows=read_tsv(paths[3],META)
        for i in range(stop):
            row=next(rows,None)
            if row is None: raise ValueError('Metadata has fewer rows than binary arrays')
            data=normalize(row,c,w,i)
            read,chrom,pos,strand,source,label,seq,ref=data
            if label!=labels[i]: raise ValueError('Metadata/binary label mismatch')
            key=(c['library_id'],chrom,pos,strand)
            if key not in sites: sites[key]=[len(sites),label,ref,0]
            site=sites[key]
            if site[1:3]!=[label,ref]: raise ValueError('Conflicting reference label/context within a site')
            site[3]+=1; ids[i]=site[0]
            pending.append((i,c['library_id'],*data,site[0]))
            if len(pending)==8192:
                db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',pending); pending.clear()
            if (i+1)%200000==0: db.commit(); print(f'[INDEX] {v} {i+1}/{stop}',flush=True)
        if stop==n and next(rows,None) is not None: raise ValueError('Extra metadata rows')
        db.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',pending); db.commit()
    except sqlite3.IntegrityError as e:
        raise ValueError('Duplicate original read/position') from e
    finally:
        db.close(); ids.flush()
    fields=['site_index',*SITE,'label','ref','n_reads']
    records=[dict(zip(fields,[x[0],*k,x[1],x[2],x[3]])) for k,x in sites.items()]
    tsv(output/'sites.tsv',records,fields)
    np.save(output/'site_labels.npy',np.asarray([r['label'] for r in records],dtype='int8'))
    counts=Counter(int(v) for v in labels[:stop]); unchanged(before)
    result=dict(version=v,records=stop,original_records=n,width=w,hidden_dim=h,partial_test=partial or stop<n,
                label_counts={str(k):counts[k] for k in (0,1)},n_sites=len(sites),source_signatures=before,feature_summary=summary)
    write_json(output/'source_manifest.json',result)
    return result


def inferred_probabilities(c,meta,paths,checkpoint,stop):
    """Generate complete-window probabilities once; every depth reuses them."""
    import torch
    sys.path.insert(0,str(PROJECT/'train'))
    from model import build_model,FUSION_STRATEGY
    cfg=checkpoint['config']; n,w,h=meta['feature_shape']
    if (w,h)!=(cfg['source_window_size'],cfg['hidden_dim']): raise ValueError('Classifier/cache dimensions differ')
    requested=c['device']; device=('cuda:0' if torch.cuda.is_available() else 'cpu') if requested=='auto' else requested
    torch.set_num_threads(c['cpu_threads']); torch.set_float32_matmul_precision('highest')
    deterministic=bool(cfg['training'].get('deterministic',True))
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=deterministic
    torch.use_deterministic_algorithms(deterministic)
    model=build_model(cfg['strategy'],h,w,cfg['model']).to(device); model.load_state_dict(checkpoint['model'],strict=True); model.eval()
    x=np.memmap(paths[1],mode='r',dtype=meta['feature_dtype'],shape=(n,w,h))
    fusion=cfg['strategy']==FUSION_STRATEGY; fw=cfg.get('fusion_window_size',5); base=cfg.get('base_source','ref')
    if fusion and (fw<1 or fw>w or fw%2!=1 or base not in ('seq','ref')): raise ValueError('Invalid fusion settings')
    codes=dict(zip('ACGTNWSMKRYBVDHZ',range(16))); offset=(w-fw)//2
    rows=read_tsv(paths[3],META)
    with torch.inference_mode():
        for lo in range(0,stop,c['batch_size']):
            hi=min(stop,lo+c['batch_size']); batch=list(itertools.islice(rows,hi-lo))
            if len(batch)!=hi-lo: raise ValueError('Missing inference metadata')
            values=np.array(x[lo:hi],dtype='float32',copy=True)
            if not np.isfinite(values).all(): raise ValueError('Nonfinite features')
            tensor=torch.from_numpy(values).to(device)
            if fusion:
                bases=torch.tensor([[codes[b] for b in r[base][offset:offset+fw].upper()] for r in batch],dtype=torch.long,device=device)
                p=torch.softmax(model(tensor[:,offset:offset+fw],bases),dim=-1)[:,1]
            else: p=torch.sigmoid(model(tensor))
            p=p.cpu().numpy()
            if p.shape!=(len(batch),): raise ValueError('Invalid classifier output')
            for row,value in zip(batch,p): yield dict(row,prob=float(value))


def audit_run(c,item,version_dir,output,source):
    import torch
    run=Path(item['run_dir']); cfg=read_json(run/'config.json'); old=read_json(run/'metrics.json')
    meta,paths,summary,_=source_cache(c,item['version'])
    before=[signature(run/name) for name in ('config.json','metrics.json','best.pt')]
    if c['prediction_source']=='saved': before.append(signature(run/'test_predictions.tsv'))
    checkpoint=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    for k in ('strategy','seed','fingerprint','model','training','feature_dir','source_window_size','hidden_dim','base_source','fusion_window_size'):
        if cfg.get(k)!=checkpoint['config'].get(k): raise ValueError(f'Checkpoint/config mismatch: {k}')
    if checkpoint['epoch']!=old['best_epoch']: raise ValueError('Checkpoint epoch differs')
    binding=audit_cache_binding(c,item['version'],cfg)
    before.extend(binding['audit_inputs'])
    stop=source['records']; width=source['width']
    probs=np.lib.format.open_memmap(output/'probabilities.npy',mode='w+',dtype='float64',shape=(stop,))
    db=readonly_db(version_dir/'observations.sqlite'); db.row_factory=sqlite3.Row
    rows=(read_tsv(run/'test_predictions.tsv',META+['prob','split']) if c['prediction_source']=='saved'
          else inferred_probabilities(c,meta,paths,checkpoint,stop))
    y=np.empty(stop,dtype='int8')
    try:
        for i,expected in enumerate(db.execute('SELECT * FROM observations ORDER BY source_sample_index')):
            row=next(rows,None)
            if row is None: raise ValueError('Missing prediction rows')
            observed=normalize(row,c,width,i)
            if observed!=tuple(expected[k] for k in ('read_id','chrom','position','strand','source','label','seq','ref')):
                raise ValueError(f'Prediction/metadata mismatch at row {i}')
            p=float(row['prob'])
            if not math.isfinite(p) or not 0<=p<=1: raise ValueError('Nonfinite or out-of-range probability')
            probs[i]=p; y[i]=expected['label']
            if (i+1)%200000==0: print(f'[AUDIT] {item["version"]}/{item["run_name"]} {i+1}/{stop}',flush=True)
        if stop==meta['feature_shape'][0] and next(rows,None) is not None: raise ValueError('Extra prediction rows')
    finally:
        db.close()
        if hasattr(rows,'close'): rows.close()
    probs.flush(); t=cfg['training']['threshold']
    read_counts=confusion(y,probs,t)
    ids=np.load(version_dir/'site_indices.npy',mmap_mode='r'); site_y=np.load(version_dir/'site_labels.npy')
    score=np.bincount(ids,weights=probs,minlength=len(site_y))/np.bincount(ids,minlength=len(site_y))
    site_counts=confusion(site_y,score,t)
    verified=False
    if not source['partial_test'] and c['prediction_source']=='saved':
        for name,counts in [('test_read_level',read_counts),('test_site_level',site_counts)]:
            previous=old[name]
            if tuple(map(int,counts))!=tuple(previous[k] for k in ('tp','tn','fp','fn')):
                raise ValueError(f'Original completed metrics do not match: {name}')
        verified=True
    unchanged(before); unchanged(source['source_signatures'])
    result=dict(**item,checkpoint_sha256=sha256(run/'best.pt'),checkpoint_epoch=checkpoint['epoch'],configuration=cfg,
        input_signatures=before,n_records=stop,partial_test=source['partial_test'],original_counts_verified=verified,
        original_read_confusion=read_counts.tolist(),original_site_confusion=site_counts.tolist(),
        probability_source=c['prediction_source'],historical_prediction_binding='consistency_checks_only' if c['prediction_source']=='saved' else 'loaded_checkpoint_sha256')
    result['cache_binding']=binding
    write_json(output/'run_manifest.json',result)
    print(f'[DONE] audited {item["version"]}/{item["run_name"]}: {stop} windows',flush=True)
    return result


def run(c,root,maximum=None,dry_run=False):
    if maximum is not None and maximum<1: raise ValueError('max-records must be positive')
    inventory=discover(c)
    if dry_run:
        caches={}
        for v in c['enabled_versions']:
            item=next(r for r in inventory if r['version']==v)
            cfg=read_json(Path(item['run_dir'])/'config.json')
            binding=audit_cache_binding(c,v,cfg,verify_content=False)
            meta,_,_,_=source_cache(c,v)
            caches[v]=dict(feature_dir=binding['feature_dir'],training_feature_dir=binding['training_feature_dir'],
                           relocated=binding['relocated'],source_shape=meta['feature_shape'],
                           content_hashes='header/summary checked; full metadata/labels deferred to formal audit')
        print(json.dumps(dict(output_dir=str(root),source=c['prediction_source'],n_runs=len(inventory),max_records=maximum,
                              feature_caches=caches,runs=inventory),indent=2)); return
    with output_directory(root) as stage:
        (stage/'manifests').mkdir(); versions=stage/'audit/versions'; runs=stage/'audit/runs'
        stage.joinpath('protocol.yaml').write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
        sources={}; records=[]
        for v in c['enabled_versions']:
            dest=versions/v; dest.mkdir(parents=True)
            sources[v]=index_source(c,v,dest,maximum)
        for item in inventory:
            dest=runs/item['version']/item['run_name']; dest.mkdir(parents=True)
            records.append(audit_run(c,item,versions/item['version'],dest,sources[item['version']]))
        for source in sources.values(): unchanged(source['source_signatures'])
        for record in records: unchanged(record['input_signatures'])
        for record in records:
            record['probabilities_sha256']=sha256(runs/record['version']/record['run_name']/'probabilities.npy')
        tsv(stage/'manifests/run_inventory.tsv',inventory,['version','strategy','training_seed','run_name','run_dir'])
        import torch
        write_json(stage/'manifests/environment.json',dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),pyyaml=yaml.__version__))
        write_json(stage/'manifests/audit_manifest.json',dict(config_sha256=digest(c),partial_test=any(s['partial_test'] for s in sources.values()),
            runs=records,sources=sources,max_records=maximum,code_sha256=code_hashes()))
        manifest(c,stage/'audit',n_runs=len(records))
    return records


if __name__=='__main__':
    p=parser(__doc__); p.add_argument('--max-records',type=int,help='仅扫描每个原 test 前N行；partial，只用于临时联调')
    args=p.parse_args(); c,root=load_config(args); run(c,root,args.max_records,args.dry_run)
