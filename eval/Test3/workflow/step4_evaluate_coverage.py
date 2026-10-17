#!/usr/bin/env python3
"""Aggregate shared sampled read probabilities; evaluate fixed sites at every depth."""
from __future__ import annotations

from common import *


def evaluate(c,root,item,common,sampling):
    version,name=item['version'],item['run_name']
    target=root/'results'/version/name
    source=root/'audit/runs'/version/name/'probabilities.npy'
    if sha256(source)!=item['probabilities_sha256']: raise ValueError('Audited probability vector changed')
    before=[signature(source)]
    p0=np.load(source,mmap_mode='r')
    mapping=np.load(root/'common'/f'mapping_{version}.npy',mmap_mode='r')
    if len(p0)!=item['n_records'] or mapping.min()<0 or mapping.max()>=len(p0): raise ValueError('Invalid original/common mapping')
    p=np.asarray(p0[mapping],dtype='float64')
    if len(p)!=common['n_observations'] or not np.isfinite(p).all() or np.any((p<0)|(p>1)): raise ValueError('Invalid common probabilities')
    sites=list(read_tsv(root/'common/fixed_sites.tsv'))
    y=np.load(root/'common/labels.npy'); offsets=np.load(root/'common/offsets.npy')
    n=len(y); depths=c['depths']; D=len(depths); R=c['repeats']; threshold=c['site_threshold']
    selected=np.load(root/'sampling/selected_indices.npy',mmap_mode='r')
    if list(selected.shape)!=[R,n,max(depths)] or list(sampling['shape'])!=list(selected.shape): raise ValueError('Sampling shape differs')
    blocks=sorted({(s['library_id'],s['chrom'],int(s['position'])//c['bootstrap']['block_size_bp']) for s in sites})
    lookup={k:i for i,k in enumerate(blocks)}
    bidx=np.asarray([lookup[(s['library_id'],s['chrom'],int(s['position'])//c['bootstrap']['block_size_bp'])] for s in sites])
    contexts=sorted({s['ref5'] for s in sites}); clook={k:i for i,k in enumerate(contexts)}
    cidx=np.asarray([clook[s['ref5']] for s in sites]); B=len(blocks); C=len(contexts)
    full=np.add.reduceat(p,offsets[:-1])/np.diff(offsets)
    full_metrics=binary_metrics(y,full,threshold)
    with output_directory(target) as stage:
        mean=np.zeros((D,n)); m2=np.zeros((D,n)); calls=np.zeros((D,n),dtype='int32')
        errors=np.zeros_like(calls); flips=np.zeros_like(calls); abs_full=np.zeros_like(mean)
        bc=np.zeros((R,D,B,4),dtype='int64'); cc=np.zeros((D,C,4),dtype='int64')
        scores=(np.lib.format.open_memmap(stage/'site_scores.npy',mode='w+',dtype='float64',shape=(R,D,n)) if c['save_all_site_scores'] else None)
        rows=[]
        for repeat in range(R):
            ix=selected[repeat]
            if np.any(ix<offsets[:-1,None]) or np.any(ix>=offsets[1:,None]) or np.any(np.diff(np.sort(ix,axis=1),axis=1)==0):
                raise ValueError('Sampling is out of site bounds or repeats original read')
            sums=np.cumsum(p[ix],axis=1,dtype='float64')
            qmax=sums[:,-1]/max(depths)
            for di,d in enumerate(depths):
                q=sums[:,d-1]/d; pred=q>=threshold
                stats=binary_metrics(y,q,threshold)
                code=np.where(y==1,np.where(pred,0,3),np.where(pred,2,1))
                bc[repeat,di]=np.bincount(bidx*4+code,minlength=B*4).reshape(B,4)
                cc[di]+=np.bincount(cidx*4+code,minlength=C*4).reshape(C,4)
                delta=q-mean[di]; mean[di]+=delta/(repeat+1); m2[di]+=delta*(q-mean[di])
                calls[di]+=pred; errors[di]+=(pred!=y); flips[di]+=(pred!=(qmax>=threshold)); abs_full[di]+=np.abs(q-full)
                if scores is not None: scores[repeat,di]=q
                rows.append(dict(protocol_id=c['protocol_id'],version=version,strategy=item['strategy'],training_seed=item['training_seed'],
                    repeat_id=repeat,depth=d,site_threshold=threshold,**stats,
                    score_abs_diff_full=float(np.mean(np.abs(q-full))),flip_vs_max_depth=float(np.mean(pred!=(qmax>=threshold)))))
            if repeat==0 or repeat==R-1 or (repeat+1)%10==0: print(f'[EVAL] {version}/{name} repeat={repeat+1}/{R}',flush=True)
        if scores is not None: scores.flush(); del scores
        tsv(stage/'metrics_by_repeat.tsv',rows,list(rows[0]))
        write_json(stage/'metrics_full_common.json',dict(**full_metrics,n_common_observations=len(p),site_threshold=threshold))
        np.savez_compressed(stage/'block_confusion.npz',counts=bc,blocks=np.asarray([json.dumps(b) for b in blocks]),
                            depths=np.asarray(depths),count_order=np.asarray(['tp','tn','fp','fn']))
        stability=[]; context_rows=[]
        for di,d in enumerate(depths):
            sd=np.sqrt(np.maximum(m2[di],0)/(R-1)) if R>1 else None
            positive=calls[di]/R; instability=np.minimum(positive,1-positive); error=errors[di]/R
            stability.append(dict(depth=d,score_sd_mean=float(sd.mean()) if sd is not None else None,
                score_sd_p95=float(np.quantile(sd,.95)) if sd is not None else None,
                call_instability_mean=float(instability.mean()),error_frequency_mean=float(error.mean()),
                flip_vs_max_depth_mean=float(np.mean(flips[di]/R)),score_abs_diff_full_mean=float(np.mean(abs_full[di]/R))))
            for ci,ref in enumerate(contexts):
                tp,tn,fp,fn=cc[di,ci]/R; mask=cidx==ci
                vals=count_metrics(np.asarray([tp,tn,fp,fn]))
                context_rows.append(dict(depth=d,ref5=ref,n_sites=int(mask.sum()),n_label0=int(np.sum(y[mask]==0)),n_label1=int(np.sum(y[mask]==1)),
                    tp_mean=float(tp),tn_mean=float(tn),fp_mean=float(fp),fn_mean=float(fn),
                    **{k:float(v) if np.isfinite(v) else None for k,v in vals.items()},
                    score_sd_mean=float(sd[mask].mean()) if sd is not None else None))
        tsv(stage/'context_metrics.tsv',context_rows,list(context_rows[0]))
        write_json(stage/'stability_summary.json',stability)
        if c['save_site_stability']:
            def site_rows():
                for di,d in enumerate(depths):
                    for i,s in enumerate(sites):
                        pi=calls[di,i]/R
                        yield dict(s,depth=d,score_mean=float(mean[di,i]),score_sd=math.sqrt(max(0,m2[di,i])/(R-1)) if R>1 else None,
                            positive_call_fraction=float(pi),call_instability=float(min(pi,1-pi)),error_frequency=float(errors[di,i]/R),
                            flip_vs_max_depth=float(flips[di,i]/R),score_abs_diff_full=float(abs_full[di,i]/R),full_common_score=float(full[i]))
            tsv(stage/'site_stability.tsv.gz',site_rows(),[*sites[0].keys(),'depth','score_mean','score_sd','positive_call_fraction',
                'call_instability','error_frequency','flip_vs_max_depth','score_abs_diff_full','full_common_score'])
        unchanged(before)
        result=manifest(c,stage,version=version,strategy=item['strategy'],training_seed=item['training_seed'],
            checkpoint_sha256=item['checkpoint_sha256'],partial_test=item['partial_test'],
            common_manifest_sha256=sha256(root/'common/manifest.json'),sampling_manifest_sha256=sha256(root/'sampling/manifest.json'),
            audit_manifest_sha256=sha256(root/'manifests/audit_manifest.json'),n_sites=n,n_label0=int(np.sum(y==0)),n_label1=int(np.sum(y==1)),
            read_threshold=None,site_threshold=threshold,code_sha256=code_hashes(),
            probability_source=item['probability_source'],bce_epsilon=1e-15)
    return result


def run(c,root,run_name=None,version=None,dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=4,input=str(root/'sampling'),output=str(root/'results'),version=version,run_name=run_name,depths=c['depths']))); return
    audit=frozen(c,root); common=verify(root/'common',c); sampling=verify(root/'sampling',c)
    if sampling['common_manifest_sha256']!=sha256(root/'common/manifest.json'): raise ValueError('Sampling/common provenance mismatch')
    chosen=[r for r in audit['runs'] if (version is None or r['version']==version) and (run_name is None or r['run_name']==run_name)]
    if not chosen: raise ValueError('No frozen runs match the requested subset')
    for r in chosen:
        if (root/'results'/r['version']/r['run_name']).exists(): raise FileExistsError('Result exists; use a new protocol or process only missing runs')
    before=[*stage_signatures(root/'common',common),*stage_signatures(root/'sampling',sampling)]
    result=[evaluate(c,root,r,common,sampling) for r in chosen]
    unchanged(before)
    return result


if __name__=='__main__':
    p=parser(__doc__)
    p.add_argument('--version',help='只处理冻结清单中的一个模型，不改变共同位点/抽样')
    p.add_argument('--run-name',help='只处理指定 strategy_seedN，可配合 version 分配多个独立进程')
    args=p.parse_args(); c,root=load_config(args); run(c,root,args.run_name,args.version,args.dry_run)
