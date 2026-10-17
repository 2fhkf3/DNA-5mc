#!/usr/bin/env python3
"""Separate training/sampling variation; pair depths/models and bootstrap genomic blocks."""
from __future__ import annotations

import itertools
from collections import defaultdict

from common import *


def as_float(value):
    return float(value) if value not in ('',None) else np.nan


def mean_finite(a,axis=0):
    a=np.asarray(a,dtype=float); valid=np.isfinite(a); n=valid.sum(axis=axis)
    return np.divide(np.where(valid,a,0).sum(axis=axis),n,out=np.full(np.shape(n),np.nan),where=n!=0)


def paired_stats(a,b,sign=1):
    return statistics(sign*(np.asarray(a)-np.asarray(b)))


def bootstrap_groups(c,states,common):
    K=c['bootstrap']['repeats']; out={}; blocks_reference=None; weights=None
    for key,items in states.items():
        sums={metric:None for metric in c['bootstrap']['metrics']}
        for item in items:
            with np.load(item['path']/'block_confusion.npz') as f:
                blocks=f['blocks'].tolist(); counts=f['counts']; depths=f['depths'].tolist()
                if f['count_order'].tolist()!=['tp','tn','fp','fn'] or depths!=c['depths'] or counts.shape!=(c['repeats'],len(depths),len(blocks),4):
                    raise ValueError('Invalid block confusion layout')
            if blocks_reference is None:
                blocks_reference=blocks
                if K and len(blocks)>=2:
                    rng=np.random.default_rng(c['bootstrap']['seed'])
                    weights=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=K)
            elif blocks!=blocks_reference: raise ValueError('Genomic block identities differ between runs')
            if weights is None: continue
            totals=np.einsum('kb,rdbc->krdc',weights,counts,optimize=True)
            metrics=count_metrics(totals)
            for metric in sums:
                values=mean_finite(metrics[metric],axis=1) # [bootstrap,depth], average sampling repetitions
                sums[metric]=values if sums[metric] is None else sums[metric]+values
        for metric,value in sums.items():
            out[(*key,metric)]=value/len(items) if value is not None else np.empty((0,len(c['depths'])))
    return out


def plot_results(c,stage,model_rows,seed_rows):
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/f'dna5mc-test3-mpl-{os.getuid()}'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=stage/'figures'; figures.mkdir()
    strategies=sorted({r['strategy'] for r in model_rows})
    colors=dict(zip(c['enabled_versions'],['#2563a6','#d37416','#27835c','#8a55a4','#b9474b','#65717f']))
    for metric in ('auroc','balanced_accuracy','fpr','fnr'):
        fig,axes=plt.subplots(1,len(strategies),figsize=(max(5,4*len(strategies)),4),squeeze=False)
        for ax,strategy in zip(axes[0],strategies):
            for version in c['enabled_versions']:
                rows=sorted((r for r in model_rows if (r['version'],r['strategy'],r['metric'])==(version,strategy,metric)),key=lambda r:r['depth'])
                if not rows: continue
                x=[r['depth'] for r in rows]; means=[r['mean'] for r in rows]
                if any(v is None for v in means): continue
                ax.plot(x,means,marker='o',label=version,color=colors.get(version))
                ax.fill_between(x,[r['sampling_q025'] for r in rows],[r['sampling_q975'] for r in rows],alpha=.12,color=colors.get(version))
                for seed in {r['training_seed'] for r in seed_rows if r['version']==version and r['strategy']==strategy}:
                    sr=sorted((r for r in seed_rows if (r['version'],r['strategy'],r['metric'],r['training_seed'])==(version,strategy,metric,seed)),key=lambda r:r['depth'])
                    ax.plot([r['depth'] for r in sr],[r['mean'] for r in sr],linewidth=.6,alpha=.3,color=colors.get(version))
            ax.set_title(strategy.replace('center_','').replace('_raw768',''),fontsize=9)
            ax.set_xticks(c['depths']); ax.set_xlabel('Reads per site'); ax.set_ylabel(metric)
            ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.15)
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle('Fixed common test sites; bands: 2.5–97.5% sampling range',fontsize=10)
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(figures/f'{metric}_coverage.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)


def run(c,root,dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=5,input=str(root/'results'),output=str(root/'summary'),bootstrap=c['bootstrap'],plots=c['plots']))); return
    audit=frozen(c,root); common=verify(root/'common',c); sampling=verify(root/'sampling',c)
    states=defaultdict(list); before=[*stage_signatures(root/'common',common),*stage_signatures(root/'sampling',sampling)]; contexts=[]
    D=len(c['depths']); R=c['repeats']
    for item in audit['runs']:
        path=root/'results'/item['version']/item['run_name']; m=verify(path,c)
        if (m['version'],m['strategy'],m['training_seed'],m['checkpoint_sha256'])!=(item['version'],item['strategy'],item['training_seed'],item['checkpoint_sha256']):
            raise ValueError('Result identity differs from frozen run inventory')
        if m['common_manifest_sha256']!=sha256(root/'common/manifest.json') or m['sampling_manifest_sha256']!=sha256(root/'sampling/manifest.json'):
            raise ValueError('Results use different fixed sites or samples')
        if m['partial_test'] and not c['allow_partial_results']: raise ValueError('Partial results forbidden')
        before.extend(stage_signatures(path,m))
        rows=list(read_tsv(path/'metrics_by_repeat.tsv')); seen=set()
        cube={metric:np.full((R,D),np.nan) for metric in METRICS}
        for row in rows:
            r,d=int(row['repeat_id']),int(row['depth'])
            if not 0<=r<R or d not in c['depths'] or (r,d) in seen: raise ValueError('Missing/duplicate or invalid repeat/depth identity')
            seen.add((r,d)); di=c['depths'].index(d)
            if (int(row['n_sites']),int(row['n_label0']),int(row['n_label1']))!=(common['n_sites'],common['n_label0'],common['n_label1']):
                raise ValueError('Class balance or fixed site denominator differs between results')
            for metric in METRICS: cube[metric][r,di]=as_float(row[metric])
        if len(seen)!=R*D: raise ValueError('Incomplete sampling result cube')
        states[(item['version'],item['strategy'])].append(dict(item,path=path,cube=cube,
            stability=read_json(path/'stability_summary.json'),full_common=read_json(path/'metrics_full_common.json')))
        for row in read_tsv(path/'context_metrics.tsv'):
            contexts.append(dict(version=item['version'],strategy=item['strategy'],training_seed=item['training_seed'],**row))
    boot=bootstrap_groups(c,states,common)
    seed_rows=[]; model_rows=[]; losses=[]; comparisons=[]; means={}; stability_rows=[]; full_rows=[]
    base=dict(n_sites=common['n_sites'],n_label0=common['n_label0'],n_label1=common['n_label1'])
    for (version,strategy),items in states.items():
        for metric in METRICS:
            full_rows.append(dict(version=version,strategy=strategy,metric=metric,n_seeds=len(items),**base,
                                  **statistics([i['full_common'][metric] for i in items])))
        for depth in c['depths']:
            records=[next(r for r in i['stability'] if r['depth']==depth) for i in items]
            for metric in records[0]:
                if metric=='depth': continue
                stability_rows.append(dict(version=version,strategy=strategy,depth=depth,metric=metric,n_seeds=len(items),**base,
                                            **statistics([r[metric] for r in records])))
        for metric in METRICS:
            stacks=np.asarray([i['cube'][metric] for i in items]) # seed,repeat,depth
            group=mean_finite(stacks,axis=0); means[(version,strategy,metric)]=group
            seed_means=mean_finite(stacks,axis=1)
            for si,item in enumerate(items):
                for di,d in enumerate(c['depths']):
                    seed_rows.append(dict(version=version,strategy=strategy,training_seed=item['training_seed'],depth=d,metric=metric,
                                          **base,**statistics(stacks[si,:,di])))
            for di,d in enumerate(c['depths']):
                group_stats=statistics(group[:,di]); seed_stats=statistics(seed_means[:,di])
                variances=[np.var(x[:,di],ddof=1) for x in stacks if R>1 and np.isfinite(x[:,di]).all()]
                model_rows.append(dict(version=version,strategy=strategy,depth=d,metric=metric,**base,
                    seeds=','.join(str(i['training_seed']) for i in items),n_seeds=len(items),repeats=R,
                    mean=group_stats['mean'],training_seed_sd=seed_stats['sd'],
                    sampling_sd_rms=float(np.sqrt(np.mean(variances))) if variances else None,
                    sampling_q025=group_stats['q025'],sampling_q975=group_stats['q975'],
                    **interval(boot.get((version,strategy,metric),np.empty((0,D)))[:,di])))
            for di,d in enumerate(c['depths'][:-1]):
                sign=1 if metric in LOWER else -1 # E(d)-E(max), Z(max)-Z(d)
                bd=boot.get((version,strategy,metric),np.empty((0,D)))
                losses.append(dict(version=version,strategy=strategy,metric=metric,depth=d,reference_depth=max(c['depths']),
                    n_seeds=len(items),**base,**paired_stats(group[:,di],group[:,-1],sign),
                    training_seed_sd=statistics(sign*(seed_means[:,di]-seed_means[:,-1]))['sd'],
                    **interval(sign*(bd[:,di]-bd[:,-1]))))
    for strategy in sorted({k[1] for k in states}):
        for va,vb in itertools.combinations(c['enabled_versions'],2):
            a,b=states[(va,strategy)],states[(vb,strategy)]
            shared=sorted({i['training_seed'] for i in a}&{i['training_seed'] for i in b})
            for metric in METRICS:
                ga,gb=means[(va,strategy,metric)],means[(vb,strategy,metric)]
                ba=boot.get((va,strategy,metric),np.empty((0,D))); bb=boot.get((vb,strategy,metric),np.empty((0,D)))
                for di,d in enumerate(c['depths']):
                    paired_seeds=[]
                    for seed in shared:
                        sa=next(i for i in a if i['training_seed']==seed)['cube'][metric][:,di]
                        sb=next(i for i in b if i['training_seed']==seed)['cube'][metric][:,di]
                        paired_seeds.append(float(mean_finite(sb-sa)))
                    ps=statistics(paired_seeds)
                    comparisons.append(dict(version_a=va,version_b=vb,strategy=strategy,metric=metric,depth=d,**base,
                        difference='B-A (all frozen training seeds; matched sites/sampling)',
                        seeds_a=','.join(str(i['training_seed']) for i in a),seeds_b=','.join(str(i['training_seed']) for i in b),
                        matching_training_seeds=','.join(map(str,shared)),n_matching_training_seeds=len(shared),
                        matching_seed_difference_mean=ps['mean'],matching_seed_difference_sd=ps['sd'],
                        **paired_stats(gb[:,di],ga[:,di]),**interval(bb[:,di]-ba[:,di])))
    context_groups=defaultdict(list)
    for row in contexts: context_groups[(row['version'],row['strategy'],row['depth'],row['ref5'])].append(row)
    context_summary=[]
    for key,rows in context_groups.items():
        for metric in ('balanced_accuracy','fpr','fnr','score_sd_mean'):
            context_summary.append(dict(zip(['version','strategy','depth','ref5'],key),metric=metric,
                n_sites=int(rows[0]['n_sites']),n_label0=int(rows[0]['n_label0']),n_label1=int(rows[0]['n_label1']),
                n_seeds=len(rows),**statistics([as_float(r[metric]) for r in rows])))
    with output_directory(root/'summary') as stage:
        for name,rows in [('seed_summary',seed_rows),('model_summary',model_rows),('depth_differences',losses),
                          ('model_differences',comparisons),('context_metrics',context_summary),
                          ('full_common_summary',full_rows),('stability_summary',stability_rows)]:
            tsv(stage/(name+'.tsv'),rows,list(rows[0]) if rows else ['status'])
        np.savez_compressed(stage/'bootstrap_model_draws.npz',**{'__'.join(k):v for k,v in boot.items()})
        write_json(stage/'reporting_notes.json',dict(
            uncertainty='Training seed SD, sampling ranges and within-library block intervals are separate; no biological replication implied.',
            group_mean='Mean of per-run metrics, first across sampling repeats then across all frozen training seeds; no probability ensembling.',
            differences='Model B-A; depth loss positive means worse at lower depth.',
            rank_metrics='AUROC/AP have sampling ranges; no block confidence interval is inferred from block confusion counts.',
            independent_libraries=1,partial_test=audit['partial_test'],site_balance=c['site_balance']))
        if c['plots']['enabled']: plot_results(c,stage,model_rows,seed_rows)
        unchanged(before)
        result=manifest(c,stage,n_runs=len(audit['runs']),partial_test=audit['partial_test'],
            common_manifest_sha256=sha256(root/'common/manifest.json'),sampling_manifest_sha256=sha256(root/'sampling/manifest.json'),code_sha256=code_hashes())
    print(f'[SUMMARY] {root/"summary"}',flush=True)
    return dict(seed_summary=seed_rows,model_summary=model_rows,depth_differences=losses,model_differences=comparisons)


if __name__=='__main__':
    args=parser(__doc__).parse_args(); c,root=load_config(args); run(c,root,args.dry_run)
