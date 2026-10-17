#!/usr/bin/env python3
"""Create one reproducible, no-replacement, nested read selection for all runs."""
from __future__ import annotations

from common import *


def select_indices(rows,seed,repeat,maximum):
    ordered=sorted(rows,key=lambda r:(ranking(seed,repeat,r['library_id'],r['chrom'],r['position'],r['strand'],r['read_id']),r['read_id']))
    if len(ordered)<maximum: raise ValueError('Site cannot support maximum depth')
    return [r['obs_index'] for r in ordered[:maximum]]


def run(c,root,dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=3,input=str(root/'common'),output=str(root/'sampling'),repeats=c['repeats'],depths=c['depths'],sampling_seed=c['sampling_seed']))); return
    frozen(c,root); m=verify(root/'common',c)
    before=stage_signatures(root/'common',m)
    nsite,nobs=m['n_sites'],m['n_observations']; maximum=max(c['depths'])
    offsets=np.load(root/'common/offsets.npy'); dtype='int32' if nobs<=np.iinfo(np.int32).max else 'int64'
    with output_directory(root/'sampling') as stage:
        selected=np.lib.format.open_memmap(stage/'selected_indices.npy',mode='w+',dtype=dtype,shape=(c['repeats'],nsite,maximum))
        db=readonly_db(root/'common/pool.sqlite'); db.row_factory=sqlite3.Row
        try:
            for lo in range(0,nsite,c['site_chunk_size']):
                hi=min(nsite,lo+c['site_chunk_size'])
                groups={i:[] for i in range(lo,hi)}
                for row in db.execute('SELECT * FROM observations WHERE obs_index>=? AND obs_index<? ORDER BY obs_index',
                                      (int(offsets[lo]),int(offsets[hi]))):
                    groups[row['site_index']].append(dict(row))
                for i,rows in groups.items():
                    if len(rows)!=offsets[i+1]-offsets[i]: raise ValueError('Incomplete site pool')
                    for repeat in range(c['repeats']):
                        selected[repeat,i]=select_indices(rows,c['sampling_seed'],repeat,maximum)
                print(f'[SAMPLE] sites={hi}/{nsite} repeats={c["repeats"]}',flush=True)
            selected.flush(); del selected
        finally: db.close()
        unchanged(before)
        result=manifest(c,stage,shape=[c['repeats'],nsite,maximum],dtype=dtype,depths=c['depths'],
            common_manifest_sha256=sha256(root/'common/manifest.json'),
            algorithm='sha256 canonical JSON [seed,repeat,library,chrom,position,strand,read_id], ascending digest then read_id',
            nested=True,replace=False,code_sha256=code_hashes())
    return result


if __name__=='__main__':
    args=parser(__doc__).parse_args(); c,root=load_config(args); run(c,root,args.dry_run)
