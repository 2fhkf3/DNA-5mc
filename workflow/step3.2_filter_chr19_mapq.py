#!/usr/bin/env python3
"""Filter verified Step-2/Step-3 windows by MAPQ and retained actual site depth.

Keep original splits and all eligible rows; do not resample or normalize signals.
Only actual_coverage changes; parent_actual_coverage records the input depth.
Count distinct-read coverage through checksum-verified unique read/site input.
Publish a new directory only after full source and output verification.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {name: re.compile(rb'"'+name.encode()+rb'"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+)')
            for name in ('chrom','center_genome_pos0','label','split','mapq','actual_coverage','parent_actual_coverage')}


def field(line, name):
    tail = name not in ('chrom','center_genome_pos0')
    offset = max(0,len(line)-2048) if tail else 0
    small = line[offset:] if tail else line[:1024]
    match = PATTERNS[name].search(small)
    if match is None:
        offset = 0
        match = PATTERNS[name].search(line)
    if match is None:
        raise ValueError(f'Missing field: {name}')
    value = match.group(1)
    if value.startswith(b'"'):
        parsed=json.loads(value) if b'\\' in value else value[1:-1].decode('utf-8')
    else:
        parsed=int(value)
    return parsed,offset+match.start(1),offset+match.end(1)


def row_metadata(line):
    key = (field(line,'chrom')[0],field(line,'center_genome_pos0')[0])
    label = field(line,'label')[0]
    split = field(line,'split')[0]
    mapq = field(line,'mapq')[0]
    coverage,a,b = field(line,'actual_coverage')
    if label not in (0,1) or not isinstance(mapq,int) or mapq < 0 or not line.rstrip().endswith(b'}'):
        raise ValueError('Invalid verified JSONL metadata')
    return key,label,split,mapq,coverage,a,b


def read_lines(path):
    # Buffered gzip reads validate every concatenated member's CRC at EOF.
    return io.BufferedReader(gzip.open(path,'rb'),buffer_size=4*1024*1024)


def atomic_json(value, path):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


class ParallelGzipWriter:
    def __init__(self,path,workers,chunk_bytes,level):
        self.handle=path.open('xb')
        self.pool=ThreadPoolExecutor(max_workers=workers)
        self.pending=deque()
        self.buffer=bytearray()
        self.chunk_bytes=chunk_bytes
        self.level=level
        self.maximum_pending=max(1,2*workers)
        self.closed=False

    def append(self,line):
        self.buffer.extend(line)
        if len(self.buffer)>=self.chunk_bytes:
            self.submit()

    def submit(self):
        if not self.buffer:
            return
        block=bytes(self.buffer)
        self.buffer.clear()
        self.pending.append(self.pool.submit(gzip.compress,block,compresslevel=self.level,mtime=0))
        if len(self.pending)>=self.maximum_pending:
            self.handle.write(self.pending.popleft().result())

    def close(self):
        try:
            self.submit()
            while self.pending:
                self.handle.write(self.pending.popleft().result())
        finally:
            self.pool.shutdown(wait=True)
            self.handle.close()
            self.closed=True

    def abort(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.handle.close()
        self.closed=True


def stats(depths,labels):
    rows=Counter()
    sites=Counter()
    histogram=Counter()
    for key,depth in depths.items():
        rows[labels[key]]+=depth
        sites[labels[key]]+=1
        histogram[depth]+=1
    values=list(depths.values())
    return dict(records=sum(rows.values()),written_by_label={str(k):rows[k] for k in (0,1)},
                unique_sites=len(depths),sites_by_label={str(k):sites[k] for k in (0,1)},
                minimum_actual_coverage=min(values) if values else None,
                maximum_actual_coverage=max(values) if values else None,
                mean_actual_coverage=sum(values)/len(values) if values else None,
                actual_coverage_histogram={str(k):v for k,v in sorted(histogram.items())},
                positive_negative_ratio=rows[1]/rows[0] if rows[0] else None)


def load_source(source):
    summary=json.loads((source/'build_summary.json').read_text())
    verified=summary.get('verification',{})
    if (verified.get('original_read_site_duplicates') != 0 or verified.get('cross_split_sites') != 0
            or not verified.get('output_sha256_matched') or not verified.get('gzip_crc_checked')):
        raise ValueError('Input must have verified SHA256/CRC, unique original read/site pairs and disjoint sites')
    splits=[s for s in ('train','valid','test') if s in summary.get('write_stats',{})]
    if not splits:
        raise ValueError('No source splits in build_summary.json')
    depths={s:{} for s in splits}
    labels={}
    table={}
    with gzip.open(source/'candidate_sites.tsv.gz','rt') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            split=row['split']
            if split not in depths:
                raise ValueError('Site table refers to an unknown split')
            key=(row['chrom'],int(row['genome_pos0']))
            label=int(row['label'])
            depth=int(row['actual_coverage'])
            if key in labels or label not in (0,1) or depth < 1:
                raise ValueError(f'Duplicate/cross-split or invalid site: {key}')
            labels[key]=label
            depths[split][key]=depth
            table[key]=row
    for split in splits:
        actual=stats(depths[split],labels)
        expected=summary['write_stats'][split]
        for name in ('records','unique_sites','written_by_label'):
            if actual[name]!=expected[name]:
                raise ValueError(f'{split}: candidate site table differs from source summary ({name})')
        if not (source/f'{split}.jsonl.gz').is_file():
            raise FileNotFoundError(f'Missing source split: {split}')
    return summary,splits,depths,labels,table


def progress(stage,phase,split,count,total,started,**extra):
    info=dict(phase=phase,split=split,scanned=count,total=total,
              percent=round(100*count/total,2) if total else 100,
              seconds=round(time.monotonic()-started,1),**extra)
    atomic_json(info,stage/'progress.json')
    print(phase.upper()+' '+json.dumps(info),flush=True)


def scan(split,path,depths,labels,expected,args,stage):
    before=path.stat()
    observed=Counter()
    high=Counter()
    low_labels=Counter()
    source_hash=hashlib.sha256()
    seen=0
    started=time.monotonic()
    with read_lines(path) as reader:
        for seen,line in enumerate(reader,1):
            source_hash.update(line)
            key,label,row_split,mapq,coverage,_,_=row_metadata(line)
            if key not in depths or label!=labels[key] or row_split!=split or coverage!=depths[key]:
                raise ValueError(f'{path}:{seen}: record metadata disagrees with candidate site table')
            if PATTERNS['parent_actual_coverage'].search(line[-2048:]):
                raise ValueError('Input already contains parent_actual_coverage; use the original source dataset')
            observed[key]+=1
            if mapq>=args.min_mapq:
                high[key]+=1
            else:
                low_labels[label]+=1
            if args.log_every and seen%args.log_every==0:
                progress(stage,'scan',split,seen,expected['records'],started,mapq_removed=sum(low_labels.values()))
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError(f'Input changed during scan: {path}')
    if seen!=expected['records'] or source_hash.hexdigest()!=expected['uncompressed_sha256'] or observed!=depths:
        raise ValueError(f'{split}: source SHA256/count/site depth mismatch')
    kept={key:n for key,n in high.items() if n>=args.min_coverage}
    if not kept:
        raise ValueError(f'{split}: no sites retain coverage >= {args.min_coverage}')
    plan=dict(input=stats(depths,labels),after_mapq_filter=stats(high,labels),planned_output=stats(kept,labels),
              removed_mapq_records=sum(low_labels.values()),removed_mapq_by_label={str(k):low_labels[k] for k in (0,1)},
              removed_coverage_records=sum(high.values())-sum(kept.values()),
              removed_sites=len(depths)-len(kept),source_sha256_matched=True)
    progress(stage,'scan',split,seen,expected['records'],started,mapq_removed=sum(low_labels.values()))
    print('PLAN '+json.dumps(dict(split=split,**plan['planned_output'],
                                 removed_mapq=plan['removed_mapq_records'],
                                 removed_coverage=plan['removed_coverage_records'])),flush=True)
    return high,kept,plan


def update_coverage(line,coverage,parent,start,end):
    changed=line[:start]+str(coverage).encode()+line[end:]
    end_object=changed.rfind(b'}')
    return changed[:end_object]+b',"parent_actual_coverage":'+str(parent).encode()+changed[end_object:]


def write_split(split,source,path,final_path,depths,quotas,labels,expected,args,stage):
    writer=ParallelGzipWriter(path,args.workers,args.chunk_mb*1024*1024,args.gzip_level)
    actual=Counter()
    source_hash=hashlib.sha256()
    output_hash=hashlib.sha256()
    seen=0
    started=time.monotonic()
    before=source.stat()
    try:
        with read_lines(source) as reader:
            for seen,line in enumerate(reader,1):
                source_hash.update(line)
                key,label,row_split,mapq,coverage,a,b=row_metadata(line)
                if key in quotas and mapq>=args.min_mapq:
                    if label!=labels[key] or row_split!=split or coverage!=depths[key]:
                        raise ValueError('Input metadata changed between passes')
                    changed=update_coverage(line,quotas[key],coverage,a,b)
                    writer.append(changed)
                    output_hash.update(changed)
                    actual[key]+=1
                if args.log_every and seen%args.log_every==0:
                    progress(stage,'write',split,seen,expected['records'],started,kept=sum(actual.values()))
        writer.close()
    except BaseException:
        if not writer.closed:
            writer.abort()
        raise
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError('Input changed during output writing')
    if seen!=expected['records'] or source_hash.hexdigest()!=expected['uncompressed_sha256'] or actual!=quotas:
        raise RuntimeError('Source SHA256/record count or selected site counts mismatch')
    result=stats(actual,labels)
    result.update(output_path=str(final_path),compressed_bytes=path.stat().st_size,
                  uncompressed_sha256=output_hash.hexdigest())
    progress(stage,'write',split,seen,expected['records'],started,kept=result['records'])
    return result


def verify_split(split,path,depths,quotas,labels,expected,args,stage):
    actual=Counter()
    output_hash=hashlib.sha256()
    seen=0
    started=time.monotonic()
    with read_lines(path) as reader:
        for seen,line in enumerate(reader,1):
            output_hash.update(line)
            key,label,row_split,mapq,coverage,_,_=row_metadata(line)
            if (key not in quotas or label!=labels[key] or row_split!=split or mapq<args.min_mapq
                    or coverage!=quotas[key] or coverage<args.min_coverage
                    or field(line,'parent_actual_coverage')[0]!=depths[key]):
                raise ValueError(f'Output failed MAPQ/coverage/metadata validation: {path}:{seen}')
            actual[key]+=1
            if args.log_every and seen%args.log_every==0:
                progress(stage,'verify',split,seen,expected['records'],started)
    if actual!=quotas or seen!=expected['records'] or output_hash.hexdigest()!=expected['uncompressed_sha256']:
        raise RuntimeError('Output CRC/SHA256/actual site depth validation failed')
    progress(stage,'verify',split,seen,expected['records'],started)


def write_tables(stage,splits,table,depths,high,kept):
    with gzip.open(stage/'candidate_sites.tsv.gz','wt',compresslevel=3) as c, gzip.open(stage/'site_coverage.tsv.gz','wt',compresslevel=3) as a:
        candidate_fields=['chrom','genome_pos0','label','bed_coverage','bed_percent','actual_coverage','split',
                          'parent_actual_coverage']
        if any('input_actual_coverage' in row for row in table.values()):
            candidate_fields.insert(-1,'input_actual_coverage')
        all_fields=['chrom','genome_pos0','label','split','parent_actual_coverage','coverage_after_mapq',
                    'actual_coverage','removed_mapq_records','removed_coverage_records','retained']
        cw=csv.DictWriter(c,fieldnames=candidate_fields,delimiter='\t',extrasaction='ignore')
        aw=csv.DictWriter(a,fieldnames=all_fields,delimiter='\t')
        cw.writeheader();aw.writeheader()
        for key in sorted(table):
            row=table[key]
            split=row['split']
            parent=depths[split][key]
            post_mapq=high[split].get(key,0)
            final=kept[split].get(key,0)
            aw.writerow(dict(chrom=key[0],genome_pos0=key[1],label=row['label'],split=split,
                             parent_actual_coverage=parent,coverage_after_mapq=post_mapq,actual_coverage=final,
                             removed_mapq_records=parent-post_mapq,removed_coverage_records=post_mapq-final,
                             retained=int(final>0)))
            if final:
                cw.writerow(dict(row,actual_coverage=final,parent_actual_coverage=parent))


def run(args):
    if args.min_mapq<0 or args.min_coverage<1 or args.workers<1 or args.chunk_mb<1 or args.log_every<0 or not 0<=args.gzip_level<=9:
        raise ValueError('Invalid filter/compression settings')
    source=args.input_dir.resolve()
    destination=args.out_dir.resolve()
    stage=destination.with_name(destination.name+'.inprogress')
    if destination.exists() or stage.exists():
        raise FileExistsError(f'Destination or staging already exists: {destination}')
    summary,splits,depths,labels,table=load_source(source)
    stage.mkdir(parents=True)
    started=time.monotonic()
    high={}; kept={}; plans={}; written={}
    try:
        for split in splits:
            high[split],kept[split],plans[split]=scan(split,source/f'{split}.jsonl.gz',depths[split],labels,
                                                    summary['write_stats'][split],args,stage)
        atomic_json(dict(min_mapq=args.min_mapq,min_coverage=args.min_coverage,splits=plans),stage/'filter_plan.json')
        for split in splits:
            written[split]=write_split(split,source/f'{split}.jsonl.gz',stage/f'{split}.jsonl.gz',
                                       destination/f'{split}.jsonl.gz',depths[split],kept[split],labels,
                                       summary['write_stats'][split],args,stage)
        for split in splits:
            verify_split(split,stage/f'{split}.jsonl.gz',depths[split],kept[split],labels,written[split],args,stage)
        combined={}
        for split in splits:
            if combined.keys() & kept[split].keys():
                raise RuntimeError('Cross-split site leakage after filtering')
            combined.update(kept[split])
        write_tables(stage,splits,table,depths,high,kept)
        result=dict(format='step3_mapq_actual_coverage_filtered_jsonl_v1',input_dir=str(source),out_dir=str(destination),
                    source_summary=str(source/'build_summary.json'),source_format=summary.get('format'),min_mapq=args.min_mapq,
                    minimum_actual_coverage_after_filtering=args.min_coverage,window_nt=summary.get('window_nt'),
                    split_strategy='Preserve original split for every chromosome/position pair',
                    coverage_definition='Distinct original reads per (chrom, center_genome_pos0) in retained output',
                    coverage_fields=dict(actual_coverage='Recomputed retained coverage',parent_actual_coverage='Direct source dataset retained coverage',
                                         input_actual_coverage='Earlier Step-2 coverage, preserved only if already present',bed_coverage='Unchanged BED coverage'),
                    sampling='All rows that satisfy both filters; no class balancing or sampling',
                    signal_processing='Signal/sequence/span bytes copied unchanged; no normalization or extraction',
                    filters=plans,write_stats=written,overall=stats(combined,labels),
                    verification=dict(gzip_crc_checked=True,output_sha256_matched=True,all_input_counts_and_sha256_matched=True,
                                      all_output_metadata_and_site_counts_checked=True,minimum_actual_coverage=args.min_coverage,
                                      mapq_at_least_threshold=True,cross_split_sites=0,original_read_site_duplicates=0,
                                      duplicate_verification='Inherited from SHA256-verified unique-read/site source; output is a subset',
                                      balanced_in_each_split=all(v['written_by_label']['0']==v['written_by_label']['1'] for v in written.values())),
                    compression=dict(workers=args.workers,chunk_mb=args.chunk_mb,gzip_level=args.gzip_level,
                                     ordered_concatenated_gzip_members=True),seconds=round(time.monotonic()-started,2))
        atomic_json(result,stage/'build_summary.json')
        chromosomes=', '.join(sorted({key[0] for key in table}))
        source_stage='Step2' if str(summary.get('format','')).startswith('step2_') else 'Step3'
        text=[f'# {chromosomes} MAPQ 和实际 coverage 过滤结果','',f'来源：`{source}`。',
              f'先保留 MAPQ≥{args.min_mapq} 的记录，再仅保留过滤后实际 coverage≥{args.min_coverage} 的位点。',
              f"保持原 {'/'.join(splits)} 集合归属，保留全部满足条件的记录。未再做正负样本抽样平衡。",
              '信号、序列、span、BED 标签和 BED coverage 保持不变。`actual_coverage` 已重新计算；',
              f'`parent_actual_coverage` 是直接来源 {source_stage} 的过滤前深度。若输入已有 `input_actual_coverage`，原值继续保留；否则不新增该字段。','',
              '| 集合 | 记录数 | 位点数 | label=0 | label=1 | 最低实际 coverage |',
              '|---|---:|---:|---:|---:|---:|']
        for split,v in written.items():
            text.append(f"| {split} | {v['records']:,} | {v['unique_sites']:,} | {v['written_by_label']['0']:,} | {v['written_by_label']['1']:,} | {v['minimum_actual_coverage']} |")
        text += ['', '`candidate_sites.tsv.gz` 仅包含保留位点；`site_coverage.tsv.gz` 包含所有来源位点及逐阶段删除量。',
                 '`filter_plan.json` 和 `build_summary.json` 记录过滤统计、SHA256 与完整验证结果。','']
        (stage/'README.md').write_text('\n'.join(text))
        atomic_json(dict(phase='complete',seconds=result['seconds']),stage/'progress.json')
        if destination.exists():
            raise FileExistsError(f'Destination appeared during processing: {destination}')
        stage.rename(destination)
        print('DONE '+json.dumps(dict(output=str(destination),overall=result['overall'],seconds=result['seconds'])),flush=True)
        return result
    except BaseException as exc:
        atomic_json(dict(phase='failed',error=str(exc),seconds=round(time.monotonic()-started,2)),stage/'progress.json')
        raise


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=PROJECT_ROOT/'output/step3_output/chr19')
    p.add_argument('--out-dir',type=Path,default=PROJECT_ROOT/'output/step3_output/chr19_mapq60_cov10')
    p.add_argument('--min-mapq',type=int,default=60)
    p.add_argument('--min-coverage',type=int,default=10)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--chunk-mb',type=int,default=16)
    p.add_argument('--gzip-level',type=int,default=3)
    p.add_argument('--log-every',type=int,default=250000)
    return p.parse_args(argv)


if __name__=='__main__':
    run(parse_args())
