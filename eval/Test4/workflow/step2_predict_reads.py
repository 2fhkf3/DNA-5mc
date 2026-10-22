#!/usr/bin/env python3
"""Extract each backbone once per shard, sharing features across its trained heads."""
from __future__ import annotations
from collections import Counter
from common import *


def run(c, root, version=None, dry_run=False):
    selected = [v for v in c['enabled_versions'] if version is None or version == v]
    if not selected:
        raise ValueError('Requested version is not in the frozen protocol')
    if dry_run:
        print(json.dumps(dict(step=2, versions=selected, output=str(root/'predictions'),
                              device=c['runtime']['device'], resume='verified completed backbone shards')))
        return
    audit = frozen(c, root); windows = verify(root/'windows', c)
    if windows['audit_manifest_sha256'] != sha(root/'audit/manifest.json'):
        raise ValueError('Window/audit binding mismatch')
    from inference import FeatureExtractor, load_heads, predict_head
    for v in selected:
        with lock(root/f'.locks/predictions_{v}.lock'):
            target = root/'predictions'/v
            if target.exists():
                m = verify(target, c)
                if m['window_manifest_sha256'] != sha(root/'windows/manifest.json'):
                    raise ValueError('Predictions came from another window set')
                continue
            work = root/'predictions'/('.'+v+'.work'); work.mkdir(parents=True, exist_ok=True)
            for orphan in work.glob('.part_*.partial-*'):
                shutil.rmtree(orphan)
            runs = [r for r in audit['runs'] if r['version'] == v]
            binding = dict(config_sha256=digest(c), window_manifest_sha256=sha(root/'windows/manifest.json'),
                           audit_manifest_sha256=sha(root/'audit/manifest.json'), version=v,
                           run_names=[r['run_name'] for r in runs])
            extractor = None; heads = None; parts = []
            for part in windows['parts']:
                name = Path(part['name']).name.replace('.jsonl.gz', '')
                target_part = work/name
                if target_part.exists():
                    m = verify(target_part, c)
                    if m['binding'] != binding or m['source_sha256'] != part['sha256']:
                        raise ValueError('Completed prediction shard binding changed')
                    parts.append(dict(name=name, n_rows=m['n_rows'], manifest_sha256=sha(target_part/'manifest.json')))
                    continue
                if extractor is None:
                    extractor = FeatureExtractor(audit['profiles'][v], c)
                    heads = load_heads(runs, extractor.device)
                with open_text(root/'windows'/part['name']) as f:
                    records = [json.loads(line) for line in f if line.strip()]
                if len(records) != part['n_rows'] or sha(root/'windows'/part['name']) != part['sha256']:
                    raise ValueError('Window shard changed')
                if any('label' in r or 'wgbs_beta' in r for r in records):
                    raise ValueError('Reference labels/beta must not enter the model input shard')
                features, valid, reasons = extractor.features(records)
                probabilities = np.full((len(records), len(runs)), np.nan, dtype=np.float32)
                ix = np.flatnonzero(valid)
                if len(ix):
                    kept_records = [records[i] for i in ix]
                    for col, (item, head) in enumerate(heads):
                        probabilities[ix, col] = predict_head(head, features[ix], kept_records,
                            item['configuration'], extractor.device, c['runtime']['head_batch_size'])
                with stage_directory(target_part) as stage:
                    np.savez_compressed(stage/'probabilities.npz', indices=np.array([r['idx'] for r in records], dtype=np.int64),
                                        valid=valid, probabilities=probabilities)
                    seal(stage, c, binding=binding, source_sha256=part['sha256'], n_rows=len(records),
                         window_identity_sha256=digest([r['window_id'] for r in records]),
                         excluded_reasons=dict(Counter(x for x in reasons if x)))
                parts.append(dict(name=name, n_rows=len(records), manifest_sha256=sha(target_part/'manifest.json')))
                print(f'[PREDICT] {v}/{name}: {len(ix)}/{len(records)} valid; {len(runs)} heads/seeds', flush=True)
            result = seal(work, c, **{k:val for k,val in binding.items() if k != 'config_sha256'},
                          parts=parts, partial_test=windows['partial_test'], feature_protocol=audit['profiles'][v])
            work.rename(target)
            del heads, extractor
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f'[DONE] Step2 {v}: {len(parts)} shards', flush=True)
    return True


if __name__ == '__main__':
    p = parser(__doc__); p.add_argument('--version'); a = p.parse_args()
    c, root = load_config(a); run(c, root, a.version, a.dry_run)
