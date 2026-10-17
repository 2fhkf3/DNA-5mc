#!/usr/bin/env python3
"""Audit saved test predictions or infer native-positive caches, then report FNR."""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

from common import (PROJECT, ROOT, KIND, TRUTH, REQUIRED, cli, export_results, identity,
                    infer_architecture, insert, load_config, observation, output_directory,
                    prediction_db, read_json, reader, sha256, signature, validate_row, write_json)


def parser():
    p = cli(__doc__)
    p.add_argument('--source', choices=['predictions', 'features'], help='覆盖 evaluation.source')
    p.add_argument('--run-dir', type=Path, help='只处理此已完成 run，须同时指定 --version')
    p.add_argument('--feature-dir', type=Path, help='覆盖所选单模型的阳性特征缓存路径')
    p.add_argument('--device', help='仅重新推理时使用；覆盖 evaluation.device')
    return p


def discover(c, version, single=None):
    root = Path(c['models'][version]['runs_root'])
    if (root/'runs').is_dir():
        root /= 'runs'
    paths = [single.resolve()] if single else sorted(p.parent for p in root.glob('*/metrics.json'))
    result = []
    for path in paths:
        m = read_json(path/'metrics.json')
        if not single and ((c['evaluation']['seeds'] and m['seed'] not in c['evaluation']['seeds']) or
                           (c['evaluation']['strategies'] and m['strategy'] not in c['evaluation']['strategies'])):
            continue
        for name in ('best.pt', 'config.json'):
            if not (path/name).is_file():
                raise FileNotFoundError(path/name)
        cfg = read_json(path/'config.json')
        if not re.fullmatch(re.escape(version)+r'(?:[-_].*)?', cfg.get('experiment', ''), re.I):
            raise ValueError(f'Checkpoint experiment does not match {version}: {path}')
        for key in ('strategy', 'seed', 'fingerprint'):
            if cfg.get(key) != m.get(key):
                raise ValueError(f'Completed run/config mismatch: {key}')
        result.append(path)
    if not result:
        raise ValueError(f'No completed runs selected for {version}')
    if len({(read_json(p/'config.json')['strategy'], read_json(p/'config.json')['seed']) for p in result}) != len(result):
        raise ValueError('Duplicate strategy/seed runs; select one frozen experiment per version')
    return result


def cache(root):
    import numpy as np
    import torch
    m = torch.load(root/'test.pt', map_location='cpu', weights_only=False)
    if m.get('storage_format') != 'streaming_memmap_v1' or m.get('split', 'test') != 'test':
        raise ValueError('Expected streaming_memmap_v1 test features')
    shape = m['feature_shape']
    if len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape) or shape[1] % 2 != 1 or m.get('records', shape[0]) != shape[0]:
        raise ValueError('Invalid cache feature shape/count')
    dtype = np.dtype(m['feature_dtype'])
    if dtype.name not in ('float16','float32') or np.dtype(m['label_dtype']).name != 'float32':
        raise ValueError('Unsupported cache dtype')
    x, y = root/m['feature_path'], root/m['label_path']
    if x.stat().st_size != math_product(shape)*dtype.itemsize or y.stat().st_size != shape[0]*4:
        raise ValueError('Truncated feature/label arrays')
    paths = [root/'test.pt', x, y, root/'metadata_test.tsv', root/'feature_summary.json']
    return m, np.memmap(y, mode='r', dtype='float32', shape=(shape[0],)), paths


def math_product(shape):
    import math
    return math.prod(shape)


def audit_protocol(c, v, cfg, summary):
    old_path = Path(c['models'][v]['feature_dir'])/'feature_summary.json'
    old = read_json(old_path)
    if infer_architecture(old) != c['models'][v]['architecture'] or infer_architecture(summary) != infer_architecture(old):
        raise ValueError('Backbone architecture mismatch')
    keys = ['encoder_layer','hidden_layer','hidden_source','decoder_layers','ode_steps','ode_start_t',
            'ode_self_cond_cfg_scale','last_t','preprocess','token_offset','resolved_token_id_mode',
            'input_layout','selected_layer_name','window_size','feature_source']
    for k in keys:
        if k in old and summary.get(k) != old[k]:
            raise ValueError(f'Feature protocol mismatch: {k}')
    for k in ('encoder_dir', 'decoder_dir'):
        if k in old and Path(summary.get(k, '')).resolve() != Path(old[k]).resolve():
            raise ValueError(f'Backbone path mismatch: {k}')
    expected_root = Path(c['models'][v]['feature_dir']).resolve()
    if Path(cfg['feature_dir']).resolve() != expected_root:
        raise ValueError('Training run and configured original feature directory differ')
    return dict(training_summary=signature(old_path), compared_fields=[*keys,'encoder_dir','decoder_dir'])


def saved_rows(c, v, cfg, run_dir, db, maximum):
    """Align every original prediction with its cached metadata and binary label."""
    root = Path(c['models'][v]['feature_dir'])
    m, y, paths = cache(root)
    summary = read_json(paths[-1])
    protocol = audit_protocol(c, v, cfg, summary)
    signed = cfg.get('input_signature', {}).get('test', [])
    actual = {s['path']: s for s in [signature(p) for p in paths]}
    for s in signed:
        # Historical paths may have been moved: require current file identity instead of guessing.
        if s['path'] not in actual or actual[s['path']] != s:
            raise ValueError('Original test feature signature differs from the training run')
    pred = run_dir/'test_predictions.tsv'
    if not pred.is_file():
        raise FileNotFoundError(f'{pred}: use --source features after Step1 when predictions are unavailable')
    inputs = [signature(p) for p in [*paths, pred]]
    n = m['feature_shape'][0]
    stop = min(n, maximum) if maximum is not None else n
    counts = {'0': 0, '1': 0}
    pending = []
    with paths[-2].open(newline='', encoding='utf-8-sig') as mf, pred.open(newline='', encoding='utf-8-sig') as pf:
        metadata, predictions = reader(mf), reader(pf, REQUIRED | {'prob', 'split'})
        for i in range(stop):
            a, b = next(metadata, None), next(predictions, None)
            if a is None or b is None:
                raise ValueError('Missing metadata/prediction rows')
            label = validate_row(a, i, c['library_id'], c['chromosome'], m['feature_shape'][1])
            validate_row(b, i, c['library_id'], c['chromosome'], m['feature_shape'][1])
            if any(a[k] != b[k] for k in REQUIRED) or label != y[i]:
                raise ValueError(f'Prediction/metadata/binary label mismatch at source row {i}')
            p = float(b['prob'])
            obs = observation(a, p, i, c['library_id']) # validate probabilities for BOTH labels
            counts[str(label)] += 1
            if label == 1:
                pending.append(obs)
            if (i+1) % 8192 == 0:
                insert(db, pending); pending.clear()
            if (i+1) % 200000 == 0:
                db.commit()
                print(f'[IMPORT] {v}/{run_dir.name} {i+1}/{stop}', flush=True)
        insert(db, pending)
        if stop == n and (next(metadata, None) is not None or next(predictions, None) is not None):
            raise ValueError('Extra metadata/prediction rows')
    partial = stop < n or source_partial(summary)
    return dict(inputs=inputs, protocol=protocol, source_feature_summary=summary,
                source_records=n, scanned_records=stop, scanned_label_counts=counts, partial_test=partial,
                probability_source='original test_predictions.tsv; positive class unchanged')


def source_partial(summary):
    while isinstance(summary, dict):
        if summary.get('limit') is not None or any(summary.get(k, {}).get('partial_test', False) for k in ('test1_dataset','test2_dataset')):
            return True
        summary = summary.get('source_feature_summary')
    return False


def infer_rows(c, v, cfg, checkpoint, root, db, maximum, device_override):
    import numpy as np
    import torch
    sys.path.insert(0, str(PROJECT/'train'))
    from model import build_model, FUSION_STRATEGY
    m, y, paths = cache(root)
    summary = read_json(paths[-1])
    dataset = summary.get('test2_dataset', {})
    if (summary.get('eval_version') != v or dataset.get('dataset_kind') != KIND or
            dataset.get('truth_source') != TRUTH or dataset.get('library_id') != c['library_id']):
        raise ValueError('Use Step1 native-positive caches of the same version/library')
    protocol = audit_protocol(c, v, cfg, summary)
    inputs = [signature(p) for p in paths]
    n, w, h = m['feature_shape']
    window = cfg['source_window_size']
    if w != window or h != cfg['hidden_dim']:
        raise ValueError('Classifier/feature dimensions differ')
    stop = min(n, maximum) if maximum is not None else n
    x = np.memmap(paths[1], mode='r', dtype=m['feature_dtype'], shape=(n,w,h))
    e = c['evaluation']
    torch.set_num_threads(e['cpu_threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    deterministic = bool(cfg['training'].get('deterministic', True))
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    requested = device_override or e['device']
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    model = build_model(cfg['strategy'], h, window, cfg['model']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    fusion = cfg['strategy'] == FUSION_STRATEGY
    fw, base = cfg.get('fusion_window_size',5), cfg.get('base_source','ref')
    if fusion and (fw < 1 or fw > w or fw % 2 != 1 or base not in ('ref','seq')):
        raise ValueError('Invalid fusion configuration')
    offset = (w-fw)//2
    codes = dict(zip('ACGTNWSMKRYBVDHZ',range(16)))
    with paths[-2].open(newline='', encoding='utf-8-sig') as f, torch.inference_mode():
        r = reader(f)
        for lo in range(0,stop,e['batch_size']):
            hi = min(stop,lo+e['batch_size'])
            rows = list(itertools.islice(r,hi-lo))
            if len(rows) != hi-lo:
                raise ValueError('Missing positive metadata rows')
            for i,row in enumerate(rows,lo):
                if validate_row(row,i,c['library_id'],c['chromosome'],w) != 1 or y[i] != 1:
                    raise ValueError('Expected positive labels in both metadata and binary cache')
            values = np.array(x[lo:hi],dtype='float32',copy=True)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite positive features')
            tensor = torch.from_numpy(values).to(device)
            if fusion:
                bases = torch.tensor([[codes[b] for b in row[base][offset:offset+fw].upper()] for row in rows],device=device,dtype=torch.long)
                probs = torch.softmax(model(tensor[:,offset:offset+fw],bases),dim=-1)[:,1]
            else:
                probs = torch.sigmoid(model(tensor))
            probs = probs.cpu().numpy()
            if probs.shape != (len(rows),):
                raise ValueError('Invalid model output shape')
            insert(db,[observation(row,float(p),i,c['library_id']) for i,(row,p) in enumerate(zip(rows,probs),lo)])
            if hi == stop or hi % 100000 < e['batch_size']:
                db.commit()
                print(f'[PREDICT] {v}/{cfg["strategy"]} {hi}/{stop}',flush=True)
        if stop == n and next(r,None) is not None:
            raise ValueError('Extra positive metadata rows')
    return dict(inputs=inputs,protocol=protocol,source_feature_summary=summary,
                source_records=n,scanned_records=stop,partial_test=stop<n or source_partial(summary),
                probability_source='best.pt classifier, eval/inference_mode, float32',
                inference=dict(device=str(device),batch_size=e['batch_size'],precision='float32',
                    torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
                    cudnn_version=torch.backends.cudnn.version(),deterministic=deterministic,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))


def evaluate(c, v, run_dir, output, source, maximum=None, feature_dir=None, device=None):
    import torch
    cfg, old = read_json(run_dir/'config.json'), read_json(run_dir/'metrics.json')
    snapshots = [signature(run_dir/p) for p in ('config.json','metrics.json','best.pt')]
    checkpoint = torch.load(run_dir/'best.pt',map_location='cpu',weights_only=False)
    for k in ('strategy','seed','fingerprint','feature_dir','model','training','source_window_size','hidden_dim','base_source','fusion_window_size'):
        if checkpoint['config'].get(k) != cfg.get(k):
            raise ValueError(f'Saved checkpoint/config mismatch: {k}')
    if checkpoint['epoch'] != old['best_epoch']:
        raise ValueError('Checkpoint epoch differs from completed metrics')
    weight_hash = sha256(run_dir/'best.pt')
    rt = c['evaluation']['read_threshold']
    rt = float(cfg['training']['threshold']) if rt is None else float(rt)
    st = c['evaluation']['site_threshold']
    if not 0 < rt < 1:
        raise ValueError('Invalid saved training threshold')
    with output_directory(output) as stage:
        db = prediction_db(stage/'predictions.sqlite')
        try:
            if source == 'predictions':
                audit = saved_rows(c,v,cfg,run_dir,db,maximum)
            else:
                audit = infer_rows(c,v,cfg,checkpoint,feature_dir or Path(c['data_root'])/v,db,maximum,device)
            if not db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]:
                raise ValueError('No positive observations selected')
            db.commit()
            metrics = export_results(db,stage,c,rt,st)
            verified = None
            if source == 'predictions' and not audit['partial_test']:
                previous = old['test_read_level']
                if metrics['read_level']['n'] != previous['tp']+previous['fn']:
                    raise ValueError('Positive count differs from original completed metrics')
                if rt == cfg['training']['threshold']:
                    if (metrics['read_level']['fn'],metrics['read_level']['tp']) != (previous['fn'],previous['tp']):
                        raise ValueError('Positive calls differ from original completed metrics')
                    verified = True
                if 'test_site_level' in old:
                    previous_site = old['test_site_level']
                    current_site = metrics['site_level']
                    if current_site['n'] != previous_site['tp']+previous_site['fn']:
                        raise ValueError('Positive site count differs from original completed metrics')
                    if st == cfg['training']['threshold'] and (current_site['fn'],current_site['tp']) != (previous_site['fn'],previous_site['tp']):
                        raise ValueError('Positive site calls differ from original completed metrics')
            metrics.update(identity(c,audit['partial_test']))
            metrics.update(version=v,architecture=c['models'][v]['architecture'],strategy=cfg['strategy'],seed=cfg['seed'],
                           checkpoint_epoch=checkpoint['epoch'],checkpoint_sha256=weight_hash,
                           source_mode=source,original_positive_counts_verified=verified,
                           collection='raw_callable_positives')
            write_json(stage/'metrics.json',metrics)
            manifest = dict(**identity(c,audit['partial_test']),run_dir=str(run_dir.resolve()),configuration=cfg,
                run_inputs=snapshots,checkpoint_sha256=weight_hash,statistics=c['statistics'],audit=audit,
                threshold_source='training configuration' if c['evaluation']['read_threshold'] is None else 'explicit frozen override',
                historical_prediction_binding=('consistency_checks_only: original TSV has no embedded checkpoint SHA256'
                    if source == 'predictions' else 'weights loaded from recorded checkpoint SHA256'),
                code_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('common.py'),PROJECT/'eval/Test1/workflow/common.py']},
                model_code_sha256={str(p.relative_to(PROJECT)):sha256(p) for p in sorted((PROJECT/'train/model').glob('*.py'))})
            write_json(stage/'run_manifest.json',manifest)
            if [signature(Path(s['path'])) for s in snapshots] != snapshots or [signature(Path(s['path'])) for s in audit['inputs']] != audit['inputs']:
                raise ValueError('Source files changed during evaluation')
        finally:
            db.close()
    print(f'[DONE] {v}/{run_dir.name}: positives={metrics["read_level"]["n"]} FN={metrics["read_level"]["fn"]} partial={metrics["partial_test"]}',flush=True)
    return metrics


def main():
    args = parser().parse_args()
    c,versions = load_config(args)
    if (args.run_dir or args.feature_dir) and len(versions) != 1:
        raise ValueError('--run-dir/--feature-dir require a single --version')
    e = c['evaluation']
    if e['batch_size'] < 1 or e['cpu_threads'] < 1:
        raise ValueError('Invalid inference settings')
    source = args.source or e['source']
    if source not in ('predictions','features'):
        raise ValueError('Invalid evaluation.source')
    tasks = [(v,r,(args.out_dir or Path(c['results_root']))/v/r.name) for v in versions for r in discover(c,v,args.run_dir)]
    if args.dry_run:
        print(json.dumps(dict(source=source,max_records=args.max_records,runs=[dict(version=v,run_dir=str(r),output=str(o),
            predictions_exist=(r/'test_predictions.tsv').is_file(),positive_cache_exists=((args.feature_dir or Path(c['data_root'])/v)/'test.pt').is_file()) for v,r,o in tasks]),indent=2))
        return
    # Refuse occupied destinations before beginning a large batch.
    for _,_,out in tasks:
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite results: {out}')
    for v,r,o in tasks:
        evaluate(c,v,r,o,source,args.max_records,args.feature_dir,args.device)


if __name__ == '__main__':
    main()
