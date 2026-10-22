#!/usr/bin/env python3
"""YAML-configured BERT / Diffusion / Flow Matching / GPT window extraction.

Reuses the existing architecture-specific extractors without Test1's WGA-only
assumptions. Relative YAML paths are project-relative; CLI paths are cwd-relative.
Only publish a new output directory after every requested split succeeds.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
# Reuse config validation and I/O helpers, not the WGA extraction entry point.
sys.path.insert(0, str(PROJECT / 'eval/Test1/workflow'))
from common import (add_extraction_performance_args, infer_architecture as infer_legacy_architecture, module,
                    output_directory, read_json, signature, valid_model_name, write_json)
from extraction_config import (FAMILY_DEFAULTS, FAMILY_FIELDS, MODEL_COMMON, PATHS,
                               as_path, decoder_directory, mapping, reject_unknown,
                               snapshot, validate)

DEFAULT_CONFIG = SCRIPT_DIR / 'config/step4_config.yaml'
PATHS = PATHS | {'token_cache_dir', 'train_config'}
# Flow uses the same ODE controls, but a distinct loader and runtime. Keep this
# dispatch local: Test1's legacy extraction entry point has its own backends.
FAMILY_DEFAULTS = {**FAMILY_DEFAULTS, 'flow_matching': {
    **FAMILY_DEFAULTS['diffusion'], 'ode_steps': 2, 'ode_start_t': .98}}
GENERAL = dict(data_dir='output/step3_output/new', out_dir=None,
               output_root='output/step4_output', splits=['train', 'valid', 'test'],
               device='auto', cpu_threads=4, batch_size=8, dtype='float16',
               window_size=15, preprocess='none', save_every=1024, log_every=1000,
               limit=None, codec_batch_size=1, codec_buffer_size=1024, pooling_device='model',
               codec_max_samples=524288, token_cache_mode='off',
               token_cache_dir='output/step4_token_cache', inference_precision='fp32',
               output_window_size=None, train_config=None, experiment=None, split_devices=None)
EXTRACTORS = dict(bert='step4_extract_window_hidden_states_BERT-js.py',
                  diffusion='step4_extract_window_hidden_states_Diffusion-kx.py',
                  flow_matching='step4_extract_window_hidden_states_Flow-matching.py',
                  gpt='step4_extract_window_hidden_states_GPT.py')


def infer_architecture(config):
    dlm = config.get('dlm_config', {})
    if str(dlm.get('model', '')).startswith('flow_matching'):
        return 'flow_matching'
    return infer_legacy_architecture(config)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='统一 Step4 YAML 配置')
    p.add_argument('--model', '--version', dest='version', help='选择 models 中的条目')
    p.add_argument('--architecture', '--arch', choices=['auto', 'bert', 'diffusion', 'flow_matching', 'gpt'])
    for key in ('data_dir', 'out_dir', 'output_root', 'model_root', 'encoder_dir', 'tokenizer_file', 'token_cache_dir', 'train_config'):
        p.add_argument('--' + key.replace('_', '-'), type=Path)
    p.add_argument('--decoder-dir', '--bert-dir', '--dlm-dir', dest='decoder_dir', type=Path)
    p.add_argument('--splits', nargs='+', choices=['train', 'valid', 'test'],
                   help='需要处理的 split；只测试可用 --splits test')
    p.add_argument('--device', help='auto / cpu / cuda:0 / cuda:1')
    p.add_argument('--split-devices', nargs='+', metavar='SPLIT=DEVICE',
                   help='按split分配设备，如 train=cuda:0 valid=cuda:1 test=cuda:1；off关闭分配')
    for key in ('cpu_threads', 'batch_size', 'window_size', 'output_window_size', 'save_every', 'log_every',
                'limit', 'encoder_layer', 'hidden_layer', 'token_offset', 'ode_steps', 'max_tokens'):
        p.add_argument('--' + key.replace('_', '-'), type=int)
    p.add_argument('--experiment', help='生成训练配置的实验名；默认使用输出目录名和输出长度')
    p.add_argument('--dtype', choices=['float16', 'float32'], help='特征落盘精度，与主干推理精度独立')
    p.add_argument('--inference-precision', choices=['fp32', 'fp16'],
                   help='主干FP32或CUDA FP16 autocast；codec和池化保持FP32，CPU使用FP32')
    p.add_argument('--preprocess', choices=['none', 'auto'])
    p.add_argument('--hidden-source', choices=['ode_hidden', 'context_hidden', 'last_hidden', 'last_hidden_state'])
    for key in ('ode_start_t', 'ode_self_cond_cfg_scale', 'last_t'):
        p.add_argument('--' + key.replace('_', '-'), type=float)
    p.add_argument('--decoder-layers', help='GPT: last / embed / 层编号 / 逗号分隔列表')
    p.add_argument('--selected-layer-name', help='GPT 多层选择时明确落盘的一个层名')
    p.add_argument('--token-id-mode', choices=['auto', 'tokenizer', 'offset', 'direct'])
    p.add_argument('--special-tokens', choices=['none', 'bos_eos'])
    p.add_argument('--codec-mode', choices=['auto', 'reference'])
    p.add_argument('--overlong-policy', choices=['skip', 'error'])
    p.add_argument('--token-cache-mode', choices=['off', 'readwrite', 'readonly'],
                   help='Raw codec token cache: off / readwrite / readonly')
    p.add_argument('--codec-max-samples', type=int,
                   help='Maximum signal samples per codec batch; a longer single window runs alone')
    add_extraction_performance_args(p)
    # Unspecified CLI options must not override the YAML values.
    p.set_defaults(codec_batch_size=None, codec_buffer_size=None, pooling_device=None)
    p.add_argument('--print-plan', '--dry-run', action='store_true',
                   help='显示最终配置、split 路径及缺失项；不加载权重或创建输出')
    return p


def resolve(args):
    import yaml
    config = args.config.expanduser().resolve()
    initial = signature(config)
    raw = mapping(yaml.safe_load(config.read_text(encoding='utf-8')), 'config')
    reject_unknown(raw, {'model', 'extraction', 'models'}, 'config')
    profiles = mapping(raw.get('models', {}), 'models')
    name = valid_model_name(args.version if args.version is not None else raw.get('model'))
    if name not in profiles:
        raise ValueError(f'Unknown model {name!r}; available: {list(profiles)}')
    profile = mapping(profiles[name], f'models.{name}')
    general = mapping(raw.get('extraction', {}), 'extraction')
    reject_unknown(general, GENERAL, 'extraction')
    reject_unknown(profile, set(GENERAL) | MODEL_COMMON | FAMILY_FIELDS, f'models.{name}')
    cli = {k: v for k, v in vars(args).items()
           if v is not None and k not in ('config', 'version', 'print_plan')}
    settings = {**GENERAL, **general, **profile, **cli, 'version': name}
    # An explicit DEVICE remains a convenient way to force a single-device run.
    if 'device' in cli and 'split_devices' not in cli:
        settings['split_devices'] = None
    for key in PATHS:
        if settings.get(key) is not None:
            settings[key] = as_path(settings[key], from_cli=key in cli)
    root = settings.get('model_root')
    if root is None and not (settings.get('encoder_dir') and settings.get('decoder_dir')):
        raise ValueError('Set model_root or both encoder_dir and decoder_dir')
    encoder = settings.get('encoder_dir') or root / 'encoder'
    decoder = settings.get('decoder_dir') or decoder_directory(root)
    for path in (encoder / 'config.json', decoder / 'config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    detected = infer_architecture(read_json(decoder / 'config.json'))
    family = settings.get('architecture', 'auto')
    if family == 'auto':
        family = detected
    if family not in FAMILY_DEFAULTS or (detected and detected != family):
        raise ValueError(f'Architecture mismatch/unknown: configured={family}, detected={detected}')
    invalid = (set(profile) | set(cli)) & (FAMILY_FIELDS - set(FAMILY_DEFAULTS[family]))
    if invalid:
        raise ValueError(f'Options do not apply to {family}: {sorted(invalid)}')
    settings = {**FAMILY_DEFAULTS[family], **settings, 'architecture': family,
                'encoder_dir': encoder, 'decoder_dir': decoder}
    settings['out_dir'] = settings['out_dir'] or settings['output_root'] / name
    splits = settings['splits']
    if (not isinstance(splits, list) or not splits
            or any(s not in ('train', 'valid', 'test') for s in splits)
            or len(set(splits)) != len(splits)):
        raise ValueError('splits must be a nonempty list without duplicates: train, valid, test')
    settings['split_devices'] = resolve_split_devices(settings['split_devices'], splits, settings['device'])
    if family == 'gpt':
        layers = settings['decoder_layers']
        if isinstance(layers, list):
            if not layers or any(type(v) not in (int, str) for v in layers):
                raise ValueError('decoder_layers must contain layer numbers or last/embed')
            layers = ','.join(map(str, layers))
        elif type(layers) is int:
            layers = str(layers)
        if not isinstance(layers, str) or not layers.strip():
            raise ValueError('Invalid decoder_layers')
        settings['decoder_layers'] = layers.strip().lower()
        if (',' in layers or layers.strip().lower() == 'all') and not settings['selected_layer_name']:
            raise ValueError('Multiple GPT layers require selected_layer_name; each output stores one layer')
    validate(settings)
    from step4_center_crop import output_length
    settings['output_window_size'] = output_length(settings['window_size'], settings['output_window_size'])
    if settings['experiment'] is not None and (
            not isinstance(settings['experiment'], str)
            or not re.fullmatch(r'[A-Za-z0-9_.-]+', settings['experiment'])):
        raise ValueError('experiment may contain only letters, digits, underscore, dot and hyphen')
    if settings['inference_precision'] not in ('fp32', 'fp16'):
        raise ValueError('inference_precision must be fp32 or fp16')
    if settings['token_cache_mode'] not in ('off', 'readwrite', 'readonly'):
        raise ValueError('token_cache_mode must be off, readwrite or readonly')
    if type(settings['codec_max_samples']) is not int or settings['codec_max_samples'] < 1:
        raise ValueError('codec_max_samples must be a positive integer')
    cache_path = settings['token_cache_dir']
    if not isinstance(cache_path, Path):
        raise ValueError('token_cache_dir must be a nonempty path; use token_cache_mode=off to disable caching')
    if settings['token_cache_mode'] != 'off' and (
            cache_path == settings['out_dir'] or settings['out_dir'] in cache_path.parents):
        raise ValueError('token_cache_dir must be outside the feature out_dir')
    if signature(config) != initial:
        raise ValueError('Configuration changed while being read; retry')
    return argparse.Namespace(**settings, config=config, config_signature=initial,
                              print_plan=args.print_plan)


def input_files(args):
    files, missing = {}, []
    for split in args.splits:
        candidates = [args.data_dir / (split + suffix) for suffix in ('.jsonl.gz', '.jsonl')]
        path = next((p for p in candidates if p.is_file()), None)
        files[split] = path
        if path is None:
            missing.append(f'{args.data_dir}/{split}.jsonl[.gz]')
    return files, missing


def extract(args, stage):
    """Load one encoder/backbone once, then stream each requested split."""
    import torch
    from step4_token_cache import CodecTokenCache, bind_token_stream
    from step4_precision import DecoderPrecision, bind_decoder_precision, annotate_outputs
    family = args.architecture
    impl = module('_unified_step4_' + family, SCRIPT_DIR / EXTRACTORS[family])
    opts = argparse.Namespace(**vars(args))
    opts.out_dir = stage
    opts.bert_dir = opts.decoder_dir
    if family == 'gpt':
        opts.decoder_layers = impl.parse_layers(args.decoder_layers)
    impl.validate_args(opts)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision('high' if family == 'bert' else 'highest')
    device = impl.resolve_device(args.device)
    if device.type == 'cuda':
        # Custom checkpoint code may allocate on the current CUDA device.
        torch.cuda.set_device(device)
    precision = DecoderPrecision(args.inference_precision, device)
    preprocessor = impl.load_preprocessor(args.preprocess, args.encoder_dir)
    print(f'[INFO] model={args.version} architecture={family} device={device} splits={args.splits}', flush=True)
    print(f'[INFO] requested_inference_precision={precision.requested} '
          f'effective_inference_precision={precision.effective}; codec=fp32 pooling=fp32 storage={args.dtype}', flush=True)
    print(f'[INFO] input_window_size={args.window_size} output_window_size={args.output_window_size}; '
          'full input context retained; center positions selected before storage', flush=True)
    if family == 'gpt':
        protocol = impl.load_token_protocol(opts)
        encoder = impl.load_encoder(str(args.encoder_dir)).to(device).eval()
        decoder = impl.load_decoder(str(args.decoder_dir)).to(device).eval()
    else:
        encoder, decoder = impl.load_models(args.encoder_dir, args.decoder_dir, device)
        if family in ('diffusion', 'flow_matching'):
            protocol = impl.load_token_protocol(args.encoder_dir, args.decoder_dir)
    backend = getattr(impl, 'shared', impl)
    cache = CodecTokenCache(opts, encoder, device, backend)
    if args.codec_batch_size > 1:
        print('[INFO] Equal-length batched codec enabled; tokens may differ from serial encoding. '
              'Use matching extraction protocols for training/evaluation.', flush=True)
    files, _ = input_files(args)
    def build(split, layer=None):
        stream = cache.for_split(split, files[split])
        with bind_token_stream(impl, stream), bind_decoder_precision(impl, family, precision):
            if family == 'bert':
                row = impl.build_split(split, args.data_dir, stage, encoder, decoder,
                                       preprocessor, opts, device, layer)
            else:
                row = impl.build_split(split, encoder, decoder, preprocessor, opts, device, protocol)
        row['token_cache'] = stream.stats
        return row
    if family == 'bert':
        index, layer = impl.resolve_hidden_layer_name(args.hidden_layer, int(decoder.config.num_hidden_layers))
        splits = [build(s, layer) for s in args.splits]
        summary = dict(model_family='PoreBERT', encoder_layer=args.encoder_layer,
                       hidden_layer=args.hidden_layer, resolved_hidden_state_index=index,
                       selected_layer_name=layer, token_offset=args.token_offset,
                       special_tokens=dict(pad=impl.PAD_TOKEN_ID, bos=impl.BOS_TOKEN_ID,
                                           eos=impl.EOS_TOKEN_ID, cls=impl.CLS_TOKEN_ID),
                       bert_dir=str(args.decoder_dir), hidden_size=int(decoder.config.hidden_size),
                       max_position_embeddings=int(decoder.config.max_position_embeddings),
                       feature_source=f'chrom19_signal_window_v548_{layer}_window{args.window_size}_refspan')
    else:
        splits = [build(s) for s in args.splits]
        summary = impl.extraction_metadata(opts, protocol)
        summary.update(hidden_size=protocol.hidden_size,
                       max_position_embeddings=protocol.maximum if family == 'gpt' else protocol.max_positions,
                       feature_source=impl.feature_source(opts, protocol) if family == 'gpt' else impl.feature_source(opts))
    annotate_outputs(stage, splits, precision)
    # Override each backend's legacy hard-coded FP32 metadata with the actual
    # requested/effective precision in the public summary and .pt manifests.
    summary.update(precision.metadata())
    # Relocate staging paths. Bin/metadata paths inside *.pt already are relative.
    for row in splits:
        for key in ('pt_path', 'feature_path', 'label_path', 'metadata_path'):
            if key in row:
                row[key] = str(args.out_dir / Path(row[key]).name)
    summary.update(model_version=args.version, model_architecture=family, splits=splits,
                   data_dir=str(args.data_dir), out_dir=str(args.out_dir),
                   encoder_dir=str(args.encoder_dir), decoder_dir=str(args.decoder_dir),
                   preprocess=args.preprocess, window_size=args.output_window_size, dtype=args.dtype,
                   input_window_size=args.window_size, output_window_size=args.output_window_size,
                   batch_size=args.batch_size, cpu_threads=args.cpu_threads, device=str(device),
                   save_every=args.save_every, limit=args.limit, partial_extraction=args.limit is not None,
                   codec_batch_size=args.codec_batch_size, codec_buffer_size=args.codec_buffer_size,
                   codec_max_samples=args.codec_max_samples, token_cache_mode=args.token_cache_mode,
                   token_cache_dir=str(args.token_cache_dir), token_cache_protocol=cache.contract,
                   pooling_device=args.pooling_device, extraction_config=snapshot(args))
    if args.output_window_size != args.window_size:
        start = (args.window_size - args.output_window_size) // 2
        summary.update(operation='extract_and_center_crop',
                       feature_source=summary['feature_source'] + f'_center{args.output_window_size}',
                       preserves_original_bert_context=True, preserves_original_model_context=True,
                       center_crop=dict(input_window_size=args.window_size, start0=start,
                                        end0=start + args.output_window_size,
                                        implementation='in_memory_before_storage'),
                       metadata_coordinate_policy='Original genomic/signal coordinates preserved; seq/ref cropped; '
                                                  'offsets in crop_start0/crop_end0')
        for row in splits:
            row.update(input_window_size=args.window_size, output_window_size=args.output_window_size,
                       crop_start0=start, crop_end0=start + args.output_window_size)
    return summary


def resolve_split_devices(value, splits, default_device):
    if value is None or value == ['off']:
        return None
    if isinstance(value, list):
        result = {}
        for item in value:
            if not isinstance(item, str) or item.count('=') != 1:
                raise ValueError('split_devices CLI format: train=cuda:0 valid=cuda:1 test=cuda:1, or off')
            split, device = item.split('=')
            if split in result:
                raise ValueError(f'Duplicate split_devices entry: {split}')
            result[split] = device
        value = result
    value = mapping(value, 'split_devices')
    reject_unknown(value, {'train', 'valid', 'test'}, 'split_devices (use valid, not val)')
    for split, device in value.items():
        if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda:(?:0|[1-9][0-9]*)', device):
            raise ValueError(f'split_devices.{split} must be cpu or an explicit cuda:N')
    resolved = {split: value.get(split, default_device) for split in splits}
    if any(not isinstance(device, str) or not re.fullmatch(r'cpu|cuda:(?:0|[1-9][0-9]*)', device)
           for device in resolved.values()):
        raise ValueError('Unmapped splits need an explicit device (cpu/cuda:N), or map every selected split')
    return resolved


def device_groups(args):
    groups = {}
    for split in args.splits:
        device = args.split_devices[split] if args.split_devices is not None else args.device
        groups.setdefault(device, []).append(split)
    return groups


def _extract_worker(args, stage, report):
    """Spawned once per device; valid/test on the same GPU reuse one model."""
    os.environ['HF_MODULES_CACHE'] = str(PROJECT / 'output/.hf_modules' / ('step4_' + args.device.replace(':', '_')))
    summary = extract(args, stage)
    write_json(report, summary)


def merge_worker_summaries(args, summaries):
    # Device, token cache hardware identity and split paths can differ; the
    # actual backbone, feature layout and numerical protocol must agree.
    ignored = {'splits', 'device', 'extraction_config', 'split_devices', 'token_cache_protocol'}
    first = summaries[0]
    for other in summaries[1:]:
        differences = [key for key in set(first) | set(other)
                       if key not in ignored and first.get(key) != other.get(key)]
        if differences:
            raise ValueError(f'Incompatible worker feature protocols: {sorted(differences)}')
    rows = {}
    protocols = {}
    for summary in summaries:
        for row in summary['splits']:
            split = row['split']
            if split in rows or split not in args.splits:
                raise ValueError(f'Duplicate/unexpected worker split: {split}')
            if summary['device'] != args.split_devices[split]:
                raise ValueError(f'Worker device mismatch for {split}')
            rows[split] = {**row, 'device': summary['device']}
            protocols[split] = summary.get('token_cache_protocol')
    if set(rows) != set(args.splits):
        raise ValueError(f'Missing worker splits: {sorted(set(args.splits) - set(rows))}')
    result = dict(first)
    result.update(splits=[rows[s] for s in args.splits], device='per_split' if len(summaries) > 1 else first['device'],
                  split_devices=dict(args.split_devices), extraction_config=snapshot(args),
                  device_groups=[dict(device=s['device'], splits=[r['split'] for r in s['splits']]) for s in summaries],
                  token_cache_protocol_by_split=protocols)
    if any(p != first.get('token_cache_protocol') for p in protocols.values()):
        result['token_cache_protocol'] = None
    return result


def extract_scheduled(args, stage, *, worker_target=None):
    """Concurrent devices, sequential splits on each device, one atomic output."""
    if args.split_devices is None:
        return extract(args, stage)
    groups = device_groups(args)
    if len(groups) == 1:
        device, splits = next(iter(groups.items()))
        single = argparse.Namespace(**{**vars(args), 'device': device, 'splits': splits})
        return merge_worker_summaries(args, [extract(single, stage)])
    import multiprocessing
    from multiprocessing.connection import wait
    context = multiprocessing.get_context('spawn')
    reports = stage / '.worker_summaries'
    reports.mkdir()
    workers = []
    try:
        for i, (device, splits) in enumerate(groups.items()):
            worker_args = argparse.Namespace(**{**vars(args), 'device': device, 'splits': splits})
            report = reports / f'worker_{i}.json'
            process = context.Process(target=worker_target or _extract_worker,
                                      args=(worker_args, stage, report), name=f'step4_{device}')
            process.start()
            workers.append((process, report))
            print(f'[WORKER] pid={process.pid} device={device} splits={splits}', flush=True)
        running = {process.sentinel: process for process, _ in workers}
        while running:
            for sentinel in wait(list(running), timeout=1):
                process = running.pop(sentinel)
                process.join()
                if process.exitcode:
                    raise RuntimeError(f'{process.name} failed (exit={process.exitcode}); cancelling other workers')
        summaries = [read_json(report) for _, report in workers]
        return merge_worker_summaries(args, summaries)
    finally:
        for process, _ in workers:
            if process.is_alive():
                process.terminate()
        for process, _ in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        shutil.rmtree(reports, ignore_errors=True)


def training_config(args):
    """Read the template once at startup; never modify it or launch training."""
    if args.train_config is None or set(args.splits) != {'train', 'valid', 'test'}:
        return None
    import yaml
    experiment = args.experiment or re.sub(r'[^A-Za-z0-9_.-]', '_',
                                          f'{args.out_dir.name}_center{args.output_window_size}')
    config = mapping(yaml.safe_load(args.train_config.read_text(encoding='utf-8')), 'train_config')
    config.update(feature_dir=str(args.out_dir), metadata_dir=None,
                  fusion_window_size=args.output_window_size, experiment=experiment)
    return config


def run(args):
    args = resolve(args)
    files, missing = input_files(args)
    train = training_config(args)
    if args.print_plan:
        plan = snapshot(args)
        plan.update(extractor=str(SCRIPT_DIR / EXTRACTORS[args.architecture]),
                    input_files={k: str(v) if v else None for k, v in files.items()},
                    missing_inputs=missing, output_exists=args.out_dir.exists(),
                    device_groups=[dict(device=d, splits=s) for d, s in device_groups(args).items()],
                    generated_train_config=str(args.out_dir / 'train_config.yaml') if train is not None else None)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if missing:
        raise FileNotFoundError('Missing input splits; set extraction.data_dir / --data-dir: ' + ', '.join(missing))
    watched = list(files.values())
    for folder in (args.encoder_dir, args.decoder_dir):
        watched.extend(folder / name for name in ('config.json', 'preprocessor_config.json', 'tokenizer.json')
                       if (folder / name).is_file())
    before = [signature(p) for p in watched]
    # YAML is a startup snapshot, so subsequent edits do not invalidate this run.
    with output_directory(args.out_dir) as stage:
        write_json(stage / 'resolved_extraction_config.json', snapshot(args))
        summary = extract_scheduled(args, stage)
        changed = [s['path'] for s in before if signature(Path(s['path'])) != s]
        if changed:
            raise RuntimeError(f'Input or model configuration changed during extraction: {changed}')
        summary['input_and_model_config_signatures'] = before
        if train is not None:
            import yaml
            (stage / 'train_config.yaml').write_text(
                '# Step4 自动生成；相对训练路径仍以 train/ 为基准。\n'
                + yaml.safe_dump(train, allow_unicode=True, sort_keys=False), encoding='utf-8')
            summary['training_config'] = dict(path=str(args.out_dir / 'train_config.yaml'),
                                             template=str(args.train_config), experiment=train['experiment'])
        write_json(stage / 'feature_summary.json', summary)
    counts = {row['split']: row['records'] for row in summary['splits']}
    print(f'[DONE] {args.version}: {counts}; output={args.out_dir}', flush=True)
    if train is not None:
        print('[TRAIN] ' + shlex.join(['bash', str(PROJECT / 'train/run/example/run.sh'),
                                      str(args.out_dir / 'train_config.yaml')]), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    # Keep downloaded dynamic-module caches out of the read-only checkpoint trees.
    os.environ.setdefault('HF_MODULES_CACHE', str(PROJECT / 'output/.hf_modules'))
    run(parser().parse_args())
