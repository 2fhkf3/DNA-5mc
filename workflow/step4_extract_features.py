#!/usr/bin/env python3
"""YAML-configured BERT / Diffusion / GPT window extraction for train/valid/test.

Reuses the existing architecture-specific extractors without Test1's WGA-only
assumptions. Relative YAML paths are project-relative; CLI paths are cwd-relative.
Only publish a new output directory after every requested split succeeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
# Reuse config validation and I/O helpers, not the WGA extraction entry point.
sys.path.insert(0, str(PROJECT / 'eval/Test1/workflow'))
from common import (add_extraction_performance_args, infer_architecture, module,
                    output_directory, read_json, signature, valid_model_name, write_json)
from extraction_config import (FAMILY_DEFAULTS, FAMILY_FIELDS, MODEL_COMMON, PATHS,
                               as_path, decoder_directory, mapping, reject_unknown,
                               snapshot, validate)

DEFAULT_CONFIG = SCRIPT_DIR / 'step4_config.yaml'
GENERAL = dict(data_dir='output/step3_output/new', out_dir=None,
               output_root='output/step4_output', splits=['train', 'valid', 'test'],
               device='auto', cpu_threads=4, batch_size=8, dtype='float16',
               window_size=15, preprocess='none', save_every=1024, log_every=1000,
               limit=None, codec_batch_size=1, codec_buffer_size=1024, pooling_device='model')
EXTRACTORS = dict(bert='step4_extract_window_hidden_states_BERT-js.py',
                  diffusion='step4_extract_window_hidden_states_Diffusion-kx.py',
                  gpt='step4_extract_window_hidden_states_GPT.py')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='统一 Step4 YAML 配置')
    p.add_argument('--model', '--version', dest='version', help='选择 models 中的条目')
    p.add_argument('--architecture', '--arch', choices=['auto', 'bert', 'diffusion', 'gpt'])
    for key in ('data_dir', 'out_dir', 'output_root', 'model_root', 'encoder_dir', 'tokenizer_file'):
        p.add_argument('--' + key.replace('_', '-'), type=Path)
    p.add_argument('--decoder-dir', '--bert-dir', '--dlm-dir', dest='decoder_dir', type=Path)
    p.add_argument('--splits', nargs='+', choices=['train', 'valid', 'test'],
                   help='需要处理的 split；只测试可用 --splits test')
    p.add_argument('--device', help='auto / cpu / cuda:0 / cuda:1')
    for key in ('cpu_threads', 'batch_size', 'window_size', 'save_every', 'log_every',
                'limit', 'encoder_layer', 'hidden_layer', 'token_offset', 'ode_steps', 'max_tokens'):
        p.add_argument('--' + key.replace('_', '-'), type=int)
    p.add_argument('--dtype', choices=['float16', 'float32'], help='落盘精度，保持原主干计算精度')
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
    preprocessor = impl.load_preprocessor(args.preprocess, args.encoder_dir)
    print(f'[INFO] model={args.version} architecture={family} device={device} splits={args.splits}', flush=True)
    if family == 'gpt':
        protocol = impl.load_token_protocol(opts)
        encoder = impl.load_encoder(str(args.encoder_dir)).to(device).eval()
        decoder = impl.load_decoder(str(args.decoder_dir)).to(device).eval()
    else:
        encoder, decoder = impl.load_models(args.encoder_dir, args.decoder_dir, device)
        if family == 'diffusion':
            protocol = impl.load_token_protocol(args.encoder_dir, args.decoder_dir)
    if family == 'bert':
        index, layer = impl.resolve_hidden_layer_name(args.hidden_layer, int(decoder.config.num_hidden_layers))
        splits = [impl.build_split(s, args.data_dir, stage, encoder, decoder, preprocessor,
                                   opts, device, layer) for s in args.splits]
        summary = dict(model_family='PoreBERT', encoder_layer=args.encoder_layer,
                       hidden_layer=args.hidden_layer, resolved_hidden_state_index=index,
                       selected_layer_name=layer, token_offset=args.token_offset,
                       special_tokens=dict(pad=impl.PAD_TOKEN_ID, bos=impl.BOS_TOKEN_ID,
                                           eos=impl.EOS_TOKEN_ID, cls=impl.CLS_TOKEN_ID),
                       bert_dir=str(args.decoder_dir), hidden_size=int(decoder.config.hidden_size),
                       max_position_embeddings=int(decoder.config.max_position_embeddings),
                       feature_source=f'chrom19_signal_window_v548_{layer}_window{args.window_size}_refspan')
    else:
        splits = [impl.build_split(s, encoder, decoder, preprocessor, opts, device, protocol)
                  for s in args.splits]
        summary = impl.extraction_metadata(opts, protocol)
        summary.update(hidden_size=protocol.hidden_size,
                       max_position_embeddings=protocol.maximum if family == 'gpt' else protocol.max_positions,
                       feature_source=impl.feature_source(opts, protocol) if family == 'gpt' else impl.feature_source(opts))
    # Relocate staging paths. Bin/metadata paths inside *.pt already are relative.
    for row in splits:
        for key in ('pt_path', 'feature_path', 'label_path', 'metadata_path'):
            if key in row:
                row[key] = str(args.out_dir / Path(row[key]).name)
    summary.update(model_version=args.version, model_architecture=family, splits=splits,
                   data_dir=str(args.data_dir), out_dir=str(args.out_dir),
                   encoder_dir=str(args.encoder_dir), decoder_dir=str(args.decoder_dir),
                   preprocess=args.preprocess, window_size=args.window_size, dtype=args.dtype,
                   batch_size=args.batch_size, cpu_threads=args.cpu_threads, device=str(device),
                   save_every=args.save_every, limit=args.limit, partial_extraction=args.limit is not None,
                   codec_batch_size=args.codec_batch_size, codec_buffer_size=args.codec_buffer_size,
                   pooling_device=args.pooling_device, extraction_config=snapshot(args))
    return summary


def run(args):
    args = resolve(args)
    files, missing = input_files(args)
    if args.print_plan:
        plan = snapshot(args)
        plan.update(extractor=str(SCRIPT_DIR / EXTRACTORS[args.architecture]),
                    input_files={k: str(v) if v else None for k, v in files.items()},
                    missing_inputs=missing, output_exists=args.out_dir.exists())
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
        summary = extract(args, stage)
        changed = [s['path'] for s in before if signature(Path(s['path'])) != s]
        if changed:
            raise RuntimeError(f'Input or model configuration changed during extraction: {changed}')
        summary['input_and_model_config_signatures'] = before
        write_json(stage / 'feature_summary.json', summary)
    counts = {row['split']: row['records'] for row in summary['splits']}
    print(f'[DONE] {args.version}: {counts}; output={args.out_dir}', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    # Keep downloaded dynamic-module caches out of the read-only checkpoint trees.
    os.environ.setdefault('HF_MODULES_CACHE', str(PROJECT / 'output/.hf_modules'))
    run(parser().parse_args())
