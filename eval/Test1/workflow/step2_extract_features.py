#!/usr/bin/env python3
"""Extract one WGA test split using YAML-configured model/architecture profiles."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from common import add_extraction_performance_args, PROJECT, module, output_directory, read_json, readonly_db, signature, write_json
from extraction_config import DEFAULT_CONFIG, resolve, snapshot

EXTRACTORS = {
    'diffusion': 'step4_extract_window_hidden_states_Diffusion-kx.py',
    'flow_matching': 'step4_extract_window_hidden_states_Flow-matching.py',
    'gpt': 'step4_extract_window_hidden_states_GPT.py',
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='Step2 YAML 配置；相对路径以项目根目录为基准')
    p.add_argument('--version', '--model', help='选择 YAML models 条目，不限制 Vxxx 编号')
    p.add_argument('--architecture', '--arch', choices=['auto', 'diffusion', 'flow_matching', 'bert', 'gpt'], help='主干架构，与版本编号分开')
    p.add_argument('--data-dir', type=Path, help='Step1 生成的 test.jsonl.gz 和数据清单目录')
    p.add_argument('--out-dir', type=Path, help='新建特征目录，禁止覆盖')
    p.add_argument('--output-root', type=Path, help='默认输出根目录，最终路径为 features/<version>')
    p.add_argument('--model-root', type=Path, help='对应预训练模型根目录；默认 train/backbone 的版本链接')
    p.add_argument('--encoder-dir', type=Path, help='独立 encoder 路径，覆盖 model_root/encoder')
    p.add_argument('--decoder-dir', type=Path, help='独立主干 checkpoint 路径，支持不同目录名')
    p.add_argument('--device', help='auto / cpu / cuda:0 等；仅用于特征提取')
    p.add_argument('--batch-size', type=int, help='主干提取批次大小；CPU 测试可设为 1')
    p.add_argument('--cpu-threads', type=int, help='CPU 算子线程数')
    p.add_argument('--dtype', choices=['float16', 'float32'], help='特征落盘精度，不改变原主干算法')
    p.add_argument('--window-size', type=int, help='输入窗口长度，必须匹配语料清单')
    p.add_argument('--preprocess', choices=['none', 'auto'], help='已处理语料使用 none')
    p.add_argument('--save-every', type=int, help='提取器缓存写入间隔')
    p.add_argument('--log-every', type=int, help='提取器进度日志间隔')
    p.add_argument('--limit', type=int, help='最多处理 N 个窗口，仅测试用')
    p.add_argument('--encoder-layer', type=int, help='encoder 量化轨道参数')
    p.add_argument('--hidden-layer', type=int, help='BERT hidden_states 索引；-1 为最后层')
    p.add_argument('--token-offset', type=int, help='BERT/GPT codec 词表偏移')
    p.add_argument('--hidden-source', help='Diffusion / Flow Matching 的 ode_hidden/context_hidden/last_hidden 等')
    p.add_argument('--ode-steps', type=int)
    p.add_argument('--ode-start-t', type=float)
    p.add_argument('--ode-self-cond-cfg-scale', type=float)
    p.add_argument('--last-t', type=float)
    p.add_argument('--decoder-layers', help='GPT last、embed、层编号或逗号分隔列表')
    p.add_argument('--selected-layer-name', help='GPT 多层选项中要落盘的一层')
    p.add_argument('--token-id-mode', choices=['auto', 'tokenizer', 'offset', 'direct'])
    p.add_argument('--tokenizer-file', type=Path)
    p.add_argument('--special-tokens', choices=['none', 'bos_eos'])
    p.add_argument('--codec-mode', choices=['auto', 'reference'])
    p.add_argument('--max-tokens', type=int)
    p.add_argument('--overlong-policy', choices=['skip', 'error'])
    p.add_argument('--print-plan', action='store_true', help='仅解析配置并检查模型 config 路径，不加载权重或写输出')
    add_extraction_performance_args(p)
    p.set_defaults(codec_batch_size=None, codec_buffer_size=None, pooling_device=None)
    return p


def extractor_command(args, data, stage):
    """Call the matching architecture's existing extractor, limited to test."""
    script = EXTRACTORS[args.architecture]
    command = [sys.executable, '-u', str(PROJECT/'workflow'/script), '--data-dir', str(data),
               '--encoder-dir', str(args.encoder_dir), '--decoder-dir', str(args.decoder_dir), '--out-dir', str(stage),
               '--splits', 'test', '--device', args.device, '--cpu-threads', str(args.cpu_threads),
               '--window-size', str(args.window_size), '--batch-size', str(args.batch_size),
               '--preprocess', args.preprocess, '--dtype', args.dtype,
               '--save-every', str(args.save_every), '--log-every', str(args.log_every)]
    for key in ('codec_batch_size', 'codec_buffer_size', 'pooling_device'):
        command += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    from extraction_config import FAMILY_DEFAULTS
    for key in FAMILY_DEFAULTS[args.architecture]:
        value = getattr(args, key)
        if value is not None:
            command += ['--'+key.replace('_', '-'), str(value)]
    if args.limit is not None:
        command += ['--limit', str(args.limit)]
    return command


def run(args):
    args = resolve(args)
    if args.print_plan:
        print(json.dumps(snapshot(args), ensure_ascii=False, indent=2))
        return
    data = args.data_dir.resolve()
    manifest = read_json(data/'dataset_manifest.json')
    if manifest['label_counts']['1'] or manifest['strand'] != '+' or manifest['window_size'] != args.window_size:
        raise ValueError('Expected all-negative positive-strand data with configured window_size')
    source = data/'test.jsonl.gz'
    signatures = [signature(source), signature(data/'dataset_manifest.json'), signature(data/'sample_audit.sqlite'),
                  signature(args.encoder_dir/'config.json'), signature(args.decoder_dir/'config.json')]
    encoder, decoder = args.encoder_dir, args.decoder_dir
    with output_directory(args.out_dir) as stage:
        # All runtime options are resolved once; editing YAML affects only future jobs.
        write_json(stage/'resolved_extraction_config.json', snapshot(args))
        if args.architecture == 'bert':
            import torch
            bert = module('_test1_bert', PROJECT/'workflow/step4_extract_window_hidden_states_BERT-js.py')
            opts = argparse.Namespace(data_dir=data, encoder_dir=encoder, bert_dir=decoder,
                                      out_dir=stage, device=args.device, encoder_layer=args.encoder_layer,
                                      hidden_layer=args.hidden_layer, token_offset=args.token_offset, window_size=args.window_size,
                                      batch_size=args.batch_size, preprocess=args.preprocess, dtype=args.dtype,
                                      save_every=args.save_every, limit=args.limit, log_every=args.log_every,
                                      codec_batch_size=args.codec_batch_size, codec_buffer_size=args.codec_buffer_size,
                                      pooling_device=args.pooling_device)
            bert.validate_args(opts)
            torch.set_num_threads(args.cpu_threads)
            torch.set_float32_matmul_precision('high')  # Same as the original BERT entry point.
            device = bert.resolve_device(args.device)
            codec, backbone = bert.load_models(encoder, decoder, device)
            index, layer = bert.resolve_hidden_layer_name(args.hidden_layer, int(backbone.config.num_hidden_layers))
            split = bert.build_split('test', data, stage, codec, backbone,
                                     bert.load_preprocessor(args.preprocess, encoder), opts, device, layer)
            summary = dict(model_family='PoreBERT', encoder_dir=str(encoder), decoder_dir=str(decoder),
                           encoder_layer=args.encoder_layer, hidden_layer=args.hidden_layer, resolved_hidden_state_index=index,
                           selected_layer_name=layer, preprocess=args.preprocess, token_offset=args.token_offset,
                           special_tokens=dict(pad=bert.PAD_TOKEN_ID, bos=bert.BOS_TOKEN_ID,
                                               eos=bert.EOS_TOKEN_ID, cls=bert.CLS_TOKEN_ID),
                           window_size=args.window_size, hidden_size=int(backbone.config.hidden_size), dtype=args.dtype,
                           splits=[split])
        else:
            subprocess.run(extractor_command(args, data, stage), check=True)
            summary = read_json(stage/'feature_summary.json')
        # Preserve per-window QC fields that the original extractors omit from their TSVs.
        audit = readonly_db(data/'sample_audit.sqlite')
        try:
            metadata = stage/'metadata_test.tsv'
            with metadata.open(newline='') as src, (stage/'metadata_test.enriched.tsv').open('w', newline='') as dst:
                reader = csv.DictReader(src, delimiter='\t')
                fields = list(reader.fieldnames)
                for key in ['library_id', 'mapq', 'input_signal_len']:
                    if key not in fields:
                        fields.append(key)
                writer = csv.DictWriter(dst, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                count = 0
                for row in reader:
                    qc = audit.execute('SELECT mapq, signal_len FROM audit WHERE chrom=? AND position=? AND read_id=?',
                                       (row['chrom'], int(row['center_genome_pos0']), row['read_id'])).fetchone()
                    if qc is None or int(row['label']) != 0:
                        raise ValueError('Extracted sample not found in WGA audit, or nonnegative label')
                    row.update(library_id=manifest['library_id'], mapq=qc[0], input_signal_len=qc[1])
                    writer.writerow(row)
                    count += 1
            (stage/'metadata_test.enriched.tsv').replace(metadata)
        finally:
            audit.close()
        if not count:
            raise ValueError('No extracted features')
        changed = [s['path'] for s in signatures if signature(Path(s['path'])) != s]
        if changed:
            raise ValueError(f'Dataset or model config changed during feature extraction: {changed}')
        # Relocated manifests use relative bin paths; remove stale staging metadata paths.
        import torch
        pt = torch.load(stage/'test.pt', map_location='cpu', weights_only=False)
        pt['metadata_path'] = 'metadata_test.tsv'
        for row in summary['splits']:
            for key in ['pt_path', 'feature_path', 'label_path', 'metadata_path']:
                if key in row:
                    row[key] = str(args.out_dir.resolve()/Path(row[key]).name)
        dataset = dict(manifest)
        if args.limit is not None:
            dataset.update(partial_test=True, extraction_limit=args.limit)
        pt.update(eval_version=args.version, model_architecture=args.architecture)
        torch.save(pt, stage/'test.pt')
        summary.update(eval_version=args.version, model_architecture=args.architecture, out_dir=str(args.out_dir.resolve()),
                       extraction_config=snapshot(args), test1_dataset=dataset, records=count, source_dataset=str(data),
                       model_config_signatures=[signature(encoder/'config.json'), signature(decoder/'config.json')])
        write_json(stage/'feature_summary.json', summary)
        write_json(stage/'resolved_extraction_config.json', snapshot(args))
    print(f'[DONE] {args.version} test features={count}', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    run(parser().parse_args())
