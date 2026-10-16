#!/usr/bin/env python3
"""Extract one WGA test split with the original version-specific backbone code."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from common import PROJECT, VERSIONS, module, output_directory, read_json, readonly_db, signature, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', choices=VERSIONS, required=True, help='必须与后续分类头相同的主干版本')
    p.add_argument('--data-dir', type=Path, required=True, help='Step1 生成的 test.jsonl.gz 和数据清单目录')
    p.add_argument('--out-dir', type=Path, required=True, help='新建特征目录，禁止覆盖')
    p.add_argument('--model-root', type=Path, help='对应预训练模型根目录；默认 train/backbone 的版本链接')
    p.add_argument('--device', default='auto', help='auto / cpu / cuda:0 等；仅用于特征提取')
    p.add_argument('--batch-size', type=int, default=4, help='主干提取批次大小；CPU 测试可设为 1')
    p.add_argument('--cpu-threads', type=int, default=4, help='CPU 算子线程数')
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16', help='特征落盘精度，不改变原主干算法')
    return p


def run(args):
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError('Batch size and CPU threads must be positive')
    data = args.data_dir.resolve()
    manifest = read_json(data/'dataset_manifest.json')
    if manifest['label_counts']['1'] or manifest['strand'] != '+' or manifest['window_size'] != 15:
        raise ValueError('Expected all-negative, positive-strand, 15-nt Test1 data')
    source = data/'test.jsonl.gz'
    signatures = [signature(source), signature(data/'dataset_manifest.json'), signature(data/'sample_audit.sqlite')]
    roots = {'V011': 'V011-Diffusion', 'V543': 'V543-BERT', 'V610': 'V610-GPT'}
    root = (args.model_root or PROJECT/'train/backbone'/roots[args.version]).resolve()
    encoder = root/'encoder'
    decoder = root/('hf_dlm_elf_mixed_with_cls_token_step65000-unsharded' if args.version == 'V011' else 'base')
    for path in (encoder/'config.json', decoder/'config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    with output_directory(args.out_dir) as stage:
        if args.version == 'V543':
            import torch
            bert = module('_test1_bert', PROJECT/'workflow/step4_extract_window_hidden_states_BERT-js.py')
            opts = argparse.Namespace(data_dir=data, encoder_dir=encoder, bert_dir=decoder,
                                      out_dir=stage, device=args.device, encoder_layer=0,
                                      hidden_layer=-1, token_offset=128, window_size=15,
                                      batch_size=args.batch_size, preprocess='none', dtype=args.dtype,
                                      save_every=512, limit=None, log_every=1000)
            bert.validate_args(opts)
            torch.set_num_threads(args.cpu_threads)
            torch.set_float32_matmul_precision('high')  # Same as the original BERT entry point.
            device = bert.resolve_device(args.device)
            codec, backbone = bert.load_models(encoder, decoder, device)
            index, layer = bert.resolve_hidden_layer_name(-1, int(backbone.config.num_hidden_layers))
            split = bert.build_split('test', data, stage, codec, backbone,
                                     bert.load_preprocessor('none', encoder), opts, device, layer)
            summary = dict(model_family='V543_PoreBERT', encoder_dir=str(encoder), decoder_dir=str(decoder),
                           encoder_layer=0, hidden_layer=-1, resolved_hidden_state_index=index,
                           selected_layer_name=layer, preprocess='none', token_offset=128,
                           special_tokens=dict(pad=bert.PAD_TOKEN_ID, bos=bert.BOS_TOKEN_ID,
                                               eos=bert.EOS_TOKEN_ID, cls=bert.CLS_TOKEN_ID),
                           window_size=15, hidden_size=int(backbone.config.hidden_size), dtype=args.dtype,
                           splits=[split])
        else:
            script = ('step4_extract_window_hidden_states_Diffusion-kx.py' if args.version == 'V011'
                      else 'step4_extract_window_hidden_states_GPT.py')
            command = [sys.executable, '-u', str(PROJECT/'workflow'/script), '--data-dir', str(data),
                       '--encoder-dir', str(encoder), '--decoder-dir', str(decoder), '--out-dir', str(stage),
                       '--splits', 'test', '--device', args.device, '--cpu-threads', str(args.cpu_threads),
                       '--window-size', '15', '--batch-size', str(args.batch_size), '--preprocess', 'none',
                       '--dtype', args.dtype]
            if args.version == 'V011':
                command += ['--encoder-layer', '0', '--hidden-source', 'ode_hidden', '--ode-steps', '1',
                            '--ode-start-t', '0.97', '--ode-self-cond-cfg-scale', '0.5', '--last-t', '1.0']
            else:
                command += ['--encoder-layer', '1', '--decoder-layers', 'last', '--token-id-mode', 'auto',
                            '--special-tokens', 'none', '--overlong-policy', 'skip']
            subprocess.run(command, check=True)
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
        if [signature(Path(s['path'])) for s in signatures] != signatures:
            raise ValueError('Dataset changed during feature extraction')
        # Relocated manifests use relative bin paths; remove stale staging metadata paths.
        import torch
        pt = torch.load(stage/'test.pt', map_location='cpu', weights_only=False)
        pt['metadata_path'] = 'metadata_test.tsv'
        torch.save(pt, stage/'test.pt')
        for row in summary['splits']:
            for key in ['pt_path', 'feature_path', 'label_path', 'metadata_path']:
                if key in row:
                    row[key] = str(args.out_dir.resolve()/Path(row[key]).name)
        summary.update(eval_version=args.version, out_dir=str(args.out_dir.resolve()),
                       test1_dataset=manifest, records=count, source_dataset=str(data),
                       model_config_signatures=[signature(encoder/'config.json'), signature(decoder/'config.json')])
        write_json(stage/'feature_summary.json', summary)
    print(f'[DONE] {args.version} test features={count}', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    run(parser().parse_args())
