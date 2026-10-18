#!/usr/bin/env python3
"""Extract V013 Flow Matching features; preserve the Step4 memmap contract.

The exported V013 loader cannot construct its embedded `porebert` configuration.
This project-local adapter reconstructs that exact BERT structure, strictly loads
ALL checkpoint tensors, and uses the checkpoint's original Flow/ODE methods.
It neither changes the checkpoint nor substitutes weights from another model.
Window alignment, truncation and streaming I/O are shared with Diffusion-kx.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
sys.dont_write_bytecode = True  # Checkpoint is often an external, read-only symlink target.
os.environ.setdefault('HF_MODULES_CACHE', str(PROJECT / 'output/.hf_modules'))
sys.path.insert(0, str(PROJECT))

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file
from transformers import AutoModel
from eval.Test1.workflow.common import (
    add_extraction_performance_args, module, signature,
)

shared = module('_flow_step4_window_io', SCRIPT_DIR / 'step4_extract_window_hidden_states_Diffusion-kx.py')
DEFAULT_ROOT = PROJECT / 'train/backbone/V013-Flow-matching'
resolve_device = shared.resolve_device
load_preprocessor = shared.load_preprocessor


@dataclass(frozen=True)
class FlowTokenProtocol(shared.TokenProtocol):
    context_pad: int

    def metadata(self):
        return {**super().metadata(), 'context_pad_token_id': self.context_pad,
                'padding_mask': 'explicit sequence-length mask; tokenizer pad ID used for batching'}


def load_token_protocol(encoder_dir, dlm_dir):
    enc = json.loads((encoder_dir / 'config.json').read_text())
    cfg = json.loads((dlm_dir / 'config.json').read_text())
    vocab = json.loads((dlm_dir / 'tokenizer.json').read_text())['model']['vocab']
    context, dlm, model = (cfg[k] for k in ('context_encoder_config', 'dlm_config', 'model_config'))
    if not str(dlm.get('model', '')).startswith('flow_matching') or not dlm.get('add_cls_token'):
        raise ValueError('Expected a Flow Matching checkpoint trained with CLS')
    size = int(enc['codebook_size'])
    lookup = torch.tensor([vocab[f'<|bwav:{i}|>'] for i in range(size)], dtype=torch.long)
    special = {k: int(vocab[f'<|{k}|>']) for k in ('bos', 'eos', 'pad')}
    special['cls'] = int(dlm['cls_token_id'])
    n_vocab = int(model['vocab_size'])
    if int(context['vocab_size']) != n_vocab:
        raise ValueError('Flow/context vocabulary sizes disagree')
    # The context embedding's padding_idx is a training setting, NOT the
    # tokenization/padding rule. Preserve it; always supply an explicit mask.
    context_pad = int(context.get('pad_token_id', 0))
    if not 0 <= context_pad < n_vocab:
        raise ValueError('context pad_token_id is outside the vocabulary')
    for name in ('bos', 'eos', 'pad'):
        for recorded in (cfg, model):
            if recorded.get(name + '_token_id') not in (None, special[name]):
                raise ValueError(f'Tokenizer/Flow config disagree on {name} token ID')
    all_ids = [*lookup.tolist(), *special.values()]
    if min(all_ids) < 0 or max(all_ids) >= n_vocab or len(set(special.values())) != 4:
        raise ValueError('Invalid Flow special/codec token IDs')
    if lookup.unique().numel() != size or set(lookup.tolist()) & set(special.values()):
        raise ValueError('Duplicate codec mappings or overlap with special tokens')
    maximum = min(int(context['max_position_embeddings']), int(dlm['max_length']),
                  int(model['max_sequence_length']))
    if maximum <= shared.SPECIAL_TOKEN_COUNT:
        raise ValueError('Flow sequence capacity cannot hold CLS/BOS/content/EOS')
    return FlowTokenProtocol(lookup, max_positions=maximum, hidden_size=int(context['hidden_size']),
                             context_pad=context_pad, **special)


class PoreBertLayers(nn.Module):
    """Exact V554 SimpleMLM tensor layout, including embedding/final LayerNorm."""

    def __init__(self, cfg):
        super().__init__()
        width = int(cfg['hidden_size'])
        self.token_embedding = nn.Embedding(int(cfg['vocab_size']), width,
                                            padding_idx=int(cfg.get('pad_token_id', 0)))
        self.position_embedding = nn.Embedding(int(cfg['max_position_embeddings']), width)
        self.embedding_norm = nn.LayerNorm(width)
        self.embedding_dropout = nn.Dropout(float(cfg.get('dropout', .1)))
        layer = nn.TransformerEncoderLayer(width, int(cfg['num_attention_heads']),
                   dim_feedforward=width * 4, dropout=float(cfg.get('dropout', .1)),
                   activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, int(cfg['num_hidden_layers']))
        self.norm = nn.LayerNorm(width)

    def forward(self, input_ids, attention_mask):
        if attention_mask is None or attention_mask.shape != input_ids.shape:
            raise ValueError('Flow extraction requires an explicit 2D attention_mask')
        if input_ids.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError('Input exceeds the context position embedding capacity')
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(self.embedding_norm(hidden))
        return self.norm(self.encoder(hidden, src_key_padding_mask=attention_mask.eq(0)))


class PoreBertModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bert = PoreBertLayers(cfg)
        self.config = SimpleNamespace(**cfg)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        hidden = self.bert(input_ids, attention_mask)
        return SimpleNamespace(last_hidden_state=hidden) if return_dict else (hidden,)


def load_models(encoder_dir, dlm_dir, device):
    dlm_dir = dlm_dir.resolve()
    raw = json.loads((dlm_dir / 'config.json').read_text())
    if raw.get('context_encoder_type') != 'hf_auto' or raw['context_encoder_config'].get('model_type') != 'porebert':
        raise ValueError('This adapter expects the V013 hf_auto/porebert checkpoint layout')
    if not (dlm_dir / 'flow_matching_core/src').is_dir():
        raise FileNotFoundError(f'Missing packaged Flow runtime: {dlm_dir}/flow_matching_core/src')
    export = module('_v013_checkpoint_flow_export', dlm_dir / 'modeling_poredlm.py')
    config = export.PoreLMConfig(**raw)
    config.name_or_path = str(dlm_dir)

    class CompatibleFlow(export.PoreLMForDiffusion):
        def __init__(self, cfg):
            # Only replace the incompatible context-construction branch. Reuse
            # the exported forward, ODE, velocity conversion and self-condition.
            export.PreTrainedModel.__init__(self, cfg)
            self._ensure_flow_matching_runtime(cfg)
            from porelm.flow_matching_core.model import FLOW_MATCHING_DENOISERS
            self.context_encoder = export.HFContextEncoderAdapter(PoreBertModel(cfg.context_encoder_config))
            self.context_hidden_size = int(cfg.context_encoder_config['hidden_size'])
            dlm, net = cfg.dlm_config, cfg.model_config
            self.flow_denoiser = FLOW_MATCHING_DENOISERS[dlm['model']](
                text_encoder_dim=self.context_hidden_size, max_length=int(dlm['max_length']),
                attn_drop=float(dlm.get('attn_dropout', 0.)), proj_drop=float(dlm.get('proj_dropout', 0.)),
                num_time_tokens=int(dlm.get('num_time_tokens', 4)),
                num_self_cond_cfg_tokens=int(dlm.get('num_self_cond_cfg_tokens', 4)),
                vocab_size=int(net['vocab_size']), num_model_mode_tokens=int(dlm.get('num_model_mode_tokens', 0)),
                bottleneck_dim=int(dlm.get('bottleneck_dim', 128)))

    decoder = CompatibleFlow(config)
    # Strict loading prevents silently using random/missing context or Flow
    # weights. Keep the independently saved tensors and original padding_idx.
    decoder.load_state_dict(load_file(str(dlm_dir / 'model.safetensors'), device='cpu'),
                            strict=True, assign=True)
    decoder.requires_grad_(False).to(device).eval()
    codec = AutoModel.from_pretrained(str(encoder_dir), trust_remote_code=True,
                local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    return codec, decoder


def infer_batch(items, decoder, args, protocol, device):
    try:
        sequences = [shared.add_special_tokens(item.token_ids, protocol) for item in items]
        ids = pad_sequence(sequences, batch_first=True, padding_value=protocol.pad).to(device)
        lengths = torch.tensor([seq.numel() for seq in sequences], device=device)
        mask = (torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]).long()
        if args.hidden_source in ('context_hidden', 'ode_hidden'):
            hidden = decoder.context_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            hidden = hidden.to(dtype=next(decoder.flow_denoiser.parameters()).dtype)
            if args.hidden_source == 'ode_hidden':
                hidden = decoder.ode_from_context_hidden(hidden, attention_mask=mask,
                    ode_steps=args.ode_steps, ode_start_t=args.ode_start_t,
                    self_cond_cfg_scale=args.ode_self_cond_cfg_scale)
        else:
            time = torch.full((len(items),), args.last_t, device=device, dtype=torch.float32)
            hidden = decoder(input_ids=ids, attention_mask=mask, t=time,
                             return_last_hidden_state=True)['last_hidden_state']
        if hidden.shape != (*ids.shape, protocol.hidden_size) or not torch.isfinite(hidden).all():
            raise RuntimeError('Invalid/non-finite Flow hidden states')
        if getattr(args, 'pooling_device', 'cpu') == 'model':
            return shared.pool_hidden_on_model(hidden, [shared.base_token_intervals(x) for x in items], offset=2)
        return [hidden[i, 2:2 + item.token_len].detach().float().cpu().contiguous()
                for i, item in enumerate(items)]
    except torch.cuda.OutOfMemoryError:
        if device.type != 'cuda' or len(items) < 2:
            raise
    torch.cuda.empty_cache()
    middle = len(items) // 2
    print(f'[WARN] Flow CUDA OOM: splitting batch {len(items)} into {middle}+{len(items)-middle}', flush=True)
    return (infer_batch(items[:middle], decoder, args, protocol, device)
            + infer_batch(items[middle:], decoder, args, protocol, device))


def feature_source(args):
    return (f'signal_window_flow_matching_{args.hidden_source}_ode{args.ode_start_t:g}_'
            f'steps{args.ode_steps}_window{args.window_size}_refspan_cls')


def extraction_metadata(args, protocol):
    import hashlib
    decoder_dir = args.decoder_dir.resolve()
    files = [Path(__file__), decoder_dir / 'modeling_poredlm.py',
             decoder_dir / 'flow_matching_core/src/porelm/flow_matching_core/model.py',
             decoder_dir / 'flow_matching_core/src/porelm/flow_matching_core/layers.py']
    return {**shared.extraction_metadata(args, protocol),
            'model_family': 'PoreLM_FlowMatching', 'model_architecture': 'flow_matching',
            'embedding_reference': str(decoder_dir / 'readme.md'),
            'embedding_implementation': str(Path(__file__).resolve()),
            'embedding_integration': 'project-local PoreBERT adapter; original checkpoint Flow/ODE runtime',
            'checkpoint_loading': 'strict=True; all context and flow tensors loaded from this checkpoint',
            'checkpoint_weight_signature': signature(decoder_dir / 'model.safetensors'),
            'implementation_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def validate_args(args):
    shared.validate_args(args, model_name='V013 Flow Matching')
    load_token_protocol(args.encoder_dir, args.decoder_dir)


def build_split(split, encoder, decoder, preprocessor, args, device, protocol):
    return shared.build_split(split, encoder, decoder, preprocessor, args, device, protocol,
                              backend=sys.modules[__name__])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=PROJECT / 'output/step3_output/new')
    p.add_argument('--encoder-dir', type=Path, default=DEFAULT_ROOT / 'encoder')
    p.add_argument('--decoder-dir', '--dlm-dir', dest='decoder_dir', type=Path, default=DEFAULT_ROOT / 'hf_dlm_step10000')
    p.add_argument('--out-dir', type=Path, default=PROJECT / 'output/step4_output/V013')
    p.add_argument('--device', default='auto')
    for key, default in dict(cpu_threads=4, encoder_layer=0, ode_steps=2, window_size=15,
                             batch_size=4, save_every=512, log_every=200).items():
        p.add_argument('--' + key.replace('_', '-'), type=int, default=default)
    p.add_argument('--hidden-source', '--encoding-type', dest='hidden_source', default='ode_hidden',
                   choices=['ode_hidden', 'context_hidden', 'last_hidden', 'last_hidden_state'])
    for key, default in dict(ode_start_t=.98, ode_self_cond_cfg_scale=.5, last_t=1.).items():
        p.add_argument('--' + key.replace('_', '-'), type=float, default=default)
    p.add_argument('--preprocess', choices=['none', 'auto'], default='none')
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--limit', type=int)
    p.add_argument('--splits', nargs='+', choices=['train', 'valid', 'test'], default=['train', 'valid', 'test'])
    p.add_argument('--force', action='store_true', help='Explicitly replace selected Step4 split outputs')
    add_extraction_performance_args(p)
    return p


def main():
    args = parser().parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision('highest')
    protocol = load_token_protocol(args.encoder_dir, args.decoder_dir)
    preprocessor = load_preprocessor(args.preprocess, args.encoder_dir)
    encoder, decoder = load_models(args.encoder_dir, args.decoder_dir, device)
    shared.prepare_output(args)
    splits = [build_split(s, encoder, decoder, preprocessor, args, device, protocol) for s in args.splits]
    summary = {**extraction_metadata(args, protocol), 'feature_source': feature_source(args),
               'data_dir': str(args.data_dir.resolve()), 'out_dir': str(args.out_dir.resolve()),
               'window_size': args.window_size, 'dtype': args.dtype, 'batch_size': args.batch_size,
               'save_every': args.save_every, 'limit': args.limit, 'partial_extraction': args.limit is not None,
               'device': str(device), 'cpu_threads': args.cpu_threads,
               'max_position_embeddings': protocol.max_positions, 'splits': splits}
    shared.atomic_dump(summary, args.out_dir / 'feature_summary.json', as_json=True)
    print(f'[DONE] Flow Matching: {[(r["split"], r["records"]) for r in splits]}; output={args.out_dir}')


if __name__ == '__main__':
    main()
