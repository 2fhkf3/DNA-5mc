"""Unlabeled adapters around the existing BERT/ELF/Flow/GPT feature implementations."""
from __future__ import annotations
from types import SimpleNamespace
from common import *

IMPLEMENTATIONS = dict(bert='BERT-js', diffusion='Diffusion-kx', flow_matching='Flow-matching', gpt='GPT')


class FeatureExtractor:
    def __init__(self, profile, c):
        import torch
        self.torch = torch; self.profile = profile; self.family = profile['architecture']
        self.impl = module('_test4_extractor_' + self.family,
             PROJECT/'workflow'/f'step4_extract_window_hidden_states_{IMPLEMENTATIONS[self.family]}.py')
        self.shared = self.impl.shared if self.family == 'flow_matching' else self.impl
        self.device = self.impl.resolve_device(c['runtime']['device'])
        torch.set_num_threads(c['runtime']['cpu_threads'])
        torch.set_float32_matmul_precision('high' if self.family == 'bert' else 'highest')
        self.args = SimpleNamespace(**profile)
        self.args.encoder_dir = Path(profile['encoder_dir']); self.args.decoder_dir = Path(profile['decoder_dir'])
        self.args.window_size = 15
        if self.family == 'gpt':
            self.args.decoder_layers = self.impl.parse_layers(profile['decoder_layers'])
            self.protocol = self.impl.load_token_protocol(self.args)
            self.encoder = self.impl.load_encoder(str(self.args.encoder_dir)).to(self.device).eval()
            self.decoder = self.impl.load_decoder(str(self.args.decoder_dir)).to(self.device).eval()
        else:
            self.encoder, self.decoder = self.impl.load_models(self.args.encoder_dir, self.args.decoder_dir, self.device)
            self.protocol = None if self.family == 'bert' else self.impl.load_token_protocol(self.args.encoder_dir, self.args.decoder_dir)
        self.encoder.requires_grad_(False); self.decoder.requires_grad_(False)

    def features(self, records):
        torch = self.torch; impl = self.impl; base = self.shared
        torch.set_float32_matmul_precision('high' if self.family == 'bert' else 'highest')
        n = len(records); features = np.full((n, 5, 768), np.nan, dtype=np.float16)
        valid = np.zeros(n, dtype=bool); reasons = ['decoder_context_limit']*n
        pending = []; positions = []
        def flush():
            if not pending:
                return
            if self.family == 'bert':
                output = impl.infer_with_oom_fallback(pending, self.decoder, self.args.hidden_layer,
                                                     self.device, self.args.pooling_device)
            elif self.family == 'gpt':
                output = impl.infer_with_oom_fallback(pending, self.decoder, self.protocol,
                                                     self.device, self.args.pooling_device)
            else:
                output = impl.infer_batch(pending, self.decoder, self.args, self.protocol, self.device)
            for i, item, hidden in zip(positions, pending, output):
                pooled = hidden.numpy() if self.args.pooling_device == 'model' else base.hidden_to_base_window(hidden, item)
                if pooled.shape != (15, 768) or not np.isfinite(pooled).all():
                    raise ValueError('Invalid extracted 15x768 feature')
                value = pooled[5:10].astype(np.float16)
                if not np.isfinite(value).all():
                    raise ValueError('FP16 feature storage overflow')
                features[i] = value; valid[i] = True; reasons[i] = ''
            pending.clear(); positions.clear()
        with torch.inference_mode():
            for i, row in enumerate(records):
                signal = base.prepare_signal(row, None, Path('<Test4-window-shard>'), i+1)
                if signal is None:
                    raise ValueError('Audited window is missing signal')
                spans = base.normalize_spans(row, signal.numel(), 15, Path('<Test4-window-shard>'), i+1)
                if spans is None:
                    raise ValueError('Audited window is missing signal/spans')
                if self.family == 'bert':
                    item = impl.encode_sample(row, signal, spans, self.encoder, self.args.encoder_layer,
                               self.args.token_offset, int(self.decoder.config.max_position_embeddings),
                               int(self.decoder.config.vocab_size), self.device)
                elif self.family == 'gpt':
                    item = impl.encode_sample(row, signal, spans, self.encoder, self.args, self.protocol, self.device)
                else:
                    item = base.encode_sample(row, signal, spans, self.encoder, self.args.encoder_layer, self.protocol, self.device)
                # Never silently truncate and stretch token coordinates over the original window.
                if item is None or getattr(item, 'was_truncated', False):
                    continue
                pending.append(item); positions.append(i)
                if len(pending) >= self.args.batch_size:
                    flush()
            flush()
        return features, valid, reasons


def load_heads(runs, device):
    import torch
    registry = module('_test4_head_models', PROJECT/'train/model/__init__.py')
    loaded = []
    for item in runs:
        cfg = item['configuration']; path = Path(item['run_dir'])/'best.pt'
        if sha(path) != item['checkpoint_sha256']:
            raise ValueError('Frozen classifier checkpoint changed')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('strategy', 'seed', 'fingerprint', 'model', 'training', 'feature_dir',
                    'source_window_size', 'hidden_dim', 'base_source', 'fusion_window_size'):
            if checkpoint['config'].get(key) != cfg.get(key):
                raise ValueError(f'Classifier config/checkpoint mismatch: {key}')
        if checkpoint['epoch'] != item['checkpoint_epoch']:
            raise ValueError('Classifier best epoch differs')
        model = registry.build_model(cfg['strategy'], cfg['hidden_dim'], cfg['source_window_size'], cfg['model'])
        model.load_state_dict(checkpoint['model'], strict=True)
        loaded.append((item, model.requires_grad_(False).to(device).eval()))
    return loaded


def predict_head(model, features, records, cfg, device, batch_size):
    import torch
    torch.set_float32_matmul_precision('highest')
    fusion = cfg['strategy'] == 'center_5mer_emb8_none'
    if fusion and cfg.get('fusion_window_size', 5) != 5:
        raise ValueError('Only existing 5mer fusion heads are supported')
    codes = dict(zip('ACGTNWSMKRYBVDHZ', range(16)))
    def infer(lo, hi):
        try:
            x = torch.from_numpy(np.array(features[lo:hi], dtype=np.float32, copy=True)).to(device)
            if fusion:
                source = cfg.get('base_source', 'ref')
                bases = torch.tensor([[codes[b] for b in r[source][5:10].upper()] for r in records[lo:hi]],
                                     dtype=torch.long, device=device)
                p = torch.softmax(model(x, bases), -1)[:, 1]
            else:
                p = torch.sigmoid(model(x))
            return p.detach().float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if device.type != 'cuda' or hi-lo <= 1:
                raise
        torch.cuda.empty_cache(); mid = (lo+hi)//2
        return np.concatenate([infer(lo, mid), infer(mid, hi)])
    with torch.inference_mode():
        p = np.concatenate([infer(lo, min(lo+batch_size, len(features))) for lo in range(0, len(features), batch_size)])
    if p.shape != (len(features),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid classifier probabilities')
    return p
