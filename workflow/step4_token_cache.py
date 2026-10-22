"""Bounded exact-length codec batches and atomic, protocol-bound raw-token shards.

Used only by the unified Step4 entry point. Decoder IDs/special tokens, spans,
labels, feature precision and backbone batching remain owned by each extractor.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

SCHEMA = 'step4-raw-codec-tokens-v1'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def stat_signature(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def encoder_fingerprint(folder):
    """Content-based: identical encoder trees can share tokens across backbones."""
    folder = Path(folder).resolve()
    files = sorted(p for p in folder.rglob('*') if p.is_file() and
                   p.suffix in ('.json', '.py', '.safetensors', '.bin', '.pt'))
    if not any(p.suffix in ('.safetensors', '.bin', '.pt') for p in files):
        raise FileNotFoundError(f'No encoder weights found below {folder}')
    before = [stat_signature(p) for p in files]
    contents = {str(p.relative_to(folder)): file_hash(p) for p in files}
    if before != [stat_signature(p) for p in files]:
        raise RuntimeError('Encoder changed while computing token-cache fingerprint')
    return fingerprint(contents), before


def record_digest(records):
    """Bind raw IDs to prepared FP32 signal, spans, identity and source order."""
    h = hashlib.sha256()
    for line, row, signal, spans in records:
        identity = {k: row.get(k) for k in ('window_id', 'read_id', 'fast5_raw_read_id',
                    'chrom', 'center_genome_pos0', 'strand', 'label', 'seq', 'ref')}
        h.update(fingerprint([line, identity, spans, int(signal.numel())]).encode())
        values = signal.detach().cpu().contiguous().numpy().astype('<f4', copy=False)
        h.update(values.tobytes())
    return h.hexdigest()


class CodecTokenCache:
    def __init__(self, args, encoder, device, backend):
        import torch
        self.args = args; self.encoder = encoder; self.device = device; self.torch = torch
        self.mode = args.token_cache_mode
        self.backend = backend
        self.encoder_signature = []
        self.contract = None
        if self.mode != 'off':
            enc, self.encoder_signature = encoder_fingerprint(args.encoder_dir)
            capability = list(torch.cuda.get_device_capability(device)) if device.type == 'cuda' else None
            # Batch-dependent codec rounding is a numerical protocol, not just I/O.
            self.contract = dict(schema=SCHEMA, encoder_sha256=enc,
                implementation_sha256=file_hash(__file__),
                prepare_signal_sha256=hashlib.sha256(inspect.getsource(backend.prepare_signal).encode()).hexdigest(),
                encoder_layer=args.encoder_layer, preprocess=args.preprocess,
                window_size=args.window_size, codec_mode=getattr(args, 'codec_mode', 'auto'),
                codec_batch_size=args.codec_batch_size, codec_buffer_size=args.codec_buffer_size,
                codec_max_samples=args.codec_max_samples,
                torch=str(torch.__version__), cuda=torch.version.cuda, device_type=device.type,
                compute_capability=capability, cudnn=torch.backends.cudnn.version(),
                device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                cpu_threads=torch.get_num_threads(),
                matmul_precision=torch.get_float32_matmul_precision(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
            if args.codec_batch_size == 1:
                self.contract['serial_encoder_sha256'] = hashlib.sha256(inspect.getsource(
                    backend.encode_tokens_without_reconstruction).encode()).hexdigest()

    def for_split(self, split, input_path):
        return TokenStream(self, split, input_path)

    def check_encoder(self):
        for old in self.encoder_signature:
            if stat_signature(old['path']) != old:
                raise RuntimeError(f'Encoder changed during token caching: {old["path"]}')


class TokenStream:
    def __init__(self, owner, split, input_path):
        self.owner = owner; self.args = owner.args; self.split = split
        self.source = stat_signature(input_path)
        self.contract = dict(codec=owner.contract, input=self.source, limit=self.args.limit)
        self.key = fingerprint(self.contract)
        self.directory = Path(self.args.token_cache_dir)/self.key/split
        self.stats = dict(mode=owner.mode, cache_key=self.key if owner.mode != 'off' else None,
            directory=str(self.directory) if owner.mode != 'off' else None,
            rows_encoded=0, rows_from_cache=0, cache_hits=0, cache_misses=0,
            codec_forward_batches=0, max_codec_batch=0, oom_splits=0, chunks=[])

    def check_source(self):
        if stat_signature(self.source['path']) != self.source:
            raise RuntimeError('Input changed during token caching')
        self.owner.check_encoder()

    def encode_group(self, signals, encode_one):
        torch = self.owner.torch; codec = self.owner.encoder
        if self.args.codec_batch_size == 1:
            self.stats['codec_forward_batches'] += 1
            self.stats['max_codec_batch'] = max(1, self.stats['max_codec_batch'])
            return [encode_one(signals[0])]
        failure = False
        try:
            x = torch.stack(signals).to(device=self.owner.device, dtype=torch.float32).unsqueeze(1)
            can_skip = all(hasattr(codec, k) for k in
                           ('pore_head', 'cnn_model', 'project_in', 'vq', 'tokenize_indices'))
            if getattr(self.args, 'codec_mode', 'auto') != 'reference' and can_skip:
                latent = codec.cnn_model.encode(codec.pore_head(x)).permute(0, 2, 1)
                _, indices = codec.vq(codec.project_in(latent))
                ids = codec.tokenize_indices(indices.long(), layer=self.args.encoder_layer)
            else:
                ids = codec.encode_signal(x, layer=self.args.encoder_layer)
            if ids.ndim != 2 or ids.shape[0] != len(signals) or ids.shape[1] == 0:
                raise ValueError('Codec must return nonempty [batch,tokens] IDs')
            if ids.is_floating_point() and (not torch.isfinite(ids).all() or not torch.equal(ids, ids.round())):
                raise ValueError('Nonintegral/nonfinite codec IDs')
            self.stats['codec_forward_batches'] += 1
            self.stats['max_codec_batch'] = max(self.stats['max_codec_batch'], len(signals))
            return list(ids.detach().to(device='cpu', dtype=torch.long).unbind(0))
        except torch.cuda.OutOfMemoryError:
            if self.owner.device.type != 'cuda' or len(signals) == 1:
                raise
            failure = True
        if failure:
            # Clear failed tensors outside the exception handler before retrying.
            x = latent = indices = ids = _ = None
            torch.cuda.empty_cache(); self.stats['oom_splits'] += 1
            middle = len(signals)//2
            print(f'[TOKEN-CACHE] codec OOM: split {len(signals)} -> {middle}+{len(signals)-middle}', flush=True)
            return self.encode_group(signals[:middle], encode_one) + self.encode_group(signals[middle:], encode_one)

    def encode(self, records, encode_one):
        groups = defaultdict(list)
        for i, row in enumerate(records):
            groups[int(row[2].numel())].append(i)
        result = [None]*len(records)
        for length, group in groups.items():
            cap = min(self.args.codec_batch_size, max(1, self.args.codec_max_samples//length))
            for start in range(0, len(group), cap):
                indices = group[start:start+cap]
                values = self.encode_group([records[i][2] for i in indices], encode_one)
                if len(values) != len(indices):
                    raise ValueError('Codec returned a different row count')
                for i, value in zip(indices, values):
                    a = value.detach().cpu().numpy()
                    if a.ndim != 1 or not len(a) or a.dtype.kind not in 'iu' or np.any(a < 0) or np.any(a > 2**31-1):
                        raise ValueError('Raw codec IDs must be nonnegative int32-compatible vectors')
                    result[i] = np.asarray(a, dtype=np.int32)
        self.stats['rows_encoded'] += len(records)
        return result

    def read(self, folder, records, identity):
        m = json.loads((folder/'manifest.json').read_text())
        if (m['schema'] != SCHEMA or m['cache_key'] != self.key or
                m['rows'] != len(records) or m['records_sha256'] != identity or
                m['contract'] != self.contract):
            raise ValueError(f'Token cache identity/protocol mismatch: {folder}')
        path = folder/'tokens.npz'
        if file_hash(path) != m['tokens_sha256']:
            raise ValueError(f'Token cache hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            tokens, offsets, lines = data['tokens'], data['offsets'], data['line_numbers']
        if (tokens.ndim != 1 or tokens.dtype.kind not in 'iu' or np.any(tokens < 0) or
                np.any(tokens > 2**31-1) or offsets.dtype != np.int64 or
                offsets.shape != (len(records)+1,) or offsets[0] != 0 or
                offsets[-1] != len(tokens) or np.any(np.diff(offsets) <= 0) or
                lines.dtype != np.int64 or not np.array_equal(lines, [r[0] for r in records])):
            raise ValueError(f'Malformed token-cache arrays: {folder}')
        self.stats['cache_hits'] += 1; self.stats['rows_from_cache'] += len(records)
        return [tokens[a:b].astype(np.int64) for a, b in zip(offsets[:-1], offsets[1:])]

    def write(self, folder, records, identity, values):
        stage = Path(tempfile.mkdtemp(prefix='.'+folder.name+'.partial-', dir=folder.parent))
        try:
            offsets = np.r_[0, np.cumsum([len(v) for v in values], dtype=np.int64)]
            tokens = np.concatenate(values)
            if tokens.max() <= 65535:
                tokens = tokens.astype(np.uint16)
            np.savez(stage/'tokens.npz', tokens=tokens, offsets=offsets,
                     line_numbers=np.array([r[0] for r in records], dtype=np.int64))
            m = dict(schema=SCHEMA, cache_key=self.key, rows=len(records),
                records_sha256=identity, contract=self.contract,
                tokens_sha256=file_hash(stage/'tokens.npz'), n_tokens=len(tokens),
                encoding_stats={k:self.stats[k] for k in ('codec_forward_batches','max_codec_batch','oom_splits')})
            (stage/'manifest.json').write_text(json.dumps(m, indent=2, allow_nan=False)+'\n')
            self.check_source()
            os.replace(stage, folder)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def chunk(self, index, records, encode_one):
        self.check_source()
        if self.owner.mode == 'off':
            return self.encode(records, encode_one)
        folder = self.directory/f'part_{index:06d}'; identity = record_digest(records)
        if folder.exists():
            values = self.read(folder, records, identity)
        elif self.owner.mode == 'readonly':
            raise FileNotFoundError(f'Token cache miss in readonly mode: {folder}')
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory/f'.part_{index:06d}.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if folder.exists():
                    values = self.read(folder, records, identity)
                else:
                    for orphan in self.directory.glob('.'+folder.name+'.partial-*'):
                        shutil.rmtree(orphan)
                    self.stats['cache_misses'] += 1
                    started = time.monotonic(); values = self.encode(records, encode_one)
                    self.write(folder, records, identity, values)
                    print(f'[TOKEN-CACHE] {self.split}/{folder.name}: wrote {len(records)} rows in '
                          f'{time.monotonic()-started:.2f}s', flush=True)
        self.stats['chunks'].append(dict(path=str(folder), manifest_sha256=file_hash(folder/'manifest.json')))
        return values

    def __call__(self, records, codec, args, device, encode_one):
        if codec is not self.owner.encoder or device != self.owner.device:
            raise ValueError('Token stream bound to a different encoder/device')
        if codec.training:
            raise ValueError('Codec token extraction requires eval() mode')
        buffer = []; index = 0
        for record in records:
            buffer.append(record)
            if len(buffer) >= args.codec_buffer_size:
                values = self.chunk(index, buffer, encode_one)
                for row, ids in zip(buffer, values):
                    yield row, self.owner.torch.as_tensor(ids, dtype=self.owner.torch.long)
                buffer.clear(); index += 1
        if buffer:
            values = self.chunk(index, buffer, encode_one)
            for row, ids in zip(buffer, values):
                yield row, self.owner.torch.as_tensor(ids, dtype=self.owner.torch.long)
        self.check_source()
        print(f'[TOKEN-CACHE] {self.split}: encoded={self.stats["rows_encoded"]}, '
              f'cached={self.stats["rows_from_cache"]}, batches={self.stats["codec_forward_batches"]}', flush=True)


@contextmanager
def bind_token_stream(implementation, stream):
    """Scope the adapter to one unified split; standalone/Test1 behavior is intact."""
    backend = getattr(implementation, 'shared', implementation)
    original = backend.iter_encoded_signals
    backend.iter_encoded_signals = stream
    try:
        yield
    finally:
        backend.iter_encoded_signals = original
