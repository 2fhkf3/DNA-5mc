#!/usr/bin/env python3
"""Extract V011 ELF DLM token features and pool them into base windows.

The embedding implementation from ModificationDetectionBenchmark's hidden-stage.py
is integrated directly here: tokenizer lookup, codec encoding, normalized context,
ODE refinement, last-hidden inference, masking and CUDA OOM batch splitting.
This file has no runtime import dependency on hidden-stage.py or BERT-js.py.
Outputs retain the Step-4 streaming_memmap_v1 format [N, window_size, 768].
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, Iterable, Sequence, TextIO

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Shared performance options live in the existing Test1 common module.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from eval.Test1.workflow.common import (
    add_extraction_performance_args, validate_extraction_performance,
    extraction_performance_metadata, iter_encoded_signals, pool_hidden_on_model,
)
DEFAULT_MODEL_ROOT = PROJECT_ROOT / 'train/backbone/V011-Diffusion'
DEFAULT_DATA_DIR = PROJECT_ROOT / 'output/step3_output/new'
DEFAULT_OUT_DIR = PROJECT_ROOT / 'output/step4_output/V011-ode097'
DEFAULT_DECODER_DIR = DEFAULT_MODEL_ROOT / 'hf_dlm_elf_mixed_with_cls_token_step65000-unsharded'


# Embedded inference contract from hidden-stage.py.

SPECIAL_TOKEN_COUNT = 3

@dataclass(frozen=True)
class TokenProtocol:
    # Indexed by raw codec ID; this performs exactly one codebook-to-vocab mapping.
    lookup: torch.Tensor
    cls: int
    bos: int
    eos: int
    pad: int
    max_positions: int
    hidden_size: int

    def metadata(self) -> dict[str, Any]:
        offsets = self.lookup - torch.arange(self.lookup.numel())
        return {
            'mapping': 'tokenizer.model.vocab[<|bwav:{raw_codec_id}|>]',
            'token_offset': int(offsets[0]) if torch.all(offsets == offsets[0]) else None,
            'codebook_size': self.lookup.numel(),
            'special_tokens': {'cls': self.cls, 'bos': self.bos, 'eos': self.eos, 'pad': self.pad},
            'max_decoder_positions': self.max_positions,
            'hidden_size': self.hidden_size,
        }

def resolve_device(value: str) -> torch.device:
    if value == 'auto':
        value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if value.isdigit():
        value = f'cuda:{value}'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA requested but unavailable: {device}')
    return device

def load_token_protocol(encoder_dir: Path, dlm_dir: Path) -> TokenProtocol:
    encoder = json.loads((encoder_dir / 'config.json').read_text())
    config = json.loads((dlm_dir / 'config.json').read_text())
    tokenizer = json.loads((dlm_dir / 'tokenizer.json').read_text())
    vocab = tokenizer['model']['vocab']
    context, dlm, model = (config[k] for k in ('context_encoder_config', 'dlm_config', 'model_config'))
    if config.get('model_type') != 'poredlm_dlm' or not dlm.get('add_cls_token'):
        raise ValueError('Expected a V011 poredlm_dlm checkpoint trained with CLS')
    size = int(encoder['codebook_size'])
    try:
        lookup = torch.tensor([vocab[f'<|bwav:{i}|>'] for i in range(size)], dtype=torch.long)
        special = {name: int(vocab[f'<|{name}|>']) for name in ('bos', 'eos', 'pad')}
    except KeyError as exc:
        raise ValueError(f'Checkpoint tokenizer is missing codec/special token: {exc}') from exc
    special['cls'] = int(dlm['cls_token_id'])  # Role defined by training config, not the token string.
    vocab_size = int(model['vocab_size'])
    if int(context['vocab_size']) != vocab_size:
        raise ValueError('DLM/context vocabulary sizes disagree')
    for name in ('bos', 'eos', 'pad'):
        for cfg in (config, context, model):
            configured = cfg.get(f'{name}_token_id')
            if configured is not None and int(configured) != special[name]:
                raise ValueError(f'Tokenizer/config disagree on {name} token ID')
    all_ids = [*lookup.tolist(), *special.values()]
    if min(all_ids) < 0 or max(all_ids) >= vocab_size or len(set(special.values())) != 4:
        raise ValueError('Token IDs outside vocabulary, or special tokens are not distinct')
    if lookup.unique().numel() != size or set(lookup.tolist()) & set(special.values()):
        raise ValueError('Codec mapping contains duplicate/special token IDs')
    maximum = min(int(context['max_position_embeddings']), int(dlm['max_length']),
                  int(model['max_sequence_length']))
    if maximum <= SPECIAL_TOKEN_COUNT:
        raise ValueError('Model context cannot hold CLS/BOS/content/EOS')
    return TokenProtocol(lookup, max_positions=maximum, hidden_size=int(context['hidden_size']), **special)

def load_models(encoder_dir: Path, dlm_dir: Path, device: torch.device):
    # Override the training machine's ELF path with the copy packaged in this checkpoint.
    elf_src = dlm_dir.resolve() / 'ELF-pytorch-port' / 'src'
    if not elf_src.is_dir():
        raise FileNotFoundError(f'Packaged ELF source not found: {elf_src}')
    config = AutoConfig.from_pretrained(str(dlm_dir), trust_remote_code=True, local_files_only=True)
    config.elf_src_path = str(elf_src)
    codec = AutoModel.from_pretrained(str(encoder_dir), trust_remote_code=True,
                                     local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    dlm = AutoModel.from_pretrained(str(dlm_dir), config=config, trust_remote_code=True,
                                   local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    return codec, dlm

def iter_jsonl(path: Path, limit: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.name.endswith('.gz') else open
    emitted = 0
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if limit is not None and emitted >= limit:
                break
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_no}: {exc}') from exc
            if not isinstance(source, dict):
                raise ValueError(f'Expected JSON object at {path}:{line_no}')
            yield line_no, source
            emitted += 1

def source_signal(source: dict[str, Any], path: Path, line_no: int) -> torch.Tensor | None:
    values = source.get('signal')
    if values is None or values == []:
        return None
    if not isinstance(values, list):
        raise ValueError(f'signal must be a list at {path}:{line_no}')
    signal = torch.as_tensor(values, dtype=torch.float32)
    if signal.ndim != 1 or not torch.isfinite(signal).all():
        raise ValueError(f'signal must be finite and 1-D at {path}:{line_no}')
    return signal

def encode_tokens_without_reconstruction(codec, signal, encoder_layer, device) -> torch.Tensor:
    x = signal.to(device=device, dtype=torch.float32).view(1, 1, -1)
    # Skip the unused CNN decoder; retain the codec's exact quantization pipeline.
    if all(hasattr(codec, name) for name in ('pore_head', 'cnn_model', 'project_in', 'vq', 'tokenize_indices')):
        latent = codec.cnn_model.encode(codec.pore_head(x)).permute(0, 2, 1)
        _, indices = codec.vq(codec.project_in(latent))
        ids = codec.tokenize_indices(indices.long(), layer=encoder_layer)
    else:
        ids = codec.encode_signal(x, layer=encoder_layer)
    return ids.squeeze(0).detach().to(device='cpu', dtype=torch.long)

def add_special_tokens(raw_ids: torch.Tensor, protocol: TokenProtocol) -> torch.Tensor:
    return torch.cat((torch.tensor([protocol.cls, protocol.bos]),
                      protocol.lookup[raw_ids], torch.tensor([protocol.eos])))

def infer_dlm_batch(items, dlm, args, protocol, device) -> list[np.ndarray]:
    sequences = [add_special_tokens(item.token_ids, protocol) for item in items]
    ids = pad_sequence(sequences, batch_first=True, padding_value=protocol.pad).to(device)
    lengths = torch.tensor([len(seq) for seq in sequences], device=device)
    mask = (torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]).long()
    if args.encoding_type in ('context_hidden', 'ode_hidden'):
        # HF forward also computes an unused last_hidden + vocabulary logits.
        # Calling these public methods directly avoids that extra ELF pass.
        context = dlm.context_encoder(input_ids=ids, attention_mask=mask,
                                      return_dict=True).last_hidden_state
        context = context.to(dtype=next(dlm.elf_denoiser.parameters()).dtype)
        selected = context
        if args.encoding_type == 'ode_hidden':
            selected = dlm.ode_from_context_hidden(context, attention_mask=mask,
                ode_steps=args.ode_steps, ode_start_t=args.ode_start_t,
                self_cond_cfg_scale=args.ode_self_cond_cfg_scale)
    else:
        time = torch.full((len(items),), args.last_t, dtype=torch.float32, device=device)
        selected = dlm(input_ids=ids, attention_mask=mask, t=time)['last_hidden_state']
    if selected.shape != (len(items), ids.shape[1], protocol.hidden_size):
        raise RuntimeError(f'Unexpected DLM feature shape: {tuple(selected.shape)}')
    if getattr(args, 'pooling_device', 'cpu') == 'model':
        pooled = pool_hidden_on_model(selected, [base_token_intervals(item) for item in items], offset=2)
        return [row.numpy() for row in pooled]
    dtype = torch.float16 if args.hidden_dtype == 'float16' else torch.float32
    arrays = []
    for row, item in enumerate(items):
        hidden = selected[row, 2:2 + item.token_len].detach().to(device='cpu', dtype=dtype).contiguous()
        if not torch.isfinite(hidden).all():
            raise RuntimeError(f'Non-finite {args.encoding_type} for read_id={item.source.get("read_id")}')
        arrays.append(hidden.numpy())
    return arrays

def infer_with_oom_fallback(items, dlm, args, protocol, device):
    try:
        return infer_dlm_batch(items, dlm, args, protocol, device)
    except torch.cuda.OutOfMemoryError:
        if device.type != 'cuda' or len(items) == 1:
            raise
    torch.cuda.empty_cache()
    midpoint = len(items) // 2
    print(f'[WARN] CUDA OOM: splitting batch {len(items)} into {midpoint}+{len(items)-midpoint}', flush=True)
    return (infer_with_oom_fallback(items[:midpoint], dlm, args, protocol, device)
            + infer_with_oom_fallback(items[midpoint:], dlm, args, protocol, device))

def atomic_dump(value, path: Path, *, as_json=False):
    temporary = path.with_suffix(path.suffix + '.tmp')
    if as_json:
        temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    else:
        with temporary.open('wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)

# Base-window input and streaming output contract from BERT-js.py.

METADATA_KEYS = [
    "read_id",
    "source",
    "label",
    "chrom",
    "center_genome_pos0",
    "genome_start0",
    "genome_end0",
    "strand",
    "seq",
    "ref",
    "split_key",
    "signal_offset",
    "signal_len",
]

def split_input_path(data_dir: Path, split: str) -> Path:
    for suffix in (".jsonl.gz", ".jsonl"):
        path = data_dir / f"{split}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {data_dir / f'{split}.jsonl.gz'} or "
        f"{data_dir / f'{split}.jsonl'}"
    )

def load_preprocessor(mode: str, encoder_dir: Path) -> Any | None:
    if mode == "none":
        return None
    return AutoFeatureExtractor.from_pretrained(
        str(encoder_dir), trust_remote_code=True, local_files_only=True
    )

def prepare_signal(
    sample: dict[str, Any],
    preprocessor: Any | None,
    input_path: Path,
    line_no: int,
) -> torch.Tensor | None:
    values = sample.get("signal")
    if values is None or values == []:
        return None
    if not isinstance(values, list):
        raise ValueError(
            f"signal must be a list at {input_path}:{line_no}; "
            f"read_id={sample.get('read_id')}"
        )
    signal = np.asarray(values, dtype=np.float32)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError(
            f"signal must be finite and 1-D at {input_path}:{line_no}; "
            f"read_id={sample.get('read_id')}"
        )
    original_len = int(signal.size)
    if preprocessor is None:
        result = torch.from_numpy(np.ascontiguousarray(signal))
    else:
        processed = preprocessor(signal, return_tensors="pt")
        if "signal" not in processed:
            raise KeyError("V548 AutoFeatureExtractor output has no 'signal' field")
        result = processed["signal"].detach().to(dtype=torch.float32).reshape(-1).cpu()
        if result.numel() != original_len:
            raise ValueError(
                "Preprocessing changed signal length, so base spans would no longer align: "
                f"before={original_len}, after={result.numel()}, "
                f"read_id={sample.get('read_id')}"
            )
    return result.contiguous()

def normalize_spans(
    sample: dict[str, Any],
    signal_len: int,
    window_size: int,
    input_path: Path,
    line_no: int,
) -> list[list[int]] | None:
    raw_spans = sample.get("base_sample_span_ref") or sample.get("base_sample_span_seq")
    if not isinstance(raw_spans, list):
        return None
    if len(raw_spans) != window_size:
        raise ValueError(
            f"Expected {window_size} base spans at {input_path}:{line_no}, got "
            f"{len(raw_spans)}; read_id={sample.get('read_id')}"
        )
    spans: list[list[int]] = []
    for index, span in enumerate(raw_spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(
                f"Invalid base span #{index} at {input_path}:{line_no}: {span!r}"
            )
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start or end > signal_len:
            raise ValueError(
                f"Base span #{index} [{start}, {end}) is outside signal length "
                f"{signal_len} at {input_path}:{line_no}"
            )
        spans.append([start, end])
    return spans

def span_to_token_interval(
    span: Sequence[int], signal_len: int, token_len: int
) -> tuple[int, int]:
    start, end = int(span[0]), int(span[1])
    token_start = int(math.floor(start * token_len / signal_len))
    token_end = int(math.ceil(end * token_len / signal_len))
    token_start = max(0, min(token_start, token_len - 1))
    token_end = max(token_start + 1, min(token_end, token_len))
    return token_start, token_end

class StreamingSplitWriter:
    """Append one split to bounded RAM buffers and publish it atomically."""

    FORMAT = "streaming_memmap_v1"

    def __init__(
        self,
        out_dir: Path,
        split: str,
        window_size: int,
        hidden_size: int,
        dtype: str,
        save_every: int,
        feature_source: str,
    ) -> None:
        self.out_dir = out_dir
        self.split = split
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.feature_dtype = np.dtype(dtype)
        self.save_every = save_every
        self.feature_source = feature_source
        dtype_tag = "f16" if self.feature_dtype == np.dtype("float16") else "f32"

        self.feature_path = out_dir / f"{split}.features.{dtype_tag}.bin"
        self.label_path = out_dir / f"{split}.labels.f32.bin"
        self.metadata_path = out_dir / f"metadata_{split}.tsv"
        self.manifest_path = out_dir / f"{split}.pt"
        self.feature_partial = self.feature_path.with_name(self.feature_path.name + ".partial")
        self.label_partial = self.label_path.with_name(self.label_path.name + ".partial")
        self.metadata_partial = self.metadata_path.with_name(
            self.metadata_path.name + ".partial"
        )
        self.manifest_partial = self.manifest_path.with_name(
            self.manifest_path.name + ".partial"
        )

        for path in (
            self.feature_partial,
            self.label_partial,
            self.metadata_partial,
            self.manifest_partial,
        ):
            path.unlink(missing_ok=True)

        self._feature_handle: BinaryIO = self.feature_partial.open("wb")
        self._label_handle: BinaryIO = self.label_partial.open("wb")
        self._metadata_handle: TextIO = self.metadata_partial.open(
            "w", newline="", encoding="utf-8"
        )
        self._metadata_writer = csv.DictWriter(
            self._metadata_handle,
            fieldnames=METADATA_KEYS + ["split"],
            delimiter="\t",
        )
        self._metadata_writer.writeheader()
        self._features: list[np.ndarray] = []
        self._labels: list[float] = []
        self._metadata: list[dict[str, Any]] = []
        self.records = 0
        self.closed = False

    @property
    def buffered(self) -> int:
        return len(self._features)

    def append(
        self, feature: np.ndarray, label: int, metadata: dict[str, Any]
    ) -> None:
        expected_shape = (self.window_size, self.hidden_size)
        if feature.shape != expected_shape:
            raise ValueError(
                f"Feature shape changed in split={self.split}: "
                f"expected {expected_shape}, got {feature.shape}"
            )
        # Convert before buffering: the old implementation retained float32
        # arrays and doubled peak memory again during the final np.stack.
        self._features.append(
            np.asarray(feature, dtype=self.feature_dtype, order="C")
        )
        self._labels.append(float(label))
        self._metadata.append(metadata)
        self.records += 1
        if self.buffered >= self.save_every:
            self.flush()

    def flush(self) -> None:
        if not self._features:
            return
        feature_block = np.stack(self._features, axis=0)
        label_block = np.asarray(self._labels, dtype=np.float32)
        feature_block.tofile(self._feature_handle)
        label_block.tofile(self._label_handle)
        self._metadata_writer.writerows(self._metadata)
        self._feature_handle.flush()
        self._label_handle.flush()
        self._metadata_handle.flush()
        self._features.clear()
        self._labels.clear()
        self._metadata.clear()

    def _close_handles(self) -> None:
        if self.closed:
            return
        self._feature_handle.close()
        self._label_handle.close()
        self._metadata_handle.close()
        self.closed = True

    def abort(self) -> None:
        self._close_handles()

    def finalize(self) -> dict[str, Any]:
        self.flush()
        for handle in (self._feature_handle, self._label_handle):
            os.fsync(handle.fileno())
        os.fsync(self._metadata_handle.fileno())
        self._close_handles()

        expected_feature_bytes = (
            self.records
            * self.window_size
            * self.hidden_size
            * self.feature_dtype.itemsize
        )
        expected_label_bytes = self.records * np.dtype("float32").itemsize
        if self.feature_partial.stat().st_size != expected_feature_bytes:
            raise RuntimeError(
                f"Incomplete feature file for split={self.split}: expected "
                f"{expected_feature_bytes} bytes, got {self.feature_partial.stat().st_size}"
            )
        if self.label_partial.stat().st_size != expected_label_bytes:
            raise RuntimeError(
                f"Incomplete label file for split={self.split}: expected "
                f"{expected_label_bytes} bytes, got {self.label_partial.stat().st_size}"
            )

        os.replace(self.feature_partial, self.feature_path)
        os.replace(self.label_partial, self.label_path)
        os.replace(self.metadata_partial, self.metadata_path)
        manifest = {
            "storage_format": self.FORMAT,
            "feature_path": self.feature_path.name,
            "feature_dtype": self.feature_dtype.name,
            "label_path": self.label_path.name,
            "label_dtype": "float32",
            "metadata_path": self.metadata_path.name,
            "split": self.split,
            "records": self.records,
            "feature_shape": [self.records, self.window_size, self.hidden_size],
            "feature_source": self.feature_source,
        }
        torch.save(manifest, self.manifest_partial)
        os.replace(self.manifest_partial, self.manifest_path)
        return manifest


@dataclass
class EncodedSample:
    source: dict[str, Any]
    signal_len: int
    spans: list[list[int]]
    token_ids: torch.Tensor  # Raw codec IDs on CPU, before tokenizer mapping.
    original_token_len: int
    was_truncated: bool

    @property
    def token_len(self) -> int:
        return self.token_ids.numel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--encoder-dir', type=Path, default=DEFAULT_MODEL_ROOT / 'encoder')
    parser.add_argument('--decoder-dir', '--dlm-dir', dest='decoder_dir', type=Path, default=DEFAULT_DECODER_DIR)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--encoder-layer', type=int, default=0)
    parser.add_argument('--hidden-source', '--encoding-type', dest='hidden_source',
                        choices=['last_hidden_state', 'last_hidden', 'context_hidden', 'ode_hidden'],
                        default='ode_hidden')
    parser.add_argument('--ode-steps', type=int, default=1)
    parser.add_argument('--ode-start-t', type=float, default=0.97)
    parser.add_argument('--ode-self-cond-cfg-scale', type=float, default=0.5)
    parser.add_argument('--last-t', type=float, default=1.)
    parser.add_argument('--window-size', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--preprocess', choices=['none', 'auto'], default='none',
                        help='none uses the supplied normalized signals, matching hidden-stage.')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float32',
                        help='Storage dtype only; codec, DLM and pooling use float32.')
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--splits', nargs='+', choices=['train', 'valid', 'test'],
                        default=['train', 'valid', 'test'])
    parser.add_argument('--force', action='store_true', help='Replace selected splits in --out-dir.')
    add_extraction_performance_args(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_extraction_performance(args)
    for path, label in ((args.encoder_dir, 'V011 encoder'), (args.decoder_dir, 'V011 ELF DLM')):
        for filename in ('config.json', 'model.safetensors'):
            if not (path / filename).is_file():
                raise FileNotFoundError(f'{label}: missing {path / filename}')
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f'Data directory not found: {args.data_dir}')
    for name in ('window_size', 'batch_size', 'save_every', 'ode_steps', 'cpu_threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be >= 1')
    if args.window_size % 2 != 1:
        raise ValueError('--window-size must be odd for a centered base window')
    if args.encoder_layer != 0:
        raise ValueError('V011 has one FSQ codebook; --encoder-layer must be 0')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be >= 1')
    if args.log_every < 0:
        raise ValueError('--log-every must be >= 0')
    if not 0 < args.ode_start_t <= 1 or not 0 < args.last_t <= 1:
        raise ValueError('--ode-start-t and --last-t must be in (0, 1]')
    if not math.isfinite(args.ode_self_cond_cfg_scale) or args.ode_self_cond_cfg_scale < 0:
        raise ValueError('--ode-self-cond-cfg-scale must be finite and >= 0')
    if len(set(args.splits)) != len(args.splits):
        raise ValueError('--splits must not contain duplicates')
    for split in args.splits:
        split_input_path(args.data_dir, split)


def inference_args(args: argparse.Namespace) -> SimpleNamespace:
    # Window features are pooled in float32 even when the storage dtype is float16.
    return SimpleNamespace(
        encoding_type='last_hidden' if args.hidden_source == 'last_hidden_state' else args.hidden_source,
        ode_steps=args.ode_steps, ode_start_t=args.ode_start_t,
        ode_self_cond_cfg_scale=args.ode_self_cond_cfg_scale,
        last_t=args.last_t, hidden_dtype='float32', pooling_device=getattr(args, 'pooling_device', 'cpu'),
    )


def encode_sample(sample, signal, spans, encoder, encoder_layer, protocol, device, raw_ids=None) -> EncodedSample:
    ids = raw_ids if raw_ids is not None else encode_tokens_without_reconstruction(encoder, signal, encoder_layer, device)
    if ids.ndim != 1 or not ids.numel() or ids.min() < 0 or ids.max() >= protocol.lookup.numel():
        raise ValueError(f'Invalid V011 raw codec tokens for read_id={sample.get("read_id")}')
    original_len = ids.numel()
    maximum = protocol.max_positions - SPECIAL_TOKEN_COUNT
    return EncodedSample(sample, signal.numel(), spans, ids[:maximum].contiguous(),
                         original_len, original_len > maximum)


def infer_batch(items: Sequence[EncodedSample], decoder, args, protocol, device) -> list[torch.Tensor]:
    # Shared function maps raw IDs, adds CLS/BOS/EOS once, masks padding and
    # removes two prefix tokens. context_hidden includes the final LayerNorm.
    arrays = infer_with_oom_fallback(items, decoder, inference_args(args), protocol, device)
    return [torch.from_numpy(array) for array in arrays]


def base_token_intervals(item: EncodedSample) -> list[tuple[int, int]] | None:
    intervals = []
    for span in item.spans:
        start, end = span_to_token_interval(span, item.signal_len, item.original_token_len)
        # Truncation preserves a prefix. Never stretch that prefix across the
        # original full signal or pool a base from tokens that were discarded.
        if end > item.token_len:
            return None
        intervals.append((start, end))
    return intervals


def hidden_to_base_window(hidden: torch.Tensor, item: EncodedSample) -> np.ndarray:
    intervals = base_token_intervals(item)
    if intervals is None:
        raise ValueError('Base spans extend beyond retained codec tokens')
    return np.stack([hidden[start:end].mean(dim=0).numpy() for start, end in intervals])


def sample_metadata(item: EncodedSample, split: str) -> dict[str, Any]:
    row = {key: item.source.get(key, '') for key in METADATA_KEYS}
    row.update(signal_len=item.signal_len, split=split)
    return row


def feature_source(args) -> str:
    return (f'signal_window_v011_{args.hidden_source}_ode{args.ode_start_t:g}_'
            f'steps{args.ode_steps}_window{args.window_size}_refspan_cls')


def extraction_metadata(args, protocol) -> dict[str, Any]:
    return {
        **extraction_performance_metadata(args),
        'model_family': 'V011_PoreDLM', 'encoder_dir': str(args.encoder_dir.resolve()),
        'decoder_dir': str(args.decoder_dir.resolve()), 'encoder_layer': args.encoder_layer,
        'embedding_reference': 'ModificationDetectionBenchmark/workflow/hidden-stage/hidden-stage.py',
        'embedding_implementation': str(Path(__file__).resolve()),
        'embedding_integration': 'inlined; no runtime dependency on the reference scripts',
        'hidden_source': args.hidden_source, 'ode_steps': args.ode_steps,
        'ode_start_t': args.ode_start_t, 'ode_self_cond_cfg_scale': args.ode_self_cond_cfg_scale,
        'last_t': args.last_t, 'preprocess': args.preprocess,
        'input_layout': '[CLS, BOS, tokenizer-mapped codec tokens, EOS]',
        'context_hidden_selection': 'last context Transformer layer after final LayerNorm',
        'special_token_removal': 'hidden[:, 2:2+token_len]',
        'inference_dtype': 'float32', 'matmul_precision': torch.get_float32_matmul_precision(),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'pooling': 'mean over base spans; coordinates mapped using original codec token count',
        **protocol.metadata(),
    }


def prepare_output(args):
    owned = []
    for split in args.splits:
        names = [f'{split}.pt', f'{split}.features.f16.bin', f'{split}.features.f32.bin',
                 f'{split}.labels.f32.bin', f'metadata_{split}.tsv']
        owned.extend(args.out_dir / name for name in names)
        owned.extend(args.out_dir / (name + '.partial') for name in names)
    owned += [args.out_dir / 'feature_summary.json', args.out_dir / 'feature_summary.json.tmp']
    existing = [p for p in owned if p.exists()]
    if existing and not args.force:
        raise FileExistsError(f'Existing Step-4 outputs in {args.out_dir}; choose another OUT_DIR '
                              'or explicitly use --force')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in existing:
            path.unlink()


def build_split(split, encoder, decoder, preprocessor, args, device, protocol) -> dict[str, Any]:
    input_path = split_input_path(args.data_dir, split)
    input_stat = input_path.stat()
    writer = StreamingSplitWriter(
        out_dir=args.out_dir, split=split, window_size=args.window_size,
        hidden_size=protocol.hidden_size, dtype=args.dtype, save_every=args.save_every,
        feature_source=feature_source(args),
    )
    pending = []
    seen = skipped = truncated = truncated_span_skips = 0
    labels = {0: 0, 1: 0}

    def flush_batch():
        if not pending:
            return
        arrays = infer_batch(pending, decoder, args, protocol, device)
        for item, hidden in zip(pending, arrays):
            label = int(item.source['label'])
            feature = hidden.numpy() if getattr(args, 'pooling_device', 'cpu') == 'model' else hidden_to_base_window(hidden, item)
            writer.append(feature, label, sample_metadata(item, split))
            labels[label] += 1
        pending.clear()

    def prepared_records():
        nonlocal seen, skipped
        for line_no, sample in iter_jsonl(input_path, args.limit):
            seen += 1
            if not isinstance(sample, dict):
                raise ValueError(f'Expected JSON object at {input_path}:{line_no}')
            if sample.get('label') not in (0, 1, '0', '1'):
                raise ValueError(f'Binary label required at {input_path}:{line_no}')
            signal = prepare_signal(sample, preprocessor, input_path, line_no)
            if signal is None:
                skipped += 1
                continue
            spans = normalize_spans(sample, signal.numel(), args.window_size, input_path, line_no)
            if spans is None:
                skipped += 1
                continue
            yield line_no, sample, signal, spans

    print(f'[INFO] split={split} input={input_path}', flush=True)
    try:
        with torch.inference_mode():
            for (line_no, sample, signal, spans), raw_ids in iter_encoded_signals(
                    prepared_records(), encoder, args, device,
                    lambda signal: encode_tokens_without_reconstruction(encoder, signal, args.encoder_layer, device)):
                item = encode_sample(sample, signal, spans, encoder, args.encoder_layer, protocol, device, raw_ids)
                truncated += int(item.was_truncated)
                if base_token_intervals(item) is None:
                    skipped += 1
                    truncated_span_skips += 1
                    continue
                pending.append(item)
                if len(pending) >= args.batch_size:
                    flush_batch()
                if args.log_every and line_no % args.log_every == 0:
                    print(f'[INFO] split={split} seen={seen} kept={writer.records} '
                          f'pending={len(pending)} skipped={skipped} truncated={truncated}', flush=True)
            flush_batch()
        if not writer.records:
            raise RuntimeError(f'No features built for split={split}')
        current_stat = input_path.stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (input_stat.st_size, input_stat.st_mtime_ns):
            raise RuntimeError(f'Input changed during extraction: {input_path}')
        manifest = writer.finalize()
        manifest.update(extraction_metadata(args, protocol))
        torch.save(manifest, writer.manifest_partial)
        writer.manifest_partial.replace(writer.manifest_path)
    except BaseException:
        writer.abort()
        raise
    return {
        'split': split, 'records': writer.records, 'seen': seen, 'skipped': skipped,
        'truncated_to_decoder_max_positions': truncated,
        'skipped_spans_outside_retained_tokens': truncated_span_skips,
        'label_counts': labels, 'feature_shape': manifest['feature_shape'],
        'storage_format': manifest['storage_format'], 'pt_path': str(writer.manifest_path),
        'feature_path': str(writer.feature_path), 'label_path': str(writer.label_path),
        'metadata_path': str(writer.metadata_path), 'input_path': str(input_path.resolve()),
        'input_size': input_stat.st_size, 'input_mtime_ns': input_stat.st_mtime_ns,
    }


def main():
    args = parse_args()
    validate_args(args)
    protocol = load_token_protocol(args.encoder_dir, args.decoder_dir)
    device = resolve_device(args.device)
    # Match hidden-stage's default FP32 policy, avoiding the old implicit 'high'.
    torch.set_float32_matmul_precision('highest')
    if device.type == 'cpu':
        torch.set_num_threads(args.cpu_threads)
    prepare_output(args)
    print(f'[INFO] device={device}; encoder={args.encoder_dir}; decoder={args.decoder_dir}', flush=True)
    print(f'[INFO] source={args.hidden_source}; ODE steps={args.ode_steps}, '
          f'start_t={args.ode_start_t}, CFG={args.ode_self_cond_cfg_scale}; protocol={protocol.metadata()}', flush=True)
    preprocessor = load_preprocessor(args.preprocess, args.encoder_dir)
    encoder, decoder = load_models(args.encoder_dir, args.decoder_dir, device)
    splits = [build_split(split, encoder, decoder, preprocessor, args, device, protocol) for split in args.splits]
    summary = {
        **extraction_metadata(args, protocol), 'data_dir': str(args.data_dir.resolve()),
        'out_dir': str(args.out_dir.resolve()), 'feature_source': feature_source(args),
        'window_size': args.window_size, 'batch_size': args.batch_size,
        'save_every': args.save_every, 'dtype': args.dtype, 'limit': args.limit,
        'device': str(device), 'cpu_threads': args.cpu_threads,
        'max_position_embeddings': protocol.max_positions, 'splits': splits,
    }
    atomic_dump(summary, args.out_dir / 'feature_summary.json', as_json=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
