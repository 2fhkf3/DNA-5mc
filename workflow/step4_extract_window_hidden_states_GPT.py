#!/usr/bin/env python3
"""Extract GPT/OLMo base-window features using the Diffusion-kx I/O contract.

Read selected {train,valid,test}.jsonl[.gz] windows. Encode each unpadded signal,
map codec IDs through the checkpoint tokenizer, batch the causal decoder and
pool token features over base_sample_span_ref (or base_sample_span_seq).
Write streaming_memmap_v1 manifests, feature/label binaries and metadata TSVs.
JSONL validation and streaming writers are inlined; this file is standalone.
The default GPT token layout has no added CLS/BOS/EOS, matching the old GPT
extractor; optional BOS/EOS must be explicitly requested for another protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Sequence, TextIO

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast

# JSONL validation and streaming output are included below for standalone use.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_ROOT = PROJECT_ROOT / 'train/backbone/V610-GPT'
DEFAULT_DATA_DIR = PROJECT_ROOT / 'output/step3_output/new'
DEFAULT_OUT_DIR = PROJECT_ROOT / 'output/step4_output/V610'
DEFAULT_ENCODER_DIR = DEFAULT_MODEL_ROOT / 'encoder'
DEFAULT_DECODER_DIR = DEFAULT_MODEL_ROOT / 'base'

# Inlined Step-4 input/output contract; no dependency on step4_window_io.py.
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
            raise KeyError("GPT encoder AutoFeatureExtractor output has no 'signal' field")
        result = processed["signal"].detach().to(dtype=torch.float32).reshape(-1).cpu()
        if result.numel() != original_len:
            raise ValueError(
                "Preprocessing changed signal length, so base spans would no longer align: "
                f"before={original_len}, after={result.numel()}, "
                f"read_id={sample.get('read_id')}"
            )
    if not torch.isfinite(result).all():
        raise ValueError(f"Preprocessed signal contains non-finite values at {input_path}:{line_no}")
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
        if label not in (0, 1):
            raise ValueError("Labels must be binary")
        if not np.isfinite(feature).all():
            raise ValueError("Non-finite pooled features")
        if not np.isfinite(np.asarray(feature, dtype=self.feature_dtype)).all():
            raise ValueError("Pooled features overflow the requested storage dtype")
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

    def finalize(self, extra_manifest=None) -> dict[str, Any]:
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
        manifest.update(extra_manifest or {})
        torch.save(manifest, self.manifest_partial)
        os.replace(self.manifest_partial, self.manifest_path)
        return manifest


# Original checkpoint loaders and explicit codec-to-vocab conversion are kept.
def parse_layers(value: str) -> List[int]:
    """Parse decoder layer specification.

    Supported examples:
      last      -> final hidden state
      11        -> transformer layer 11
      embed,11  -> embedding output and transformer layer 11
      all       -> all hidden states
    """
    value = value.strip().lower()
    if value == "all":
        return [-10_000]
    parsed: List[int] = []
    for item in value.split(","):
        item = item.strip().lower()
        if item == "embed":
            parsed.append(-1)
        elif item == "last":
            parsed.append(-2)
        else:
            parsed.append(int(item))
    return parsed

def layer_name(hidden_index: int) -> str:
    return "embed" if hidden_index == 0 else f"layer_{hidden_index - 1}"

def load_encoder(encoder_dir: str) -> torch.nn.Module:
    """Load the custom pore encoder without scanning unrelated AutoModel classes."""
    encoder_path = Path(encoder_dir)
    modeling_file = encoder_path / "modeling_pore_codec.py"
    if not modeling_file.exists():
        AutoConfig.from_pretrained(encoder_dir, trust_remote_code=True, local_files_only=True)
        return AutoModel.from_pretrained(encoder_dir, trust_remote_code=True, local_files_only=True)

    module_name = "local_pore_codec_gpt_" + hashlib.sha256(str(encoder_path.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, modeling_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import encoder module from {modeling_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    config = module.PoreRSQCodecConfig.from_pretrained(str(encoder_path))
    config.output_hidden_states = False
    return module.PoreRSQCodec.from_pretrained(str(encoder_path), config=config, local_files_only=True, torch_dtype=torch.float32)

def load_decoder(decoder_dir: str) -> torch.nn.Module:
    """Load the decoder used by the existing pore foundation model."""
    config = AutoConfig.from_pretrained(decoder_dir, trust_remote_code=True, local_files_only=True)
    if getattr(config, "model_type", None) == "olmo2":
        from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM

        return Olmo2ForCausalLM.from_pretrained(decoder_dir, config=config, local_files_only=True, torch_dtype=torch.float32)
    return AutoModel.from_pretrained(decoder_dir, trust_remote_code=True, local_files_only=True)

def load_bwav_tokenizer(decoder_dir: str, tokenizer_file: str | None) -> PreTrainedTokenizerFast:
    """Load the bwav WordLevel tokenizer used to map codec ids to model ids."""
    path = Path(tokenizer_file) if tokenizer_file else Path(decoder_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file does not exist: {path}")
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )

def resolve_token_id_mode(decoder_dir: str, tokenizer_file: str | None, token_id_mode: str) -> str:
    """Resolve auto token-id handling from local decoder assets."""
    if token_id_mode != "auto":
        return token_id_mode
    path = Path(tokenizer_file) if tokenizer_file else Path(decoder_dir) / "tokenizer.json"
    if path.exists():
        return "tokenizer"
    raise FileNotFoundError(
        "token_id_mode=auto requires a tokenizer file for codec-id to decoder-id mapping. "
        f"Expected tokenizer at {path}. Set --tokenizer-file explicitly, or use "
        "--token-id-mode direct only after manually confirming this decoder expects raw codec ids."
    )

def codebook_ids_to_decoder_input_ids(
    codebook_ids: torch.Tensor,
    *,
    mode: str,
    tokenizer: PreTrainedTokenizerFast | None,
    token_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert encoder codebook ids into decoder input ids.

    ``direct`` preserves the historical behavior.  ``tokenizer`` is the safest
    mode for V600/V601-style OLMo bases because it asks tokenizer.json for the
    exact ``<|bwav:x|>`` ids.  ``offset`` is kept as a fast equivalence check
    when the tokenizer is known to map ``<|bwav:x|>`` to ``x + token_offset``.
    """
    if mode == "direct":
        return codebook_ids.to(device).long()
    if mode == "offset":
        return codebook_ids.to(device).long() + int(token_offset)
    if mode != "tokenizer":
        raise ValueError(f"Unsupported token_id_mode: {mode}")
    if tokenizer is None:
        raise ValueError("tokenizer mode requires a loaded tokenizer")

    original_shape = tuple(codebook_ids.shape)
    flat_ids = codebook_ids.detach().cpu().long().reshape(-1).tolist()
    tokens = [f"<|bwav:{int(item)}|>" for item in flat_ids]
    mapped = tokenizer.convert_tokens_to_ids(tokens)
    unk_id = tokenizer.unk_token_id
    bad = [(tok, idx) for tok, idx in zip(tokens, mapped) if idx == unk_id]
    if bad:
        examples = ", ".join(tok for tok, _ in bad[:5])
        raise ValueError(
            f"Tokenizer mapped {len(bad)} bwav tokens to unk_id={unk_id}. "
            f"Examples: {examples}. Check codec/tokenizer compatibility."
        )
    return torch.tensor(mapped, dtype=torch.long, device=device).reshape(original_shape)


def resolve_device(value):
    if value == 'auto':
        value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if value.isdigit():
        value = f'cuda:{value}'
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use cpu, auto or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if (device.index or 0) >= torch.cuda.device_count():
            raise ValueError(f'GPU index outside available devices: {device}')
    return device


def iter_jsonl(path, limit=None):
    opener = gzip.open if path.name.endswith('.gz') else open
    count = 0
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if limit is not None and count >= limit:
                break
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_no}: {exc}') from exc
            if not isinstance(sample, dict):
                raise ValueError(f'Expected JSON object at {path}:{line_no}')
            yield line_no, sample
            count += 1


def selected_layer(config, layer_spec, requested_name):
    depth = int(getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 0)))
    if depth < 1:
        raise ValueError('Decoder config must expose num_hidden_layers or n_layer')
    if layer_spec == [-10_000]:
        indices = list(range(depth + 1))
    else:
        indices = []
        for layer in layer_spec:
            if layer == -1:
                indices.append(0)
            elif layer == -2:
                indices.append(depth)
            elif 0 <= layer < depth:
                indices.append(layer + 1)
            else:
                raise ValueError(f'Decoder layer {layer} outside 0..{depth-1}')
    if requested_name is None:
        if len(indices) != 1:
            raise ValueError('Multiple decoder layers require --selected-layer-name; output stores one layer')
        index = indices[0]
    else:
        matches = [i for i in indices if layer_name(i) == requested_name]
        if not matches:
            raise ValueError(f'Selected layer {requested_name!r} not in {[layer_name(i) for i in indices]}')
        index = matches[0]
    return index, layer_name(index), depth


@dataclass
class TokenProtocol:
    lookup: torch.Tensor
    pad: int
    bos: int | None
    eos: int | None
    maximum: int
    hidden_size: int
    vocab_size: int
    layer_index: int
    layer_name: str
    layer_depth: int
    mode: str
    special_tokens: str

    def metadata(self):
        delta = self.lookup - torch.arange(self.lookup.numel())
        return dict(resolved_token_id_mode=self.mode,
                    token_offset=int(delta[0]) if torch.all(delta == delta[0]) else None,
                    codebook_size=self.lookup.numel(),
                    special_tokens=dict(pad=self.pad, bos=self.bos, eos=self.eos),
                    input_layout='mapped codec tokens' if self.special_tokens == 'none' else '[BOS, mapped codec tokens, EOS]',
                    max_decoder_positions=self.maximum, hidden_size=self.hidden_size,
                    selected_layer_name=self.layer_name, selected_hidden_index=self.layer_index,
                    last_layer_includes_final_norm=self.layer_index == self.layer_depth)


def load_token_protocol(args):
    config = AutoConfig.from_pretrained(str(args.decoder_dir), trust_remote_code=True, local_files_only=True)
    codec_config = json.loads((args.encoder_dir / 'config.json').read_text())
    base_size = int(codec_config['codebook_size'])
    quantizers = int(codec_config.get('codebook_nqtz', 1))
    layers = args.encoder_layer or quantizers
    if not 1 <= layers <= quantizers:
        raise ValueError(f'Encoder layer must be 0..{quantizers}')
    size = base_size ** layers
    vocab_size = int(config.vocab_size)
    if not 1 <= size <= vocab_size:
        raise ValueError(f'Effective codec codebook {size} cannot fit decoder vocab {vocab_size}')
    mode = resolve_token_id_mode(str(args.decoder_dir), args.tokenizer_file, args.token_id_mode)
    tokenizer_path = Path(args.tokenizer_file) if args.tokenizer_file else args.decoder_dir / 'tokenizer.json'
    tokenizer = load_bwav_tokenizer(str(args.decoder_dir), str(tokenizer_path)) if tokenizer_path.is_file() else None
    raw = torch.arange(size, dtype=torch.long)
    lookup = codebook_ids_to_decoder_input_ids(raw, mode=mode, tokenizer=tokenizer,
                                              token_offset=args.token_offset, device=torch.device('cpu'))
    if lookup.unique().numel() != size or int(lookup.min()) < 0 or int(lookup.max()) >= vocab_size:
        raise ValueError('Codec lookup contains duplicate or out-of-vocabulary token IDs')
    # A claimed offset must be equivalent to tokenizer.json when assets exist.
    if mode == 'offset' and tokenizer is not None:
        reference = codebook_ids_to_decoder_input_ids(raw, mode='tokenizer', tokenizer=tokenizer,
                                                      token_offset=0, device=torch.device('cpu'))
        if not torch.equal(reference, lookup):
            raise ValueError('--token-offset disagrees with tokenizer.json')
    specials = {}
    for name in ('pad', 'bos', 'eos'):
        configured = getattr(config, f'{name}_token_id', None)
        from_tokenizer = getattr(tokenizer, f'{name}_token_id', None) if tokenizer is not None else None
        if isinstance(configured, list):
            if from_tokenizer is not None and from_tokenizer not in configured:
                raise ValueError(f'Tokenizer/config disagree on {name}')
            configured = from_tokenizer if from_tokenizer is not None else configured[0]
        if configured is not None and from_tokenizer is not None and int(configured) != from_tokenizer:
            raise ValueError(f'Tokenizer/config disagree on {name}')
        value = configured if configured is not None else from_tokenizer
        specials[name] = int(value) if value is not None else None
        if value is not None and not 0 <= int(value) < vocab_size:
            raise ValueError(f'Invalid {name} token ID')
    if specials['pad'] is None:
        specials['pad'] = specials['eos'] if specials['eos'] is not None else 0
    if args.special_tokens == 'bos_eos':
        if specials['bos'] is None or specials['eos'] is None:
            raise ValueError('bos_eos layout requires BOS and EOS IDs in checkpoint assets')
        if specials['bos'] == specials['eos'] or set(lookup.tolist()) & {specials['bos'], specials['eos']}:
            raise ValueError('BOS/EOS must be distinct and outside the codec vocabulary')
    maximum = next((int(getattr(config, k)) for k in ('max_position_embeddings', 'max_sequence_length', 'n_positions')
                    if getattr(config, k, None) is not None), None)
    if maximum is None:
        if args.max_tokens is None:
            raise ValueError('Decoder context limit unknown; set --max-tokens explicitly')
        maximum = args.max_tokens
    elif args.max_tokens is not None:
        maximum = min(maximum, args.max_tokens)
    if maximum <= (2 if args.special_tokens == 'bos_eos' else 0):
        raise ValueError('Decoder context cannot hold signal tokens')
    index, name, depth = selected_layer(config, args.decoder_layers, args.selected_layer_name)
    hidden_size = int(getattr(config, 'hidden_size', getattr(config, 'n_embd', 0)))
    if hidden_size <= 0:
        raise ValueError('Unknown decoder hidden size')
    return TokenProtocol(lookup, maximum=maximum, hidden_size=hidden_size, vocab_size=vocab_size,
                         layer_index=index, layer_name=name, layer_depth=depth, mode=mode,
                         special_tokens=args.special_tokens, **specials)


def encode_tokens_without_reconstruction(codec, signal, layer, device, mode='auto'):
    x = signal.to(device=device, dtype=torch.float32).view(1, 1, -1)
    can_skip = all(hasattr(codec, name) for name in ('pore_head', 'cnn_model', 'project_in', 'vq', 'tokenize_indices'))
    if mode != 'reference' and can_skip:
        latent = codec.cnn_model.encode(codec.pore_head(x)).permute(0, 2, 1)
        _, indices = codec.vq(codec.project_in(latent))
        ids = codec.tokenize_indices(indices.long(), layer=layer)
    else:
        ids = codec.encode_signal(x, layer=layer)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError(f'Encoder must return nonempty [1, K] token IDs, got {tuple(ids.shape)}')
    if ids.is_floating_point() and not torch.equal(ids, ids.round()):
        raise ValueError('Encoder returned nonintegral token IDs')
    return ids[0].detach().to(device='cpu', dtype=torch.long)


@dataclass
class EncodedSample:
    source: dict
    spans: list
    signal_len: int
    token_ids: torch.Tensor


def encode_sample(sample, signal, spans, encoder, args, protocol, device):
    raw = encode_tokens_without_reconstruction(encoder, signal, args.encoder_layer, device, args.codec_mode)
    if int(raw.min()) < 0 or int(raw.max()) >= len(protocol.lookup):
        raise ValueError(f'Codec token outside configured codebook; read_id={sample.get("read_id")}')
    overhead = 2 if protocol.special_tokens == 'bos_eos' else 0
    if len(raw) + overhead > protocol.maximum:
        if args.overlong_policy == 'error':
            raise ValueError(f'{len(raw)+overhead} tokens exceed decoder limit {protocol.maximum}; '
                             f'read_id={sample.get("read_id")}')
        return None
    return EncodedSample(sample, spans, signal.numel(), protocol.lookup[raw].contiguous())


def infer_batch(items, decoder, protocol, device):
    sequences = [item.token_ids for item in items]
    offset = 0
    if protocol.special_tokens == 'bos_eos':
        offset = 1
        sequences = [torch.cat((torch.tensor([protocol.bos]), ids, torch.tensor([protocol.eos]))) for ids in sequences]
    ids = pad_sequence(sequences, batch_first=True, padding_value=protocol.pad).to(device)
    lengths = torch.tensor([len(seq) for seq in sequences], device=device)
    mask = (torch.arange(ids.shape[1], device=device)[None, :] < lengths[:, None]).long()
    # CausalLM checkpoints expose a backbone: skip unused vocabulary logits.
    backbone = getattr(decoder, 'base_model', decoder)
    last = protocol.layer_index == protocol.layer_depth
    output = backbone(input_ids=ids, attention_mask=mask, use_cache=False,
                      output_hidden_states=not last, return_dict=True)
    hidden = output.last_hidden_state if last else output.hidden_states[protocol.layer_index]
    if hidden.shape != (len(items), ids.shape[1], protocol.hidden_size):
        raise ValueError(f'Unexpected decoder hidden shape: {tuple(hidden.shape)}')
    if not torch.isfinite(hidden).all():
        raise ValueError('Decoder returned non-finite hidden states')
    return [hidden[i, offset:offset+len(item.token_ids)].detach().to(device='cpu', dtype=torch.float32)
            for i, item in enumerate(items)]


def infer_with_oom_fallback(items, decoder, protocol, device):
    try:
        return infer_batch(items, decoder, protocol, device)
    except torch.cuda.OutOfMemoryError:
        if device.type != 'cuda' or len(items) <= 1:
            raise
        print(f'[WARN] CUDA OOM: splitting decoder batch of {len(items)}', flush=True)
    # Leave the except block before retrying, releasing the failed traceback.
    torch.cuda.empty_cache()
    mid = len(items) // 2
    return (infer_with_oom_fallback(items[:mid], decoder, protocol, device)
            + infer_with_oom_fallback(items[mid:], decoder, protocol, device))


def hidden_to_base_window(hidden, item):
    vectors = []
    for span in item.spans:
        start, end = span_to_token_interval(span, item.signal_len, len(item.token_ids))
        vectors.append(hidden[start:end].float().mean(dim=0).numpy())
    return np.stack(vectors)


def sample_metadata(item, split):
    row = {key: item.source.get(key, '') for key in METADATA_KEYS}
    row.update(signal_len=item.signal_len, label=int(item.source['label']), split=split)
    return row


def feature_source(args, protocol):
    return f'signal_window_gpt_{protocol.layer_name}_window{args.window_size}_refspan_{args.special_tokens}'


def extraction_metadata(args, protocol):
    return dict(model_family='GPT_OLMo', encoder_dir=str(args.encoder_dir.resolve()),
                decoder_dir=str(args.decoder_dir.resolve()), encoder_layer=args.encoder_layer,
                decoder_layers=args.decoder_layers, token_id_mode=args.token_id_mode,
                tokenizer_file=str(Path(args.tokenizer_file).resolve()) if args.tokenizer_file else str((args.decoder_dir/'tokenizer.json').resolve()),
                preprocess=args.preprocess, codec_mode=args.codec_mode,
                overlong_policy=args.overlong_policy, inference_dtype='float32',
                storage_dtype=args.dtype, matmul_precision=torch.get_float32_matmul_precision(),
                pooling='mean over base spans using actual signal length and original codec token count',
                embedding_implementation=str(Path(__file__).resolve()), **protocol.metadata())


def owned_outputs(args):
    paths = []
    for split in args.splits:
        for name in (f'{split}.pt', f'{split}.features.f16.bin', f'{split}.features.f32.bin',
                     f'{split}.labels.f32.bin', f'metadata_{split}.tsv'):
            paths.extend((args.out_dir/name, args.out_dir/(name+'.partial')))
    return paths + [args.out_dir/'feature_summary.json', args.out_dir/'feature_summary.json.tmp']


@contextlib.contextmanager
def output_lock(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir/'.extract.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'An extractor is already active in {args.out_dir}') from exc
        existing = [p for p in owned_outputs(args) if p.exists()]
        if existing and not args.force:
            raise FileExistsError(f'Existing Step-4 outputs in {args.out_dir}; choose another OUT_DIR or use --force')
        yield existing


def build_split(split, encoder, decoder, preprocessor, args, device, protocol):
    input_path = split_input_path(args.data_dir, split)
    initial = input_path.stat()
    writer = StreamingSplitWriter(args.out_dir, split, args.window_size, protocol.hidden_size,
                                  args.dtype, args.save_every, feature_source(args, protocol))
    pending = []
    seen = skipped = overlong = 0
    skipped_reasons = dict(missing_signal=0, missing_spans=0, decoder_context_limit=0)
    labels = {0: 0, 1: 0}

    def flush_batch():
        if not pending:
            return
        arrays = infer_with_oom_fallback(pending, decoder, protocol, device)
        if len(arrays) != len(pending):
            raise RuntimeError('Decoder batch output count mismatch')
        for item, hidden in zip(pending, arrays):
            label = int(item.source['label'])
            writer.append(hidden_to_base_window(hidden, item), label, sample_metadata(item, split))
            labels[label] += 1
        pending.clear()

    print(f'[INFO] split={split} input={input_path}', flush=True)
    try:
        with torch.inference_mode():
            for line_no, sample in iter_jsonl(input_path, args.limit):
                seen += 1
                if args.log_every and seen % args.log_every == 0:
                    print(f'[INFO] split={split} seen={seen} kept={writer.records} '
                          f'pending={len(pending)} skipped={skipped}', flush=True)
                if sample.get('label') not in (0, 1, '0', '1'):
                    raise ValueError(f'Binary label required at {input_path}:{line_no}')
                if not sample.get('read_id') or not sample.get('chrom') or 'center_genome_pos0' not in sample:
                    raise ValueError(f'Missing read/site identity at {input_path}:{line_no}')
                if int(sample['center_genome_pos0']) < 0:
                    raise ValueError(f'Invalid genome position at {input_path}:{line_no}')
                for name in ('ref', 'seq'):
                    if sample.get(name) and (not isinstance(sample[name], str) or len(sample[name]) != args.window_size):
                        raise ValueError(f'{name} length does not match window at {input_path}:{line_no}')
                signal = prepare_signal(sample, preprocessor, input_path, line_no)
                if signal is None:
                    skipped += 1; skipped_reasons['missing_signal'] += 1
                    continue
                spans = normalize_spans(sample, signal.numel(), args.window_size, input_path, line_no)
                if spans is None:
                    skipped += 1; skipped_reasons['missing_spans'] += 1
                    continue
                item = encode_sample(sample, signal, spans, encoder, args, protocol, device)
                if item is None:
                    skipped += 1; overlong += 1; skipped_reasons['decoder_context_limit'] += 1
                    continue
                pending.append(item)
                if len(pending) >= args.batch_size:
                    flush_batch()
            flush_batch()
        if not writer.records:
            raise RuntimeError(f'No features built for split={split}')
        current = input_path.stat()
        if (current.st_size, current.st_mtime_ns) != (initial.st_size, initial.st_mtime_ns):
            raise RuntimeError(f'Input changed during extraction: {input_path}')
        manifest = writer.finalize(extraction_metadata(args, protocol))
    except BaseException:
        writer.abort()
        raise
    return dict(split=split, records=writer.records, seen=seen, skipped=skipped, skipped_reasons=skipped_reasons,
                skipped_decoder_context_limit=overlong, label_counts=labels,
                feature_shape=manifest['feature_shape'], storage_format=manifest['storage_format'],
                pt_path=str(writer.manifest_path), feature_path=str(writer.feature_path),
                label_path=str(writer.label_path), metadata_path=str(writer.metadata_path),
                input_path=str(input_path.resolve()), input_size=initial.st_size, input_mtime_ns=initial.st_mtime_ns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--encoder-dir', type=Path, default=DEFAULT_ENCODER_DIR)
    parser.add_argument('--decoder-dir', type=Path, default=DEFAULT_DECODER_DIR)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--encoder-layer', type=int, default=1)
    parser.add_argument('--decoder-layers', type=parse_layers, default=parse_layers('last'))
    parser.add_argument('--selected-layer-name', help='Required when decoder-layers selects multiple layers')
    parser.add_argument('--token-id-mode', choices=['auto', 'tokenizer', 'offset', 'direct'], default='auto')
    parser.add_argument('--tokenizer-file')
    parser.add_argument('--token-offset', type=int, default=128)
    parser.add_argument('--special-tokens', choices=['none', 'bos_eos'], default='none')
    parser.add_argument('--codec-mode', choices=['auto', 'reference'], default='auto',
                        help='auto skips unused CNN reconstruction when supported; reference calls encode_signal')
    parser.add_argument('--preprocess', choices=['none', 'auto'], default='none',
                        help='none for supplied normalized Step-3 signals; auto applies the codec feature extractor')
    parser.add_argument('--window-size', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float32', help='Storage dtype; inference is FP32')
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-tokens', type=int, help='Optional tighter context cap, including any special tokens')
    parser.add_argument('--overlong-policy', choices=['skip', 'error'], default='skip')
    parser.add_argument('--splits', nargs='+', choices=['train', 'valid', 'test'], default=['train', 'valid', 'test'])
    parser.add_argument('--force', action='store_true', help='Replace only selected split outputs and feature_summary.json')
    return parser.parse_args(argv)


def validate_args(args):
    for name in ('window_size', 'batch_size', 'save_every', 'cpu_threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if args.window_size % 2 != 1:
        raise ValueError('--window-size must be odd')
    if args.encoder_layer < 0 or args.log_every < 0:
        raise ValueError('encoder-layer and log-every must be nonnegative')
    if args.limit is not None and args.limit < 1 or args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError('limit and max-tokens must be positive when supplied')
    if len(set(args.splits)) != len(args.splits):
        raise ValueError('--splits must not contain duplicates')
    for split in args.splits:
        split_input_path(args.data_dir, split)
    for path in (args.encoder_dir, args.decoder_dir):
        if not (path/'config.json').is_file():
            raise FileNotFoundError(f'Missing model config: {path}/config.json')
    if args.tokenizer_file is not None and not Path(args.tokenizer_file).is_file():
        raise FileNotFoundError(f'Missing tokenizer: {args.tokenizer_file}')


def main():
    args = parse_args()
    validate_args(args)
    protocol = load_token_protocol(args)
    device = resolve_device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision('highest')
    with output_lock(args) as existing:
        print(f'[INFO] device={device}; protocol={protocol.metadata()}; preprocess={args.preprocess}', flush=True)
        preprocessor = load_preprocessor(args.preprocess, args.encoder_dir)
        encoder = load_encoder(str(args.encoder_dir)).to(device).eval()
        decoder = load_decoder(str(args.decoder_dir)).to(device).eval()
        # Load all models successfully before replacing any existing outputs.
        for path in existing:
            path.unlink()
        splits = [build_split(split, encoder, decoder, preprocessor, args, device, protocol) for split in args.splits]
        summary = dict(extraction_metadata(args, protocol), data_dir=str(args.data_dir.resolve()),
                       out_dir=str(args.out_dir.resolve()), feature_source=feature_source(args, protocol),
                       window_size=args.window_size, batch_size=args.batch_size, save_every=args.save_every,
                       dtype=args.dtype, limit=args.limit, device=str(device), cpu_threads=args.cpu_threads,
                       max_position_embeddings=protocol.maximum, splits=splits)
        temporary = args.out_dir/'feature_summary.json.tmp'
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(args.out_dir/'feature_summary.json')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
