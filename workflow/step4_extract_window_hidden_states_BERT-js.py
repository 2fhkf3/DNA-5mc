#!/usr/bin/env python3
"""Extract base-level V548 PoreBERT features from 15-nt signal windows.

Input
-----
``data_dir/{train,valid,test}.jsonl[.gz]``.  Every record must contain an
already-normalized one-dimensional ``signal``, a binary ``label``, and 15
signal spans in ``base_sample_span_ref`` (or ``base_sample_span_seq`` as a
fallback).

For each record this script follows the V548 inference path used by
``extract_v548_hidden_states.py``:

1. encode the signal with the V548 RSQ codec without running its decoder;
2. add 128 to codec IDs and wrap them with ``[CLS, BOS, EOS]``;
3. run the PoreBERT backbone and select one entry from ``hidden_states``;
4. remove special/padding tokens and mean-pool token states into 15 reference
   base embeddings using the supplied signal spans.

Output
------
``out_dir/{train,valid,test}.pt`` is a small manifest pointing to streamed,
memory-mappable feature and label files.  Features have logical shape
``[N, 15, hidden_size]``.  TSV metadata files and ``feature_summary.json`` are
written alongside them.  This avoids retaining an entire split in RAM.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence, TextIO

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoFeatureExtractor, AutoModel, AutoModelForMaskedLM


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "train" / "backbone" / "V554-BERT"
DEFAULT_DATA_DIR = PROJECT_ROOT / "output" / "step3_output" / "new"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "step4_output" / "V554"

PAD_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
CLS_TOKEN_ID = 5
SPECIAL_TOKEN_COUNT = 3
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


@dataclass
class EncodedSample:
    source: dict[str, Any]
    signal_len: int
    spans: list[list[int]]
    # IDs already include token_offset. Kept on CPU until PoreBERT batching.
    token_ids: torch.Tensor
    was_truncated: bool

    @property
    def token_len(self) -> int:
        return int(self.token_ids.numel())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--encoder-dir", type=Path, default=DEFAULT_MODEL_ROOT / "encoder")
    parser.add_argument("--bert-dir", type=Path, default=DEFAULT_MODEL_ROOT / "base")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda:0, or a bare CUDA index such as 0.",
    )
    parser.add_argument(
        "--encoder-layer",
        type=int,
        default=0,
        help="RSQ tokenize_indices layer; 0 uses all available quantizer tracks.",
    )
    parser.add_argument(
        "--hidden-layer",
        type=int,
        default=-1,
        help=(
            "Index into PoreBERT output_hidden_states. The default -1 selects "
            "the final Transformer layer, matching extract_v548_hidden_states.py."
        ),
    )
    parser.add_argument("--token-offset", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--preprocess",
        choices=["none", "auto"],
        default="none",
        help=(
            "none expects signals already normalized by step 1; auto applies "
            "the V548 AutoFeatureExtractor before codec inference."
        ),
    )
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1024,
        help=(
            "Number of completed samples buffered before appending them to disk. "
            "This controls host RAM usage, not model inference batch size."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path, label in (
        (args.data_dir, "data directory"),
        (args.encoder_dir, "V548 encoder"),
        (args.bert_dir, "V548 PoreBERT"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested, but CUDA is unavailable")
    return device


def split_input_path(data_dir: Path, split: str) -> Path:
    for suffix in (".jsonl.gz", ".jsonl"):
        path = data_dir / f"{split}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {data_dir / f'{split}.jsonl.gz'} or "
        f"{data_dir / f'{split}.jsonl'}"
    )


def iter_jsonl(path: Path, limit: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    emitted = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and emitted >= limit:
                return
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            yield line_no, sample
            emitted += 1


def load_models(
    encoder_dir: Path,
    bert_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    codec = AutoModel.from_pretrained(
        str(encoder_dir), trust_remote_code=True, local_files_only=True
    )
    bert_mlm = AutoModelForMaskedLM.from_pretrained(
        str(bert_dir), trust_remote_code=True, local_files_only=True
    )
    codec.to(device).eval()
    bert_mlm.to(device).eval()
    return codec, bert_mlm


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


def encode_tokens_without_reconstruction(
    codec: torch.nn.Module,
    signal: torch.Tensor,
    encoder_layer: int,
    device: torch.device,
) -> torch.Tensor:
    """Match V548 tokenization while bypassing its unused signal decoder."""
    x = signal.to(device=device, dtype=torch.float32).view(1, 1, -1)
    try:
        headed = codec.pore_head(x)
        latent = codec.cnn_model.encode(headed).permute(0, 2, 1)
        projected = codec.project_in(latent)
        _, level_indices = codec.vq(projected)
        token_ids = codec.tokenize_indices(
            level_indices.to(torch.long), layer=encoder_layer
        )
    except AttributeError:
        token_ids = codec.encode_signal(x, layer=encoder_layer)
    return token_ids.squeeze(0).detach().to(device="cpu", dtype=torch.long)


def encode_sample(
    sample: dict[str, Any],
    signal: torch.Tensor,
    spans: list[list[int]],
    codec: torch.nn.Module,
    encoder_layer: int,
    token_offset: int,
    max_positions: int,
    vocab_size: int,
    device: torch.device,
) -> EncodedSample:
    token_ids = encode_tokens_without_reconstruction(
        codec, signal, encoder_layer, device
    )
    if token_ids.ndim != 1 or token_ids.numel() == 0:
        raise ValueError(
            f"V548 codec produced invalid tokens for read_id={sample.get('read_id')}"
        )

    token_ids = token_ids + token_offset
    token_min = int(token_ids.min().item())
    token_max = int(token_ids.max().item())
    if token_min < token_offset or token_max >= vocab_size:
        raise ValueError(
            f"Token range [{token_min}, {token_max}] is outside PoreBERT vocabulary "
            f"[0, {vocab_size}) for read_id={sample.get('read_id')}"
        )

    max_signal_tokens = max_positions - SPECIAL_TOKEN_COUNT
    if max_signal_tokens < 1:
        raise ValueError(
            f"max_position_embeddings={max_positions} cannot hold CLS/BOS/EOS"
        )
    was_truncated = token_ids.numel() > max_signal_tokens
    if was_truncated:
        token_ids = token_ids[:max_signal_tokens]

    return EncodedSample(
        source=sample,
        signal_len=int(signal.numel()),
        spans=spans,
        token_ids=token_ids.contiguous(),
        was_truncated=was_truncated,
    )


def add_special_tokens(token_ids: torch.Tensor) -> torch.Tensor:
    full = torch.empty(token_ids.numel() + SPECIAL_TOKEN_COUNT, dtype=torch.long)
    full[0] = CLS_TOKEN_ID
    full[1] = BOS_TOKEN_ID
    full[2:-1] = token_ids
    full[-1] = EOS_TOKEN_ID
    return full


def infer_bert_batch(
    items: Sequence[EncodedSample],
    bert_mlm: torch.nn.Module,
    hidden_layer: int,
    device: torch.device,
) -> list[torch.Tensor]:
    sequences = [add_special_tokens(item.token_ids) for item in items]
    lengths = torch.tensor([seq.numel() for seq in sequences], dtype=torch.long)
    input_ids = pad_sequence(
        sequences, batch_first=True, padding_value=PAD_TOKEN_ID
    ).to(device, non_blocking=True)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    attention_mask = positions < lengths.to(device).unsqueeze(1)

    # Match extract_v548_hidden_states.py: call SimpleMLM directly so no MLM
    # logits are allocated, and select the requested hidden_states entry.
    backbone = getattr(bert_mlm, "bert", None)
    if backbone is None:
        raise AttributeError("V548 masked-LM model has no .bert backbone")
    _, hidden_states = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask.to(torch.long),
        output_hidden_states=True,
    )
    if hidden_states is None:
        raise RuntimeError("V548 PoreBERT returned no hidden_states")
    try:
        selected = hidden_states[hidden_layer]
    except IndexError as exc:
        raise IndexError(
            f"--hidden-layer {hidden_layer} is invalid for "
            f"{len(hidden_states)} hidden states"
        ) from exc

    arrays: list[torch.Tensor] = []
    for row_index, item in enumerate(items):
        # 0=CLS, 1=BOS, then signal tokens; EOS and batch padding are excluded.
        arrays.append(
            selected[row_index, 2 : 2 + item.token_len]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
    return arrays


def infer_with_oom_fallback(
    items: Sequence[EncodedSample],
    bert_mlm: torch.nn.Module,
    hidden_layer: int,
    device: torch.device,
) -> list[torch.Tensor]:
    try:
        return infer_bert_batch(items, bert_mlm, hidden_layer, device)
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda" or len(items) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(items) // 2
        print(
            f"[WARN] CUDA OOM for batch={len(items)}; retrying as "
            f"{midpoint}+{len(items) - midpoint}",
            flush=True,
        )
        return infer_with_oom_fallback(
            items[:midpoint], bert_mlm, hidden_layer, device
        ) + infer_with_oom_fallback(
            items[midpoint:], bert_mlm, hidden_layer, device
        )


def span_to_token_interval(
    span: Sequence[int], signal_len: int, token_len: int
) -> tuple[int, int]:
    start, end = int(span[0]), int(span[1])
    token_start = int(math.floor(start * token_len / signal_len))
    token_end = int(math.ceil(end * token_len / signal_len))
    token_start = max(0, min(token_start, token_len - 1))
    token_end = max(token_start + 1, min(token_end, token_len))
    return token_start, token_end


def hidden_to_base_window(hidden: torch.Tensor, item: EncodedSample) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for span in item.spans:
        token_start, token_end = span_to_token_interval(
            span, item.signal_len, item.token_len
        )
        vector = hidden[token_start:token_end].mean(dim=0).numpy()
        vectors.append(vector)
    return np.stack(vectors, axis=0)


def sample_metadata(item: EncodedSample, split: str) -> dict[str, Any]:
    row = {key: item.source.get(key, "") for key in METADATA_KEYS}
    # Step-3 examples do not carry signal_len explicitly; record the actual
    # model input length while preserving the existing TSV column schema.
    row["signal_len"] = item.signal_len
    row["split"] = split
    return row


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


def resolve_hidden_layer_name(hidden_layer: int, num_hidden_layers: int) -> tuple[int, str]:
    total_states = num_hidden_layers + 1  # embedding plus every Transformer layer
    resolved_index = hidden_layer if hidden_layer >= 0 else total_states + hidden_layer
    if resolved_index < 0 or resolved_index >= total_states:
        raise ValueError(
            f"--hidden-layer {hidden_layer} is invalid for {total_states} hidden states"
        )
    name = "embed" if resolved_index == 0 else f"layer_{resolved_index - 1}"
    return resolved_index, name


def build_split(
    split: str,
    data_dir: Path,
    out_dir: Path,
    codec: torch.nn.Module,
    bert_mlm: torch.nn.Module,
    preprocessor: Any | None,
    args: argparse.Namespace,
    device: torch.device,
    selected_layer_name: str,
) -> dict[str, Any]:
    input_path = split_input_path(data_dir, split)
    max_positions = int(bert_mlm.config.max_position_embeddings)
    vocab_size = int(bert_mlm.config.vocab_size)
    hidden_size = int(bert_mlm.config.hidden_size)
    feature_source = (
        f"chrom19_signal_window_v548_{selected_layer_name}_"
        f"window{args.window_size}_refspan"
    )

    writer = StreamingSplitWriter(
        out_dir=out_dir,
        split=split,
        window_size=args.window_size,
        hidden_size=hidden_size,
        dtype=args.dtype,
        save_every=args.save_every,
        feature_source=feature_source,
    )
    pending: list[EncodedSample] = []
    seen = 0
    skipped = 0
    truncated = 0

    def flush_batch() -> None:
        nonlocal pending
        if not pending:
            return
        hidden_arrays = infer_with_oom_fallback(
            pending, bert_mlm, args.hidden_layer, device
        )
        for item, hidden in zip(pending, hidden_arrays):
            try:
                label = int(item.source["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"label must be integer-like; read_id={item.source.get('read_id')}, "
                    f"label={item.source.get('label')!r}"
                ) from exc
            writer.append(
                hidden_to_base_window(hidden, item),
                label,
                sample_metadata(item, split),
            )
        pending = []

    print(f"[INFO] split={split} input={input_path}", flush=True)
    try:
        with torch.inference_mode():
            for line_no, sample in iter_jsonl(input_path, args.limit):
                seen += 1
                signal = prepare_signal(sample, preprocessor, input_path, line_no)
                if signal is None:
                    skipped += 1
                    continue
                spans = normalize_spans(
                    sample, int(signal.numel()), args.window_size, input_path, line_no
                )
                if spans is None:
                    skipped += 1
                    continue
                item = encode_sample(
                    sample=sample,
                    signal=signal,
                    spans=spans,
                    codec=codec,
                    encoder_layer=args.encoder_layer,
                    token_offset=args.token_offset,
                    max_positions=max_positions,
                    vocab_size=vocab_size,
                    device=device,
                )
                truncated += int(item.was_truncated)
                pending.append(item)
                if len(pending) >= args.batch_size:
                    flush_batch()
                if args.log_every > 0 and seen % args.log_every == 0:
                    print(
                        f"[INFO] split={split} seen={seen} kept={writer.records} "
                        f"pending={len(pending)} disk_buffer={writer.buffered} "
                        f"skipped={skipped} truncated={truncated}",
                        flush=True,
                    )
            flush_batch()
        if not writer.records:
            raise RuntimeError(f"No features built for split={split}")
        manifest = writer.finalize()
    except BaseException:
        writer.abort()
        raise

    return {
        "split": split,
        "records": writer.records,
        "skipped": skipped,
        "truncated_to_bert_max_positions": truncated,
        "feature_shape": manifest["feature_shape"],
        "storage_format": manifest["storage_format"],
        "pt_path": str(writer.manifest_path),
        "feature_path": str(writer.feature_path),
        "label_path": str(writer.label_path),
        "metadata_path": str(writer.metadata_path),
        "input_path": str(input_path),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] data_dir={args.data_dir}", flush=True)
    print(f"[INFO] encoder_dir={args.encoder_dir}", flush=True)
    print(f"[INFO] bert_dir={args.bert_dir}", flush=True)
    print(f"[INFO] out_dir={args.out_dir}", flush=True)
    print(f"[INFO] preprocess={args.preprocess}", flush=True)

    preprocessor = load_preprocessor(args.preprocess, args.encoder_dir)
    codec, bert_mlm = load_models(args.encoder_dir, args.bert_dir, device)
    resolved_hidden_index, selected_layer_name = resolve_hidden_layer_name(
        args.hidden_layer, int(bert_mlm.config.num_hidden_layers)
    )
    print(
        f"[INFO] loaded codec={type(codec).__name__}; "
        f"PoreBERT={type(bert_mlm).__name__} "
        f"layers={bert_mlm.config.num_hidden_layers} "
        f"hidden={bert_mlm.config.hidden_size} "
        f"max_positions={bert_mlm.config.max_position_embeddings} "
        f"selected={selected_layer_name}",
        flush=True,
    )

    split_summaries = [
        build_split(
            split=split,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            codec=codec,
            bert_mlm=bert_mlm,
            preprocessor=preprocessor,
            args=args,
            device=device,
            selected_layer_name=selected_layer_name,
        )
        for split in ("train", "valid", "test")
    ]

    feature_source = (
        f"chrom19_signal_window_v548_{selected_layer_name}_"
        f"window{args.window_size}_refspan"
    )
    summary = {
        "data_dir": str(args.data_dir),
        "encoder_dir": str(args.encoder_dir),
        "bert_dir": str(args.bert_dir),
        # Keep decoder_dir for consumers of the previous Step-4 summary schema.
        "decoder_dir": str(args.bert_dir),
        "out_dir": str(args.out_dir),
        "model_family": "V548_PoreBERT",
        "feature_source": feature_source,
        "encoder_layer": args.encoder_layer,
        "hidden_layer": args.hidden_layer,
        "resolved_hidden_state_index": resolved_hidden_index,
        "selected_layer_name": selected_layer_name,
        "token_offset": args.token_offset,
        "special_tokens": {
            "pad": PAD_TOKEN_ID,
            "bos": BOS_TOKEN_ID,
            "eos": EOS_TOKEN_ID,
            "cls": CLS_TOKEN_ID,
        },
        "preprocess": args.preprocess,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "save_every": args.save_every,
        "hidden_size": int(bert_mlm.config.hidden_size),
        "max_position_embeddings": int(bert_mlm.config.max_position_embeddings),
        "dtype": args.dtype,
        "splits": split_summaries,
    }
    summary_path = args.out_dir / "feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
