#!/usr/bin/env python3
"""Train any of the seven classifiers using one YAML configuration.

Paths in YAML are relative to this script's directory. Step-4 manifests and
legacy tensor .pt inputs are supported, each with metadata_{split}.tsv.
None of the old trainers are imported.
Multiple devices run independent strategy/seed experiments through a spawn
worker queue. Each worker runs one experiment at a time; this is not DDP.
"""
from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate

from model import FUSION_STRATEGY, STRATEGIES, build_model
import wandb_tracking as tracking
import gpu_feature_cache as feature_caching

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")
BASE_TO_CODE = dict(zip("ACGTNWSMKRYBVDHZ", range(16)))
DEFAULTS = {
    "feature_dir": "data/V543-V2",
    "metadata_dir": None,  # None means the same directory as feature_dir.
    "output_dir": "output", "experiment": "test_BERT",
    "strategies": list(STRATEGIES), "seeds": [1], "device": "auto", "devices": None,
    "base_source": "ref", "fusion_window_size": 5,
    "check_site_overlap": True, "skip_completed": True,
    "evaluate_train": False, "save_predictions": "test",
    "wandb": tracking.DEFAULTS,
    "training": {
        "epochs": 80, "patience": 12, "batch_size": 2048,
        "eval_batch_size": 2048, "lr": 1e-4, "weight_decay": 1e-4,
        "grad_clip": 0.0, "monitor_metric": "auroc", "threshold": 0.5,
        "num_workers": 0, "cpu_threads": 4, "deterministic": True,
        "log_every_steps": 50,
        "batch_read": True, "eval_num_workers": 0,
        "prefetch_factor": 2, "persistent_workers": True,
        **feature_caching.DEFAULTS,
    },
    "model": {
        "gaussian_sigma": 1.0, "mlp_hidden_dims": [512, 128],
        "conv_channels": 128, "conv_head_dim": 128, "dropout": 0.2,
        "deepplant": {
            "signal_mlp_hidden": 256, "base_hidden": 128, "fusion_hidden": 256,
            "base_layers": 2, "fusion_layers": 3, "dropout": 0.5,
            "signal_mlp_dropout": 0.1,
        },
    },
    "strategy_overrides": {FUSION_STRATEGY: {"batch_size": 256, "eval_batch_size": 256, "grad_clip": 0.5}},
}


def merge_config(defaults, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown config keys in {prefix or 'config'}: {sorted(unknown)}")
    result = {}
    for key, value in defaults.items():
        new = copy.deepcopy(supplied.get(key, value))
        if isinstance(value, dict) and key != "strategy_overrides":
            new = merge_config(value, new, f"{prefix}{key}.")
        result[key] = new
    return result


def positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_training(settings):
    feature_caching.validate_settings(settings)
    for key in ("epochs", "patience", "batch_size", "eval_batch_size", "cpu_threads"):
        positive_integer(settings[key], key)
    positive_integer(settings["num_workers"], "num_workers", 0)
    positive_integer(settings["eval_num_workers"], "eval_num_workers", 0)
    positive_integer(settings["prefetch_factor"], "prefetch_factor")
    positive_integer(settings["log_every_steps"], "log_every_steps", 0)
    for key in ("lr", "weight_decay", "grad_clip", "threshold"):
        value = settings[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if settings["lr"] <= 0 or min(settings["weight_decay"], settings["grad_clip"]) < 0:
        raise ValueError("lr must be positive; weight_decay and grad_clip must be >= 0")
    if not 0 < settings["threshold"] < 1:
        raise ValueError("threshold must be in (0,1)")
    if settings["monitor_metric"] not in ("auroc", "auprc", "f1", "accuracy", "loss"):
        raise ValueError("Unsupported monitor_metric")
    if type(settings["deterministic"]) is not bool:
        raise ValueError("deterministic must be boolean")
    for key in ("batch_read", "persistent_workers"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")


def load_config(path, args=None):
    supplied = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("YAML must contain a mapping")
    supplied = dict(supplied)
    for singular, plural in (("strategy", "strategies"), ("seed", "seeds")):
        if singular in supplied:
            if plural in supplied:
                raise ValueError(f"Specify {singular} or {plural}, not both")
            supplied[plural] = [supplied.pop(singular)]
    cfg = merge_config(DEFAULTS, supplied)
    tracking.validate_config(cfg["wandb"])
    if args is not None:
        for key in ("strategies", "seeds", "device", "devices", "experiment", "output_dir", "metadata_dir"):
            if getattr(args, key, None) is not None:
                cfg[key] = getattr(args, key)
        if getattr(args, "device", None) is not None:
            cfg["devices"] = None
        for key in ("epochs", "batch_size", "log_every_steps"):
            if getattr(args, key, None) is not None:
                cfg["training"][key] = getattr(args, key)
        if getattr(args, "batch_size", None) is not None:
            cfg["training"]["eval_batch_size"] = args.batch_size
            for override in cfg["strategy_overrides"].values():
                override["batch_size"] = args.batch_size
                override["eval_batch_size"] = args.batch_size
        if getattr(args, "log_every_steps", None) is not None:
            for override in cfg["strategy_overrides"].values():
                override["log_every_steps"] = args.log_every_steps
        for key in ("num_workers", "eval_num_workers", "prefetch_factor", "batch_read", "persistent_workers",
                    *feature_caching.DEFAULTS):
            if getattr(args, key, None) is not None:
                cfg["training"][key] = getattr(args, key)
                for override in cfg["strategy_overrides"].values():
                    override[key] = getattr(args, key)
    if cfg["devices"] is not None:
        if not isinstance(cfg["devices"], list) or not cfg["devices"]:
            raise ValueError("devices must be a nonempty list or null")
        cfg["devices"] = [device_name(v) for v in cfg["devices"]]
        gpu_devices = [v for v in cfg["devices"] if v != "cpu"]
        if "auto" in cfg["devices"] or len(set(gpu_devices)) != len(gpu_devices):
            raise ValueError("devices requires explicit, distinct GPU indices (e.g. [cuda:0, cuda:1])")
    cfg["device"] = device_name(cfg["device"])
    for key in ("feature_dir", "output_dir"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    if cfg["metadata_dir"] is None:
        cfg["metadata_dir"] = cfg["feature_dir"]
    else:
        if not isinstance(cfg["metadata_dir"], str) or not cfg["metadata_dir"].strip():
            raise ValueError("metadata_dir must be a directory path or null")
        p = Path(cfg["metadata_dir"]).expanduser()
        cfg["metadata_dir"] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    if not isinstance(cfg["experiment"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg["experiment"]) or cfg["experiment"] in (".", ".."):
        raise ValueError("experiment must be a simple directory name")
    for key in ("strategies", "seeds"):
        if not isinstance(cfg[key], list) or not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    for strategy in cfg["strategies"]:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; choices: {STRATEGIES}")
    for seed in cfg["seeds"]:
        positive_integer(seed, "seed", 0)
        if seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
    for key in ("check_site_overlap", "skip_completed", "evaluate_train"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if cfg["save_predictions"] not in ("none", "test", "all"):
        raise ValueError("save_predictions must be none, test, or all")
    if cfg["base_source"] not in ("ref", "seq"):
        raise ValueError("base_source must be ref or seq")
    positive_integer(cfg["fusion_window_size"], "fusion_window_size")
    if cfg["fusion_window_size"] % 2 != 1:
        raise ValueError("fusion_window_size must be odd")
    validate_training(cfg["training"])
    if not isinstance(cfg["strategy_overrides"], dict):
        raise ValueError("strategy_overrides must be a mapping")
    for strategy, overrides in cfg["strategy_overrides"].items():
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown override strategy {strategy}")
        validate_training(merge_config(cfg["training"], overrides))
    m = cfg["model"]
    if not isinstance(m["mlp_hidden_dims"], list) or len(m["mlp_hidden_dims"]) != 2:
        raise ValueError("mlp_hidden_dims must contain two integers")
    for value in m["mlp_hidden_dims"]:
        positive_integer(value, "mlp_hidden_dims")
    for key in ("conv_channels", "conv_head_dim"):
        positive_integer(m[key], key)
    if not isinstance(m["gaussian_sigma"], (int, float)) or not math.isfinite(m["gaussian_sigma"]) or m["gaussian_sigma"] <= 0:
        raise ValueError("gaussian_sigma must be finite and positive")
    for key in ("signal_mlp_hidden", "base_hidden", "fusion_hidden", "base_layers", "fusion_layers"):
        positive_integer(m["deepplant"][key], key)
    for value in (m["dropout"], m["deepplant"]["dropout"], m["deepplant"]["signal_mlp_dropout"]):
        if not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError("dropout must be in [0,1)")
    return cfg


def torch_load(path, **kwargs):
    return torch.load(path, weights_only=False, **kwargs)


def signature(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_write(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


class FeatureSplit:
    def __init__(self, root, split, metadata_dir=None):
        self.name = split
        # Use the split's TSV explicitly, rather than a metadata path embedded
        # in a moved .pt manifest or an embedded Python metadata list.
        self.metadata_path = Path(metadata_dir if metadata_dir is not None else root) / f"metadata_{split}.tsv"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata TSV for split={split}: {self.metadata_path}")
        path = root / f"{split}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed Step-4 manifest: {path} (a .partial file is not ready)")
        self.data = torch_load(path, map_location="cpu")
        self.files = [path, self.metadata_path]
        if self.data.get("storage_format") == "streaming_memmap_v1":
            self.shape = tuple(int(v) for v in self.data["feature_shape"])
            if len(self.shape) != 3 or min(self.shape) < 1:
                raise ValueError(f"Invalid feature shape: {self.shape}")
            fp, lp = [root / self.data[k] for k in ("feature_path", "label_path")]
            self.files.extend((fp, lp))
            for file, shape, dtype in ((fp, self.shape, self.data["feature_dtype"]),
                                       (lp, (self.shape[0],), self.data["label_dtype"])):
                if not file.is_file() or file.stat().st_size != math.prod(shape) * np.dtype(dtype).itemsize:
                    raise ValueError(f"Incomplete or invalid array: {file}")
            self.x = np.memmap(fp, mode="r", dtype=self.data["feature_dtype"], shape=self.shape)
            self.y = np.memmap(lp, mode="r", dtype=self.data["label_dtype"], shape=(self.shape[0],))
        else:
            self.x = self.data["x"].cpu().numpy()
            self.y = self.data["y"].cpu().numpy().reshape(-1)
            self.shape = tuple(self.x.shape)
            if len(self.shape) != 3 or min(self.shape) < 1 or len(self.y) != self.shape[0]:
                raise ValueError(f"Invalid tensor dimensions in {path}")
        for start in range(0, len(self.y), 100000):
            values = self.y[start:start + 100000]
            if not np.isin(values, (0, 1)).all():
                raise ValueError(f"Nonbinary labels in {split}")
        self.input_signature = [signature(p) for p in self.files]
        self.base_ids = None

    def __getstate__(self):
        # NumPy pickles memmaps as array contents. Spawned workers must reopen
        # the files instead of serializing tens of GB for every DataLoader.
        state = self.__dict__.copy()
        state.pop("x")
        state.pop("y")
        if self.data.get("storage_format") != "streaming_memmap_v1":
            # Sending legacy torch tensors through multiprocessing can move
            # their storage and invalidate the parent's existing NumPy views.
            # Reopen the legacy .pt in each worker instead.
            state["data"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.data is None:
            self.data = torch_load(self.files[0], map_location="cpu")
        if self.data.get("storage_format") == "streaming_memmap_v1":
            self.x = np.memmap(self.files[2], mode="r", dtype=self.data["feature_dtype"], shape=self.shape)
            self.y = np.memmap(self.files[3], mode="r", dtype=self.data["label_dtype"], shape=(self.shape[0],))
        else:
            self.x = self.data["x"].cpu().numpy()
            self.y = self.data["y"].cpu().numpy().reshape(-1)

    def metadata(self, required_columns=()):
        with self.metadata_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter="\t")
            columns = reader.fieldnames or []
            if len(set(columns)) != len(columns):
                raise ValueError(f"Duplicate TSV column names in {self.metadata_path}")
            required = {"read_id", "label", "chrom", "center_genome_pos0", *required_columns}
            missing = required - set(columns)
            if missing:
                raise ValueError(f"Missing TSV columns in {self.metadata_path}: {sorted(missing)}")
            for line_number, row in enumerate(reader, 2):
                if None in row or any(row.get(k) is None for k in required):
                    raise ValueError(f"Malformed TSV row at {self.metadata_path}:{line_number}")
                yield row

    def audit(self, need_bases, base_source):
        sites = set()
        if need_bases:
            # 15 bytes/sample instead of parsing raw signal/statistic TSVs.
            self.base_ids = np.empty(self.shape[:2], dtype=np.uint8)
        count = 0
        for count, row in enumerate(self.metadata((base_source,) if need_bases else ()), 1):
            index = count - 1
            if count > self.shape[0] or int(row["label"]) != int(self.y[index]):
                raise ValueError(f"Feature/metadata label or count mismatch in {self.name}, row {count}")
            if row.get("split") and row["split"] != self.name:
                raise ValueError(f"Wrong split tag in {self.name}, row {count}")
            if not row["read_id"].strip():
                raise ValueError(f"Empty read_id in {self.metadata_path}, row {count}")
            if "sample_index" in row and int(row["sample_index"]) != index:
                raise ValueError(f"TSV sample_index does not match feature row {index}: {self.metadata_path}")
            sites.add((row["chrom"], int(row["center_genome_pos0"])))
            if need_bases:
                seq = row[base_source].strip().upper()
                if len(seq) != self.shape[1] or any(b not in BASE_TO_CODE for b in seq):
                    raise ValueError(f"Invalid {base_source} sequence in {self.name}, row {count}")
                self.base_ids[index] = [BASE_TO_CODE[b] for b in seq]
        if count != self.shape[0]:
            raise ValueError(f"Metadata count mismatch in {self.name}: {count} != {self.shape[0]}")
        if self.input_signature != [signature(p) for p in self.files]:
            raise ValueError("Input changed during metadata audit")
        return sites


class FeatureDataset(Dataset):
    def __init__(self, split, fusion_window_size=None, batch_read=True):
        self.split = split
        self.batch_read = batch_read
        self.fusion = fusion_window_size is not None
        self.start = 0 if not self.fusion else (split.shape[1] - fusion_window_size) // 2
        self.end = split.shape[1] if not self.fusion else self.start + fusion_window_size

    def __len__(self):
        return self.split.shape[0]

    def __getitem__(self, index):
        x = torch.from_numpy(np.array(self.split.x[index, self.start:self.end], dtype=np.float32, copy=True))
        if self.fusion:
            bases = torch.from_numpy(self.split.base_ids[index, self.start:self.end].astype(np.int64))
        else:
            bases = torch.empty(0, dtype=torch.long)
        return x, bases, torch.tensor(float(self.split.y[index]), dtype=torch.float32)

    def __getitems__(self, indices):
        if not self.batch_read:
            return [self[index] for index in indices]
        indices = np.asarray(indices, dtype=np.intp)
        if indices.ndim != 1 or not indices.size or indices.min() < 0 or indices.max() >= len(self):
            raise IndexError("Batch indices must be a nonempty 1-D list within the dataset")
        if indices.size == 1 or np.all(np.diff(indices) == 1):
            # Validation/test batches normally use one contiguous slice.
            x = np.array(self.split.x[indices[0]:indices[-1]+1, self.start:self.end],
                         dtype=np.float32, copy=True)
        else:
            # Read random rows in disk-offset order, then restore the EXACT
            # sampler order (including repeated indices) before the forward pass.
            order = np.argsort(indices, kind="stable")
            x = np.empty((len(indices), self.end-self.start, self.split.shape[2]), dtype=np.float32)
            x[order] = self.split.x[indices[order], self.start:self.end]
        if self.fusion:
            bases = torch.from_numpy(np.array(self.split.base_ids[indices, self.start:self.end],
                                              dtype=np.int64, copy=True))
        else:
            bases = torch.empty((len(indices), 0), dtype=torch.long)
        y = torch.from_numpy(np.array(self.split.y[indices], dtype=np.float32, copy=True))
        return torch.from_numpy(x), bases, y


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if y.size == 0 or y.shape != p.shape or not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise ValueError("Invalid labels/probabilities for metrics")
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    npos = int(np.sum(y == 1)); nneg = y.size - npos
    order = np.argsort(-p, kind="stable")
    scores = p[order]; positives = (y[order] == 1)
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), y.size - 1]
    tps = np.cumsum(positives)[ends]; fps = ends + 1 - tps
    # Score ties share one threshold, so AP is independent of row ordering.
    ap = float(np.sum(np.diff(np.r_[0, tps]) * tps / (ends + 1)) / npos) if npos else None
    if npos and nneg:
        tpr = np.r_[0, tps / npos]; fpr = np.r_[0, fps / nneg]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    else:
        auc = None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"accuracy": (tp + tn) / y.size, "precision": precision, "recall": recall,
            "specificity": tn / max(1, tn + fp),
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
            "auroc": auc, "auprc": ap, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n": int(y.size), "label_0": int(nneg), "label_1": int(npos)}


def site_predictions(split, probabilities):
    grouped = {}
    sentinel = object()
    for row, prob in zip_longest(split.metadata(), probabilities, fillvalue=sentinel):
        if row is sentinel or prob is sentinel:
            raise ValueError("Metadata/prediction length mismatch")
        # Preserve the existing scripts' grouping by position, strand and source.
        key = (row["chrom"], int(row["center_genome_pos0"]), row.get("strand", ""), row.get("source", ""))
        label = int(row["label"])
        if key not in grouped:
            grouped[key] = [label, 0.0, 0]
        entry = grouped[key]
        if entry[0] != label:
            raise ValueError(f"Conflicting site labels at {key}")
        entry[1] += float(prob); entry[2] += 1
    rows = []
    for key, (label, total, count) in sorted(grouped.items()):
        chrom, position, strand, source = key
        rows.append(dict(chrom=chrom, center_genome_pos0=position, strand=strand, source=source,
                         split_key=f"{chrom}:{position}", label=label, prob_mean=total/count, n_reads=count))
    return np.array([r["label"] for r in rows]), np.array([r["prob_mean"] for r in rows]), rows


def collate_feature_batch(data):
    # __getitems__ already produces a batch; avoid a second stack/copy.
    return data if isinstance(data, tuple) else default_collate(data)


class FeatureDataLoader(DataLoader):
    """Reuse workers while retaining the old loader's per-epoch RNG draws."""
    def __iter__(self):
        # Standard non-persistent loaders draw a base seed on every iterator.
        # PyTorch persistent workers skip it after the first epoch. Preserve
        # that draw, so shuffle order AND model dropout RNG remain unchanged.
        if self.persistent_workers and getattr(self, "_has_iterated", False):
            torch.empty((), dtype=torch.int64).random_(generator=self.generator).item()
        self._has_iterated = True
        return super().__iter__()


def loader(dataset, training, batch_size, shuffle, device, generator=None, cached=None):
    if cached is not None:
        return feature_caching.GPUCachedLoader(dataset, cached, batch_size, shuffle, generator)
    workers = training["num_workers"] if shuffle else training.get("eval_num_workers", 0)
    options = {}
    if workers:
        options.update(prefetch_factor=training.get("prefetch_factor", 2),
                       persistent_workers=training.get("persistent_workers", True),
                       multiprocessing_context="spawn")
    return FeatureDataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=workers, pin_memory=device.type == "cuda",
                             generator=generator, worker_init_fn=seed_worker,
                             collate_fn=collate_feature_batch, **options)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def batch_forward(model, x, bases, labels, fusion, device):
    x = x.to(device, non_blocking=True)
    if not torch.isfinite(x).all():
        raise ValueError("Input features contain non-finite values")
    y = labels.to(device, non_blocking=True)
    logits = model(x, bases.to(device, non_blocking=True)) if fusion else model(x)
    if fusion:
        loss = nn.functional.cross_entropy(logits, y.long())
        probabilities = torch.softmax(logits, dim=-1)[:, 1]
    else:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        probabilities = torch.sigmoid(logits)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite classifier loss")
    return loss, probabilities


@torch.no_grad()
def predict(model, dataset, training, fusion, device, data_loader=None):
    model.eval()
    labels, probs = [], []
    total_loss = 0.0
    if data_loader is None:
        data_loader = loader(dataset, training, training["eval_batch_size"], False, device)
    for x, bases, y in data_loader:
        loss, p = batch_forward(model, x, bases, y, fusion, device)
        total_loss += float(loss.item()) * len(y)
        labels.append(y.cpu().numpy()); probs.append(p.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probs), total_loss / len(dataset)


def write_tsv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def save_predictions(run_dir, split, probabilities, site_rows):
    fields = ["read_id", "chrom", "center_genome_pos0", "strand", "source", "label", "prob", "split", "seq", "ref"]
    def rows():
        for row, prob in zip(split.metadata(), probabilities):
            yield dict(row, prob=float(prob))
    write_tsv(run_dir / f"{split.name}_predictions.tsv", rows(), fields)
    write_tsv(run_dir / f"{split.name}_site_predictions.tsv", site_rows,
              ["chrom", "center_genome_pos0", "strand", "source", "split_key", "label", "prob_mean", "n_reads"])


def device_name(value):
    if isinstance(value, bool):
        raise ValueError("Invalid device")
    value = str(value)
    if value.isdigit():
        value = f"cuda:{value}"
    if value == "cuda":
        value = "cuda:0"
    if value not in ("auto", "cpu") and not re.fullmatch(r"cuda:[0-9]+", value):
        raise ValueError(f"Unsupported device: {value}; use cpu, auto, or cuda:N")
    if value.startswith("cuda:"):
        value = f"cuda:{int(value.split(':')[1])}"
    return value


def resolve_device(value):
    value = device_name(value)
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; set device: cpu or auto")
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        raise ValueError(f"Device {device} is outside the {torch.cuda.device_count()} visible CUDA devices")
    return device


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# Hash the numerical/data code independently of scheduling and logging. This
# permits changing GPU assignment without invalidating completed experiments.
TRAINING_DEFINITIONS = (
    "positive_integer", "validate_training", "torch_load", "signature", "digest",
    "json_write", "FeatureSplit", "FeatureDataset", "binary_metrics", "site_predictions",
    "collate_feature_batch", "FeatureDataLoader", "loader", "seed_worker", "batch_forward",
    "predict", "write_tsv", "save_predictions", "fit",
)
LEGACY_CODE_HASH = "a892a884f238498983b4e0a421c5774c1870cf7b3bf7e8180c741ca7970c6b26"
# AST serialization in the production PoreDLM Python 3.10 environment.
LEGACY_TRAINING_HASH = "6e9514516a4d0f8d38ec6674fe345da468d4378b27bdf468052503b895d7e56a"


def code_hashes():
    source = Path(__file__).read_bytes()
    models = b"".join(p.read_bytes() for p in sorted((ROOT / "model").glob("*.py")))
    tree = ast.parse(source)
    class WithoutTracking(ast.NodeTransformer):
        def visit_Expr(self, node):
            # These four observation-only hooks do not affect numerical/data
            # semantics. Keep pre-W&B completed experiments reusable.
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "tracking"
                    and call.func.attr in ("update_config", "log_train", "log_epoch", "log_results")):
                return None
            return self.generic_visit(node)
    tree = WithoutTracking().visit(tree)
    numerical = "".join(ast.dump(n, include_attributes=False) for n in tree.body
                        if getattr(n, "name", None) in TRAINING_DEFINITIONS).encode()
    base_codes = json.dumps(BASE_TO_CODE, sort_keys=True).encode()
    cache_code = Path(feature_caching.__file__).read_bytes()
    return dict(source=hashlib.sha256(source + models + Path(tracking.__file__).read_bytes() + cache_code).hexdigest(),
                training=hashlib.sha256(numerical + models + base_codes + cache_code).hexdigest())


def run_configuration(cfg, strategy, seed, splits, device, hashes):
    training = merge_config(cfg["training"], cfg["strategy_overrides"].get(strategy, {}))
    run_config = {k: v for k, v in cfg.items() if k not in ("strategies", "seeds", "strategy_overrides", "skip_completed", "devices")}
    run_config.update(strategy=strategy, seed=seed, training=training, device_resolved=str(device),
                      input_signature={s: d.input_signature for s, d in splits.items()},
                      code_sha256=hashes["source"], training_code_sha256=hashes["training"])
    fingerprint = digest({k: v for k, v in run_config.items()
                          if k not in ("device", "device_resolved", "code_sha256", "wandb")})
    return training, run_config, fingerprint


def completed_result(run_dir, cfg, run_config, fingerprint):
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    previous = json.loads(path.read_text())
    saved = json.loads((run_dir / "config.json").read_text())
    compatible = previous.get("fingerprint") == fingerprint == saved.get("fingerprint")
    # Only the immediately preceding, verified implementation is eligible for
    # legacy reuse; arbitrary older code/configurations are never accepted.
    if (not compatible and saved.get("code_sha256") == LEGACY_CODE_HASH
            and run_config["training_code_sha256"] == LEGACY_TRAINING_HASH):
        old_keys = set(run_config) - {"training_code_sha256", "wandb"}
        old_config = {k: saved.get(k) for k in old_keys}
        compare_keys = old_keys - {"device", "device_resolved", "code_sha256"}
        compatible = (digest(old_config) == previous.get("fingerprint") == saved.get("fingerprint")
                      and all(saved.get(k) == run_config[k] for k in compare_keys))
    if cfg["skip_completed"] and compatible:
        print(f"[SKIP] {run_dir.name}: compatible completed configuration", flush=True)
        return previous
    raise FileExistsError(f"Existing run differs or skip_completed=false: {run_dir}. Choose another experiment name.")


def run_directory(cfg, strategy, seed):
    return Path(cfg["output_dir"]) / cfg["experiment"] / "runs" / f"{strategy}_seed{seed}"


def train_run(cfg, strategy, seed, splits, device, hashes, feature_cache=None):
    training, run_config, fingerprint = run_configuration(cfg, strategy, seed, splits, device, hashes)
    run_dir = run_directory(cfg, strategy, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Run is already active: {run_dir}") from exc
        previous = completed_result(run_dir, cfg, run_config, fingerprint)
        if previous is not None:
            return previous
        if any(p.name != ".run.lock" for p in run_dir.iterdir()):
            raise FileExistsError(f"Incomplete run exists: {run_dir}. Choose another experiment name.")
        with (run_dir / "train.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                with tracking.track_run(cfg, run_config, run_dir):
                    return fit(cfg, strategy, seed, splits, device, training, run_config, fingerprint, run_dir,
                               feature_cache)


def fit(cfg, strategy, seed, splits, device, training, run_config, fingerprint, run_dir, feature_cache=None):
    # Also support callers that run one task directly rather than the scheduler.
    if feature_cache is None:
        with feature_caching.GPUFeatureCache(device) as owned_cache:
            return fit(cfg, strategy, seed, splits, device, training, run_config, fingerprint, run_dir,
                       owned_cache)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = training["deterministic"]
    torch.use_deterministic_algorithms(training["deterministic"])
    # Avoid TF32 token/feature comparison surprises; training heads use FP32.
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(training["cpu_threads"])
    window, hidden_dim = splits["train"].shape[1:]
    fusion = strategy == FUSION_STRATEGY
    datasets = {s: FeatureDataset(d, cfg["fusion_window_size"] if fusion else None,
                                 batch_read=training["batch_read"]) for s, d in splits.items()}
    model = build_model(strategy, hidden_dim, window, cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    cached_splits, cache_info = feature_cache.prepare(splits, training)
    generator = torch.Generator().manual_seed(seed)
    train_loader = loader(datasets["train"], training, training["batch_size"], True, device, generator,
                          cached=cached_splits.get("train"))
    eval_loaders = {"valid": loader(datasets["valid"], training, training["eval_batch_size"], False, device,
                                   cached=cached_splits.get("valid"))}
    run_config.update(fingerprint=fingerprint, source_window_size=window, hidden_dim=hidden_dim,
                      model_window_size=cfg["fusion_window_size"] if fusion else (1 if strategy == STRATEGIES[0] else 5),
                      base_embedding_dim=8 if fusion else None,
                      loss="cross_entropy" if fusion else "binary_cross_entropy_with_logits",
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      gpu_cache=cache_info)
    json_write(run_dir / "config.json", run_config)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    tracking.update_config(run_config)
    print(f"[RUN] {strategy} seed={seed} device={device} parameters={run_config['trainable_parameters']}", flush=True)
    train_workers = 0 if "train" in cached_splits else training["num_workers"]
    print(f"[DATA] batch_read={training['batch_read']} workers={train_workers} "
          f"eval_workers={training['eval_num_workers']} "
          f"prefetch_factor={training['prefetch_factor'] if train_workers else 0} "
          f"persistent_workers={bool(train_workers and training['persistent_workers'])} "
          f"gpu_cache_splits={list(cached_splits)}", flush=True)
    history = []; best_score = -math.inf; best_epoch = 0; stale = 0
    monitor_metric = training["monitor_metric"]
    best_path = run_dir / "best.pt"
    steps_per_epoch = len(train_loader)
    log_every_steps = training["log_every_steps"]
    log_context = f"{strategy} seed={seed} device={device}"
    for epoch in range(1, training["epochs"] + 1):
        model.train(); total_loss = 0.0
        print(f"[EPOCH] {log_context} epoch={epoch}/{training['epochs']} "
              f"steps={steps_per_epoch} batch_size={training['batch_size']} "
              "starting training; waiting for first batch", flush=True)
        epoch_start = interval_start = batch_end = time.perf_counter()
        samples_seen = interval_samples = 0
        interval_loss = data_wait_s = 0.0
        for step, (x, bases, y) in enumerate(train_loader, 1):
            data_wait_s += time.perf_counter() - batch_end
            optimizer.zero_grad(set_to_none=True)
            loss, _ = batch_forward(model, x, bases, y, fusion, device)
            loss.backward()
            if training["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
            optimizer.step()
            batch_loss = float(loss.item())
            batch_samples = len(y)
            total_loss += batch_loss * batch_samples
            samples_seen += batch_samples
            interval_loss += batch_loss * batch_samples
            interval_samples += batch_samples
            if log_every_steps and (step == 1 or step % log_every_steps == 0 or step == steps_per_epoch):
                now = time.perf_counter()
                elapsed_s = now - epoch_start
                interval_s = max(now - interval_start, 1e-9)
                eta_s = elapsed_s / samples_seen * (len(datasets['train']) - samples_seen)
                print(f"[TRAIN] {log_context} epoch={epoch}/{training['epochs']} "
                      f"step={step}/{steps_per_epoch} global_step={(epoch - 1) * steps_per_epoch + step} "
                      f"samples={samples_seen}/{len(datasets['train'])} "
                      f"loss={batch_loss:.6f} interval_loss={interval_loss / interval_samples:.6f} "
                      f"train_loss={total_loss / samples_seen:.6f} lr={optimizer.param_groups[0]['lr']:.6g} "
                      f"samples_per_s={interval_samples / interval_s:.1f} "
                      f"data_wait_s={data_wait_s:.2f} elapsed_s={elapsed_s:.1f} eta_s={eta_s:.1f}",
                      flush=True)
                tracking.log_train(epoch, (epoch - 1) * steps_per_epoch + step,
                                   batch_loss, interval_loss / interval_samples, total_loss / samples_seen,
                                   optimizer.param_groups[0]['lr'], interval_samples / interval_s,
                                   data_wait_s, elapsed_s, eta_s)
                interval_start = time.perf_counter()
                interval_loss = data_wait_s = 0.0
                interval_samples = 0
            batch_end = time.perf_counter()
        print(f"[VALID] {log_context} epoch={epoch}/{training['epochs']} "
              "starting validation", flush=True)
        labels, probs, valid_loss = predict(model, datasets["valid"], training, fusion, device,
                                           data_loader=eval_loaders["valid"])
        valid_metrics = binary_metrics(labels, probs, training["threshold"])
        monitor = None if monitor_metric == "loss" else valid_metrics[monitor_metric]
        effective_monitor = "loss" if monitor is None else monitor_metric
        score = -valid_loss if monitor is None else monitor
        row = dict(epoch=epoch, train_loss=total_loss/len(datasets["train"]), valid_loss=valid_loss,
                   effective_monitor=effective_monitor, **{f"valid_{k}": v for k, v in valid_metrics.items()})
        history.append(row)
        with (run_dir / "history.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        tracking.log_epoch(row, epoch * steps_per_epoch)
        if best_epoch == 0 or score > best_score:
            best_score = score; best_epoch = epoch; stale = 0
            partial = best_path.with_suffix(".pt.partial")
            torch.save({"model": model.state_dict(), "config": run_config, "epoch": epoch}, partial)
            partial.replace(best_path)
        else:
            stale += 1
            if stale >= training["patience"]:
                break
    model.load_state_dict(torch_load(best_path, map_location=device)["model"])
    best_row = history[best_epoch - 1]
    metrics = dict(strategy=strategy, seed=seed, fingerprint=fingerprint, best_epoch=best_epoch,
                   monitor_metric=monitor_metric, effective_monitor=best_row["effective_monitor"],
                   best_valid_metric=best_row["valid_loss"] if best_row["effective_monitor"] == "loss" else best_row[f"valid_{monitor_metric}"],
                   history=history)
    for name in (("train", "valid", "test") if cfg["evaluate_train"] else ("valid", "test")):
        if name not in eval_loaders:
            eval_loaders[name] = loader(datasets[name], training, training["eval_batch_size"], False, device,
                                        cached=cached_splits.get(name))
        labels, probs, loss = predict(model, datasets[name], training, fusion, device,
                                     data_loader=eval_loaders[name])
        metrics[f"{name}_read_level"] = dict(binary_metrics(labels, probs, training["threshold"]), loss=loss)
        sy, sp, rows = site_predictions(splits[name], probs)
        metrics[f"{name}_site_level"] = binary_metrics(sy, sp, training["threshold"])
        if cfg["save_predictions"] == "all" or (cfg["save_predictions"] == "test" and name == "test"):
            save_predictions(run_dir, splits[name], probs, rows)
    for split in splits.values():
        if split.input_signature != [signature(p) for p in split.files]:
            raise ValueError("Input files changed during training")
    json_write(run_dir / "metrics.json", metrics)
    tracking.log_results(metrics, run_config)
    print(f"[DONE] {run_dir}", flush=True)
    return metrics


def summarize(metrics, experiment_dir):
    fields = ["strategy", "seed", "best_epoch"]
    metric_fields = [f"{split}_{level}_{metric}" for split in ("valid", "test")
                     for level in ("read_level", "site_level")
                     for metric in ("accuracy", "f1", "auroc", "auprc")]
    rows = []
    groups = defaultdict(list)
    for result in metrics:
        row = {k: result[k] for k in fields}
        for key in metric_fields:
            split_level, metric = key.rsplit("_", 1)
            row[key] = result[split_level][metric]
            if row[key] is not None:
                groups[(result["strategy"], key)].append(row[key])
        rows.append(row)
    partial = experiment_dir / "summary.tsv.partial"
    write_tsv(partial, rows, fields + metric_fields)
    partial.replace(experiment_dir / "summary.tsv")
    aggregated = []
    for (strategy, metric), values in sorted(groups.items()):
        n = len(values)
        aggregated.append(dict(strategy=strategy, metric=metric, n=n, mean=float(np.mean(values)),
                               std=float(np.std(values, ddof=1)) if n > 1 else None,
                               variance=float(np.var(values, ddof=1)) if n > 1 else None,
                               variance_ddof=1))
    partial = experiment_dir / "aggregate.tsv.partial"
    write_tsv(partial, aggregated,
              ["strategy", "metric", "n", "mean", "std", "variance", "variance_ddof"])
    partial.replace(experiment_dir / "aggregate.tsv")


def load_splits(cfg):
    splits = {s: FeatureSplit(Path(cfg["feature_dir"]), s, Path(cfg["metadata_dir"])) for s in SPLITS}
    shape = splits["train"].shape[1:]
    if shape[0] < 5 or shape[0] % 2 != 1 or any(d.shape[1:] != shape for d in splits.values()):
        raise ValueError("All splits must share an odd feature window >= 5 and the same hidden dimension")
    need_bases = FUSION_STRATEGY in cfg["strategies"]
    if need_bases and cfg["fusion_window_size"] > shape[0]:
        raise ValueError("fusion_window_size exceeds the input feature window")
    sites = {s: d.audit(need_bases, cfg["base_source"]) for s, d in splits.items()}
    if cfg["check_site_overlap"]:
        for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
            overlap = sites[left] & sites[right]
            if overlap:
                raise ValueError(f"Site leakage: {left}/{right} share {len(overlap)} chromosome/position pairs")
    return splits, {s: len(v) for s, v in sites.items()}


def check_active_runs(experiment_dir):
    # Check every run, including runs outside the requested strategy/seed list:
    # an older serial launcher may still have additional jobs scheduled.
    for path in sorted((experiment_dir / "runs").glob("*/.run.lock")):
        with path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Experiment has an active run: {path.parent}. "
                                   "Wait for it to finish or use a new experiment name.") from exc


@contextlib.contextmanager
def experiment_lock(experiment_dir):
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with (experiment_dir / ".scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another launcher owns this experiment: {experiment_dir}") from exc
        check_active_runs(experiment_dir)
        yield


def gpu_worker(worker_id, device_name_, cfg, jobs, events, hashes, input_signature):
    """Spawned process: open inputs locally; never send feature arrays over IPC."""
    job = None
    try:
        if code_hashes() != hashes:
            raise RuntimeError("Training source changed while starting workers")
        device = resolve_device(device_name_)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.set_num_threads(cfg["training"]["cpu_threads"])
        splits, _ = load_splits(cfg)
        if {s: d.input_signature for s, d in splits.items()} != input_signature:
            raise ValueError("Input files changed while starting workers")
        with feature_caching.GPUFeatureCache(device) as feature_cache:
            while True:
                job = jobs.get()
                if job is None:
                    break
                index, strategy, seed = job
                events.put(("start", worker_id, index, device_name_))
                train_run(cfg, strategy, seed, splits, device, hashes, feature_cache)
                # metrics/history stay on disk; only a small status crosses IPC.
                events.put(("done", worker_id, index, device_name_))
        events.put(("exit", worker_id, None, device_name_))
    except BaseException:
        events.put(("error", worker_id, job[0] if job is not None else None, traceback.format_exc()))


def parallel_runs(cfg, pending, devices, hashes, input_signature, results, state, experiment_dir):
    context = mp.get_context("spawn")
    jobs, events = context.Queue(), context.Queue()
    workers = []
    try:
        for job in pending:
            jobs.put(job)
        count = min(len(devices), len(pending))
        for _ in range(count):
            jobs.put(None)
        for worker_id, device in enumerate(devices[:count]):
            worker = context.Process(target=gpu_worker,
                                     args=(worker_id, str(device), cfg, jobs, events, hashes, input_signature),
                                     name=f"trainer-{worker_id}")
            worker.start()
            workers.append(worker)
        finished = 0
        while finished < len(pending):
            try:
                event, worker_id, index, detail = events.get(timeout=1)
            except queue.Empty:
                bad = [w for w in workers if w.exitcode not in (None, 0)]
                if bad or all(not w.is_alive() for w in workers):
                    raise RuntimeError(f"Training workers exited before completing all jobs: "
                                       f"{[(w.name, w.exitcode) for w in workers]}")
                continue
            if event == "error":
                raise RuntimeError(f"Worker {worker_id} failed on task {index}:\n{detail}")
            if event in ("start", "done"):
                task = state["tasks"][index]
                task.update(status="running" if event == "start" else "completed",
                            device=detail, worker=worker_id)
                if event == "done":
                    path = run_directory(cfg, task["strategy"], task["seed"]) / "metrics.json"
                    results[index] = json.loads(path.read_text())
                    finished += 1
                    summarize([results[i] for i in sorted(results)], experiment_dir)
                json_write(experiment_dir / "scheduling.json", state)
                print(f"[SCHEDULE] {event} {task['strategy']} seed={task['seed']} "
                      f"device={detail} completed={len(results)}/{len(state['tasks'])}", flush=True)
        for worker in workers:
            worker.join(timeout=10)
            if worker.exitcode != 0:
                raise RuntimeError(f"Worker {worker.name} did not exit cleanly: {worker.exitcode}")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=10)
        # A terminated consumer may leave an undrained task queue. Do not wait
        # for its feeder thread on failure/interrupt.
        jobs.cancel_join_thread()
        jobs.close()
        events.close()


def execute_runs(cfg, splits, devices, hashes, experiment_dir):
    results = {}
    pending = []
    state = dict(status="running", devices=[str(d) for d in devices], tasks=[])
    for strategy in cfg["strategies"]:
        for seed in cfg["seeds"]:
            index = len(state["tasks"])
            run_dir = run_directory(cfg, strategy, seed)
            _, run_config, fingerprint = run_configuration(cfg, strategy, seed, splits, devices[0], hashes)
            previous = completed_result(run_dir, cfg, run_config, fingerprint)
            task = dict(strategy=strategy, seed=seed, status="skipped" if previous is not None else "pending")
            state["tasks"].append(task)
            if previous is not None:
                results[index] = previous
            else:
                if run_dir.exists() and any(p.name != ".run.lock" for p in run_dir.iterdir()):
                    raise FileExistsError(f"Incomplete run exists: {run_dir}. Choose another experiment name; "
                                          "best.pt does not contain optimizer state for resuming.")
                pending.append((index, strategy, seed))
    # Complete preflight for ALL tasks before starting any training.
    if pending:
        tracking.check_available(cfg)
    json_write(experiment_dir / "scheduling.json", state)
    input_signature = {s: d.input_signature for s, d in splits.items()}
    try:
        if len(devices) > 1 and pending:
            # Workers reopen memmaps, whose file pages are shared by the OS.
            # Avoid keeping a second legacy tensor copy in the parent.
            splits.clear()
            import gc
            gc.collect()
            parallel_runs(cfg, pending, devices, hashes, input_signature, results, state, experiment_dir)
        else:
            with feature_caching.GPUFeatureCache(devices[0]) as feature_cache:
                for index, strategy, seed in pending:
                    state["tasks"][index].update(status="running", device=str(devices[0]), worker=0)
                    json_write(experiment_dir / "scheduling.json", state)
                    results[index] = train_run(cfg, strategy, seed, splits, devices[0], hashes, feature_cache)
                    state["tasks"][index]["status"] = "completed"
                    summarize([results[i] for i in sorted(results)], experiment_dir)
        state["status"] = "completed"
        summarize([results[i] for i in sorted(results)], experiment_dir)
    except BaseException:
        state["status"] = "failed"
        state["error"] = traceback.format_exc()
        for task in state["tasks"]:
            if task["status"] == "running":
                task["status"] = "interrupted"
        raise
    finally:
        json_write(experiment_dir / "scheduling.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "run/example/config.yaml")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, help="Training DataLoader workers per GPU task")
    parser.add_argument("--eval-num-workers", type=int, help="Validation/test DataLoader workers (default: 0)")
    parser.add_argument("--prefetch-factor", type=int, help="Batches prefetched by each worker (default: 2)")
    parser.add_argument("--batch-read", action=argparse.BooleanOptionalAction, default=None,
                        help="Read/convert complete batches (default: enabled)")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None,
                        help="Keep nonzero DataLoader workers alive between epochs (default: enabled)")
    parser.add_argument("--gpu-cache", action=argparse.BooleanOptionalAction, default=None,
                        help="Cache selected feature splits on each training GPU (default: disabled)")
    parser.add_argument("--gpu-cache-splits", nargs="+", choices=SPLITS,
                        help="Splits to cache together; default: train valid")
    parser.add_argument("--gpu-cache-max-gib", type=float, help="Maximum cache GiB per training process")
    parser.add_argument("--gpu-cache-reserve-gib", type=float, help="Keep this many GiB for training/other allocations")
    parser.add_argument("--gpu-cache-chunk-size", type=int, help="Rows copied per bounded CPU-to-GPU chunk")
    parser.add_argument("--gpu-cache-fallback", action=argparse.BooleanOptionalAction, default=None,
                        help="Fall back to original loaders if cache cannot fit; disable for a strict error")
    parser.add_argument("--log-every-steps", type=int,
                        help="Log training progress every N batches, plus first/last batch (default: 50; 0 disables step logs)")
    device_options = parser.add_mutually_exclusive_group()
    device_options.add_argument("--device", help="Single-device override; disables config.devices")
    device_options.add_argument("--devices", nargs="+", help="One worker per GPU, e.g. cuda:0 cuda:1")
    parser.add_argument("--experiment")
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-dir", help="Directory containing metadata_{train,valid,test}.tsv; defaults to feature_dir")
    parser.add_argument("--dry-run", action="store_true", help="Validate all inputs and show tasks without training/writing outputs")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve(), args)
    # Required by deterministic CUDA matmul, set before CUDA context creation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    experiment_dir = Path(cfg["output_dir"]) / cfg["experiment"]
    lock = contextlib.nullcontext() if args.dry_run else experiment_lock(experiment_dir)
    with lock:
        run_main(cfg, args.dry_run, experiment_dir)


def run_main(cfg, dry_run, experiment_dir):
    requested_devices = cfg["devices"] or [cfg["device"]]
    devices = requested_devices if dry_run else [resolve_device(v) for v in requested_devices]
    splits, site_counts = load_splits(cfg)
    print(json.dumps({"input_shapes": {s: d.shape for s, d in splits.items()},
                      "metadata_files": {s: str(d.metadata_path) for s, d in splits.items()},
                      "base_source": cfg["base_source"],
                      "sites": site_counts, "devices": [str(d) for d in devices],
                      "parallelism": "independent strategy/seed experiments",
                      "tasks": len(cfg["strategies"]) * len(cfg["seeds"]),
                      "strategies": cfg["strategies"], "seeds": cfg["seeds"],
                      "gpu_cache": {k: cfg['training'][k] for k in feature_caching.DEFAULTS},
                      "output": str(Path(cfg["output_dir"]) / cfg["experiment"])}, indent=2), flush=True)
    if dry_run:
        return
    execute_runs(cfg, splits, devices, code_hashes(), experiment_dir)


if __name__ == "__main__":
    main()
