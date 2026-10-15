"""Process-local CUDA feature cache; preserve source precision and CPU sampling."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


GIB = 1024 ** 3
DEFAULTS = {
    "gpu_cache": False,
    "gpu_cache_splits": ["train", "valid"],
    "gpu_cache_max_gib": 40.0,
    "gpu_cache_reserve_gib": 4.0,
    "gpu_cache_chunk_size": 8192,
    "gpu_cache_fallback": True,
}


def validate_settings(settings):
    for key in ("gpu_cache", "gpu_cache_fallback"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    names = settings["gpu_cache_splits"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(s, str) or s not in ("train", "valid", "test") for s in names)
            or len(set(names)) != len(names)):
        raise ValueError("gpu_cache_splits must be a nonempty list of distinct train/valid/test names")
    for key, minimum in (("gpu_cache_max_gib", 0), ("gpu_cache_reserve_gib", 0)):
        value = settings[key]
        if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
                or (key == "gpu_cache_max_gib" and value == 0)):
            raise ValueError(f"{key} must be finite and {'> 0' if key == 'gpu_cache_max_gib' else '>= 0'}")
    value = settings["gpu_cache_chunk_size"]
    if type(value) is not int or value < 1:
        raise ValueError("gpu_cache_chunk_size must be an integer >= 1")


def required_bytes(split):
    """Full source window, FP32 labels and optional uint8 base codes."""
    size = math.prod(split.shape) * split.x.dtype.itemsize + split.shape[0] * 4
    if split.base_ids is not None:
        size += split.base_ids.size  # audit() stores uint8 base codes.
    return size


@dataclass
class CachedSplit:
    features: torch.Tensor
    labels: torch.Tensor
    bases: torch.Tensor | None
    identity: tuple
    nbytes: int


class GPUFeatureCache:
    """One cache per independent training worker, never passed to DataLoader IPC."""
    def __init__(self, device):
        self.device = torch.device(device)
        self.entries = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.clear()

    def clear(self):
        had_entries = bool(self.entries)
        self.entries.clear()
        if had_entries and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    @staticmethod
    def identity(split):
        return (tuple((s["path"], s["size"], s["mtime_ns"]) for s in split.input_signature),
                tuple(split.shape), split.x.dtype.str, split.base_ids is not None)

    def copy_split(self, split, chunk_size):
        # Allocate once, then copy bounded chunks; never materialize the full
        # memmap as a host tensor or quantize an FP32 source into FP16.
        dtype = torch.from_numpy(np.empty(0, dtype=split.x.dtype)).dtype
        features = torch.empty(split.shape, dtype=dtype, device=self.device)
        labels = torch.empty(split.shape[0], dtype=torch.float32, device=self.device)
        bases = (torch.empty(split.base_ids.shape, dtype=torch.uint8, device=self.device)
                 if split.base_ids is not None else None)
        started = last_log = time.perf_counter()
        for start in range(0, split.shape[0], chunk_size):
            end = min(start + chunk_size, split.shape[0])
            features[start:end].copy_(torch.from_numpy(np.array(split.x[start:end], copy=True, order="C")))
            labels[start:end].copy_(torch.from_numpy(np.array(split.y[start:end], dtype=np.float32, copy=True)))
            if bases is not None:
                bases[start:end].copy_(torch.from_numpy(np.array(split.base_ids[start:end], copy=True)))
            now = time.perf_counter()
            if start == 0 or end == split.shape[0] or now - last_log >= 5:
                print(f"[GPU-CACHE] loading split={split.name} rows={end}/{split.shape[0]} "
                      f"dtype={dtype} elapsed_s={now-started:.1f}", flush=True)
                last_log = now
        return CachedSplit(features, labels, bases, self.identity(split), required_bytes(split))

    def _decline(self, settings, reason):
        self.clear()
        message = f"[GPU-CACHE] {reason}"
        if not settings["gpu_cache_fallback"]:
            raise RuntimeError(message + "; gpu_cache_fallback=false")
        print(message + "; falling back to the original CPU/memmap loaders", flush=True)
        return {}, dict(requested=True, active=False, splits=[], bytes=0, reason=reason)

    def prepare(self, splits, settings):
        if not settings["gpu_cache"]:
            self.clear()
            return {}, dict(requested=False, active=False, splits=[], bytes=0)
        if self.device.type != "cuda":
            return self._decline(settings, "GPU caching requires a CUDA device")
        names = settings["gpu_cache_splits"]
        needed = sum(required_bytes(splits[s]) for s in names)
        limit = int(settings["gpu_cache_max_gib"] * GIB)
        reserve = int(settings["gpu_cache_reserve_gib"] * GIB)
        if needed > limit:
            return self._decline(settings, f"requested cache {needed/GIB:.2f} GiB exceeds "
                                 f"gpu_cache_max_gib={settings['gpu_cache_max_gib']}")
        for name in list(self.entries):
            if name not in names or self.entries[name].identity != self.identity(splits[name]):
                del self.entries[name]
        missing = [s for s in names if s not in self.entries]
        new_bytes = sum(required_bytes(splits[s]) for s in missing)
        # Driver free memory excludes blocks reserved but unused by this
        # process's PyTorch allocator; those can also satisfy cache allocations.
        free, _ = torch.cuda.mem_get_info(self.device)
        reusable = max(0, torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device))
        if new_bytes + reserve > free + reusable:
            return self._decline(settings, f"additional cache needs {new_bytes/GIB:.2f} GiB plus "
                                 f"{reserve/GIB:.2f} GiB reserve, available {(free+reusable)/GIB:.2f} GiB")
        print(f"[GPU-CACHE] device={self.device} splits={names} total_gib={needed/GIB:.2f} "
              f"reused={len(names)-len(missing)} loading={len(missing)}", flush=True)
        failure = None
        try:
            for name in missing:
                self.entries[name] = self.copy_split(splits[name], settings["gpu_cache_chunk_size"])
        except torch.cuda.OutOfMemoryError as exc:
            # Leave the except block before cleanup: its traceback can retain
            # partially allocated tensors from copy_split().
            failure = str(exc).splitlines()[0] if str(exc) else "allocation failed"
        if failure is not None:
            self.clear()
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            return self._decline(settings, "CUDA out of memory while loading cache: " + failure)
        info = dict(requested=True, active=True, splits=list(names), bytes=needed,
                    storage_dtypes={s: str(self.entries[s].features.dtype) for s in names},
                    reserve_gib=settings["gpu_cache_reserve_gib"])
        return dict(self.entries), info


class _IndexDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index

    def __getitems__(self, indices):
        return torch.as_tensor(indices, dtype=torch.long)


def _collate_indices(indices):
    return torch.as_tensor(indices, dtype=torch.long)


class GPUCachedLoader:
    """CPU DataLoader RNG/sampler semantics, GPU gather and FP32 batches.

    A zero-worker index loader draws the same per-iterator base seed as the
    original FeatureDataLoader (including its persistent-worker adjustment).
    Only small index batches cross the CPU/GPU boundary; feature reads do not.
    """
    def __init__(self, dataset, cached, batch_size, shuffle, generator=None):
        self.cached = cached
        self.start, self.end, self.fusion = dataset.start, dataset.end, dataset.fusion
        if self.fusion and cached.bases is None:
            raise ValueError("Fusion cache requires audited base codes")
        self.indices = DataLoader(_IndexDataset(len(dataset)), batch_size=batch_size, shuffle=shuffle,
                                  num_workers=0, pin_memory=False, generator=generator,
                                  collate_fn=_collate_indices)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        cache = self.cached
        for indices in self.indices:
            indices = indices.to(cache.features.device)
            x = cache.features[:, self.start:self.end].index_select(0, indices).to(torch.float32)
            y = cache.labels.index_select(0, indices)
            bases = (cache.bases[:, self.start:self.end].index_select(0, indices).long()
                     if self.fusion else torch.empty((len(indices), 0), dtype=torch.long, device=x.device))
            yield x, bases, y
