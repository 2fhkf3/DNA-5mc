"""Scoped decoder autocast for unified Step4; codec and pooling stay FP32."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
from pathlib import Path

INFERENCE_FUNCTIONS = dict(bert='infer_bert_batch', diffusion='infer_dlm_batch',
                           flow_matching='infer_batch', gpt='infer_batch')


class DecoderPrecision:
    def __init__(self, requested, device):
        import torch
        if requested not in ('fp32', 'fp16'):
            raise ValueError('inference_precision must be fp32 or fp16')
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Unified Step4 precision supports cpu/cuda devices')
        self.torch = torch; self.device = device; self.requested = requested
        self.enabled = requested == 'fp16' and device.type == 'cuda'
        self.effective = 'fp16' if self.enabled else 'fp32'

    def autocast(self):
        # CPU deliberately keeps the established FP32 path, rather than silently
        # substituting CPU BF16 for the requested CUDA FP16 numerical protocol.
        return self.torch.autocast(device_type=self.device.type, dtype=self.torch.float16,
                                   enabled=self.enabled)

    def fp32(self):
        return self.torch.autocast(device_type=self.device.type, enabled=False)

    def metadata(self):
        return dict(requested_inference_precision=self.requested,
                    inference_precision=self.effective,
                    inference_dtype='float16_autocast' if self.enabled else 'float32',
                    inference_autocast_enabled=self.enabled,
                    inference_autocast_dtype='float16' if self.enabled else None,
                    weights_dtype='float32', codec_inference_dtype='float32', pooling_dtype='float32',
                    precision_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


@contextmanager
def bind_decoder_precision(implementation, family, policy):
    """Wrap the existing batch kernel, retaining its masks, pooling and OOM retry.

    Module bindings are restored even on error. Only the unified entry point
    installs this context; standalone and evaluation entry points keep their
    established defaults. Codec calls execute outside these wrappers.
    """
    changes = []
    def replace(owner, name, value):
        changes.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    backend = getattr(implementation, 'shared', implementation) if family == 'diffusion' else implementation
    name = INFERENCE_FUNCTIONS[family]
    original = getattr(backend, name)

    @wraps(original)
    def infer(*args, **kwargs):
        with policy.autocast():
            return original(*args, **kwargs)

    def full_precision_pool(pool):
        @wraps(pool)
        def call(*args, **kwargs):
            with policy.fp32():
                return pool(*args, **kwargs)
        return call

    try:
        replace(backend, name, infer)
        owners = [implementation, getattr(implementation, 'shared', implementation)]
        seen = set()
        for owner in owners:
            if id(owner) not in seen and hasattr(owner, 'pool_hidden_on_model'):
                replace(owner, 'pool_hidden_on_model', full_precision_pool(owner.pool_hidden_on_model))
            seen.add(id(owner))
        yield
    finally:
        for owner, name, value in reversed(changes):
            setattr(owner, name, value)


def annotate_outputs(stage, splits, policy):
    """The existing .pt files are small manifests, not the large feature arrays."""
    torch = policy.torch
    metadata = policy.metadata()
    for row in splits:
        path = Path(stage)/Path(row['pt_path']).name
        manifest = torch.load(path, map_location='cpu', weights_only=False)
        manifest.update(metadata)
        temporary = path.with_name(path.name+'.precision.tmp')
        torch.save(manifest, temporary)
        temporary.replace(path)
        row.update(metadata)
