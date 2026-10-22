"""Center-crop pooled positions before storage, retaining full inference context."""
from __future__ import annotations

def output_length(input_size, output_size):
    if output_size is None:
        return input_size
    if (type(output_size) is not int or output_size < 5 or output_size % 2 != 1
            or output_size > input_size or input_size % 2 != 1):
        raise ValueError('output_window_size must be an odd integer >=5 and <= window_size')
    return output_size


class CenterCropWriterMixin:
    """Shared by the three streaming writers (Flow uses the Diffusion writer).

    The writer receives full-window pooled features, but its buffers, byte-size
    checks and manifest use the output width. No full-window binary is written.
    Without an output width the standalone extractors retain their old behavior.
    """

    def initialize_center_crop(self, input_size, output_size):
        self.input_window_size = input_size
        self.window_size = output_length(input_size, output_size)
        self.crop_start = (input_size - self.window_size) // 2
        self.crop_end = self.crop_start + self.window_size
        self.cropping = self.window_size != input_size
        if self.cropping:
            self.feature_source += f'_center{self.window_size}'

    def output_metadata_fields(self, fields):
        if not self.cropping:
            return fields
        return fields + ['crop_input_window_size', 'crop_start0', 'crop_end0'] + [
            f'crop_input_{key}' for key in ('seq', 'ref') if key in fields]

    def prepare_output(self, feature, label, metadata):
        if not self.cropping:
            return feature, metadata
        import numpy as np
        expected = (self.input_window_size, self.hidden_size)
        if feature.shape != expected:
            raise ValueError(f'Full input feature shape must be {expected}, got {feature.shape}')
        if label not in (0, 1) or int(metadata['label']) != label:
            raise ValueError('Feature and metadata labels must match and be binary')
        if not str(metadata['read_id']).strip() or metadata.get('split', self.split) != self.split:
            raise ValueError('Invalid read_id or split in crop metadata')
        if not np.isfinite(feature).all():
            raise ValueError('Non-finite pooled features')
        row = dict(metadata)
        for key in ('seq', 'ref'):
            if key not in row:
                continue
            sequence = row[key].strip()
            if (sequence or key == 'ref') and len(sequence) != self.input_window_size:
                raise ValueError(f'{key} length does not match input window_size={self.input_window_size}')
            row[f'crop_input_{key}'] = row[key]
            row[key] = sequence[self.crop_start:self.crop_end] if sequence else ''
        row.update(crop_input_window_size=self.input_window_size,
                   crop_start0=self.crop_start, crop_end0=self.crop_end)
        # Copy to ensure a cropped view does not retain a full float32 window
        # in the write buffer when storage dtype is also float32.
        cropped = np.array(feature[self.crop_start:self.crop_end], dtype=self.feature_dtype,
                           order='C', copy=True)
        if not np.isfinite(cropped).all():
            raise ValueError('Pooled features overflow the requested storage dtype')
        return cropped, row

    def center_crop_metadata(self):
        if not self.cropping:
            return {}
        return dict(input_window_size=self.input_window_size, output_window_size=self.window_size,
                    center_crop=dict(source_feature_shape=[self.records, self.input_window_size, self.hidden_size],
                                     start0=self.crop_start, end0=self.crop_end,
                                     previous_crop=None, preserves_original_bert_context=True,
                                     preserves_original_model_context=True,
                                     implementation='in_memory_before_storage'))
