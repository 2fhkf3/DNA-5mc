"""Resolve Step2 YAML profiles, architecture selection and command overrides."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from common import PROJECT, infer_architecture, read_json, signature, valid_model_name

DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')
GENERAL = dict(data_dir='eval/Test1/data/step1_out/chr19', out_dir=None,
               output_root='eval/Test1/output', device='auto', cpu_threads=4, batch_size=4,
               dtype='float16', window_size=15, preprocess='none', save_every=512, log_every=1000, limit=None)
FAMILY_DEFAULTS = {
    'diffusion': dict(encoder_layer=0, hidden_source='ode_hidden', ode_steps=1, ode_start_t=.97,
                      ode_self_cond_cfg_scale=.5, last_t=1.),
    'bert': dict(encoder_layer=0, hidden_layer=-1, token_offset=128),
    'gpt': dict(encoder_layer=1, decoder_layers='last', selected_layer_name=None, token_id_mode='auto',
                tokenizer_file=None, token_offset=128, special_tokens='none', codec_mode='auto',
                max_tokens=None, overlong_policy='skip'),
}
MODEL_COMMON = {'architecture', 'model_root', 'encoder_dir', 'decoder_dir'}
FAMILY_FIELDS = set().union(*(set(v) for v in FAMILY_DEFAULTS.values()))
PATHS = {'data_dir', 'out_dir', 'output_root', 'model_root', 'encoder_dir', 'decoder_dir', 'tokenizer_file'}


def mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a YAML mapping')
    return dict(value)


def reject_unknown(value, allowed, label):
    unknown = set(value)-set(allowed)
    if unknown:
        raise ValueError(f'Unknown fields in {label}: {sorted(unknown)}')


def as_path(value, from_cli=False):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f'Expected nonempty path, got {value!r}')
    path = Path(value).expanduser()
    return ((Path.cwd() if from_cli else PROJECT)/path).resolve() if not path.is_absolute() else path.resolve()


def decoder_directory(root):
    """Prefer base/, otherwise require one unambiguous non-encoder checkpoint."""
    if (root/'base/config.json').is_file():
        return root/'base'
    candidates = [p.parent for p in root.glob('*/config.json') if p.parent.name != 'encoder']
    if (root/'config.json').is_file():
        candidates.append(root)
    if len(candidates) != 1:
        raise ValueError(f'Cannot uniquely select decoder below {root}; set decoder_dir explicitly')
    return candidates[0]


def resolve(args):
    import yaml
    config_path = Path(args.config).expanduser().resolve()
    before = signature(config_path)
    supplied = mapping(yaml.safe_load(config_path.read_text()), 'config')
    reject_unknown(supplied, {'model', 'extraction', 'models'}, 'config')
    models = mapping(supplied.get('models', {}), 'models')
    version = valid_model_name(args.version if args.version is not None else supplied.get('model'))
    profile = mapping(models.get(version, {}), f'models.{version}')
    reject_unknown(profile, MODEL_COMMON | FAMILY_FIELDS | set(GENERAL), f'models.{version}')
    general = mapping(supplied.get('extraction', {}), 'extraction')
    reject_unknown(general, GENERAL, 'extraction')
    cli = {k: v for k, v in vars(args).items() if v is not None and k not in {'config', 'version', 'print_plan'}}
    settings = {**GENERAL, **general, **profile, **cli, 'version': version}
    for key in PATHS:
        value = settings.get(key)
        if value is not None:
            settings[key] = as_path(value, from_cli=key in cli)
    # The explicit environment override retains the older TEST1_OUTPUT_ROOT convention.
    if 'output_root' not in cli and os.environ.get('TEST1_OUTPUT_ROOT'):
        settings['output_root'] = as_path(os.environ['TEST1_OUTPUT_ROOT'], from_cli=True)
    root = settings.get('model_root')
    if root is None and not (settings.get('encoder_dir') and settings.get('decoder_dir')):
        raise ValueError(f'{version}: configure model_root or both encoder_dir and decoder_dir')
    encoder = settings.get('encoder_dir') or root/'encoder'
    decoder = settings.get('decoder_dir') or decoder_directory(root)
    for path in (encoder/'config.json', decoder/'config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    detected = infer_architecture(read_json(decoder/'config.json'))
    architecture = settings.get('architecture', 'auto')
    if architecture == 'auto':
        if not detected:
            raise ValueError('Cannot infer architecture; explicitly configure diffusion, bert or gpt')
        architecture = detected
    if architecture not in FAMILY_DEFAULTS:
        raise ValueError(f'Unsupported architecture: {architecture!r}')
    if detected and architecture != detected:
        raise ValueError(f'Architecture mismatch: configured {architecture}, decoder config indicates {detected}')
    invalid = (set(profile) | set(cli)) & (FAMILY_FIELDS - set(FAMILY_DEFAULTS[architecture]))
    if invalid:
        raise ValueError(f'Options do not apply to architecture {architecture}: {sorted(invalid)}')
    settings = {**FAMILY_DEFAULTS[architecture], **settings, 'architecture': architecture,
                'encoder_dir': encoder, 'decoder_dir': decoder}
    settings['out_dir'] = settings['out_dir'] or settings['output_root']/'features'/version
    if architecture == 'gpt':
        layers = settings['decoder_layers']
        if isinstance(layers, list):
            if not layers or any(type(v) not in (str, int) for v in layers):
                raise ValueError('decoder_layers list must contain layer numbers or last/embed')
            layers = ','.join(map(str, layers))
        elif type(layers) is int:
            layers = str(layers)
        if not isinstance(layers, str) or not layers.strip():
            raise ValueError('decoder_layers must be last, embed, a layer index or a list')
        layers = layers.strip().lower()
        if (',' in layers or layers == 'all') and not settings['selected_layer_name']:
            raise ValueError('Multiple decoder_layers require selected_layer_name; each output stores one layer')
        settings['decoder_layers'] = layers
    validate(settings)
    if signature(config_path) != before:
        raise ValueError('Configuration changed while being read')
    settings.update(config=config_path, config_signature=before, print_plan=args.print_plan)
    return argparse.Namespace(**settings)


def validate(s):
    for key in ['batch_size', 'cpu_threads', 'window_size', 'save_every']:
        if type(s[key]) is not int or s[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['encoder_layer', 'log_every']:
        if type(s[key]) is not int or s[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if s['window_size']%2 != 1:
        raise ValueError('window_size must be odd')
    if s['limit'] is not None and (type(s['limit']) is not int or s['limit'] < 1):
        raise ValueError('limit must be a positive integer or null')
    if s['dtype'] not in ('float16', 'float32') or s['preprocess'] not in ('none', 'auto'):
        raise ValueError('Invalid dtype or preprocess')
    if not isinstance(s['device'], str) or not s['device'].strip():
        raise ValueError('device must be a nonempty string')
    if s['architecture'] == 'bert':
        if type(s['hidden_layer']) is not int or type(s['token_offset']) is not int:
            raise ValueError('hidden_layer and token_offset must be integers')
        depth = read_json(s['decoder_dir']/'config.json').get('num_hidden_layers')
        if depth is not None and not -(depth+1) <= s['hidden_layer'] <= depth:
            raise ValueError(f'hidden_layer outside available {depth+1} hidden states')
    elif s['architecture'] == 'diffusion':
        if s['encoder_layer'] != 0:
            raise ValueError('Current PoreDLM extractor supports encoder_layer=0')
        if s['hidden_source'] not in ('ode_hidden', 'context_hidden', 'last_hidden', 'last_hidden_state'):
            raise ValueError('Invalid hidden_source')
        if type(s['ode_steps']) is not int or s['ode_steps'] < 1:
            raise ValueError('ode_steps must be a positive integer')
        for key in ['ode_start_t', 'last_t', 'ode_self_cond_cfg_scale']:
            value = s[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'{key} must be a finite number')
        if not 0 < s['ode_start_t'] <= 1 or not 0 < s['last_t'] <= 1 or s['ode_self_cond_cfg_scale'] < 0:
            raise ValueError('Invalid ODE time or CFG scale')
    else:
        for key, allowed in dict(token_id_mode=('auto', 'tokenizer', 'offset', 'direct'),
                                 special_tokens=('none', 'bos_eos'), codec_mode=('auto', 'reference'),
                                 overlong_policy=('skip', 'error')).items():
            if s[key] not in allowed:
                raise ValueError(f'Invalid {key}: {s[key]!r}')
        if type(s['token_offset']) is not int:
            raise ValueError('token_offset must be an integer')
        if s['max_tokens'] is not None and (type(s['max_tokens']) is not int or s['max_tokens'] < 1):
            raise ValueError('max_tokens must be positive or null')
        if s['tokenizer_file'] is not None and not s['tokenizer_file'].is_file():
            raise FileNotFoundError(s['tokenizer_file'])


def snapshot(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'print_plan'}
