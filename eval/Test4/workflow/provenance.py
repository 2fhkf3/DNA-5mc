"""Resolve trained runs and freeze their actual feature protocols, not new defaults."""
from __future__ import annotations
from common import *


def summary_chain(summary):
    values = []; current = summary
    while isinstance(current, dict):
        values.append(current); current = current.get('source_feature_summary')
    return values


def feature_protocol(settings):
    path = Path(settings['feature_summary']); chain = summary_chain(read_json(path))
    def recorded(key, default=None):
        return next((s[key] for s in chain if key in s and s[key] is not None), default)
    family = settings['architecture']
    p = dict(architecture=family, encoder_dir=str((Path(settings['model_root'])/'encoder').resolve()),
             decoder_dir=str((Path(settings['model_root'])/settings['decoder_subdir']).resolve()),
             encoder_layer=recorded('encoder_layer'), token_offset=recorded('token_offset', 128),
             preprocess=recorded('preprocess'), pooling_device=recorded('pooling_device', 'cpu'),
             batch_size=int(recorded('batch_size', 8)), input_window_size=15, hidden_dim=768,
             storage_dtype=recorded('storage_dtype', recorded('dtype', 'float16')))
    if p['encoder_layer'] is None or p['preprocess'] != 'none' or p['storage_dtype'] != 'float16':
        raise ValueError(f'Unsupported/incomplete historical feature protocol: {path}')
    if p['pooling_device'] not in ('cpu', 'model') or p['batch_size'] < 1:
        raise ValueError('Invalid historical performance protocol')
    if family == 'bert':
        p['hidden_layer'] = int(recorded('hidden_layer', -1))
        if p['hidden_layer'] != -1:
            raise ValueError('This workflow currently validates last-layer BERT features only')
    elif family in ('diffusion', 'flow_matching'):
        for key in ('hidden_source', 'ode_steps', 'ode_start_t', 'ode_self_cond_cfg_scale'):
            p[key] = recorded(key)
            if p[key] is None:
                raise ValueError(f'Missing {key} in historical feature summary')
        p['last_t'] = recorded('last_t', 1.)
    elif family == 'gpt':
        layout = recorded('input_layout')
        options = {'mapped codec tokens': 'none', '[BOS, mapped codec tokens, EOS]': 'bos_eos'}
        if layout not in options:
            raise ValueError(f'Unknown historical GPT token layout: {layout}')
        p.update(special_tokens=options[layout], decoder_layers='last', selected_layer_name=None,
                 token_id_mode=recorded('resolved_token_id_mode', 'auto'), tokenizer_file=None,
                 codec_mode=recorded('codec_mode', 'auto'), max_tokens=None, overlong_policy='skip')
    else:
        raise ValueError(f'Unsupported architecture: {family}')
    return p


def weight_files(folder):
    folder = Path(folder); result = []
    for index in folder.glob('*.index.json'):
        data = read_json(index)
        if 'weight_map' in data:
            result.extend(folder/name for name in sorted(set(data['weight_map'].values())))
    if not result:
        result = list(folder.glob('*.safetensors')) or list(folder.glob('pytorch_model*.bin'))
    if not result or any(not p.is_file() for p in result):
        raise FileNotFoundError(f'Model weights missing: {folder}')
    return sorted(set(result))


def inventory(c, content=False):
    runs = []; profiles = {}; inputs = {}
    def add(path):
        p = Path(path).resolve()
        if str(p) not in inputs:
            inputs[str(p)] = signature(p, content)
    for version in c['enabled_versions']:
        setting = c['models'][version]
        profile = feature_protocol(setting); profiles[version] = profile
        add(setting['feature_summary'])
        for key in ('encoder_dir', 'decoder_dir'):
            folder = Path(profile[key]); add(folder/'config.json')
            for file in weight_files(folder):
                add(file)
            for file in folder.rglob('*'):
                if file.is_file() and file.suffix in ('.py', '.json'):
                    add(file)
        base = Path(setting['runs_root'])
        if (base/'runs').is_dir():
            base /= 'runs'
        selected = []
        for path in sorted(base.glob('*/metrics.json')):
            old = read_json(path); cfg = read_json(path.parent/'config.json')
            strategy, seed = cfg.get('strategy'), cfg.get('seed')
            if c['strategies'] and strategy not in c['strategies']:
                continue
            if c['training_seeds'] and seed not in c['training_seeds']:
                continue
            if strategy not in STRATEGIES or type(seed) is not int:
                raise ValueError('Unsupported trained strategy/seed')
            if any(cfg.get(k) != old.get(k) for k in ('strategy', 'seed', 'fingerprint')):
                raise ValueError(f'Checkpoint metadata identity differs: {path}')
            if cfg.get('hidden_dim') != 768 or cfg.get('source_window_size') != 5:
                raise ValueError('Expected existing center-5 x 768 classifier cache protocol')
            if not re.fullmatch(re.escape(version) + r'(?:[-_].*)?', cfg['experiment'], re.I):
                raise ValueError('Run experiment/version mismatch')
            if path.parent.name != f'{strategy}_seed{seed}':
                raise ValueError('Noncanonical run folder name')
            for name in ('best.pt', 'config.json', 'metrics.json'):
                add(path.parent/name)
            selected.append(dict(version=version, strategy=strategy, training_seed=seed,
                run_name=path.parent.name, run_dir=str(path.parent.resolve()),
                checkpoint_sha256=inputs[str((path.parent/'best.pt').resolve())].get('sha256'),
                checkpoint_epoch=old['best_epoch'], configuration=cfg))
        if not selected:
            raise ValueError(f'No completed runs selected: {version}')
        if c['strategies'] and {r['strategy'] for r in selected} != set(c['strategies']):
            raise ValueError(f'Missing requested strategy: {version}')
        if len({(r['strategy'], r['training_seed']) for r in selected}) != len(selected):
            raise ValueError('Duplicate trained run')
        runs.extend(selected)
    scope = {r['strategy'] for r in runs}
    for v in c['enabled_versions']:
        if {r['strategy'] for r in runs if r['version'] == v} != scope:
            raise ValueError('Every enabled backbone must have the same head scope')
        for h in scope:
            seeds = {r['training_seed'] for r in runs if (r['version'], r['strategy']) == (v, h)}
            if c['training_seeds'] and seeds != set(c['training_seeds']):
                raise ValueError('Requested training seed is missing')
    return runs, profiles, list(inputs.values())
