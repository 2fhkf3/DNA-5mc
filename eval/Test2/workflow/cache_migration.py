"""Explicit, frozen bindings between training-cache paths and relocated caches.

The original training records have size/mtime signatures, not content hashes.
Migration records preserve those records and establish a new checked baseline;
they cannot establish historical byte identity of the large feature arrays.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import importlib.util
from pathlib import Path

# Keep the binding checks independent of the importing workflow's `common`
# module, so Test3 can reuse the exact same records and validation rules.
PROJECT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('_cache_migration_io', PROJECT/'eval/Test1/workflow/common.py')
_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_io)
for _name in ('infer_architecture', 'read_json', 'sha256', 'signature', 'write_json'):
    globals()[_name] = getattr(_io, _name)


@lru_cache(maxsize=128)
def _digest(path, size, mtime_ns):
    """Hash once per current file signature, including shared inputs across runs."""
    result = sha256(path)
    after = signature(path)
    if (after['size'], after['mtime_ns']) != (size, mtime_ns):
        raise ValueError(f'Cache changed while hashing: {path}')
    return result


def checked_hash(path):
    s = signature(path)
    return _digest(s['path'], s['size'], s['mtime_ns'])


def original_signatures(cfg):
    root = Path(cfg['feature_dir']).resolve()
    signed = cfg.get('input_signature', {}).get('test', [])
    if len(signed) < 4 or len({s['path'] for s in signed}) != len(signed):
        raise ValueError('Missing or duplicate original test input signatures')
    for s in signed:
        if Path(s['path']).parent != root or Path(s['path']).name in ('', '.', '..'):
            raise ValueError('Original test signature is outside the recorded training cache')
    return {Path(s['path']).name: s for s in signed}


def audit_binding(c, version, cfg, *, verify_content=True):
    """Keep strict old-path checks, or require an explicit audited migration."""
    settings = c['models'][version]
    root = Path(settings['feature_dir']).resolve()
    original_root = Path(cfg['feature_dir']).resolve()
    signed = original_signatures(cfg)
    paths = [root / name for name in sorted(set(signed) | {'feature_summary.json'})]
    actual = {p.name: signature(p) for p in paths}
    common = dict(training_feature_dir=str(original_root), feature_dir=str(root),
                  original_input_signatures=list(signed.values()), current_input_signatures=list(actual.values()))
    if root == original_root:
        for name, s in signed.items():
            if actual[name] != s:
                raise ValueError(f'Original test feature signature differs from training: {root/name}')
        return dict(common, relocated=False, audit_inputs=[],
                    historical_identity_scope='original path/size/mtime signatures; no historical content hashes')
    manifest_path = settings.get('cache_migration_manifest')
    if not manifest_path:
        raise ValueError(f'{version}: relocated cache requires models.{version}.cache_migration_manifest')
    manifest_path = Path(manifest_path)
    manifest_sig = signature(manifest_path)
    m = read_json(manifest_path)
    if (m.get('schema') != 'test2_cache_migration_v1' or m.get('version') != version
            or m.get('architecture') != settings['architecture']
            or m.get('original_root') != str(original_root) or m.get('relocated_root') != str(root)):
        raise ValueError(f'Cache migration identity/path mismatch: {manifest_path}')
    if m.get('original_input_signatures') != list(signed.values()):
        raise ValueError(f'Cache migration does not match training input signatures: {version}')
    if m.get('relocated_input_signatures') != list(actual.values()):
        raise ValueError(f'Relocated cache changed since its migration audit: {version}; re-audit the migration explicitly')
    archive = Path(m['protocol_archive']['path'])
    if checked_hash(archive) != m['protocol_archive']['sha256']:
        raise ValueError(f'Training protocol archive changed: {archive}')
    archived = read_json(archive)
    if (archived.get('operation') != 'select_native_test_label_zero'
            or Path(archived.get('source_feature_dir', '')).resolve() != original_root
            or archived.get('source_feature_summary') != read_json(root/'feature_summary.json')):
        raise ValueError(f'Relocated feature protocol differs from its archived training source: {version}')
    if set(m.get('file_sha256', {})) != set(actual) - {m.get('feature_array_name')}:
        raise ValueError(f'Incomplete migration content hashes: {manifest_path}')
    for name, expected in m['file_sha256'].items():
        # Discovery/planning need not stream hundreds of MB of metadata. Full
        # evaluation retains all content checks, cached once per file signature.
        if not verify_content and name not in ('test.pt', 'feature_summary.json'):
            continue
        if checked_hash(root/name) != expected:
            raise ValueError(f'Relocated cache content hash mismatch: {root/name}')
    for name, s in signed.items():
        if actual[name]['size'] != s['size']:
            raise ValueError(f'Relocated cache size differs from original: {root/name}')
    return dict(common, relocated=True, migration_manifest=manifest_sig,
                migration_manifest_sha256=checked_hash(manifest_path),
                content_hashes_verified=verify_content,
                audit_inputs=[manifest_sig, signature(archive)],
                historical_identity_scope=m['historical_identity_scope'])


def audit_inputs(protocol, paths):
    """Include migration evidence and original source files in change detection."""
    binding = protocol['cache_binding']
    signed = [*binding['current_input_signatures'], *binding['audit_inputs'], *map(signature, paths)]
    return list({s['path']: s for s in signed}.values())


def prepare(c, version, refresh=False):
    # Import lazily so normal evaluations do not run preparation or import torch here.
    from step2_evaluate import cache, discover
    settings = c['models'][version]
    root = Path(settings['feature_dir']).resolve()
    destination = Path(settings['cache_migration_manifest'])
    runs = discover(c, version)
    configs = [read_json(p/'config.json') for p in runs]
    cfg = configs[0]
    original_root = Path(cfg['feature_dir']).resolve()
    if root == original_root:
        raise ValueError(f'{version}: cache is not relocated')
    signed = original_signatures(cfg)
    if any(q['feature_dir'] != cfg['feature_dir'] or original_signatures(q) != signed for q in configs):
        raise ValueError(f'Training runs do not share one frozen source cache: {version}')
    archive = PROJECT/'eval/Test1/data/step3_output/chr16-native'/f'{version}-V2'/'feature_summary.json'
    archived = read_json(archive)
    old_inputs = archived.get('test1_dataset', {}).get('source_signatures', [])
    summary = read_json(root/'feature_summary.json')
    if (archived.get('operation') != 'select_native_test_label_zero'
            or Path(archived.get('source_feature_dir', '')).resolve() != original_root
            or any(s not in old_inputs for s in signed.values())
            or archived.get('source_feature_summary') != summary):
        raise ValueError(f'Archive cannot bind the relocated cache to original training inputs: {version}')
    if infer_architecture(summary) != settings['architecture']:
        raise ValueError(f'Relocated cache architecture mismatch: {version}')
    meta, _, paths = cache(root)
    expected_names = {p.name for p in paths[:-1]}
    if set(signed) != expected_names:
        raise ValueError(f'Training signatures and relocated cache filenames differ: {version}')
    if any(meta['feature_shape'][1:] != [q['source_window_size'], q['hidden_dim']] for q in configs):
        raise ValueError(f'Relocated feature dimensions do not match classifiers: {version}')
    for p in paths[:-1]:
        if p.stat().st_size != signed[p.name]['size']:
            raise ValueError(f'Relocated size differs from training cache: {p}')
    # Pin all files, hashing the small header/labels/metadata/summary. The large
    # feature array retains size/mtime checks; its historical bytes are unproven.
    current = [signature(root/n) for n in sorted(expected_names | {'feature_summary.json'})]
    hashes = {p.name: checked_hash(p) for p in paths if p.name != meta['feature_path']}
    m = dict(schema='test2_cache_migration_v1', generated_at_utc=datetime.now(timezone.utc).isoformat(),
             version=version, architecture=settings['architecture'], original_root=str(original_root), relocated_root=str(root),
             original_input_signatures=list(signed.values()), relocated_input_signatures=current,
             feature_array_name=meta['feature_path'], file_sha256=hashes,
             protocol_archive=dict(path=str(archive.resolve()), sha256=checked_hash(archive)),
             feature_shape=meta['feature_shape'], training_runs=[str(p.resolve()) for p in runs],
             historical_identity_scope='Original size/mtime signatures match the pre-migration archive; current sizes and archived feature protocol agree. New hashes pin current header/labels/metadata/summary, not historical bytes. Large feature array uses frozen size/mtime; historical byte identity is unverified. Saved predictions are checked row-by-row against current metadata and labels during evaluation.')
    if [signature(Path(s['path'])) for s in current] != current:
        raise ValueError(f'Cache changed during migration audit: {version}')
    if destination.exists() and not refresh:
        audit_binding(c, version, cfg)
        print(f'[UNCHANGED] {version}: existing migration binding verified', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, m)
    audit_binding(c, version, cfg)
    print(f'[MIGRATION] {version}: {original_root} -> {root}; {len(runs)} runs verified', flush=True)


def main():
    from common import cli, load_config
    parser = cli('Prepare explicit cache-migration bindings; never changes NAS or training outputs.')
    parser.add_argument('--refresh', action='store_true', help='重新审计并更新现有迁移记录')
    args = parser.parse_args()
    if args.dry_run or args.max_records is not None or args.out_dir:
        parser.error('Preparation requires full cache headers and fixed manifest destinations; omit --dry-run/--max-records/--out-dir')
    c, versions = load_config(args)
    for version in versions:
        prepare(c, version, args.refresh)


if __name__ == '__main__':
    main()
