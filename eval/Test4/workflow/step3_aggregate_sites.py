#!/usr/bin/env python3
"""Intersect physical observations before coverage filtering and continuous site scoring."""
from __future__ import annotations
from common import *


def run(c, root, dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=3, output=str(root/'common'), min_common_reads=c['filters']['min_common_reads'],
              estimators=list(ESTIMATORS), read_call_threshold=c['filters']['read_call_threshold'],
              scope='all frozen backbones/heads/seeds share the same read-site intersection')))
        return
    audit = frozen(c, root); windows = verify(root/'windows', c)
    predictions = {v: verify(root/'predictions'/v, c) for v in c['enabled_versions']}
    for v, m in predictions.items():
        if (m['window_manifest_sha256'] != sha(root/'windows/manifest.json') or
                m['audit_manifest_sha256'] != sha(root/'audit/manifest.json') or
                len(m['parts']) != len(windows['parts'])):
            raise ValueError(f'Predictions do not bind the frozen windows/audit: {v}')
    with lock(root/'.locks/step3.lock'):
        if (root/'common').exists():
            m = verify(root/'common', c)
            if m['prediction_manifests'] != {v: sha(root/'predictions'/v/'manifest.json') for v in predictions}:
                raise ValueError('Prediction inputs changed since aggregation')
            return m
        refs = sqlite3.connect(f'file:{root / "audit/reference.sqlite"}?mode=ro', uri=True)
        refrows = list(refs.execute('SELECT position,coverage,beta FROM sites ORDER BY position')); refs.close()
        positions = np.array([r[0] for r in refrows], dtype=np.int64)
        n_sites, n_runs = len(positions), len(audit['runs'])
        run_columns = {(r['version'], r['run_name']): i for i, r in enumerate(audit['runs'])}
        sources = sqlite3.connect(f'file:{root / "windows/observations.sqlite"}?mode=ro', uri=True)
        with stage_directory(root/'common') as stage:
            sums = np.lib.format.open_memmap(stage/'sum_probability.work.npy', mode='w+', dtype=np.float64, shape=(n_sites, n_runs)); sums[:] = 0
            calls = np.lib.format.open_memmap(stage/'methylated_calls.work.npy', mode='w+', dtype=np.int64, shape=(n_sites, n_runs)); calls[:] = 0
            counts = np.zeros(n_sites, dtype=np.int64); cursor = 0
            exclusions = {v: 0 for v in c['enabled_versions']}; common_before_coverage = 0
            for part in windows['parts']:
                n = part['n_rows']; expected = np.arange(cursor, cursor+n, dtype=np.int64)
                source_rows = list(sources.execute('SELECT idx,position,window_id FROM observations WHERE idx>=? AND idx<? ORDER BY idx', (cursor, cursor+n)))
                if len(source_rows) != n or [r[0] for r in source_rows] != expected.tolist():
                    raise ValueError('Window identity/index order differs')
                identity = digest([r[2] for r in source_rows]); keep = np.ones(n, dtype=bool); arrays = {}
                for v in c['enabled_versions']:
                    name = part['name'].replace('.jsonl.gz', '')
                    folder = root/'predictions'/v/name; m = verify(folder, c)
                    if m['source_sha256'] != part['sha256'] or m['window_identity_sha256'] != identity:
                        raise ValueError('Cross-backbone physical window identities differ')
                    with np.load(folder/'probabilities.npz', allow_pickle=False) as data:
                        idx, valid, p = data['indices'], data['valid'], data['probabilities']
                    nr = len(predictions[v]['run_names'])
                    if not np.array_equal(idx, expected) or valid.shape != (n,) or valid.dtype != np.bool_ or p.shape != (n, nr):
                        raise ValueError('Malformed prediction shard shape/index')
                    if not np.isfinite(p[valid]).all() or np.any((p[valid] < 0) | (p[valid] > 1)) or not np.isnan(p[~valid]).all():
                        raise ValueError('Invalid valid/invalid prediction encoding')
                    keep &= valid; exclusions[v] += int((~valid).sum()); arrays[v] = p
                s = np.array([r[1] for r in source_rows], dtype=np.int64)[keep]
                site_ids = np.searchsorted(positions, s)
                if np.any(site_ids >= n_sites) or not np.array_equal(positions[site_ids], s):
                    raise ValueError('A window has no audited reference site')
                unique, inverse = np.unique(site_ids, return_inverse=True)
                counts[unique] += np.bincount(inverse, minlength=len(unique))
                for v, p in arrays.items():
                    for j, name in enumerate(predictions[v]['run_names']):
                        col = run_columns[v, name]; values = p[keep, j].astype(np.float64)
                        sums[unique, col] += np.bincount(inverse, weights=values, minlength=len(unique))
                        calls[unique, col] += np.bincount(inverse, weights=values >= c['filters']['read_call_threshold'], minlength=len(unique)).astype(np.int64)
                np.savez_compressed(stage/f'common_indices_{cursor:012d}.npz', indices=expected[keep])
                common_before_coverage += int(keep.sum()); cursor += n
            sources.close()
            if cursor != windows['n_windows']:
                raise ValueError('Missing window shards')
            selected = np.flatnonzero(counts >= c['filters']['min_common_reads'])
            if not len(selected):
                raise ValueError('No common site reaches min_common_reads; a short read prefix may have insufficient coverage')
            site_rows = [dict(site_index=j, chrom=c['chromosome'], position=int(positions[i]), strand='+',
                             wgbs_coverage=refrows[i][1], wgbs_beta=refrows[i][2], n_common_reads=int(counts[i]))
                         for j, i in enumerate(selected)]
            tsv(stage/'fixed_sites.tsv', site_rows, list(site_rows[0]))
            excluded = [dict(position=int(positions[i]), n_common_reads=int(counts[i]), reason='below_minimum_common_reads')
                        for i in np.flatnonzero(counts < c['filters']['min_common_reads'])]
            tsv(stage/'excluded_sites.tsv', excluded, ['position', 'n_common_reads', 'reason'])
            np.savez_compressed(stage/'fixed_sites.npz', position=positions[selected],
                wgbs_beta=np.array([refrows[i][2] for i in selected]), wgbs_coverage=np.array([refrows[i][1] for i in selected]),
                n_common_reads=counts[selected])
            for col, item in enumerate(audit['runs']):
                folder = stage/'scores'/item['version']; folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/(item['run_name']+'.npz'), n_methylated_calls=np.array(calls[selected, col]),
                                    beta_prob=np.array(sums[selected, col])/counts[selected])
            del calls, sums
            (stage/'sum_probability.work.npy').unlink(); (stage/'methylated_calls.work.npy').unlink()
            blocks = np.unique(positions[selected]//c['bootstrap']['block_size_bp'])
            R = c['bootstrap']['repeats'] if len(blocks) >= c['bootstrap']['minimum_blocks'] else 0
            weights = np.random.default_rng(c['bootstrap']['seed']).multinomial(len(blocks), np.full(len(blocks), 1/len(blocks)), size=R)
            np.savez_compressed(stage/'bootstrap_weights.npz', blocks=blocks, weights=weights)
            result = seal(stage, c, n_sites=len(selected), n_reference_sites=n_sites,
                n_observations=int(counts[selected].sum()), model_common_before_site_coverage=common_before_coverage,
                windows_excluded_per_backbone=exclusions, n_windows=windows['n_windows'],
                n_reference_sites_with_windows=int((counts > 0).sum()), n_blocks=len(blocks),
                bootstrap_repeats_effective=R, partial_test=windows['partial_test'],
                audit_manifest_sha256=sha(root/'audit/manifest.json'), window_manifest_sha256=sha(root/'windows/manifest.json'),
                prediction_manifests={v: sha(root/'predictions'/v/'manifest.json') for v in predictions},
                common_indices_scope='all model-valid observations before the final site-coverage filter; join fixed_sites to obtain the final cohort')
        print(f'[DONE] Step3: {len(selected)} fixed continuous-beta sites, {result["n_observations"]} common observations', flush=True)
        return result


if __name__ == '__main__':
    a = parser(__doc__).parse_args(); c, root = load_config(a); run(c, root, a.dry_run)
