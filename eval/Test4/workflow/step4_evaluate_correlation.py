#!/usr/bin/env python3
"""Continuous site correlation, fixed strata and exact paired block-moment bootstrap."""
from __future__ import annotations
from common import *


def score_arrays(root, item, fixed):
    with np.load(root/'common/scores'/item['version']/(item['run_name']+'.npz'), allow_pickle=False) as data:
        calls, probability = data['n_methylated_calls'], data['beta_prob']
    n = fixed['n_common_reads']
    if calls.shape != n.shape or probability.shape != n.shape or np.any(calls < 0) or np.any(calls > n):
        raise ValueError('Invalid methylated/total read counts')
    return dict(beta_call=calls/n, beta_prob=probability)


def strata(fixed):
    x = fixed['wgbs_beta']
    masks = [('all', np.ones(len(x), dtype=bool)), ('beta_intermediate', (x > 0) & (x < 1)),
             ('beta_midrange', (x > .2) & (x < .8)), ('beta_low', x <= .2),
             ('beta_middle', (x > .2) & (x < .8)), ('beta_high', x >= .8)]
    for field in ('wgbs_coverage', 'n_common_reads'):
        a = fixed[field]
        for lo, hi in [(5, 9), (10, 19), (20, 49), (50, None)]:
            masks.append((f'{field}_{lo}_{hi or "plus"}', (a >= lo) & ((a <= hi) if hi is not None else True)))
    return masks


def evaluate(c, root, item, fixed, common, blocks, weights):
    destination = root/'metrics'/item['version']/item['run_name']
    if destination.exists():
        m = verify(destination, c)
        if m['common_manifest_sha256'] != sha(root/'common/manifest.json'):
            raise ValueError('Existing metrics use another fixed cohort')
        return m
    scores = score_arrays(root, item, fixed); x = fixed['wgbs_beta']; rows = []; stats = {}; boot_arrays = {}
    group = np.searchsorted(blocks, fixed['position']//c['bootstrap']['block_size_bp'])
    with stage_directory(destination) as stage:
        for estimator, y in scores.items():
            stats[estimator] = correlation_metrics(x, y)
            for name, mask in strata(fixed):
                rows.append(dict(estimator=estimator, stratum=name, **correlation_metrics(x[mask], y[mask])))
            terms = np.array([np.ones(len(x)), x, y, x*x, y*y, x*y, np.abs(y-x), (y-x)**2])
            sums = np.stack([np.bincount(group, weights=t, minlength=len(blocks)) for t in terms], axis=1)
            draws = weights @ sums
            draw_metrics = [moment_metrics(row) for row in draws]
            results = np.array([[row[m] if row[m] is not None else np.nan for m in BOOT_METRICS]
                                for row in draw_metrics], dtype=float).reshape(len(draws), len(BOOT_METRICS))
            boot_arrays[estimator] = results
            stats[estimator]['intervals'] = {m: interval(results[:, i]) for i, m in enumerate(BOOT_METRICS)}
            stats[estimator]['intervals']['spearman_rho'] = dict(ci_low=None, ci_high=None,
                ci_status='not_computed_rank_bootstrap_requires_raw_rank_resampling', bootstrap_valid=0)
        np.savez_compressed(stage/'bootstrap_metrics.npz', metric_names=np.array(BOOT_METRICS), **boot_arrays)
        write_json(stage/'metrics.json', dict(version=item['version'], strategy=item['strategy'], training_seed=item['training_seed'],
                  n_common_observations=common['n_observations'], partial_test=common['partial_test'],
                  common_manifest_sha256=sha(root/'common/manifest.json'), estimates=stats))
        tsv(stage/'stratified_metrics.tsv', rows, ['estimator', 'stratum', 'n_sites', *METRICS, 'correlation_status'])
        result = seal(stage, c, version=item['version'], run_name=item['run_name'],
            checkpoint_sha256=item['checkpoint_sha256'], common_manifest_sha256=sha(root/'common/manifest.json'),
            partial_test=common['partial_test'], n_sites=common['n_sites'])
    return result


def run(c, root, version=None, run_name=None, dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=4, version=version, run_name=run_name, metrics=list(METRICS),
              output=str(root/'metrics'), bootstrap=c['bootstrap']))); return
    audit = frozen(c, root); common = verify(root/'common', c)
    with np.load(root/'common/fixed_sites.npz', allow_pickle=False) as data:
        fixed = {k: data[k] for k in data.files}
    with np.load(root/'common/bootstrap_weights.npz', allow_pickle=False) as data:
        blocks, weights = data['blocks'], data['weights']
    selected = [r for r in audit['runs'] if (version is None or r['version'] == version) and (run_name is None or r['run_name'] == run_name)]
    if not selected:
        raise ValueError('No matching frozen run')
    for item in selected:
        with lock(root/f'.locks/evaluate_{item["version"]}_{item["run_name"]}.lock'):
            evaluate(c, root, item, fixed, common, blocks, weights)
    print(f'[DONE] Step4: {len(selected)} runs; paired site metrics and block intervals', flush=True)


if __name__ == '__main__':
    p = parser(__doc__); p.add_argument('--version'); p.add_argument('--run-name'); a = p.parse_args()
    c, root = load_config(a); run(c, root, a.version, a.run_name, a.dry_run)
