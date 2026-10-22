#!/usr/bin/env python3
"""Seed summaries, paired backbone differences, tables and optional correlation figures."""
from __future__ import annotations
from collections import defaultdict
import html
from common import *


def render_html(rows, common):
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Test4 WGBS correlation</title>',
        '<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}.scroll{overflow:auto}.var{font-size:11px;color:#666}</style>',
        f'<h1>HG002 chr16 与独立 WGBS 的相关性</h1><p>固定 {common["n_sites"]:,} 个位点，{common["n_observations"]:,} 个共同 read–site 观测。上行为三个训练 seed 的均值（按实际冻结 seed 数），括号内为样本方差；模型不做概率集成。</p>']
    if common['partial_test']:
        parts.append('<p><strong>PARTIAL SMOKE：限量测试，不是正式评测。</strong></p>')
    for estimator in ESTIMATORS:
        parts.append('<h2>'+estimator+'</h2><div class="scroll"><table><tr><th>主干</th><th>分类头</th>'+''.join('<th>'+m+'</th>' for m in METRICS)+'</tr>')
        for row in rows:
            if row['estimator'] != estimator:
                continue
            cells = [html.escape(row['version']), html.escape(row['strategy'])]
            for m in METRICS:
                mean, variance = row[m+'.mean'], row[m+'.variance']
                cells.append(('NA' if mean is None else f'{mean:.5f}') + '<div class="var">('+
                             ('NA' if variance is None else f'{variance:.2e}')+')</div>')
            parts.append('<tr>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>')
        parts.append('</table></div>')
    parts.append('<p>beta_call 为固定阈值 read 调用比例，beta_prob 为概率均值。Pearson/Spearman 越大越好，MAE/RMSE 越小越好；Bias 理想为0、斜率为1、截距为0。空间区块区间与训练 seed 方差分别保存在明细中。一个 HG002 样本不构成独立生物重复。</p></html>')
    return '\n'.join(parts)


def plot_run(c, root, stage, item, fixed, estimates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from step4_evaluate_correlation import score_arrays
    scores = score_arrays(root, item, fixed); n = len(fixed['wgbs_beta'])
    selected = np.linspace(0, n-1, min(n, c['plots']['max_points']), dtype=np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), layout='constrained')
    for ax, estimator in zip(axes, ESTIMATORS):
        x, y = fixed['wgbs_beta'][selected], scores[estimator][selected]
        mesh = ax.hexbin(x, y, gridsize=45, extent=(0, 1, 0, 1), mincnt=1,
                         bins='log', vmin=1, vmax=c['plots']['max_points'], cmap='viridis', linewidths=0)
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=.9)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='WGBS methylation fraction', ylabel='Predicted methylation fraction', title=estimator)
        ax.set_aspect('equal'); m = estimates[estimator]
        value = 'NA' if m['pearson_r'] is None else f'{m["pearson_r"]:.4f}'
        ax.text(.04, .96, f'N={n:,}\nr={value}\nMAE={m["mae"]:.4f}', transform=ax.transAxes, va='top', fontsize=8)
        fig.colorbar(mesh, ax=ax, label='Displayed site count (log scale)', shrink=.75)
    fig.suptitle(item['version']+' / '+item['run_name'] + (' [PARTIAL]' if c['scan']['max_records'] else ''), fontsize=10)
    fig.text(.5, -.025, f'Metrics: all fixed sites. Display: same {len(selected):,} deterministic site indices for every run.', ha='center', fontsize=7)
    folder = stage/'figures'/item['version']; folder.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder/(item['run_name']+'.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


def run(c, root, dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=5, output=str(root/'summary'), plots=c['plots'],
              aggregation='metric per fitted seed; then arithmetic mean and sample variance; no ensemble'))); return
    audit = frozen(c, root); common = verify(root/'common', c)
    with lock(root/'.locks/step5.lock'):
        if (root/'summary').exists():
            m = verify(root/'summary', c)
            if m['common_manifest_sha256'] != sha(root/'common/manifest.json'):
                raise ValueError('Existing comparison uses a different cohort')
            return m
        per_seed = []; grouped = defaultdict(list); records = {}; bootstrap = {}; inputs = {}
        for item in audit['runs']:
            folder = root/'metrics'/item['version']/item['run_name']; m = verify(folder, c)
            if m['common_manifest_sha256'] != sha(root/'common/manifest.json') or m['checkpoint_sha256'] != item['checkpoint_sha256']:
                raise ValueError('Metric cohort/checkpoint binding differs')
            inputs[f'{item["version"]}/{item["run_name"]}'] = sha(folder/'manifest.json')
            record = read_json(folder/'metrics.json'); records[item['version'], item['run_name']] = record
            with np.load(folder/'bootstrap_metrics.npz', allow_pickle=False) as data:
                if list(data['metric_names']) != list(BOOT_METRICS):
                    raise ValueError('Bootstrap metric ordering differs')
                for estimator in ESTIMATORS:
                    boot = data[estimator]
                    if boot.shape != (common['bootstrap_repeats_effective'], len(BOOT_METRICS)):
                        raise ValueError('Missing paired bootstrap draws')
                    bootstrap[item['version'], item['run_name'], estimator] = boot
            for estimator, metrics in record['estimates'].items():
                row = dict(version=item['version'], strategy=item['strategy'], training_seed=item['training_seed'],
                           estimator=estimator, n_sites=metrics['n_sites'], **{m: metrics[m] for m in METRICS})
                per_seed.append(row); grouped[item['version'], item['strategy'], estimator].append((item, metrics))
        aggregate = []; mean_draws = {}
        for (version, strategy, estimator), values in sorted(grouped.items()):
            row = dict(version=version, strategy=strategy, estimator=estimator, n_sites=common['n_sites'])
            draws = np.stack([bootstrap[version, item['run_name'], estimator] for item, _ in values])
            # A common spatial draw is valid for the seed-mean statistic only when every seed has a finite metric.
            mean_draws[version, strategy, estimator] = np.mean(draws, axis=0)
            for metric in METRICS:
                stats = sample_summary([m[metric] for _, m in values])
                for k, v in stats.items():
                    row[metric+'.'+k] = v
                ci = (interval(mean_draws[version, strategy, estimator][:, BOOT_METRICS.index(metric)])
                      if metric in BOOT_METRICS else dict(ci_low=None, ci_high=None, ci_status='not_computed_rank_bootstrap', bootstrap_valid=0))
                for k, v in ci.items():
                    row[metric+'.'+k] = v
            aggregate.append(row)
        differences = []
        for strategy in sorted({r['strategy'] for r in audit['runs']}):
            for i, va in enumerate(c['enabled_versions']):
                for vb in c['enabled_versions'][i+1:]:
                    for estimator in ESTIMATORS:
                        a = next(r for r in aggregate if (r['version'], r['strategy'], r['estimator']) == (va, strategy, estimator))
                        b = next(r for r in aggregate if (r['version'], r['strategy'], r['estimator']) == (vb, strategy, estimator))
                        for metric in METRICS:
                            av, bv = a[metric+'.mean'], b[metric+'.mean']
                            ci = interval(mean_draws[vb, strategy, estimator][:, BOOT_METRICS.index(metric)]-
                                          mean_draws[va, strategy, estimator][:, BOOT_METRICS.index(metric)]) if metric in BOOT_METRICS else dict(ci_low=None, ci_high=None, ci_status='not_computed_rank_bootstrap', bootstrap_valid=0)
                            differences.append(dict(model_a=va, model_b=vb, strategy=strategy, estimator=estimator,
                                metric=metric, difference_b_minus_a=bv-av if av is not None and bv is not None else None, **ci))
        with stage_directory(root/'summary') as stage:
            tsv(stage/'correlation_by_seed.tsv', per_seed, list(per_seed[0]))
            tsv(stage/'correlation_mean_variance.tsv', aggregate, list(aggregate[0]))
            tsv(stage/'model_differences.tsv', differences, ['model_a', 'model_b', 'strategy', 'estimator', 'metric', 'difference_b_minus_a', 'ci_low', 'ci_high', 'ci_status', 'bootstrap_valid'])
            compare = stage/'compare'; compare.mkdir()
            with (compare/'correlation_comparison.csv').open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(aggregate[0])); w.writeheader(); w.writerows(aggregate)
            (compare/'correlation_comparison.html').write_text(render_html(aggregate, common))
            funnel = dict(reference=audit['reference_counts'], source_records=windows_count(root),
                          valid_windows_before_models=common['n_windows'],
                          model_common_observations_before_site_coverage=common['model_common_before_site_coverage'],
                          sites_before_model_coverage=common['n_reference_sites'],
                          sites_with_common_reads=common['n_reference_sites_with_windows'],
                          final_paired_sites=common['n_sites'], final_common_observations=common['n_observations'],
                          per_model_window_exclusions=common['windows_excluded_per_backbone'])
            write_json(stage/'data_funnel.json', funnel)
            if c['plots']['enabled']:
                with np.load(root/'common/fixed_sites.npz', allow_pickle=False) as data:
                    fixed = {k: data[k] for k in data.files}
                for item in audit['runs']:
                    plot_run(c, root, stage, item, fixed, records[item['version'], item['run_name']]['estimates'])
            (stage/'README.md').write_text('# Test4 WGBS correlation results\n\n'+
                f'Fixed sites: {common["n_sites"]:,}; common observations: {common["n_observations"]:,}. Partial: {common["partial_test"]}.\n\n'+
                'Main estimator: beta_call; secondary: beta_prob. Metrics are computed separately for every fitted seed; tables report the seed mean and sample variance (ddof=1). Spatial block intervals use shared block draws and remain separate from training-seed variation. Spearman block intervals are not computed.\n\n'+
                '[Comparison](compare/correlation_comparison.html) · [Full seed results](correlation_by_seed.tsv) · [Data funnel](data_funnel.json)\n')
            result = seal(stage, c, n_runs=len(audit['runs']), n_comparisons=len(aggregate),
                          common_manifest_sha256=sha(root/'common/manifest.json'), metric_manifests=inputs,
                          partial_test=common['partial_test'])
        print(f'[DONE] Step5: {len(aggregate)} model/head/estimator rows', flush=True)
        return result


def windows_count(root):
    return read_json(root/'windows/manifest.json')['n_input_records']


if __name__ == '__main__':
    a = parser(__doc__).parse_args(); c, root = load_config(a); run(c, root, a.dry_run)
