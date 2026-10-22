#!/usr/bin/env python3
"""Audit continuous WGBS beta, reference CpGs and frozen classifier provenance."""
from __future__ import annotations
from collections import Counter
from common import *
from provenance import inventory


def reference_sites(c, sequence, destination):
    db = sqlite3.connect(destination)
    db.execute('CREATE TABLE sites(position INTEGER PRIMARY KEY,coverage INTEGER,beta REAL,bed_line INTEGER)')
    counters = Counter(); region = c['scan']['region']
    before = signature(c['inputs']['wgbs_bed'])
    with open_text(c['inputs']['wgbs_bed']) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            counters['bed_records'] += 1
            a = line.rstrip('\r\n').split('\t')
            if len(a) != 11:
                raise ValueError(f'Expected verified 11-column BED at line {line_no}')
            if a[0] != c['chromosome'] or a[5] != '+' or a[3] != 'm':
                continue
            counters['chr16_plus_m_records'] += 1
            try:
                p, end, coverage = int(a[1]), int(a[2]), int(a[9]); percent = float(a[10])
            except (ValueError, OverflowError) as e:
                raise ValueError(f'Malformed BED numeric field at line {line_no}') from e
            if p < 0 or end != p+1 or coverage < 0 or not math.isfinite(percent) or not 0 <= percent <= 100:
                raise ValueError(f'Invalid BED coordinate/coverage/percent at line {line_no}')
            if p >= len(sequence) or sequence[p] != 'C':
                counters['reference_base_mismatch'] += 1; continue
            if p+1 >= len(sequence) or sequence[p:p+2] != 'CG':
                counters['not_reference_CpG'] += 1; continue
            counters['reference_CpG_records'] += 1
            if region and not region[0] <= p < region[1]:
                continue
            if coverage < c['filters']['min_wgbs_coverage']:
                counters['low_reference_coverage'] += 1; continue
            previous = db.execute('SELECT coverage,beta FROM sites WHERE position=?', (p,)).fetchone()
            if previous:
                if previous != (coverage, percent/100):
                    raise ValueError(f'Conflicting WGBS records at {p}; never average duplicate rounded beta')
                counters['identical_duplicate_reference'] += 1; continue
            db.execute('INSERT INTO sites VALUES(?,?,?,?)', (p, coverage, percent/100, line_no))
    db.commit(); n = db.execute('SELECT COUNT(*) FROM sites').fetchone()[0]
    if not n:
        db.close(); raise ValueError('No reference CpGs pass the predeclared filters')
    counters['retained_reference_sites'] = n
    counters['retained_intermediate_beta_sites'] = db.execute('SELECT COUNT(*) FROM sites WHERE beta>0 AND beta<1').fetchone()[0]
    if counters['reference_base_mismatch']:
        db.close()
        raise ValueError('BED positions disagree with the reference CpG context; verify build/coordinate before proceeding')
    db.close(); check_signatures([before])
    return dict(counters)


def run(c, root, dry_run=False):
    runs, profiles, model_inputs = inventory(c, content=False)
    issues = readiness(c)
    if dry_run:
        result = dict(step=0, output=str(root), n_runs=len(runs), versions=c['enabled_versions'],
                      readiness_issues=issues, feature_protocols=profiles,
                      partial_test=c['scan']['max_records'] is not None,
                      note='Read-only inventory; no full BED/signal scan, model loading or large weight hashes')
        print(json.dumps(result, indent=2, ensure_ascii=False)); return result
    if issues:
        raise ValueError('Input readiness is incomplete:\n- ' + '\n- '.join(issues))
    with lock(root/'.locks/step0.lock'):
        if (root/'audit').exists():
            return frozen(c, root)
        if (root/'protocol.yaml').exists() and yaml.safe_load((root/'protocol.yaml').read_text()) != c:
            raise ValueError('An incompatible protocol.yaml already exists')
        root.mkdir(parents=True, exist_ok=True)
        tmp = root/'protocol.yaml.tmp'; tmp.write_text(yaml.safe_dump(c, sort_keys=False)); os.replace(tmp, root/'protocol.yaml')
        runs, profiles, model_inputs = inventory(c, content=True)
        raw_inputs = [signature(c['inputs']['signal_jsonl']), signature(c['inputs']['wgbs_bed'], True),
                      signature(c['reference']['fasta'], True)]
        sequence = load_chromosome(c['reference']['fasta'], c['chromosome'])
        with stage_directory(root/'audit') as stage:
            counts = reference_sites(c, sequence, stage/'reference.sqlite')
            write_json(stage/'inventory.json', dict(runs=runs, profiles=profiles))
            tsv(stage/'run_inventory.tsv', runs, ['version', 'strategy', 'training_seed', 'run_name', 'run_dir', 'checkpoint_sha256'])
            write_json(stage/'environment.json', dict(python=sys.version, executable=sys.executable,
                       numpy=np.__version__, pyyaml=yaml.__version__))
            check_signatures([*raw_inputs, *model_inputs])
            result = seal(stage, c, runs=runs, profiles=profiles, inputs=[*raw_inputs, *model_inputs],
                          reference_counts=counts, reference_chromosome_length=len(sequence),
                          partial_test=c['scan']['max_records'] is not None, code_sha256=code_hashes())
        print(f'[DONE] Step0: {len(runs)} runs, {counts["retained_reference_sites"]} continuous reference CpGs', flush=True)
        return result


if __name__ == '__main__':
    a = parser(__doc__).parse_args(); c, root = load_config(a); run(c, root, a.dry_run)
