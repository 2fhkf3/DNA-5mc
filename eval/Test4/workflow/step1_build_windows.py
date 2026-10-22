#!/usr/bin/env python3
"""Single-pass, resumable unlabeled CpG signal windows; no beta-based balancing."""
from __future__ import annotations
from collections import Counter
import bisect
from common import *


def original_read_id(record):
    value = record.get('fast5_raw_read_id') or record.get('read_id')
    if not value:
        raise ValueError('Missing original read identity')
    return re.sub(r':[1-9]\d*:[1-9]\d*$', '', str(value)) if not record.get('fast5_raw_read_id') else str(value)


def extract_window(record, position, mapping, sequence, c):
    signal = record.get('signal'); spans = record.get('base_sample_span_seq'); seq = record.get('seq')
    if not isinstance(signal, list) or not signal or not isinstance(spans, list) or not isinstance(seq, str):
        return None, 'missing_signal_or_mapping'
    items = [mapping.get(p) for p in range(position-7, position+8)]
    if position < 7 or position+8 > len(sequence) or any(x is None for x in items):
        return None, 'incomplete_15nt_window'
    ref = sequence[position-7:position+8]
    query = []; ref_spans = []; seq_spans = []
    for i, item in enumerate(items):
        q = item.get('query_pos_signal'); span = item.get('signal_span')
        if type(q) is not int or not 0 <= q < len(seq) or q >= len(spans):
            return None, 'invalid_query_position'
        if item.get('ref_base', '').upper() != ref[i] or item.get('query_base', '').upper() != seq[q].upper():
            return None, 'reference_or_query_mapping_mismatch'
        for value in (span, spans[q]):
            if (not isinstance(value, (list, tuple)) or len(value) != 2 or
                    any(type(v) is not int for v in value) or not 0 <= value[0] < value[1] <= len(signal)):
                return None, 'invalid_signal_span'
        query.append(seq[q].upper()); ref_spans.append(span); seq_spans.append(spans[q])
    query = ''.join(query)
    if set(ref) - set('ACGT') or ref[7:9] != 'CG':
        return None, 'ambiguous_or_not_CpG'
    if c['filters']['require_query_reference_match'] and query != ref:
        return None, 'query_reference_mismatch'
    start = min(s[0] for s in ref_spans); end = max(s[1] for s in ref_spans)
    if any(a < start or b > end for a, b in seq_spans):
        return None, 'seq_span_outside_window'
    values = np.asarray(signal[start:end], dtype=np.float32)
    if not np.isfinite(values).all():
        return None, 'nonfinite_signal'
    identity = [c['library_id'], original_read_id(record), c['chromosome'], position, '+']
    return dict(window_id=digest(identity), library_id=c['library_id'], read_id=identity[1],
        chrom=c['chromosome'], center_genome_pos0=position, strand='+', source='mod',
        seq=query, ref=ref, signal=values.tolist(), signal_len=len(values), signal_offset=start,
        base_sample_span_ref=[[a-start, b-start] for a, b in ref_spans],
        base_sample_span_seq=[[a-start, b-start] for a, b in seq_spans]), None


def run(c, root, dry_run=False):
    if dry_run:
        print(json.dumps(dict(step=1, source=c['inputs']['signal_jsonl'], output=str(root/'windows'),
              max_records=c['scan']['max_records'], region=c['scan']['region'],
              note='Single pass over plain JSONL; completed shards and exact read-boundary checkpoints resume')))
        return
    audit = frozen(c, root)
    with lock(root/'.locks/step1.lock'):
        if (root/'windows').exists():
            return verify(root/'windows', c)
        work = root/'.windows.work'; work.mkdir(exist_ok=True)
        source = Path(c['inputs']['signal_jsonl'])
        if source.suffix == '.gz':
            raise ValueError('Resumable signal input must be plain JSONL, not gzip')
        source_sig = signature(source); sequence = load_chromosome(c['reference']['fasta'], c['chromosome'])
        refdb = sqlite3.connect(f'file:{root / "audit/reference.sqlite"}?mode=ro', uri=True)
        positions = [r[0] for r in refdb.execute('SELECT position FROM sites ORDER BY position')]; refdb.close()
        db = sqlite3.connect(work/'observations.sqlite')
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY,payload TEXT)')
        db.execute('''CREATE TABLE IF NOT EXISTS observations(idx INTEGER PRIMARY KEY,window_id TEXT UNIQUE,
                   read_id TEXT,position INTEGER,ref5 TEXT,seq5 TEXT,content_sha256 TEXT)''')
        db.execute('CREATE TABLE IF NOT EXISTS parts(name TEXT PRIMARY KEY,n_rows INTEGER,sha256 TEXT)')
        row = db.execute('SELECT payload FROM state WHERE id=1').fetchone()
        state = json.loads(row[0]) if row else dict(config_sha256=digest(c), source_signature=source_sig,
                 audit_manifest_sha256=sha(root/'audit/manifest.json'), offset=0, lines=0, records=0,
                 n_windows=0, counters={}, source_chunk_chain='0'*64, eof=False)
        if (state['config_sha256'] != digest(c) or state['source_signature'] != source_sig or
                state['audit_manifest_sha256'] != sha(root/'audit/manifest.json')):
            db.close(); raise ValueError('Cannot resume windows from different input/configuration/audit')
        committed = {r[0]: (r[1], r[2]) for r in db.execute('SELECT name,n_rows,sha256 FROM parts')}
        for name, (_, h) in committed.items():
            if sha(work/name) != h:
                db.close(); raise ValueError('Committed window shard changed')
        for p in work.glob('part_*'):
            if p.name not in committed:
                p.unlink()  # Only uncommitted files owned by this interrupted stage.
        counts = Counter(state['counters']); buffer = []; chunk_hash = hashlib.sha256()
        try:
            import orjson
            loads = orjson.loads
        except ImportError:
            loads = json.loads
        with db, source.open('rb', buffering=8*1024**2) as f:
            f.seek(state['offset']); since_commit = 0
            def checkpoint(eof=False):
                nonlocal buffer, chunk_hash, since_commit
                if buffer:
                    name = f'part_{len(committed):06d}.jsonl.gz'; tmp = work/(name+'.partial')
                    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=3) as out:
                        for w in buffer:
                            out.write(json.dumps(w, separators=(',', ':'), allow_nan=False)+'\n')
                    os.replace(tmp, work/name); h = sha(work/name)
                    db.execute('INSERT INTO parts VALUES(?,?,?)', (name, len(buffer), h))
                    committed[name] = (len(buffer), h)
                state.update(offset=f.tell(), counters=dict(counts), eof=eof,
                    source_chunk_chain=digest([state['source_chunk_chain'], chunk_hash.hexdigest(), f.tell()]))
                check_signatures([source_sig])
                db.execute('INSERT OR REPLACE INTO state VALUES(1,?)', (json.dumps(state),)); db.commit()
                buffer = []; chunk_hash = hashlib.sha256(); since_commit = 0
            while not state['eof']:
                if c['scan']['max_records'] is not None and state['records'] >= c['scan']['max_records']:
                    checkpoint(); break
                line = f.readline()
                if not line:
                    checkpoint(eof=True); break
                chunk_hash.update(line); state['lines'] += 1
                if not line.strip():
                    continue
                record = loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f'Expected JSON object at source line {state["lines"]}')
                state['records'] += 1; since_commit += 1
                reason = None
                if record.get('chrom') != c['chromosome'] or record.get('strand') != '+':
                    reason = 'wrong_chromosome_or_strand'
                elif type(record.get('mapq')) is not int or record['mapq'] < c['filters']['min_mapq']:
                    reason = 'low_or_missing_mapq'
                elif record.get('is_secondary') or record.get('is_supplementary'):
                    reason = 'nonprimary_alignment'
                if reason:
                    counts['reads_'+reason] += 1
                else:
                    mapping = {}
                    for item in record.get('per_base', []):
                        p = item.get('genome_pos0')
                        if type(p) is not int:
                            raise ValueError('Invalid per_base genome position')
                        if p in mapping and mapping[p] != item:
                            raise ValueError('Conflicting per_base entries')
                        mapping[p] = item
                    if not mapping:
                        counts['reads_missing_mapping'] += 1
                    else:
                        lo, hi = min(mapping), max(mapping)
                        for pos in positions[bisect.bisect_left(positions, lo):bisect.bisect_right(positions, hi)]:
                            counts['candidate_windows'] += 1
                            w, reason = extract_window(record, pos, mapping, sequence, c)
                            if reason:
                                counts['windows_'+reason] += 1; continue
                            h = digest(w)
                            old = db.execute('SELECT content_sha256 FROM observations WHERE window_id=?', (w['window_id'],)).fetchone()
                            if old:
                                if old[0] != h:
                                    raise ValueError(f'Conflicting duplicate read-site {w["window_id"]}')
                                counts['identical_duplicate_windows'] += 1; continue
                            w['idx'] = state['n_windows']; state['n_windows'] += 1
                            w['source_line'] = state['lines']
                            db.execute('INSERT INTO observations VALUES(?,?,?,?,?,?,?)',
                                (w['idx'], w['window_id'], w['read_id'], pos, w['ref'][5:10], w['seq'][5:10], h))
                            buffer.append(w)
                if state['records'] % c['scan']['log_every_records'] == 0:
                    print(f'[WINDOWS] records={state["records"]} bytes={f.tell()}/{source_sig["size"]} windows={state["n_windows"]}', flush=True)
                if len(buffer) >= c['scan']['shard_windows'] or since_commit >= c['scan']['checkpoint_records']:
                    checkpoint()
        try:
            if not state['n_windows']:
                raise ValueError('No legal windows; prefix may not cover qualifying CpGs')
            check_signatures([source_sig])
            parts = [dict(name=name, n_rows=n, sha256=h) for name, (n, h) in sorted(committed.items())]
            db.execute('CREATE INDEX IF NOT EXISTS obs_position ON observations(position)'); db.commit(); db.close()
            write_json(work/'scan_summary.json', state)
            result = seal(work, c, n_windows=state['n_windows'], n_input_records=state['records'],
                       partial_test=not state['eof'], source_signature=source_sig,
                       source_content_hash_scope='SHA256 chain over committed source-byte chunks, including exact offsets',
                       source_chunk_chain=state['source_chunk_chain'], parts=parts,
                       audit_manifest_sha256=sha(root/'audit/manifest.json'))
            if result['partial_test'] and not c['allow_partial_results']:
                raise ValueError('Partial scan cannot be published as formal data')
            work.rename(root/'windows')
            print(f'[DONE] Step1: {state["n_windows"]} windows in {len(parts)} shards', flush=True)
            return result
        finally:
            db.close()


if __name__ == '__main__':
    a = parser(__doc__).parse_args(); c, root = load_config(a); run(c, root, a.dry_run)
