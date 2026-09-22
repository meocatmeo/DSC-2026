"""CPU-only FTS coverage sweep; no predictions, model fitting or private input."""
import argparse
from collections import Counter
from contextlib import closing
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import fts_sweep as fts
from legalir2.duo_evidence import Evidence, index_metadata
from legalir2.h100 import digest, guard
from legalir2.store import file_hash, read_json, write_json
from audit_prior_development import load as load_development, DATASET
from evaluate_duo_boundary import load as load_short


def summarize(vectors, counts):
    before, after = np.asarray(vectors, dtype=float).T
    return {'questions': len(vectors), 'baseline_pool_coverage': float(before.mean()),
            'expanded_pool_coverage': float(after.mean()), 'delta': float((after-before).mean()), **counts}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'reports/private/20260921-fts-coverage')
    args = parser.parse_args()
    index = ROOT/'artifacts2/frozen-20260918'
    model = ROOT/'analysis/impact/duo-20260918T110105Z'
    signature, calibration = read_json(model/'inputs.json'), read_json(model/'calibration.json')
    expected = signature['source_inputs']['files']['artifacts-ranking-h100/store.sqlite']
    if file_hash(index/'store.sqlite') != expected:
        raise ValueError('Frozen store changed')
    metadata = index_metadata(index/'duo-fulltext-v2.sqlite')
    if metadata['store_sha256'] != expected:
        raise ValueError('FTS index belongs to another corpus')
    known = set(signature['known_docs'])
    datasets = {s: read_json(DATASET/(s+'.json')) for s in ('development', 'validation')}
    if set(datasets['development']) & set(datasets['validation']):
        raise ValueError('Labeled ID overlap')
    plan = {'schema': fts.SCHEMA, 'policies': fts.POLICIES, 'coverage_gain_required': .008,
        'score_deadband': .0052, 'mode': 'coverage-only; no submission authorization',
        'conversion_096': 'User planning assumption, not an estimated marginal success probability',
        'selection': 'Exploratory sweep across both historically reused labeled sets',
        'optimization': 'Search only baseline-uncovered gold questions; never use this label shortcut at inference',
        'validation_baseline': 'Reconstruct fulltext-v2 candidate additions on frozen short-duo pools; no validation answer recall claimed',
        'datasets': {s: digest(r) for s, r in datasets.items()}, 'metadata': metadata,
        'index_sha256': file_hash(index/'duo-fulltext-v2.sqlite'),
        'code': {n: file_hash(p) for n, p in [('fts_sweep.py', Path(fts.__file__)), ('sweep_fts_coverage.py', Path(__file__))]}}
    guard(args.out/'registered-plan.json', plan)
    if calibration['inputs'] != digest(signature):
        raise ValueError('Original duo calibration changed')
    rows_by_split = {'development': load_development(),
                     'validation': load_short(model, 'validation', signature, calibration)}
    guard(args.out/'inputs.json', {'plan': digest(plan), 'rows': {s: digest(rows) for s, rows in rows_by_split.items()}})
    vectors = {s: [[] for _ in fts.POLICIES] for s in rows_by_split}
    stats = {s: [Counter() for _ in fts.POLICIES] for s in rows_by_split}
    searches, started = 0, time.monotonic()
    with closing(fts.Search(index/'duo-fulltext-v2.sqlite')) as search, closing(Evidence(index, None, index/'duo-fulltext-v2.sqlite')) as baseline:
        for split, rows in rows_by_split.items():
            if len(rows) != len(datasets[split]) or {r['id'] for r in rows} != set(datasets[split]):
                raise ValueError('Labeled membership changed')
            for number, row in enumerate(rows):
                qid, gold = row['id'], set(row['gold'])
                record = datasets[split][qid]
                if row['question'] != record['question'] or gold != set(record['answer']) or not gold:
                    raise ValueError('Question or gold changed')
                existing = row['docs']
                if not set(existing) <= known:
                    raise ValueError('Unknown source document')
                result = None
                if not gold <= set(existing):
                    path = args.out/'queries'/split/(qid+'.json')
                    if path.exists():
                        result = read_json(path)
                        if result['input'] != digest(row):
                            raise ValueError('Query cache changed')
                    else:
                        if split == 'validation':
                            existing, _ = baseline.expand(row['question'], existing, row['prior'], known)
                        hits = {str(p): search.hits(row['question'], p) for p in (3, 4, 5, 6)} if not gold <= set(existing) else {}
                        result = {'input': digest(row), 'id': qid, 'existing': existing, 'hits': hits}
                        write_json(path, result)
                    existing = result['existing']
                    searches += bool(result['hits'])
                before = len(gold & set(existing))/len(gold)
                for k, policy in enumerate(fts.POLICIES):
                    new = fts.additions(result['hits'][str(policy['phrase_words'])], existing, known, policy) if result and result['hits'] else []
                    recovered = gold & set(new)
                    after = len(gold & set(existing+new))/len(gold)
                    vectors[split][k].append((before, after))
                    stats[split][k]['gold_instances_recovered'] += len(recovered)
                    stats[split][k]['questions_recovered'] += bool(recovered)
                    stats[split][k]['diagnostic_additions'] += len(new)
                if result or (number+1) % 100 == 0:
                    print(f'{split}: {number+1}/{len(rows)}; searched uncovered={searches}; elapsed={time.monotonic()-started:.1f}s', flush=True)
    trials = []
    for k, policy in enumerate(fts.POLICIES):
        measured = {s: summarize(vectors[s][k], dict(stats[s][k])) for s in vectors}
        pooled = vectors['development'][k]+vectors['validation'][k]
        counts = stats['development'][k]+stats['validation'][k]
        trials.append({'policy': policy, **measured, 'pooled': summarize(pooled, dict(counts))})
    best = max(trials, key=lambda t: (t['pooled']['delta'], -t['policy']['extra'], -t['policy']['search_limit']))
    cleared = best['pooled']['delta'] > plan['coverage_gain_required']
    report = {'plan': digest(plan), 'trials': trials, 'best_exploratory': best,
        'coverage_gate_cleared': cleared, 'score_gate': 'not evaluated; requires actual append-only predictions on both labeled sets',
        'submission_allowed': False, 'uncovered_queries_searched': searches,
        'elapsed_seconds': time.monotonic()-started,
        'limitation': 'Coverage only, label-aware compute shortcut, reused labeled sets, no private data or GPU inference.'}
    write_json(args.out/'report.json', report)
    print('Best coverage:', best, flush=True)
    print('Coverage gate >0.008:', cleared, '; submission allowed: False', flush=True)


if __name__ == '__main__':
    main()
