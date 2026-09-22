"""CPU full-ranking fusion of current components; no submission."""
import argparse
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import current_fusion as fusion, fulltext_prior as prior
from legalir2.h100 import digest, guard
from legalir2.labeled_gate import summarize
from legalir2.store import file_hash, read_json, write_json
from audit_prior_development import load as load_full, DATASET, SOURCE
from evaluate_duo_boundary import load as load_short


def compare(rows, before, after):
    pairs = [(Fraction(len(set(r['gold']) & set(a)), len(r['gold'])),
              Fraction(len(set(r['gold']) & set(b)), len(r['gold'])), set(a) != set(b))
             for r, a, b in zip(rows, before, after)]
    if len(rows) != len(before) or len(rows) != len(after):
        raise ValueError('Incomplete comparison')
    result = summarize(pairs)
    delta = np.asarray([float(b-a) for a, b, _ in pairs])
    rng = np.random.default_rng(20260921)
    samples = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)])
    result['paired_bootstrap_ci95'] = np.quantile(samples, [.025, .975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'reports/current-fusion/20260921-current-v2')
    args = parser.parse_args()
    policy = fusion.FIXED_POLICY
    plan = {'schema': 'current-component-fusion-v1', 'policy': policy,
        'selection': 'New equal-weight RRF baseline registered before this evaluation; separate from rank_fusion_probe.py',
        'code': {name: file_hash(ROOT/name) for name in ('legalir2/current_fusion.py',
            'scripts/analysis/evaluate_current_fusion.py')},
        'datasets': {s: file_hash(DATASET/(s+'.json')) for s in ('development', 'validation')},
        'score_deadband': .0052, 'submission_allowed': False}
    guard(args.out/'registered-plan.json', plan)
    run = ROOT/'analysis/impact/duo-20260918T110105Z'
    signature, selection = read_json(run/'inputs.json'), read_json(run/'calibration.json')
    if selection['inputs'] != digest(signature) or selection['chosen']['strength'] != .25:
        raise ValueError('Frozen comparator choice changed')
    full = load_full()
    short = {s: load_short(run, s, signature, selection) for s in ('development', 'validation')}
    marker = read_json(SOURCE/'inputs.json')
    if file_hash(DATASET/'v9-development.json') != marker['datasets']['v9-development.json']:
        raise ValueError('V9 cache changed')
    v9 = read_json(DATASET/'v9-development.json')
    records = {s: read_json(DATASET/(s+'.json')) for s in short}
    for split, rows in short.items():
        if {r['id']: {'question': r['question'], 'answer': r['gold']} for r in rows} != records[split]:
            raise ValueError('Labeled cache membership changed')
    if len(full) != 429 or len(short['validation']) != 1050:
        raise ValueError('Expected 429 development and 1050 validation questions')
    old, current = ({r['id']: r for r in rows} for rows in (v9, short['development']))
    if set(old) != set(current) or set(old) != {r['id'] for r in full}:
        raise ValueError('Voter membership differs')
    rows = []
    known = set(signature['known_docs'])
    for row in full:
        qid = row['id']
        rankings = fusion.cached_orderings(old[qid], current[qid], row, signature['weights'])
        if not all(set(r) <= known for r in rankings.values()):
            raise ValueError('Unknown candidate ID')
        if rankings['fulltext_duo'][:5] != prior.predict(row, prior.BASELINE):
            raise ValueError('Full-text baseline did not replay')
        rows.append({'id': qid, 'question': row['question'], 'gold': row['gold'], 'rankings': rankings})
    identity = {'plan': digest(plan), 'v9_rows': digest(v9), 'short_rows': digest(short['development']),
                'fulltext_rows': digest(full), 'weights': digest(signature['weights'])}
    guard(args.out/'inputs.json', identity)
    write_json(args.out/'development-rankings.json', {'inputs': digest(identity), 'rows': rows})
    anchor = [r['rankings']['fulltext_duo'][:5] for r in rows]
    predicted = [fusion.fuse(r['rankings'], policy)[:5] for r in rows]
    truncated = [fusion.fuse({n: o[:5] for n, o in r['rankings'].items()}, policy)[:5] for r in rows]
    components = {name: compare(rows, anchor, [r['rankings'][name][:5] for r in rows])
                  for name in fusion.COMPONENTS}
    report = {'plan': digest(plan), 'development': {'components_vs_fulltext': components,
        'full_order_fusion': compare(rows, anchor, predicted),
        'top5_only_diagnostic': compare(rows, anchor, truncated),
        'mean_unanimous_top5': float(np.mean([len(set.intersection(*(set(o[:5]) for o in r['rankings'].values()))) for r in rows])),
        'mean_depth': {n: float(np.mean([len(r['rankings'][n]) for r in rows])) for n in fusion.COMPONENTS}},
        'pooled_current_fusion_gate': 'blocked: no actual full-text validation score cache',
        'independent_positives_established': False, 'submission_allowed': False}
    write_json(args.out/'development-predictions.json', {'inputs': digest(identity), 'policy': policy,
        'predictions': {r['id']: {'baseline': a, 'proposed': b} for r, a, b in zip(rows, anchor, predicted)}})
    write_json(args.out/'report.json', report)
    print('Full-order fusion:', report['development']['full_order_fusion'], flush=True)


if __name__ == '__main__':
    main()
