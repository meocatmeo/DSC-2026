"""Freeze the exact probe's validation-selected weight, then check all 429 development questions."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.cache import normalise_question
from legalir2.h100 import digest, guard
from legalir2.store import file_hash, read_json, write_json
import rank_fusion_probe as probe
from audit_prior_development import DATASET
from evaluate_current_fusion import compare


def check(rows, records):
    if len(rows) != len(records) or {r['id'] for r in rows} != set(records):
        raise ValueError('Question membership differs')
    for row in rows:
        record = records[row['id']]
        if row['question'] != record['question'] or set(row['gold']) != set(record['answer']) or not row['gold']:
            raise ValueError('Question or gold differs')
        for name in (probe.BASE,)+probe.WEAK:
            order = probe.ordering(row, name)
            if len(order) != len(row['docs']) or len(order) != len(set(order)) or set(order) != set(row['docs']):
                raise ValueError('Expected a complete permutation for every probe voter')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'reports/current-fusion/20260921-fixed-probe-v2')
    args = parser.parse_args()
    cache = ROOT/'analysis/v11/investigation'
    plan = {'schema': 'fixed-probe-development-v1', 'probe_code': file_hash(Path(probe.__file__)),
        'wrapper_code': file_hash(Path(__file__)), 'k': probe.K, 'weak': list(probe.WEAK), 'grid': list(probe.GRID),
        'selection': 'Freeze the validation-wide grid winner, same tie rule as supplied probe; no development weight search',
        'folds': 20, 'score_deadband': .0052,
        'caches': {s: file_hash(cache/(s+'.compact.jsonl.gz')) for s in ('train', 'validation')},
        'labels': {s: file_hash(DATASET/(s+'.json')) for s in ('development', 'validation')}}
    guard(args.out/'registered-plan.json', plan)
    validation = probe.load(cache/'validation.compact.jsonl.gz')
    val_records = read_json(DATASET/'validation.json')
    check(validation, val_records)
    trials = [{'weight': w, 'recall': probe.mean_recall(validation, w)} for w in probe.GRID]
    chosen = max(trials, key=lambda t: t['recall'])['weight']
    guard(args.out/'frozen-policy.json', {'plan': digest(plan), 'weight': chosen, 'selection_split': 'validation',
        'trials': trials, 'warning': 'Selection-set gain is biased; do not call it an independent positive'})
    print('Frozen W:', chosen, '; reproducing exact 20-split cross-fit...', flush=True)
    crossfit = probe.crossfit(validation, 20)
    print('Validation cross-fitted gain:', crossfit['mean'], flush=True)
    # Development labels and orderings are loaded only after the policy is frozen.
    records = read_json(DATASET/'development.json')
    development = [r for r in probe.load(cache/'train.compact.jsonl.gz') if r['id'] in records]
    check(development, records)
    if set(records) & set(val_records) or ({normalise_question(r['question']) for r in records.values()} &
                                          {normalise_question(r['question']) for r in val_records.values()}):
        raise ValueError('Development/validation ID or normalized-text overlap')
    old = [probe.ordering(r, probe.BASE)[:5] for r in development]
    new = [probe.fuse(r, chosen) for r in development]
    development_result = compare(development, old, new)
    report = {'plan': digest(plan), 'weight': chosen, 'validation_crossfit': crossfit,
        'validation_in_sample_gain_biased': max(t['recall'] for t in trials)-probe.baseline(validation),
        'development_fixed_policy': development_result,
        'submission_allowed': False,
        'limitation': 'Disjoint from fusion-weight selection, but historically reused data and training-cache component scores; not a fresh end-to-end holdout. Repeated cross-fit splits are not independent experiments.'}
    write_json(args.out/'report.json', report)
    write_json(args.out/'development-predictions.json', {r['id']: {'baseline': a, 'proposed': b}
        for r, a, b in zip(development, old, new)})
    print('Development, unchanged W:', development_result, flush=True)


if __name__ == '__main__':
    main()
