"""Select a bounded replacement policy on development, then evaluate validation."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from legalir2 import duo
from legalir2.duo_boundary import predict
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json, write_json


def load(run, split, signature, choice):
    directory = run/'scores'/split
    marker = read_json(directory/'resume.json')
    if marker['inputs'] != digest(signature) or marker['model'] != choice['model']:
        raise ValueError('Score cache belongs to a different comparator')
    paths = sorted(directory.glob('part-*.json'))
    if [p.name for p in paths] != [f'part-{i:05d}.json' for i in range(len(paths))]:
        raise ValueError('Missing score shards')
    rows = [r for p in paths for r in read_json(p)]
    records = {r['id']: {'question': r['question'], 'answer': r['gold']} for r in rows}
    if len(records) != len(rows) or digest(records) != marker['questions']:
        raise ValueError('Incomplete or changed question/label cache')
    duo.validate_scores(rows, list(records.items()))
    if any(not r['gold'] for r in rows):
        raise ValueError('This evaluator requires labeled development/validation data')
    return rows


def answers(rows, threshold, strength):
    if threshold is None:
        return [r['predicted'] for r in duo.diagnostics(rows, strength).values()]
    return [predict(r, threshold, strength) for r in rows]


def recalls(rows, predictions):
    return np.asarray([len(set(r['gold']) & set(p)) / len(r['gold']) for r, p in zip(rows, predictions)])


def breakdown(rows, predictions):
    counts = Counter()
    for r, p in zip(rows, predictions):
        for doc in set(r['gold']) - set(p):
            counts['outside_pool' if doc not in r['docs'] else 'inside_pool_ranked_out'] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    signature, choice = read_json(args.run/'inputs.json'), read_json(args.run/'calibration.json')
    if choice['inputs'] != digest(signature):
        raise ValueError('Calibration signature mismatch')
    strength = choice['chosen']['strength']
    development = load(args.run, 'development', signature, choice)
    if {r['id'] for r in development} != set(signature['development']):
        raise ValueError('Development membership mismatch')
    base = answers(development, None, strength)
    trials = []
    for threshold in (None, .5, .75, .9):
        proposed = answers(development, threshold, strength)
        trials.append({'threshold': threshold, 'recall': float(recalls(development, proposed).mean()),
                       'changed_sets_vs_duo': sum(set(a) != set(b) for a, b in zip(base, proposed))})
    # Preserve existing duo on ties; only development results select the threshold.
    selected = max(trials, key=lambda t: (t['recall'], t['threshold'] is None, -t['changed_sets_vs_duo']))
    write_json(args.out/'development-choice.json', {'trials': trials, 'selected': selected,
               'strength': strength, 'source_inputs': file_hash(args.run/'inputs.json')})
    print('Development selection:', selected, flush=True)
    validation = load(args.run, 'validation', signature, choice)
    if {r['id'] for r in validation} & {r['id'] for r in development}:
        raise ValueError('Development/validation ID overlap')
    before = answers(validation, None, strength)
    after = answers(validation, selected['threshold'], strength)
    old, new = recalls(validation, before), recalls(validation, after)
    differences = new - old
    rng = np.random.default_rng(2026)
    means = np.asarray([rng.choice(differences, len(differences), replace=True).mean() for _ in range(10000)])
    report = {'status': 'offline experiment; private questions and leaderboard scores unused',
              'limitation': 'Existing validation was reused historically; not a fresh holdout.',
              'development': {'questions': len(development), 'trials': trials, 'selected': selected},
              'validation': {'questions': len(validation), 'v9': float(recalls(validation, [r['baseline'] for r in validation]).mean()),
                  'duo': float(old.mean()), 'candidate': float(new.mean()), 'delta': float(differences.mean()),
                  'wins': int((differences > 0).sum()), 'losses': int((differences < 0).sum()),
                  'ci95': np.quantile(means, [.025, .975]).tolist(),
                  'missed_gold_documents': breakdown(validation, before),
                  'candidate_missed_gold_documents': breakdown(validation, after)},
              'promote': bool(selected['threshold'] is not None and np.quantile(means, .025) > 0)}
    write_json(args.out/'evaluation.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
