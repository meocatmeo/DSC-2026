"""Paired macro-recall gate on complete labeled sets; no prediction generation."""
from fractions import Fraction
from .compare import bootstrap, sign_test

DEADBAND = Fraction(52, 10000)
SPLITS = ('development', 'validation')


def decision(delta, ci95=None):
    if ci95 is not None and ci95[0] <= 0 <= ci95[1]:
        return 'inconclusive'
    if delta >= DEADBAND:
        return 'promote'
    if delta <= -DEADBAND:
        return 'reject'
    return 'inconclusive'


def summarize(rows):
    """Descriptive paired statistics only; split summaries cannot authorize deployment."""
    if not rows:
        raise ValueError('Cannot summarize an empty labeled set')
    before = sum((r[0] for r in rows), Fraction())/len(rows)
    after = sum((r[1] for r in rows), Fraction())/len(rows)
    return {'questions': len(rows), 'baseline_recall': float(before),
            'proposed_recall': float(after), 'delta': float(after-before),
            'wins': sum(b > a for a, b, _ in rows),
            'losses': sum(b < a for a, b, _ in rows),
            'changed_sets': sum(changed for _, _, changed in rows),
            'paired_bootstrap_ci95': list(bootstrap([float(b-a) for a, b, _ in rows])),
            'sign_test_p': sign_test(sum(b > a for a, b, _ in rows), sum(b < a for a, b, _ in rows))}


def evaluate(datasets, predictions, known):
    """Require every labeled question and exactly five valid documents per answer."""
    if set(datasets) != set(SPLITS) or set(predictions) != set(SPLITS):
        raise ValueError('Both labeled splits are required; private data is not accepted')
    if set(datasets['development']) & set(datasets['validation']):
        raise ValueError('Overlapping labeled question IDs')
    measured, pooled = {}, []
    for split in SPLITS:
        records = datasets[split]
        if not records or set(predictions[split]) != set(records):
            raise ValueError('Missing or extra labeled predictions: '+split)
        rows = []
        for qid, record in sorted(records.items()):
            answer = record['answer']
            if not answer or len(answer) != len(set(answer)) or not set(answer) <= known:
                raise ValueError('Missing or invalid gold labels')
            gold = set(answer)
            row = predictions[split][qid]
            if set(row) != {'baseline', 'proposed'}:
                raise ValueError('Expected paired baseline/proposed predictions')
            for docs in row.values():
                if not isinstance(docs, list) or len(docs) != 5 or len(set(docs)) != 5 or not set(docs) <= known:
                    raise ValueError('Expected five unique known document IDs')
            before, after = (set(row[k]) for k in ('baseline', 'proposed'))
            rows.append((Fraction(len(gold & before), len(gold)),
                         Fraction(len(gold & after), len(gold)), before != after))
        measured[split] = summarize(rows)
        pooled.extend(rows)
    result = summarize(pooled)
    eligible = all(len(datasets[s]) == n for s, n in zip(SPLITS, (429, 1050)))
    delta = sum((b-a for a, b, _ in pooled), Fraction())/len(pooled)
    result.update(gate_eligible=eligible, decision=decision(delta, result['paired_bootstrap_ci95'])
                  if eligible else 'inconclusive')
    return {**measured, 'pooled': result, 'deadband': float(DEADBAND),
            'interpretation': 'Only complete 429+1050 pooling can decide; requires the deadband and a paired 95% interval excluding zero. Historically reused labels; no private-score guarantee.'}
