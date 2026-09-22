"""Experimental single-slot rescue with agreement between two saved comparators.

Keep the baseline's first four answers. An outsider may replace only its fifth
answer, and only when both models prefer that outsider directly and in fitted
ability. Priors, evidence, pools and model identity must be checked by the caller.
This is a CPU experiment, not an enabled submission policy.
"""
import numpy as np

POLICY = {'name': 'duo-agreement-fifth-v1', 'minimum_pair_preference': .9,
          'protected_prefix': 4, 'maximum_replacements': 1}


def scores(row):
    docs = row['docs']
    n = len(docs)
    if n < 5 or len(set(docs)) != n:
        raise ValueError('Expected distinct candidate documents')
    prior, ability = (np.asarray(row[k], dtype=float) for k in ('prior', 'ability'))
    p = np.asarray(row['probabilities'], dtype=float)
    if (prior.shape != (n,) or ability.shape != (n,) or not np.isfinite(prior).all()
            or not np.isfinite(ability).all() or p.shape != (n, n) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.allclose(p+p.T, 1., atol=1e-6)):
        raise ValueError('Invalid comparison scores')
    return prior, ability, p


def predict(baseline, candidate):
    """Return a baseline-preserving answer and diagnostic evidence; never read labels."""
    docs = baseline['docs']
    if any(baseline[k] != candidate[k] for k in ('id', 'question', 'docs', 'prior')):
        raise ValueError('Comparators must use the same query, candidate order and prior')
    prior, old_ability, old_p = scores(baseline)
    _, new_ability, new_p = scores(candidate)
    values = .75*prior + .25*old_ability
    original = sorted(range(len(docs)), key=lambda i: (-values[i], docs[i]))[:5]
    victim = original[4]
    challengers = []
    for i, doc in enumerate(docs):
        if i in original:
            continue
        confidence = min(old_p[i, victim], new_p[i, victim])
        margin = min(old_ability[i]-old_ability[victim], new_ability[i]-new_ability[victim])
        if confidence >= POLICY['minimum_pair_preference'] and margin > 0:
            challengers.append((float(confidence), float(margin), doc, i))
    answer = [docs[i] for i in original]
    trace = {'policy': POLICY['name'], 'replaced': False, 'eligible': len(challengers)}
    if challengers:
        confidence, margin, doc, i = min(challengers, key=lambda r: (-r[0], -r[1], r[2]))
        answer[4] = doc
        trace.update(replaced=True, removed=docs[victim], added=doc,
                     old_pair=float(old_p[i, victim]), new_pair=float(new_p[i, victim]),
                     minimum_ability_margin=margin)
    return answer, trace
