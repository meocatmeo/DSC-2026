"""Rank fusion of complete cached V9 and duo orderings, with optional fourth voter."""
import math

import numpy as np

COMPONENTS = ('v9', 'duo', 'fulltext_duo')
FIXED_POLICY = {'method': 'rrf', 'constant': 60.,
                'weights': {'v9': 1., 'duo': 1., 'fulltext_duo': 1.}}


def score_order(docs, values):
    scores = np.asarray(values, dtype=float)
    if len(docs) < 5 or len(docs) != len(set(docs)) or scores.shape != (len(docs),) or not np.isfinite(scores).all():
        raise ValueError('Invalid complete candidate scores')
    return sorted(docs, key=lambda d: (-float(scores[docs.index(d)]), d))


def cached_orderings(v9, short, full, weights, strength=.25):
    for row in (short, full):
        if any(row[k] != v9[k] for k in ('id', 'question', 'gold')):
            raise ValueError('Voters refer to different questions or labels')
    rankings = {'v9': score_order(v9['docs'], np.asarray(v9['features1'])@np.asarray(weights))}
    for name, row in (('duo', short), ('fulltext_duo', full)):
        scores = np.asarray(row['prior'])+strength*(np.asarray(row['ability'])-row['prior'])
        rankings[name] = score_order(row['docs'], scores)
    if rankings['v9'][:5] != short['baseline'][:5]:
        raise ValueError('V9 top five did not replay against comparator caches')
    return rankings


def fuse(rankings, policy=FIXED_POLICY):
    """Missing documents contribute zero; never invent ranks outside a voter's cache."""
    if policy.get('method') != 'rrf' or set(policy) != {'method', 'constant', 'weights'}:
        raise ValueError('Expected an explicit RRF policy')
    constant, weights = policy['constant'], policy['weights']
    if not math.isfinite(constant) or constant < 0 or set(weights) != set(rankings):
        raise ValueError('Invalid rank constant or missing voter')
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError('Voter weights must be finite and positive')
    scores = {}
    for name, ordering in rankings.items():
        if len(ordering) < 5 or len(ordering) != len(set(ordering)):
            raise ValueError('Invalid voter ordering')
        for rank, doc in enumerate(ordering, 1):
            scores[doc] = scores.get(doc, 0.)+weights[name]/(constant+rank)
    anchor = {d: i for i, d in enumerate(rankings['fulltext_duo'])}
    return sorted(scores, key=lambda d: (-scores[d], anchor.get(d, math.inf), d))


def parameter_audit(base, fourth_parameters):
    if not isinstance(fourth_parameters, int) or fourth_parameters <= 0:
        raise ValueError('Expected actual loaded fourth-model parameter count')
    if sum(base['components'].values()) != base['total']:
        raise ValueError('Inconsistent baseline parameter audit')
    total = base['total']+fourth_parameters
    if total >= 4_000_000_000:
        raise ValueError('Four-voter deployment exceeds the total parameter budget')
    return {'components': dict(base['components'], fourth_reranker=fourth_parameters),
            'total': total, 'budget': 4_000_000_000, 'headroom': 4_000_000_000-total,
            'shared_weights': 'short and full-text duo use the same comparator checkpoint'}
