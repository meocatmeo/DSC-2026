"""Experimental top-five replacement check using already scored document pairs.

Keep a proposed challenger only when its direct comparison supports replacing an
incumbent. This is evaluated offline before any production inference integration.
"""
import math

import numpy as np


def ranked(row, strength=.25):
    score = np.asarray(row['prior']) + strength * (np.asarray(row['ability']) - row['prior'])
    return sorted(range(len(row['docs'])), key=lambda i: (-score[i], row['docs'][i]))


def predict(row, threshold=.75, strength=.25):
    if not math.isfinite(threshold) or not .5 <= threshold <= 1:
        raise ValueError('Direct-comparison threshold must be between .5 and 1')
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('Invalid comparator strength')
    docs = row['docs']
    position = {doc: i for i, doc in enumerate(docs)}
    original = list(row['baseline'][:5])
    if len(original) != 5 or len(set(original)) != 5 or not set(original) <= set(docs):
        raise ValueError('Expected five distinct baseline documents in the pool')
    p = np.asarray(row['probabilities'])
    if (p.shape != (len(docs), len(docs)) or not np.isfinite(p).all() or
            (p < 0).any() or (p > 1).any() or not np.allclose(p + p.T, 1., atol=1e-6)):
        raise ValueError('Invalid pair probabilities')
    if strength == 0:
        return original
    order = ranked(row, strength)
    proposed = [docs[i] for i in order[:5]]
    # Only examine the existing blend's proposed changes. Never look at labels.
    remaining = [d for d in original if d not in proposed]
    result = original.copy()
    for challenger in (d for d in proposed if d not in original):
        supported = [d for d in remaining if p[position[challenger], position[d]] >= threshold]
        if supported:
            victim = max(supported, key=lambda d: (p[position[challenger], position[d]], original.index(d)))
            result[result.index(victim)] = challenger
            remaining.remove(victim)
    return sorted(result, key=lambda d: order.index(position[d]))
