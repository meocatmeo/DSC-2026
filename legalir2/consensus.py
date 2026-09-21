"""Validated tournament signals adapted from thong/e091868.

These signals are candidates for development calibration, not fixed overrides.
Training-label membership never removes documents from the answer space.
"""
import numpy as np


def probability_matrix(value, count):
    matrix = np.asarray(value, dtype=float)
    if (count < 5 or matrix.shape != (count, count)
            or not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any()
            or not np.allclose(matrix+matrix.T, 1, atol=1e-6, rtol=0)
            or not np.allclose(np.diag(matrix), .5, atol=1e-6, rtol=0)):
        raise ValueError('Invalid comparison probability matrix')
    return matrix


def standardize(value):
    value = np.asarray(value, dtype=float)
    return (value-value.mean())/max(float(value.std()), 1e-6)


def signals(probability):
    matrix = probability_matrix(probability, len(probability))
    count = len(matrix)
    borda = standardize((matrix.sum(axis=1)-.5)/(count-1))
    transition = matrix/matrix.sum(axis=0)[None, :]
    rank = np.full(count, 1/count)
    for _ in range(50):
        rank = .85*(transition@rank)+.15/count
    # Count wins minus losses; the original sign(diff)*abs(diff) duplicated Borda.
    copeland = standardize(np.sign(matrix-.5).sum(axis=1))
    return borda, standardize(rank), copeland


def rerank(docs, score, probability, anchor, profile):
    """Head-to-head gated swaps, with every score looked up by document ID."""
    if profile not in ('borda', 'graph'):
        raise ValueError('Unknown consensus profile')
    docs, anchor = list(docs), list(anchor)
    if (len(set(docs)) != len(docs) or len(anchor) != 5 or len(set(anchor)) != 5
            or not set(anchor) <= set(docs)):
        raise ValueError('Invalid consensus candidates or anchor')
    matrix = probability_matrix(probability, len(docs))
    score = np.asarray(score, dtype=float)
    if score.shape != (len(docs),) or not np.isfinite(score).all():
        raise ValueError('Invalid consensus scores')
    borda, pagerank, copeland = signals(matrix)
    score = score + (.25*borda if profile == 'borda' else .15*borda+.08*pagerank+.05*copeland)
    threshold = .52 if profile == 'borda' else .56
    index = {doc: i for i, doc in enumerate(docs)}
    result = anchor.copy()
    for candidate in sorted(docs, key=lambda d: (-score[index[d]], d)):
        if candidate in result:
            continue
        weakest = min(result, key=lambda d: (score[index[d]], d))
        ci, wi = index[candidate], index[weakest]
        if score[ci] > score[wi] and matrix[ci, wi] >= threshold:
            result[result.index(weakest)] = candidate
    return sorted(result, key=lambda d: (-score[index[d]], d))
