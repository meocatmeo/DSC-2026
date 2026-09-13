"""Rank and score fusion, plus passage selection for reranking."""
from collections import defaultdict


def rrf(rankings, weights=None, constant=60, depth=None):
    """Weighted reciprocal rank fusion. Robust when channel scores are not comparable."""
    weights = weights or [1.0] * len(rankings)
    scores = defaultdict(float)
    for weight, ranking in zip(weights, rankings):
        if not weight:
            continue
        for rank, (key, _) in enumerate(ranking[:depth] if depth else ranking, 1):
            scores[key] += weight / (constant + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def normalised(rankings, weights=None, depth=None):
    """Weighted sum of min-max normalised scores, for comparable channels."""
    weights = weights or [1.0] * len(rankings)
    scores = defaultdict(float)
    for weight, ranking in zip(weights, rankings):
        ranking = ranking[:depth] if depth else ranking
        if not weight or not ranking:
            continue
        values = [v for _, v in ranking]
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        for key, value in ranking:
            scores[key] += weight * (value - low) / span
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def jaccard(first, second):
    return len(first & second) / max(1, len(first | second))


def diverse(chunk_ids, text_of, count, threshold=0.8):
    """Pick distinct passages, so reranker slots are not spent on overlapping windows."""
    selected, bags, deferred = [], [], []
    for chunk in chunk_ids:
        bag = set(text_of(chunk).lower().split())
        if any(jaccard(bag, other) > threshold for other in bags):
            deferred.append(chunk)
            continue
        selected.append(chunk)
        bags.append(bag)
        if len(selected) >= count:
            return selected
    return (selected + deferred)[:count]


def with_neighbours(chunk_ids_of_document, anchor, text_of, radius=0):
    """Join the anchor passage with its neighbours inside the same document."""
    if radius <= 0:
        return text_of(anchor)
    position = chunk_ids_of_document.index(anchor)
    window = chunk_ids_of_document[max(0, position - radius):position + radius + 1]
    seen, pieces = set(), []
    for chunk in window:
        text = text_of(chunk)
        if text in seen:
            continue
        seen.add(text)
        pieces.append(text)
    return "\n".join(pieces)
