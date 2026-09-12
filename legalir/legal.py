"""Extractive legal phrase evidence; no generated queries or external lexicons."""

import math
import unicodedata
from collections import Counter

from .data import terms


def folded_terms(text):
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return terms("".join(c for c in text if not unicodedata.combining(c)))


def phrase_evidence(query, texts):
    """Score contiguous query phrases, weighted by rarity in candidate passages.

    The original query is not expanded. Folding bridges supplied ASCII titles and
    accented questions. Long phrases retain concepts lost by unigram BM25.
    """
    query_words = folded_terms(query)
    query_grams = {n: set(tuple(query_words[i:i+n]) for i in range(len(query_words)-n+1))
                   for n in (2, 3, 4)}
    matches = []
    frequency = Counter()
    for text in texts:
        words = folded_terms(text)
        found = set()
        for n, grams in query_grams.items():
            found.update(tuple(words[i:i+n]) for i in range(len(words)-n+1)
                         if tuple(words[i:i+n]) in grams)
        matches.append(found)
        frequency.update(found)
    count = len(texts)
    return [sum((len(gram)-1) * math.log1p((count+1)/(frequency[gram]+1))
                for gram in found) for found in matches]


def diverse_chunks(chunk_ids, text_lookup, count):
    """Avoid spending all reranker slots on near-identical overlapping windows."""
    selected, bags = [], []
    deferred = []
    for chunk in chunk_ids:
        bag = set(terms(text_lookup(chunk)))
        if any(len(bag & other) / max(1, len(bag | other)) > .8 for other in bags):
            deferred.append(chunk)
            continue
        selected.append(chunk)
        bags.append(bag)
        if len(selected) == count:
            return selected
    return (selected + deferred)[:count]