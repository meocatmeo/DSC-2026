"""End to end retrieval: lexical + dense candidates, fused, then reranked.

Stages
------
1. Lexical BM25F and hashed bigram channels over article chunks.
2. Optional dense bi-encoder channel over the same chunks.
3. Weighted reciprocal rank fusion to a document shortlist.
4. Cross encoder reranking of a few distinct passages per shortlisted document,
   with the document score taken as its best passage.
5. Final order blends the reranker with the fusion rank, then the top five
   document ids are returned.

Every stage is optional so each one can be ablated on the validation split.
"""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .fuse import diverse, normalised, rrf, with_neighbours
from .lexical import Lexical
from .models import audit
from .store import Store

DEFAULTS = {
    "candidates": 50,          # documents handed to the reranker
    "passages": 3,             # distinct passages scored per document
    "shortlist": 200,          # documents kept from fusion before reranking
    "chunk_candidates": 600,   # chunks pulled from each retrieval channel
    "lexical_weight": 1.0,
    "dense_weight": 1.0,
    "rrf_constant": 60,
    "rerank_blend": 0.02,      # weight of the fusion prior, used mainly to break ties
    "context_radius": 0,
    "rerank_batch": 8,
    "rerank_max_length": 768,
    "aggregate": "max_plus",   # max | max_plus | softmax, over a document's passages
    "aggregate_weight": 0.2,
    "stage2_candidates": 25,   # documents the second and later rerankers rescore
    "reranker_weights": None,  # per reranker ensemble weights; None means equal
    "top_k": 5,
    "stacker": None,           # path to fitted rank-fusion weights; None keeps the blend
}


def aggregate_passages(scores, mode="max_plus", weight=0.2):
    """Turn several passage scores for one document into a single document score.

    `max` is the usual choice; `max_plus` adds a fraction of the runner up, which
    slightly favours documents that are relevant in more than one article; `softmax`
    is a smooth maximum for when passage scores are close together.
    """
    if not scores:
        return -1e30
    ordered = sorted(scores, reverse=True)
    if mode == "max" or len(ordered) == 1:
        return ordered[0]
    if mode == "softmax":
        top = ordered[0]
        shifted = [pow(2.718281828, (s - top) / max(weight, 1e-6)) for s in ordered]
        total = sum(shifted)
        return top + weight * (total and __import__("math").log(total))
    return ordered[0] + weight * ordered[1]


def final_order(fused, passage_scores, settings, weights=None, channels=None,
                stacker=None):
    """Final document order from cached per-passage reranker scores.

    Pure: no model, no store, no I/O. `fused` is the fusion ranking, `passage_scores`
    is one {doc: [score, ...]} table per reranker, and every knob comes from
    `settings`. Both the live pipeline and the offline sweep call this, so a setting
    that wins during tuning is the setting that ships - the divergence that cost this
    project two points when mining and prediction each had their own fusion is not
    available to happen here.
    """
    shortlist = fused[:settings["candidates"]]
    docs = [d for d, _ in shortlist]
    if stacker is not None and channels is not None:
        # The learned combination replaces the hand-weighted blend entirely. Everything
        # below - aggregate, rerank_blend, the normalised sum - is the thing it exists
        # to supersede, so running both would be scoring the same evidence twice.
        ordered = stacker.order(docs, passage_scores, channels)
        return ordered + [d for d, _ in fused if d not in set(ordered)]
    count, mode = settings["passages"], settings["aggregate"]
    weight = settings["aggregate_weight"]
    weights = weights or [1.0] * len(passage_scores)

    def aggregated(index, subset):
        table = passage_scores[index]
        return [(d, aggregate_passages(table.get(d, [])[:count], mode, weight))
                for d in subset]

    first = aggregated(0, docs)
    stage2 = settings["stage2_candidates"]
    if len(passage_scores) > 1 and stage2:
        ordered = sorted(first, key=lambda x: (-x[1], x[0]))
        promoted = [d for d, _ in ordered[:stage2]]
        chosen = set(promoted)
        rankings, used = [[(d, s) for d, s in first if d in chosen]], [weights[0]]
        for index in range(1, len(passage_scores)):
            rankings.append(aggregated(index, promoted))
            used.append(weights[index])
        prior = [(d, s) for d, s in shortlist if d in chosen]
        top = normalised(rankings + [prior], used + [settings["rerank_blend"]])
        order = [d for d, _ in top]
        order += [d for d, _ in ordered if d not in set(order)]
    else:
        rankings, used = [first], [weights[0]]
        for index in range(1, len(passage_scores)):
            rankings.append(aggregated(index, docs))
            used.append(weights[index])
        blended = normalised(rankings + [shortlist], used + [settings["rerank_blend"]])
        order = [d for d, _ in blended]
    order += [d for d, _ in fused if d not in set(order)]
    return order


class Pipeline:
    def __init__(self, store_path, lexical_directory, dense_directory=None,
                 embedding_path=None, reranker_path=None, device="cuda",
                 settings=None, lexical_settings=None):
        self.settings = dict(DEFAULTS, **(settings or {}))
        self.store = Store(store_path)
        self.lexical = Lexical(lexical_directory, lexical_settings)
        self.dense = None
        self.rerankers = []
        parameters = {}
        if dense_directory:
            from .dense import DenseIndex
            self.dense = DenseIndex(dense_directory, store_path, embedding_path, device=device)
            parameters["embedding"] = self.dense.parameters
        paths = ([reranker_path] if isinstance(reranker_path, (str, type(Path())))
                 else list(reranker_path or []))
        if paths:
            from .rerank import Reranker
            for position, path in enumerate(paths):
                model = Reranker(path, device=device,
                                 max_length=self.settings["rerank_max_length"])
                self.rerankers.append(model)
                parameters[f"reranker{position + 1}"] = model.parameters
        self.stacker = None
        if self.settings.get("stacker"):
            from .stack import Stacker
            self.stacker = Stacker.load(self.settings["stacker"])
        self.audit = audit(parameters) if parameters else {"components": {}, "total": 0}
        self.chunk_document = self.store.chunk_documents()
        self.timings = defaultdict(float)

    # ----------------------------------------------------------------- retrieval

    def _dense_documents(self, query):
        scores = self.dense.scores(query)
        limit = min(self.settings["chunk_candidates"], len(scores))
        picks = np.argpartition(-scores, limit - 1)[:limit]
        picks = picks[np.argsort(-scores[picks], kind="stable")]
        chunk_ids = self.dense.chunk_ids[picks]
        values = scores[picks]
        grouped, chunk_ranking = {}, []
        for position in range(len(picks)):
            chunk = int(chunk_ids[position])
            chunk_ranking.append((chunk, float(values[position])))
            slot = grouped.setdefault(self.chunk_document[chunk], [])
            if len(slot) < self.lexical.settings["depth"]:
                slot.append(float(values[position]))
        decay = self.lexical.settings["decay"]
        documents = [(doc, group[0] + decay * sum(group[1:])) for doc, group in grouped.items()]
        documents.sort(key=lambda x: (-x[1], x[0]))
        return documents, chunk_ranking

    def candidates(self, query, channels=None):
        """Fused document shortlist plus the per-chunk evidence behind it.

        `channels`, when a dict is passed, is filled with each retrieval channel's own
        ranking. Fusion collapses them into one number, and the learned stacker needs
        them separately - a document that BM25F ranked first and the dense channel
        ranked fortieth is a different object from one both ranked twentieth, and the
        fused score cannot tell them apart.
        """
        started = time.time()
        lexical_documents, lexical_chunks = self.lexical.search(
            query, self.settings["chunk_candidates"])
        self.timings["lexical"] += time.time() - started
        rankings, weights = [lexical_documents], [self.settings["lexical_weight"]]
        chunk_prior = defaultdict(float)
        for rank, (chunk, _) in enumerate(lexical_chunks, 1):
            chunk_prior[chunk] += self.settings["lexical_weight"] / (60 + rank)
        if self.dense:
            started = time.time()
            dense_documents, dense_chunks = self._dense_documents(query)
            self.timings["dense"] += time.time() - started
            rankings.append(dense_documents)
            weights.append(self.settings["dense_weight"])
            for rank, (chunk, _) in enumerate(dense_chunks, 1):
                chunk_prior[chunk] += self.settings["dense_weight"] / (60 + rank)
        if channels is not None:
            channels["lexical"] = lexical_documents
            channels["dense"] = dense_documents if self.dense else []
        fused = rrf(rankings, weights, self.settings["rrf_constant"])
        if channels is not None:
            channels["fused"] = fused
        if len(fused) < self.settings["top_k"]:
            present = {doc for doc, _ in fused}
            fused = fused + [(doc, 0.0) for doc in self.store.doc_ids if doc not in present]
        return fused[:self.settings["shortlist"]], chunk_prior

    # ----------------------------------------------------------------- reranking

    def _passages(self, doc_id, chunk_prior, count=None):
        count = count if count is not None else self.settings["passages"]
        chunk_ids = self.store.document_chunk_ids(doc_id)
        ordered = sorted(chunk_ids, key=lambda c: (-chunk_prior.get(c, 0.0), c))
        pool = ordered[:max(8, count * 3)]
        picked = diverse(pool, lambda c: self.store.chunk_text(c, with_label=False),
                         count)
        radius = self.settings["context_radius"]
        return [with_neighbours(chunk_ids, chunk,
                                lambda c: self.store.chunk_text(c, with_label=(c == chunk)),
                                radius) if radius else self.store.chunk_text(chunk)
                for chunk in picked]

    def _rerank(self, model, query, documents, chunk_prior, cache, passages=None):
        """Raw per-passage scores for each document: {doc: [score, ...]}.

        Deliberately *not* aggregated here. Aggregation is a setting, and returning
        raw scores is what lets `tune --stage rerank` score once and then sweep every
        aggregation offline, instead of paying a GPU pass per candidate setting.
        """
        count = passages if passages is not None else self.settings["passages"]
        texts, owners = [], []
        for doc in documents:
            if doc not in cache:
                cache[doc] = self._passages(doc, chunk_prior, count)
            for passage in cache[doc]:
                texts.append(passage)
                owners.append(doc)
        started = time.time()
        scores = model.score(query, texts, self.settings["rerank_batch"],
                             self.settings["rerank_max_length"])
        self.timings["rerank"] += time.time() - started
        grouped = defaultdict(list)
        for doc, score in zip(owners, scores):
            grouped[doc].append(float(score))
        return grouped

    def search(self, query, top_k=None, diagnostics=None):
        """Fuse, rerank, and return the top document ids.

        With one reranker this is a single pass over `candidates` documents. With
        more than one, the first model is the cheap wide pass and the rest rescore
        only its best `stage2_candidates`, so a deeper - and therefore higher
        ceilinged - shortlist costs little more than a shallow one. Scores from
        different rerankers are not comparable, so they are min-max normalised
        within this question before being combined.
        """
        top_k = top_k or self.settings["top_k"]
        channels = {} if self.stacker else None
        fused, chunk_prior = self.candidates(query, channels=channels)
        if diagnostics is not None:
            diagnostics["candidates"] = [doc for doc, _ in fused]
        if not self.rerankers:
            return [doc for doc, _ in fused[:top_k]]

        shortlist = fused[:self.settings["candidates"]]
        docs = [d for d, _ in shortlist]
        passage_cache, tables = {}, []
        weights = self.settings["reranker_weights"] or [1.0] * len(self.rerankers)
        tables.append(self._rerank(self.rerankers[0], query, docs, chunk_prior, passage_cache))

        stage2 = self.settings["stage2_candidates"]
        if len(self.rerankers) > 1 and stage2:
            # Later rerankers only see the wide pass's best, so score just those.
            ranked = sorted(((d, aggregate_passages(v, self.settings["aggregate"],
                                                    self.settings["aggregate_weight"]))
                             for d, v in tables[0].items()), key=lambda x: (-x[1], x[0]))
            promoted = [d for d, _ in ranked[:stage2]]
        else:
            promoted = docs
        for model in self.rerankers[1:]:
            tables.append(self._rerank(model, query, promoted, chunk_prior, passage_cache))
        order = final_order(fused, tables, self.settings, weights,
                            channels=channels, stacker=self.stacker)
        if diagnostics is not None:
            diagnostics["reranked"] = order[:20]
        return order[:top_k]

    def manifest(self):
        # The model paths belong in here, not only in whatever shell command happened
        # to launch the run. A scored result that cannot say which checkpoints produced
        # it is not reproducible, and picking a winner from a table of numbers whose
        # provenance lives in scrollback is how the wrong model gets submitted.
        return {"settings": self.settings, "lexical": self.lexical.settings,
                "models": {
                    "stacker": self.settings.get("stacker"),
                    "rerankers": [str(r.local.path) for r in self.rerankers],
                    "embedding": str(self.dense.embedder.local.path) if self.dense else None,
                    "dense": str(self.dense.directory) if self.dense else None},
                "audit": self.audit, "store": self.store.metadata,
                "timings": {k: round(v, 2) for k, v in self.timings.items()}}

    def close(self):
        self.store.close()
        for component in [self.dense] + self.rerankers:
            if component:
                component.close()
