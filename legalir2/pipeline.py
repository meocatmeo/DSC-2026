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
    "top_k": 5,
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


class Pipeline:
    def __init__(self, store_path, lexical_directory, dense_directory=None,
                 embedding_path=None, reranker_path=None, device="cuda",
                 settings=None, lexical_settings=None):
        self.settings = dict(DEFAULTS, **(settings or {}))
        self.store = Store(store_path)
        self.lexical = Lexical(lexical_directory, lexical_settings)
        self.dense = None
        self.reranker = None
        parameters = {}
        if dense_directory:
            from .dense import DenseIndex
            self.dense = DenseIndex(dense_directory, store_path, embedding_path, device=device)
            parameters["embedding"] = self.dense.parameters
        if reranker_path:
            from .rerank import Reranker
            self.reranker = Reranker(reranker_path, device=device,
                                     max_length=self.settings["rerank_max_length"])
            parameters["reranker"] = self.reranker.parameters
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

    def candidates(self, query):
        """Fused document shortlist plus the per-chunk evidence behind it."""
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
        fused = rrf(rankings, weights, self.settings["rrf_constant"])
        if len(fused) < self.settings["top_k"]:
            present = {doc for doc, _ in fused}
            fused = fused + [(doc, 0.0) for doc in self.store.doc_ids if doc not in present]
        return fused[:self.settings["shortlist"]], chunk_prior

    # ----------------------------------------------------------------- reranking

    def _passages(self, doc_id, chunk_prior):
        chunk_ids = self.store.document_chunk_ids(doc_id)
        ordered = sorted(chunk_ids, key=lambda c: (-chunk_prior.get(c, 0.0), c))
        pool = ordered[:max(8, self.settings["passages"] * 3)]
        picked = diverse(pool, lambda c: self.store.chunk_text(c, with_label=False),
                         self.settings["passages"])
        radius = self.settings["context_radius"]
        return [with_neighbours(chunk_ids, chunk,
                                lambda c: self.store.chunk_text(c, with_label=(c == chunk)),
                                radius) if radius else self.store.chunk_text(chunk)
                for chunk in picked]

    def search(self, query, top_k=None, diagnostics=None):
        top_k = top_k or self.settings["top_k"]
        fused, chunk_prior = self.candidates(query)
        if diagnostics is not None:
            diagnostics["candidates"] = [doc for doc, _ in fused]
        if not self.reranker:
            return [doc for doc, _ in fused[:top_k]]
        shortlist = fused[:self.settings["candidates"]]
        texts, owners = [], []
        for doc, _ in shortlist:
            for passage in self._passages(doc, chunk_prior):
                texts.append(passage)
                owners.append(doc)
        started = time.time()
        scores = self.reranker.score(query, texts, self.settings["rerank_batch"],
                                     self.settings["rerank_max_length"])
        self.timings["rerank"] += time.time() - started
        grouped = defaultdict(list)
        for doc, score in zip(owners, scores):
            grouped[doc].append(score)
        neural = [(doc, aggregate_passages(grouped[doc], self.settings["aggregate"],
                                           self.settings["aggregate_weight"]))
                  for doc, _ in shortlist]
        blended = normalised([neural, shortlist], [1.0, self.settings["rerank_blend"]])
        order = [doc for doc, _ in blended]
        order += [doc for doc, _ in fused if doc not in set(order)]
        if diagnostics is not None:
            diagnostics["reranked"] = order[:20]
        return order[:top_k]

    def manifest(self):
        return {"settings": self.settings, "lexical": self.lexical.settings,
                "audit": self.audit, "store": self.store.metadata,
                "timings": {k: round(v, 2) for k, v in self.timings.items()}}

    def close(self):
        self.store.close()
        for component in (self.dense, self.reranker):
            if component:
                component.close()
