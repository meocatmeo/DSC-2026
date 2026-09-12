"""FTS5 BM25, chunk dense retrieval, document RRF, and passage reranking."""

import heapq
from collections import defaultdict
from pathlib import Path

from .data import connect, file_hash, read_json, terms, write_json
from .legal import diverse_chunks, phrase_evidence


def lexical(db, query, limit):
    tokens = list(dict.fromkeys(terms(query)))[:128]
    if not tokens:
        return []
    expression = " OR ".join('"' + x.replace('"', '""') + '"' for x in tokens)
    return [(row[0], -row[1]) for row in db.execute(
        "SELECT rowid, bm25(search) AS score FROM search WHERE search MATCH ? ORDER BY score, rowid LIMIT ?",
        (expression, limit))]


def document_ranking(chunk_ranking, chunk_documents):
    seen, result = set(), []
    for chunk_id, score in chunk_ranking:
        doc = chunk_documents[chunk_id]
        if doc not in seen:
            seen.add(doc)
            result.append((doc, score))
    return result


def fuse(rankings, constant=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, 1):
            scores[doc] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def build_dense(database, model_path, directory, device="cuda", batch_size=8, max_length=1024):
    import numpy as np
    from .neural import Embedder

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Manifest is a completion marker; a failed rebuild must not appear valid.
    (directory / "manifest.json").unlink(missing_ok=True)
    db = connect(database)
    count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    model = Embedder(model_path, device=device)
    dimension = model.model.config.hidden_size
    vectors = np.lib.format.open_memmap(directory / "vectors.npy", mode="w+", dtype="float32", shape=(count, dimension))
    chunk_ids = []
    offset = 0
    cursor = db.execute("SELECT id,text FROM chunks ORDER BY id")
    try:
        while rows := cursor.fetchmany(batch_size):
            vectors[offset:offset + len(rows)] = model.encode([r[1] for r in rows], batch_size, max_length)
            chunk_ids.extend(r[0] for r in rows)
            offset += len(rows)
            if offset % (batch_size * 100) == 0:
                print(f"Embedded {offset}/{count} chunks", flush=True)
        vectors.flush()
        np.save(directory / "chunk_ids.npy", np.asarray(chunk_ids, dtype="int64"))
        manifest = {"database_sha256": file_hash(database), "model_manifest": read_json(Path(model_path) / "provenance.json"),
                    "parameters": model.parameters, "dimension": dimension, "chunks": count, "max_length": max_length}
        write_json(directory / "manifest.json", manifest)
    finally:
        vectors._mmap.close()
        db.close()
        model.close()
    return manifest


class Retriever:
    def __init__(self, database, dense=None, embedding=None, reranker=None, device="cuda",
                 candidates=60, chunk_candidates=600, passages=2, batch_size=4,
                 legal_weight=0.0):
        if min(candidates, chunk_candidates, passages, batch_size) < 1:
            raise ValueError("Retrieval sizes must be positive")
        self.db = connect(database)
        self.documents = [x[0] for x in self.db.execute("SELECT id FROM documents ORDER BY id")]
        self.chunk_documents = dict(self.db.execute("SELECT id,document_id FROM chunks"))
        self.doc_chunks = defaultdict(list)
        for chunk, doc in self.chunk_documents.items():
            self.doc_chunks[doc].append(chunk)
        self.candidates, self.chunk_candidates = candidates, chunk_candidates
        if not 0 <= legal_weight <= 1:
            raise ValueError("legal_weight must be between zero and one")
        self.legal_weight = legal_weight
        self.passages, self.batch_size = passages, batch_size
        self.embedder = self.reranker = None
        self.manifest = {"mode": "bm25", "parameters": 0, "database_sha256": file_hash(database)}
        if dense:
            import numpy as np
            from .neural import Embedder
            manifest = read_json(Path(dense) / "manifest.json")
            if manifest["database_sha256"] != self.manifest["database_sha256"]:
                raise ValueError("Dense index belongs to a different corpus database; rebuild it")
            if read_json(Path(embedding) / "provenance.json") != manifest["model_manifest"]:
                raise ValueError("Embedding checkpoint differs from dense index")
            self.vectors = np.load(Path(dense) / "vectors.npy", mmap_mode="r")
            self.vector_ids = np.load(Path(dense) / "chunk_ids.npy")
            if len(self.vector_ids) != len(self.chunk_documents) or set(self.vector_ids.tolist()) != set(self.chunk_documents):
                raise ValueError("Dense chunk IDs do not match corpus")
            if self.vectors.shape != (manifest["chunks"], manifest["dimension"]):
                raise ValueError("Invalid dense vector dimensions")
            self.embedder = Embedder(embedding, device=device)
            self.embedding_length = manifest["max_length"]
            self.manifest.update(mode="hybrid", embedding=manifest["model_manifest"], parameters=self.embedder.parameters)
        if reranker:
            from .neural import Reranker
            self.reranker = Reranker(reranker, device=device)
            self.manifest["reranker"] = read_json(Path(reranker) / "provenance.json")
            self.manifest["parameters"] += self.reranker.parameters
        if self.manifest["parameters"] >= 4_000_000_000:
            raise ValueError("Combined system is not strictly below four billion parameters")
        self.manifest.update(candidates=candidates, chunk_candidates=chunk_candidates, passages=passages,
                             legal_weight=legal_weight)

    def search(self, query, top_k=5):
        if not 1 <= top_k <= 5:
            raise ValueError("top_k must be between one and five")
        sparse = lexical(self.db, query, self.chunk_candidates)
        rankings = [document_ranking(sparse, self.chunk_documents)]
        passage_scores = defaultdict(float)
        for rank, (chunk, _) in enumerate(sparse, 1):
            passage_scores[chunk] += 1 / (60 + rank)
        if self.embedder:
            import numpy as np
            vector = self.embedder.encode([query], 1, self.embedding_length, query=True)[0]
            scores = self.vectors @ vector
            # Stable full sorting keeps ties reproducible across runs.
            positions = np.argsort(-scores, kind="stable")[:self.chunk_candidates]
            dense = [(int(self.vector_ids[i]), float(scores[i])) for i in positions]
            rankings.append(document_ranking(dense, self.chunk_documents))
            for rank, (chunk, _) in enumerate(dense, 1):
                passage_scores[chunk] += 1 / (60 + rank)
        fused = fuse(rankings)
        seen = {doc for doc, _ in fused}
        # Deterministic fallback for empty/unmatched questions, never invented IDs.
        fused.extend((doc, 0.0) for doc in self.documents if doc not in seen)
        candidates = fused[:max(top_k, self.candidates)]
        text_cache = {}

        def text_for(chunk):
            if chunk not in text_cache:
                text_cache[chunk] = self.db.execute("SELECT text FROM chunks WHERE id=?", (chunk,)).fetchone()[0]
            return text_cache[chunk]

        selected_passages = {}
        if self.legal_weight or self.reranker:
            for doc, _ in candidates:
                shortlist = heapq.nsmallest(max(8, self.passages * 4), self.doc_chunks[doc],
                                           key=lambda c: (-passage_scores[c], c))
                selected_passages[doc] = diverse_chunks(shortlist, text_for, max(3, self.passages))
        if self.legal_weight:
            owners, evidence_text = [], []
            for doc, _ in candidates:
                for chunk in selected_passages[doc]:
                    owners.append(doc)
                    evidence_text.append(text_for(chunk))
            evidence = defaultdict(float)
            for doc, score in zip(owners, phrase_evidence(query, evidence_text)):
                evidence[doc] = max(evidence[doc], score)
            strongest = max(evidence.values(), default=0) or 1
            fusion_max = max((score for _, score in candidates), default=0) or 1
            candidates = [(doc, (1-self.legal_weight)*score/fusion_max +
                           self.legal_weight*evidence[doc]/strongest) for doc, score in candidates]
            candidates.sort(key=lambda x: (-x[1], x[0]))
        self.last_candidates = [doc for doc, _ in candidates]
        if self.reranker:
            texts, owners = [], []
            for doc, _ in candidates:
                selected = selected_passages[doc][:self.passages]
                for chunk in selected:
                    texts.append(text_for(chunk))
                    owners.append(doc)
            scores = self.reranker.score(query, texts, self.batch_size)
            best = defaultdict(float)
            for doc, score in zip(owners, scores):
                best[doc] = max(best[doc], score)
            candidates.sort(key=lambda x: (-best[x[0]], -x[1], x[0]))
        return [doc for doc, _ in candidates[:top_k]]

    def close(self):
        self.db.close()
        if hasattr(self, "vectors"):
            self.vectors._mmap.close()
        for model in (self.embedder, self.reranker):
            if model:
                model.close()