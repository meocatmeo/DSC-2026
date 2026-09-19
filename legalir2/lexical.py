"""BM25F over article chunks, plus a hashed word-bigram phrase channel.

Two numpy posting files are built from the chunk store:

* `unigram.npz` - fields label (title, document number, accented subject), crumb
  (chapter / section path) and body (the article text), scored with BM25F so each
  field gets its own weight and length normalisation.
* `bigram.npz`  - the same layout over consecutive word pairs, hashed into a fixed
  bucket space. Vietnamese words span several syllables, so unigram matching
  discards the phrase that carries the legal concept; this channel restores it
  without needing a word segmenter or any external resource.

`prepare` bakes weights and length normalisation into the postings once, after
which a query is only gathers and adds - fast enough to sweep parameters on the
full validation split with no GPU.
"""
import array
import re
import time
import unicodedata
import zlib
from pathlib import Path

import numpy as np

from .store import connect

TOKEN = re.compile(r"[0-9\w]+", re.UNICODE)
UNIGRAM_FIELDS = ("label", "crumb", "body")
BIGRAM_FIELDS = ("label", "body")
BUCKETS = 1 << 22


def tokens(text):
    return TOKEN.findall(unicodedata.normalize("NFC", text or "").lower())


def bigrams(stream):
    return [zlib.crc32((a + "\x1f" + b).encode("utf-8")) % BUCKETS
            for a, b in zip(stream, stream[1:])]


# --------------------------------------------------------------------- building

def _pack(terms, chunks, freqs, size):
    term = np.frombuffer(terms, dtype="int32")
    chunk = np.frombuffer(chunks, dtype="int32")
    freq = np.frombuffer(freqs, dtype="uint8")
    order = np.argsort(term, kind="stable")
    term, chunk, freq = term[order], chunk[order], freq[order]
    del order
    counts = np.bincount(term, minlength=size)
    start = np.zeros(size + 1, dtype="int64")
    np.cumsum(counts, out=start[1:])
    return start, chunk, freq, counts


def build(store_path, directory, bigram=True):
    """Write unigram.npz and bigram.npz next to each other in `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    db = connect(store_path)
    documents = db.execute("SELECT id,title,number,subject FROM documents ORDER BY id").fetchall()
    doc_ids = [r[0] for r in documents]
    position_of = {r[0]: i for i, r in enumerate(documents)}
    labels = {r[0]: tokens(" | ".join(x for x in (r[1], r[2], r[3]) if x)) for r in documents}
    label_grams = {doc: bigrams(stream) for doc, stream in labels.items()}

    vocabulary = {}
    uni = {f: (array.array("i"), array.array("i"), array.array("B"), array.array("i"))
           for f in UNIGRAM_FIELDS}
    big = {f: (array.array("i"), array.array("i"), array.array("B"), array.array("i"))
           for f in BIGRAM_FIELDS}
    chunk_ids, doc_index, unit = array.array("i"), array.array("i"), array.array("B")
    started, position = time.time(), 0
    cursor = db.execute("SELECT id, document_id, unit, crumb, body FROM chunks ORDER BY id")
    while batch := cursor.fetchmany(20000):
        for chunk_id, document_id, kind, crumb, body in batch:
            chunk_ids.append(chunk_id)
            doc_index.append(position_of[document_id])
            unit.append(0 if kind == "header" else 1)
            body_tokens = tokens(body)
            streams = (("label", labels[document_id]), ("crumb", tokens(crumb)),
                       ("body", body_tokens))
            for field, stream in streams:
                terms, chunks, freqs, lengths = uni[field]
                lengths.append(len(stream))
                counts = {}
                for token in stream:
                    counts[token] = counts.get(token, 0) + 1
                for token, count in counts.items():
                    terms.append(vocabulary.setdefault(token, len(vocabulary)))
                    chunks.append(position)
                    freqs.append(min(count, 255))
            if bigram:
                for field, stream in (("label", label_grams[document_id]),
                                      ("body", bigrams(body_tokens))):
                    terms, chunks, freqs, lengths = big[field]
                    lengths.append(len(stream))
                    counts = {}
                    for gram in stream:
                        counts[gram] = counts.get(gram, 0) + 1
                    for gram, count in counts.items():
                        terms.append(gram)
                        chunks.append(position)
                        freqs.append(min(count, 255))
            position += 1
        print(f"  indexed {position} chunks {time.time() - started:.0f}s", flush=True)
    db.close()

    shared = {"chunk_ids": np.frombuffer(chunk_ids, dtype="int32"),
              "doc_index": np.frombuffer(doc_index, dtype="int32"),
              "doc_ids": np.array(doc_ids, dtype=object),
              "unit": np.frombuffer(unit, dtype="uint8")}

    payload = dict(shared, vocab=np.array(sorted(vocabulary, key=vocabulary.get), dtype=object))
    df = None
    for field in UNIGRAM_FIELDS:
        terms, chunks, freqs, lengths = uni[field]
        start, chunk, freq, counts = _pack(terms, chunks, freqs, len(vocabulary))
        payload.update({f"{field}_start": start, f"{field}_chunk": chunk, f"{field}_tf": freq,
                        f"{field}_len": np.frombuffer(lengths, dtype="int32")})
        if field == "body":
            df = counts
    payload["df"] = np.maximum(df, 1)
    np.savez(directory / "unigram.npz", **payload)
    print(f"  wrote unigram.npz ({len(vocabulary)} terms) {time.time() - started:.0f}s", flush=True)

    if bigram:
        payload = dict(shared, vocab=np.array([], dtype=object), buckets=np.array([BUCKETS]))
        for field in BIGRAM_FIELDS:
            terms, chunks, freqs, lengths = big[field]
            start, chunk, freq, counts = _pack(terms, chunks, freqs, BUCKETS)
            payload.update({f"{field}_start": start, f"{field}_chunk": chunk, f"{field}_tf": freq,
                            f"{field}_len": np.frombuffer(lengths, dtype="int32")})
            if field == "body":
                df = counts
        payload["df"] = np.maximum(df, 1)
        np.savez(directory / "bigram.npz", **payload)
        print(f"  wrote bigram.npz {time.time() - started:.0f}s", flush=True)
    return {"chunks": position, "terms": len(vocabulary), "seconds": round(time.time() - started)}


# ---------------------------------------------------------------------- scoring

class Postings:
    fields = UNIGRAM_FIELDS

    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.vocab = {t: i for i, t in enumerate(data["vocab"])}
        self.chunk_ids = data["chunk_ids"]
        self.doc_index = data["doc_index"]
        self.doc_ids = list(data["doc_ids"])
        self.unit = data["unit"]
        self.chunks = len(self.chunk_ids)
        self.start, self.chunk, self.tf, self.lengths, self.average = {}, {}, {}, {}, {}
        for field in self.fields:
            self.start[field] = data[f"{field}_start"]
            self.chunk[field] = data[f"{field}_chunk"]
            self.tf[field] = data[f"{field}_tf"].astype("float32")
            self.lengths[field] = data[f"{field}_len"].astype("float32")
            self.average[field] = float(max(1.0, self.lengths[field].mean()))
        self.df = data["df"].astype("float32")
        self.idf = np.log(1.0 + (self.chunks - self.df + 0.5) / (self.df + 0.5)).astype("float32")
        self.value, self.settings, self.k1 = None, None, 1.2

    def prepare(self, weights, b, k1=1.2):
        settings = (tuple(sorted(weights.items())), tuple(sorted(b.items())), k1)
        if settings == self.settings:
            return
        self.value = {}
        for field in self.fields:
            norm = 1.0 - b[field] + b[field] * self.lengths[field] / self.average[field]
            self.value[field] = (self.tf[field] * weights[field]
                                 / norm[self.chunk[field]]).astype("float32")
        self.k1, self.settings = k1, settings

    def _accumulate(self, indexes, header_penalty):
        total = np.zeros(self.chunks, dtype="float32")
        for index in indexes:
            parts = []
            for field in self.fields:
                lo, hi = self.start[field][index], self.start[field][index + 1]
                if hi > lo:
                    parts.append((self.chunk[field][lo:hi], self.value[field][lo:hi]))
            if not parts:
                continue
            if len(parts) == 1:
                ids, weighted = parts[0]
            else:
                ids = np.concatenate([p[0] for p in parts])
                values = np.concatenate([p[1] for p in parts])
                ids, inverse = np.unique(ids, return_inverse=True)
                weighted = np.bincount(inverse, weights=values).astype("float32")
            total[ids] += self.idf[index] * weighted / (self.k1 + weighted)
        if header_penalty:
            total[self.unit == 0] *= (1.0 - header_penalty)
        return total

    def score(self, query_tokens, header_penalty=0.0):
        indexes = [self.vocab[t] for t in dict.fromkeys(query_tokens) if t in self.vocab]
        return self._accumulate(indexes, header_penalty)


class BigramPostings(Postings):
    fields = BIGRAM_FIELDS

    def score(self, query_tokens, header_penalty=0.0):
        return self._accumulate(dict.fromkeys(bigrams(list(query_tokens))), header_penalty)


# Selected by coordinate descent on 600 *training* questions (`python -m legalir2 tune`),
# so the validation split stays an unbiased check. Measured on the full 1,050 question
# holdout: Recall@5 0.8483, candidate recall 0.9608 at 50 and 0.9752 at 100.
# `crumb` at zero is a real result, not a disabled feature: the chapter/section path
# adds nothing to lexical scoring, while it still goes to the embedder and the cross
# encoder, where it tells them which part of the document a passage came from.
DEFAULTS = {
    "label": 1.0, "crumb": 0.0, "body": 1.0,
    "b_label": 0.75, "b_crumb": 0.75, "b_body": 0.9,
    "k1": 1.2, "beta": 1.5, "decay": 0.15, "depth": 8,
    "chunk_candidates": 600, "header_penalty": 0.0,
}


class Lexical:
    """The lexical stage: BM25F plus the bigram channel, aggregated to documents."""

    def __init__(self, directory, settings=None):
        directory = Path(directory)
        self.settings = dict(DEFAULTS, **(settings or {}))
        self.unigram = Postings(directory / "unigram.npz")
        bigram_path = directory / "bigram.npz"
        self.bigram = BigramPostings(bigram_path) if bigram_path.is_file() else None
        self.apply(self.settings)

    def apply(self, settings):
        self.settings.update(settings)
        s = self.settings
        self.unigram.prepare({"label": s["label"], "crumb": s["crumb"], "body": s["body"]},
                             {"label": s["b_label"], "crumb": s["b_crumb"], "body": s["b_body"]},
                             s["k1"])
        if self.bigram:
            self.bigram.prepare({"label": s["label"], "body": s["body"]},
                                {"label": s["b_label"], "body": s["b_body"]}, s["k1"])

    def chunk_scores(self, query):
        stream = tokens(query)
        scores = self.unigram.score(stream, self.settings["header_penalty"])
        if self.bigram and self.settings["beta"]:
            scores = scores + self.settings["beta"] * self.bigram.score(
                stream, self.settings["header_penalty"])
        return scores

    def top_chunks(self, scores, limit=None):
        limit = min(limit or self.settings["chunk_candidates"], self.unigram.chunks)
        picks = np.argpartition(-scores, limit - 1)[:limit]
        picks = picks[np.argsort(-scores[picks], kind="stable")]
        return picks[scores[picks] > 0]

    def documents(self, picks, scores, decay=None, depth=None):
        """Best chunk per document plus a decayed bonus from its further chunks."""
        decay = self.settings["decay"] if decay is None else decay
        depth = self.settings["depth"] if depth is None else depth
        docs = self.unigram.doc_index[picks]
        values = scores[picks]
        grouped = {}
        for position in range(len(picks)):
            slot = grouped.setdefault(int(docs[position]), [])
            if len(slot) < depth:
                slot.append(float(values[position]))
        out = [(self.unigram.doc_ids[doc], group[0] + decay * sum(group[1:]))
               for doc, group in grouped.items()]
        out.sort(key=lambda x: (-x[1], x[0]))
        return out

    def search(self, query, limit=None, scores=None):
        """Return (document ranking, chunk ranking) for one question."""
        scores = self.chunk_scores(query) if scores is None else scores
        picks = self.top_chunks(scores, limit)
        chunk_ranking = [(int(self.unigram.chunk_ids[p]), float(scores[p])) for p in picks]
        return self.documents(picks, scores), chunk_ranking
