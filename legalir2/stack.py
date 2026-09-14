"""Learned rank fusion: combine the reranker with the retrieval evidence it overrides.

The shipped system orders documents like this::

    final = normalised([reranker_scores], [1.0]) + normalised([fusion_prior], [0.02])

The reranker carries weight 1.0. Everything retrieval learned - BM25F, the hashed bigram
channel, dense similarity, the fused rank - is compressed into one number carrying 0.02.
That constant was chosen by hand and never fitted, and retrieval is not a weak signal:
its own top five already contains the right document 89% of the time.

The diagnostics say what this costs. Against the 0.9433 model, reranking rescues 80 gold
documents that retrieval ranked below fifth and discards 23 that retrieval had ranked
correctly. Net +57, so the reranker earns its place - but those 23 are recoverable by
weighting the signals instead of asserting a 50:1 ratio between them.

So: a linear model over per-question-normalised features, fitted on the organiser's own
labels with the same listwise objective the reranker uses. Linear is deliberate. With
~15 features it cannot overfit 300k rows in any interesting way, the learned weights are
readable as "how much is each signal worth", and it is arithmetic at inference - no
parameters against the deployed budget, nothing new to download.

Multi-positive targets are used here too, for the same reason as in the reranker: the
scorer gives partial credit per gold document, so the objective should reward lifting
all of them.
"""
import json
import time
from pathlib import Path

import numpy as np

from .store import write_json

FEATURES = (
    "rr_max",           # best passage score for this document
    "rr_second",        # runner up, or the best again when only one passage
    "rr_mean",
    "rr_spread",        # max - mean: is one passage carrying it, or the whole document
    "rr_near_top",      # passages within 1.0 of this document's best
    "fused_score",
    "fused_rr",         # 1 / (1 + rank), so rank 1 dominates and the tail flattens
    "lex_score",
    "lex_rr",
    "den_score",
    "den_rr",
    "lex_den_gap",      # channels disagreeing is itself informative
)


def _ranks(ranking):
    """{doc: (score, rank)} from a [(doc, score), ...] ranking."""
    return {doc: (float(score), index + 1) for index, (doc, score) in enumerate(ranking)}


def _normalise(column):
    """Per question min-max. Scores are only comparable within one question."""
    low, high = float(column.min()), float(column.max())
    span = high - low
    return (column - low) / span if span > 1e-12 else np.zeros_like(column)


def build_features(docs, passage_scores, channels):
    """Feature matrix for one question, per-question normalised. Shape (len(docs), 12).

    Missing channels are handled as "ranked worse than anything present" rather than as
    zero, because absence from a channel is evidence, not a neutral value.
    """
    table = passage_scores[0] if passage_scores else {}
    lex, den = _ranks(channels.get("lexical") or []), _ranks(channels.get("dense") or [])
    fused = _ranks(channels.get("fused") or [])
    absent = len(docs) + 1
    rows = []
    for doc in docs:
        values = sorted(table.get(doc, []), reverse=True) or [0.0]
        best, mean = values[0], sum(values) / len(values)
        lex_score, lex_rank = lex.get(doc, (0.0, absent))
        den_score, den_rank = den.get(doc, (0.0, absent))
        fus_score, fus_rank = fused.get(doc, (0.0, absent))
        rows.append([
            best,
            values[1] if len(values) > 1 else best,
            mean,
            best - mean,
            float(sum(1 for v in values if v >= best - 1.0)),
            fus_score, 1.0 / (1.0 + fus_rank),
            lex_score, 1.0 / (1.0 + lex_rank),
            den_score, 1.0 / (1.0 + den_rank),
            abs(1.0 / (1.0 + lex_rank) - 1.0 / (1.0 + den_rank)),
        ])
    matrix = np.asarray(rows, dtype="float64")
    for column in range(matrix.shape[1]):
        matrix[:, column] = _normalise(matrix[:, column])
    return matrix


class Stacker:
    """A fitted linear scorer. Arithmetic only - no model, no parameter budget."""

    def __init__(self, weights, names=FEATURES):
        self.weights = np.asarray(weights, dtype="float64")
        self.names = tuple(names)
        if self.weights.shape != (len(self.names),):
            raise ValueError(f"expected {len(self.names)} weights, got {self.weights.shape}")

    @classmethod
    def load(cls, path):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(record["weights"], record.get("features", FEATURES))

    def score(self, matrix):
        return matrix @ self.weights

    def order(self, docs, passage_scores, channels):
        scores = self.score(build_features(docs, passage_scores, channels))
        return [doc for _, doc in sorted(zip(-scores, docs), key=lambda x: (x[0], x[1]))]


def fit(rows, epochs=300, learning_rate=0.15, l2=1e-4, seed=2026, verbose=True):
    """Listwise softmax cross entropy on a linear scorer, full batch, plain gradient.

    `rows` is a list of (feature matrix, label vector) per question. The target spreads
    mass evenly over that question's gold documents, matching both the scorer's partial
    credit and the reranker's own objective.

    Full-batch gradient descent is not a compromise here: the problem is convex in the
    weights and a few hundred iterations over 300k rows is seconds of numpy.
    """
    rng = np.random.default_rng(seed)
    width = rows[0][0].shape[1]
    weights = rng.normal(0, 0.01, width)
    usable = [(matrix, labels) for matrix, labels in rows if labels.sum() > 0]
    if not usable:
        raise ValueError("no question has a gold document among its candidates")
    history = []
    for epoch in range(epochs):
        gradient, loss = np.zeros(width), 0.0
        for matrix, labels in usable:
            scores = matrix @ weights
            scores -= scores.max()
            probability = np.exp(scores)
            probability /= probability.sum()
            target = labels / labels.sum()
            loss -= float(target @ np.log(probability + 1e-12))
            gradient += matrix.T @ (probability - target)
        gradient = gradient / len(usable) + l2 * weights
        weights -= learning_rate * gradient
        loss /= len(usable)
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>4} listwise loss {loss:.4f}", flush=True)
        history.append(round(loss, 5))
    return weights, history


def collect(args, limit=2000, output=None, depth=None):
    """One GPU pass over training questions, caching features and labels.

    Training questions only. Fitting this on the validation split would tune the
    combination on the same questions used to judge it, and every number after that
    would be optimistic by an unknown amount.
    """
    from .pipeline import Pipeline
    from .predict import load_questions

    pipeline = Pipeline(args.store, args.lexical,
                        dense_directory=args.dense if (Path(args.dense) / "manifest.json").is_file() else None,
                        embedding_path=args.embedding_model,
                        reranker_path=args.reranker_model, device=args.device,
                        settings=json.loads(args.settings) if args.settings else None)
    if not pipeline.rerankers:
        raise ValueError("the stacker needs reranker scores; pass --reranker-model")
    depth = depth or pipeline.settings["candidates"]
    records = load_questions(args.questions, labeled=True)
    items = list(records.items())[:limit]

    cached, started, covered = [], time.time(), 0
    for number, (qid, record) in enumerate(items, 1):
        channels = {}
        fused, prior = pipeline.candidates(record["question"], channels=channels)
        docs = [d for d, _ in fused[:depth]]
        gold = set(record["answer"])
        labels = np.array([1.0 if d in gold else 0.0 for d in docs])
        if labels.sum() == 0:
            continue            # no gold in the pool: nothing for a re-ranking to learn
        covered += 1
        table = pipeline._rerank(pipeline.rerankers[0], record["question"], docs, prior, {})
        cached.append({"id": qid,
                       "features": build_features(docs, [table], channels).tolist(),
                       "labels": labels.tolist()})
        if number % 100 == 0:
            rate = number / max(1e-6, time.time() - started)
            print(f"  collected {number}/{len(items)}, "
                  f"eta {(len(items)-number)/max(rate,1e-9)/60:.1f} min", flush=True)
    pipeline.close()
    print(f"  {covered}/{len(items)} questions had a gold document in the pool")
    if output:
        with Path(output).open("w", encoding="utf-8") as stream:
            for row in cached:
                stream.write(json.dumps(row) + "\n")
    return cached


def train(cache_path, output, **kwargs):
    rows = []
    for line in Path(cache_path).open(encoding="utf-8"):
        record = json.loads(line)
        rows.append((np.asarray(record["features"]), np.asarray(record["labels"])))
    print(f"  fitting on {len(rows)} questions, "
          f"{sum(len(m) for m, _ in rows)} candidate rows, {len(FEATURES)} features")
    weights, history = fit(rows, **kwargs)
    order = np.argsort(-np.abs(weights))
    print("\n  learned weights, largest first:")
    for index in order:
        print(f"    {FEATURES[index]:<14} {weights[index]:+.4f}")
    write_json(output, {"features": list(FEATURES), "weights": weights.tolist(),
                        "questions": len(rows), "loss": history[-1],
                        "history": history[::25]})
    print(f"\n  wrote {output}")
    return weights
