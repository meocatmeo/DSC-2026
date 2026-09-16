"""Frozen candidate and score cache: measurement at 2.6x the resolution, for free.

Two problems ended phase four, and this fixes both.

**Resolution.** Judging on a 1,050-question holdout resolves a paired difference of
about 0.0094. Every remaining improvement is in the 0.003-0.008 band, so three of four
interventions came back "not distinguishable from no change" - not because the ideas
were wrong but because the ruler was too coarse. Scoring across all ~7,000 labelled
questions makes that 0.0036.

**Cost.** Each configuration cost a 16-minute GPU pass to evaluate, so a sweep could
only afford 400 questions and paid for it with a winner's curse that did not survive
contact with held-out data. But the expensive part - the reranker's forward pass - does
not depend on any of the settings being tuned. Score once, cache the raw numbers, and
every ranking-level question (aggregation, candidate depth, blend weights, the learned
stacker) becomes arithmetic.

**The honesty problem, and why cross-fitting is not optional.** The deployed reranker
was trained on 5,950 of those 7,000 questions. Scoring them with it measures what it
memorised. So the cache is built from *cross-fitted* models: the questions are split in
two, a reranker is trained on each half, and every question is scored by the model that
never saw it. Without that step, a larger cache would just be a more confident way to
be wrong.
"""
import json
import time
from pathlib import Path

import numpy as np

from .store import read_json, write_json


def normalise_question(text):
    """Case and whitespace folded, trailing punctuation dropped.

    Deliberately conservative. The point is to catch the same question submitted twice
    under two ids, not to decide that two differently worded questions mean the same
    thing - a normaliser that merges genuinely distinct questions would shrink the
    effective training set for no benefit.
    """
    return " ".join(str(text).split()).casefold().rstrip("?.!:;, ")


def question_texts(source):
    """{id: text} from either {id: text} or {id: {"question": text, ...}}."""
    out = {}
    for key, value in (source or {}).items():
        if isinstance(value, dict):
            value = value.get("question")
        if value:
            out[key] = value
    return out


def duplicate_groups(texts):
    """{normalised text: [ids]} for every question text that appears more than once."""
    groups = {}
    for qid, text in question_texts(texts).items():
        groups.setdefault(normalise_question(text), []).append(qid)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def folds(question_ids, count=2, seed=2026, texts=None):
    """Deterministic assignment of questions to folds.

    Hash-based rather than shuffled-and-sliced, so a question keeps its fold even if
    the question file is reordered or extended - otherwise adding data silently
    invalidates every cached score by moving questions across the train/score boundary.

    **Hashing the id is not enough.** The labelled set contains the same question text
    submitted under more than one id. Keyed on the id, those copies land in different
    folds, so a model trained on fold A is then scored on a question it has already
    seen - which is precisely the leak cross-fitting exists to prevent, reintroduced by
    the mechanism meant to prevent it. Pass `texts` and the key becomes the normalised
    question, so every copy travels together.

    `texts` may be {id: text} or {id: record}; ids missing from it fall back to the id,
    which reproduces the old behaviour exactly when it is omitted.
    """
    import zlib
    lookup = question_texts(texts)
    keys = {q: normalise_question(lookup[q]) if q in lookup else str(q)
            for q in question_ids}
    return {q: zlib.crc32(f"{seed}:{k}".encode("utf-8")) % count for q, k in keys.items()}


def build(pipeline, questions, output, depth=200, rerank_depth=100, passages=4,
          progress=100):
    """Score `questions` through `pipeline` and freeze everything a ranking needs.

    Stores per-channel rankings AND the reranker's raw per-passage scores, because the
    settings being tuned downstream need to distinguish a document BM25F ranked first
    from one the dense channel ranked first - information fusion has already destroyed.
    """
    from .stack import build_features  # noqa: F401  (import proves the pairing works)

    items = list(questions.items())
    doc_index, docs = {}, []

    def index_of(doc):
        if doc not in doc_index:
            doc_index[doc] = len(docs)
            docs.append(doc)
        return doc_index[doc]

    n = len(items)
    channels_out = {name: (np.full((n, depth), -1, dtype="int32"),
                           np.zeros((n, depth), dtype="float32"))
                    for name in ("fused", "lexical", "dense")}
    rr_ids = np.full((n, rerank_depth), -1, dtype="int32")
    rr_scores = np.zeros((n, rerank_depth, passages), dtype="float32")
    rr_counts = np.zeros((n, rerank_depth), dtype="int8")
    max_gold = max(len(v["answer"]) for v in questions.values())
    gold = np.full((n, max_gold), -1, dtype="int32")

    started = time.time()
    for row, (qid, record) in enumerate(items):
        channels = {}
        fused, prior = pipeline.candidates(record["question"], channels=channels)
        for name, (ids, scores) in channels_out.items():
            ranking = fused if name == "fused" else (channels.get(name) or [])
            for position, (doc, score) in enumerate(ranking[:depth]):
                ids[row, position] = index_of(doc)
                scores[row, position] = score
        for position, doc in enumerate(record["answer"][:max_gold]):
            gold[row, position] = index_of(doc)

        shortlist = [d for d, _ in fused[:rerank_depth]]
        table = pipeline._rerank(pipeline.rerankers[0], record["question"], shortlist,
                                 prior, {}, passages=passages)
        for position, doc in enumerate(shortlist):
            rr_ids[row, position] = index_of(doc)
            values = table.get(doc, [])[:passages]
            rr_counts[row, position] = len(values)
            for slot, value in enumerate(values):
                rr_scores[row, position, slot] = value

        if (row + 1) % progress == 0:
            rate = (row + 1) / max(1e-6, time.time() - started)
            print(f"  cached {row + 1}/{n}, "
                  f"eta {(n - row - 1) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gold=gold, rr_ids=rr_ids, rr_scores=rr_scores, rr_counts=rr_counts,
        **{f"{name}_ids": ids for name, (ids, _) in channels_out.items()},
        **{f"{name}_scores": scores for name, (_, scores) in channels_out.items()})
    write_json(output.with_suffix(".docs.json"),
               {"documents": docs, "questions": [q for q, _ in items],
                "depth": depth, "rerank_depth": rerank_depth, "passages": passages})
    size = output.stat().st_size / 1e6
    print(f"  wrote {output} ({size:.1f} MB) covering {n} questions "
          f"in {time.time() - started:.0f}s")
    return output


class Cache:
    """A frozen cache, replayable into the same ranking the live pipeline produces."""

    def __init__(self, path):
        path = Path(path)
        self.arrays = np.load(path if path.suffix == ".npz" else path.with_suffix(".npz"))
        sidecar = json.loads(path.with_suffix(".docs.json").read_text(encoding="utf-8"))
        self.documents = sidecar["documents"]
        self.questions = sidecar["questions"]
        self.passages = sidecar["passages"]

    def __len__(self):
        return len(self.questions)

    def rows(self, subset=None):
        keep = set(subset) if subset is not None else None
        for row, qid in enumerate(self.questions):
            if keep is None or qid in keep:
                yield row, qid

    def replay(self, row):
        """Rebuild the exact inputs `final_order` expects for one question."""
        out = {}
        for name in ("fused", "lexical", "dense"):
            ids = self.arrays[f"{name}_ids"][row]
            scores = self.arrays[f"{name}_scores"][row]
            out[name] = [(self.documents[i], float(s))
                         for i, s in zip(ids, scores) if i >= 0]
        table, ids = {}, self.arrays["rr_ids"][row]
        values, counts = self.arrays["rr_scores"][row], self.arrays["rr_counts"][row]
        for position, doc_id in enumerate(ids):
            if doc_id < 0:
                continue
            table[self.documents[doc_id]] = [float(v) for v in values[position][:counts[position]]]
        gold = [self.documents[i] for i in self.arrays["gold"][row] if i >= 0]
        return out, table, gold

    def evaluate(self, settings, stacker=None, subset=None, top_k=5):
        """Recall@5 over the cache, using the live pipeline's own ordering function."""
        from .pipeline import DEFAULTS, final_order
        merged = dict(DEFAULTS, **(settings or {}))
        per_question = {}
        for row, qid in self.rows(subset):
            channels, table, gold = self.replay(row)
            if not gold:
                continue
            order = final_order(channels["fused"], [table], merged,
                                channels=channels, stacker=stacker)
            per_question[qid] = len(set(gold) & set(order[:top_k])) / len(gold)
        return per_question

    def features(self, subset=None):
        """(question id, matrix, labels) per question, for fitting the stacker offline.

        The id travels with the features so a caller can hold a fold out. Fitting on
        every cached question and then scoring those same questions measures how well
        twelve weights can memorise 7,000 rows, not how the fusion generalises.
        """
        from .stack import build_features
        rows = []
        for row, qid in self.rows(subset):
            channels, table, gold = self.replay(row)
            docs = [d for d, _ in channels["fused"][:len(table)]]
            if not docs:
                continue
            labels = np.array([1.0 if d in set(gold) else 0.0 for d in docs])
            if labels.sum() == 0:
                continue
            rows.append((qid, build_features(docs, [table], channels), labels))
        return rows


def summarise(per_question, gold_sizes=None, label=""):
    values = list(per_question.values())
    mean = sum(values) / max(1, len(values))
    print(f"\n  {label or 'cached evaluation'}")
    print(f"    questions {len(values)}   Recall@5 {mean:.4f}")
    if gold_sizes:
        for name, keep in (("single-gold", lambda k: gold_sizes.get(k, 1) == 1),
                           ("multi-gold", lambda k: gold_sizes.get(k, 1) > 1)):
            subset = [v for k, v in per_question.items() if keep(k)]
            if subset:
                print(f"    {name:<12} {len(subset):>5}   "
                      f"{sum(subset) / len(subset):.4f}")
    return mean


SEARCH_GRID = {
    "candidates": (30, 50, 75, 100),
    "passages": (1, 2, 3, 4),
    "aggregate": ("max", "max_plus", "softmax"),
    "aggregate_weight": (0.1, 0.2, 0.35),
    "rerank_blend": (0.0, 0.01, 0.02, 0.05, 0.1),
}


# Knobs that only the hand-weighted blend consults. When a stacker is active,
# final_order returns before any of them is read, so searching them yields exactly
# +0.0000 for every candidate - correct, but a waste of cycles and a confusing thing
# to stare at. The stacker supersedes them by design; it does not ignore them by accident.
BLEND_ONLY = ("passages", "aggregate", "aggregate_weight", "rerank_blend")


def search(caches, base, stacker=None, grid=SEARCH_GRID, cycles=2, limits=None):
    """Coordinate ascent over cached scores, accepting only SIGNIFICANT improvements.

    The phase-four sweep took the argmax of ~130 configurations on 400 questions and
    reported +0.0107, none of which survived held-out measurement. Taking the best of
    many noisy numbers is a selection procedure, not a measurement, and it reports the
    luckiest draw as if it were the largest effect.

    So a candidate has to beat the incumbent on a PAIRED bootstrap interval that
    excludes zero before it is adopted. Ties keep the incumbent. That makes the search
    conservative by construction: it will sometimes leave a small real gain on the
    table, and it will not hand back a gain that is not there.
    """
    from .compare import bootstrap

    def evaluate(settings):
        out = {}
        for cache in caches:
            out.update(cache.evaluate(settings, stacker=stacker))
        return out

    def better(candidate, incumbent):
        keys = sorted(set(candidate) & set(incumbent))
        differences = [candidate[k] - incumbent[k] for k in keys]
        low, high = bootstrap(differences, resamples=4000)
        mean = sum(differences) / max(1, len(differences))
        return (low > 0.0), mean, (low, high)

    if stacker is not None:
        grid = {k: v for k, v in grid.items() if k not in BLEND_ONLY}
        print(f"  a stacker is active, so {', '.join(BLEND_ONLY)} are skipped:")
        print(f"  final_order returns before reading them, so every candidate would "
              f"score exactly +0.0000.", flush=True)

    best = dict(base)
    current = evaluate(best)
    start = sum(current.values()) / max(1, len(current))
    print(f"  starting point: Recall@5 {start:.4f} over {len(current)} questions", flush=True)
    limits = limits or {}
    for cycle in range(cycles):
        adopted = 0
        for key, values in grid.items():
            cap = limits.get(key)
            for value in values:
                if value == best[key] or (cap is not None and value > cap):
                    continue
                trial = evaluate(dict(best, **{key: value}))
                real, mean, (low, high) = better(trial, current)
                flag = "ADOPT" if real else "  -  "
                print(f"    {flag} {key}={value!r:<10} {mean:+.4f} "
                      f"[{low:+.4f}, {high:+.4f}]", flush=True)
                if real:
                    best[key], current = value, trial
                    adopted += 1
        # Coordinate ascent only revisits a knob because an earlier adoption may have
        # changed what is optimal for it. Adopt nothing and the next cycle is bit-for-bit
        # the same evaluations, which is what printed the duplicated block.
        if adopted == 0:
            print(f"  cycle {cycle} adopted nothing - later cycles would repeat it "
                  f"exactly, so stopping here", flush=True)
            break
    final = sum(current.values()) / max(1, len(current))
    print(f"\n  chosen: Recall@5 {final:.4f} ({final - start:+.4f} over the starting point)")
    print(f"  {best}")
    return best, current, final
