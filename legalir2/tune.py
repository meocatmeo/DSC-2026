"""Coordinate descent over the lexical and fusion settings.

Tuning runs on a sample of the *training* split by default. Those questions are
labelled and are never used for weight updates in the lexical stage, so they can
select hyperparameters without spending the held out validation split, which then
stays a clean check rather than something the settings were fitted to.

A full grid is wasteful: these settings are close to independent here. One pass per
setting, keeping whatever improved, reaches the same neighbourhood far faster.
"""
import copy
import json
import time
from pathlib import Path

from .lexical import DEFAULTS as LEXICAL_DEFAULTS
from .predict import load_questions
from .store import write_json

CANDIDATES = {
    "b_body": (0.75, 0.85, 0.9, 1.0),
    "beta": (1.0, 1.5, 2.0, 3.0),
    "decay": (0.0, 0.15, 0.3, 0.5),
    "depth": (3, 5, 8),
    "label": (0.25, 0.5, 1.0, 2.0),
    "crumb": (0.0, 0.6, 1.2),
    "k1": (0.9, 1.2, 1.6),
    "chunk_candidates": (400, 600, 1000),
}
# The reranker sees the top `candidates` documents, so its ceiling is candidate
# recall at that depth; but its ordering is not perfect, so the order the retriever
# already produces still matters. The objective weights both.
OBJECTIVE = "composite"
WEIGHTS = {"recall@5": 0.25, "recall@20": 0.30, "recall@50": 0.45}


def recall_at(rows, k):
    total = 0.0
    for gold, ranked in rows:
        total += len(set(gold) & set(ranked[:k])) / len(gold)
    return total / max(1, len(rows))


def sweep(args, limit=250, output=None, objective=OBJECTIVE):
    from .lexical import Lexical

    lexical = Lexical(args.lexical)
    records = load_questions(args.questions, labeled=True)
    items = list(records.items())[:limit]
    queries = [(r["answer"], r["question"]) for _, r in items]

    def score(settings):
        lexical.apply(settings)
        rows = []
        for gold, question in queries:
            documents, _ = lexical.search(question)
            rows.append((gold, [d for d, _ in documents]))
        metrics = {f"recall@{k}": round(recall_at(rows, k), 4) for k in (5, 10, 20, 50, 100)}
        metrics["composite"] = round(sum(w * metrics[k] for k, w in WEIGHTS.items()), 4)
        return metrics

    best = dict(LEXICAL_DEFAULTS)
    history = [{"settings": copy.copy(best), "metrics": score(best), "note": "start"}]
    print(f"start {history[-1]['metrics']}", flush=True)
    current = history[-1]["metrics"][objective]
    started = time.time()
    for key, values in CANDIDATES.items():
        for value in values:
            if best[key] == value:
                continue
            trial = dict(best, **{key: value})
            metrics = score(trial)
            improved = metrics[objective] > current
            history.append({"settings": copy.copy(trial), "metrics": metrics,
                            "note": f"{key}={value}" + (" kept" if improved else "")})
            print(f"{key}={value:<6} {metrics}" + ("  kept" if improved else ""), flush=True)
            if improved:
                best, current = trial, metrics[objective]
    final = score(best)
    print(f"\nbest lexical settings ({objective}={current:.4f}, {time.time()-started:.0f}s):")
    print(json.dumps(best, indent=1))
    if output:
        write_json(output, {"objective": objective, "best": best, "metrics": final,
                            "questions": len(queries), "history": history})
    return best, final


# ---------------------------------------------------------------- fusion stage

FUSION_CANDIDATES = {
    "lexical_weight": (0.5, 0.75, 1.0, 1.5, 2.0),
    "dense_weight": (0.5, 0.75, 1.0, 1.5, 2.0),
    "rrf_constant": (10, 20, 60, 120),
}


def _document_ranking(chunk_ids, doc_of, scores, decay, depth, limit):
    """Chunk scores -> ranked documents, mirroring the pipeline's aggregation."""
    import numpy as np

    limit = min(limit, len(scores))
    picks = np.argpartition(-scores, limit - 1)[:limit]
    picks = picks[np.argsort(-scores[picks], kind="stable")]
    picks = picks[scores[picks] > 0]
    grouped = {}
    for position in picks:
        doc = doc_of[int(chunk_ids[position])]
        slot = grouped.setdefault(doc, [])
        if len(slot) < depth:
            slot.append(float(scores[position]))
    out = [(doc, group[0] + decay * sum(group[1:])) for doc, group in grouped.items()]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def fusion_sweep(args, limit=400, output=None, depths=(5, 20, 50, 100, 200), target=100):
    """Tune how the lexical and dense channels are combined.

    Worth its own pass because of a result that is easy to get wrong: once both
    retrievers are fine-tuned in-domain, naive fusion usually *hurts*. Gains survive
    only with weights tuned for the specific pair, so the weights are fitted here
    rather than assumed to be equal.

    Each channel is scored once per question and cached, after which sweeping weights
    costs nothing - the expensive part is the dense forward pass, not the arithmetic.
    """
    import numpy as np

    from .dense import DenseIndex
    from .fuse import rrf
    from .lexical import Lexical
    from .store import Store

    store = Store(args.store)
    lexical = Lexical(args.lexical)
    dense = DenseIndex(args.dense, args.store, args.embedding_model, device=args.device)
    doc_of = store.chunk_documents()
    records = load_questions(args.questions, labeled=True)
    items = list(records.items())[:limit]

    decay, depth = lexical.settings["decay"], lexical.settings["depth"]
    chunk_limit = lexical.settings["chunk_candidates"]
    cached, started = [], time.time()
    for number, (_, record) in enumerate(items, 1):
        scores = lexical.chunk_scores(record["question"])
        lex = _document_ranking(lexical.unigram.chunk_ids, doc_of, scores, decay, depth, chunk_limit)
        vec = dense.scores(record["question"])
        den = _document_ranking(dense.chunk_ids, doc_of, vec, decay, depth, chunk_limit)
        cached.append((record["answer"], lex, den))
        if number % 50 == 0:
            print(f"  scored {number}/{len(items)} questions {time.time()-started:.0f}s", flush=True)
    dense.close()
    store.close()

    def score(settings):
        rows = []
        for gold, lex, den in cached:
            fused = rrf([lex, den], [settings["lexical_weight"], settings["dense_weight"]],
                        settings["rrf_constant"])
            rows.append((gold, [d for d, _ in fused]))
        return {f"recall@{k}": round(recall_at(rows, k), 4) for k in depths}

    def single(index):
        """Each channel on its own, so fusion can be judged against both."""
        rows = [(gold, [d for d, _ in (lex, den)[index]]) for gold, lex, den in cached]
        return {f"recall@{k}": round(recall_at(rows, k), 4) for k in depths}

    key = f"recall@{target}"
    print(f"\nlexical alone : {single(0)}")
    print(f"dense alone   : {single(1)}")

    best = {"lexical_weight": 1.0, "dense_weight": 1.0, "rrf_constant": 60}
    metrics = score(best)
    current = metrics[key]
    print(f"equal weights : {metrics}\n")
    history = [{"settings": copy.copy(best), "metrics": metrics, "note": "equal weights"}]
    for name, values in FUSION_CANDIDATES.items():
        for value in values:
            if best[name] == value:
                continue
            trial = dict(best, **{name: value})
            trial_metrics = score(trial)
            improved = trial_metrics[key] > current
            history.append({"settings": copy.copy(trial), "metrics": trial_metrics,
                            "note": f"{name}={value}" + (" kept" if improved else "")})
            print(f"{name}={value:<6} {trial_metrics}" + ("  kept" if improved else ""), flush=True)
            if improved:
                best, current = trial, trial_metrics[key]

    final = score(best)
    print(f"\nbest fusion settings ({key}={current:.4f}):")
    print(json.dumps(best, indent=1))
    print("pass these to predict with --settings, and re-check on validation once.")
    if output:
        write_json(output, {"objective": key, "best": best, "metrics": final,
                            "lexical_only": single(0), "dense_only": single(1),
                            "questions": len(cached), "history": history})
    return best, final


RERANK_GRID = {
    "candidates": (30, 50, 75, 100),
    "passages": (1, 2, 3, 4),
    "aggregate": ("max", "max_plus", "softmax"),
    "aggregate_weight": (0.1, 0.2, 0.35),
    "rerank_blend": (0.0, 0.01, 0.02, 0.05, 0.1),
}


def rerank_sweep(args, limit=400, output=None, depth=100, passages=4, radii=(0, 1)):
    """Tune the reranking stage by scoring once and sweeping the rest offline.

    Everything after the reranker's forward pass - how many documents it looks at,
    how many passages per document, how a document's passages become one score, how
    much of the fusion rank survives - was set by hand and never measured. Each is
    worth a fraction of a point and together they are worth more than that, but
    measuring them the obvious way costs a full GPU pass per combination.

    So: score `depth` documents x `passages` passages once per question and cache the
    raw numbers, then evaluate every combination against the cache. One pass buys
    hundreds of configurations. `candidates` and `passages` are swept by truncating
    the cache, which is exact for `candidates` and near-exact for `passages` - a
    shallower setting picks the same leading passages unless a document runs out of
    distinct ones, which the reported `prefix_exact` count checks rather than assumes.
    """
    # context_radius cannot be swept from a cache. Every other knob reorders numbers
    # that are already computed, but the radius changes the passage TEXT the reranker
    # reads, so each value needs its own scoring pass. Sweeping it off one cache would
    # silently report the radius-0 scores for every setting.
    results = []
    for radius in radii:
        print(f"\n--- context_radius = {radius} "
              f"({'single chunk' if radius == 0 else f'chunk plus {radius} neighbour(s)'}) ---",
              flush=True)
        chosen, recall, baseline = _rerank_pass(args, limit, depth, passages, radius)
        results.append({"context_radius": radius, "best": dict(chosen, context_radius=radius),
                        "recall": recall, "baseline": baseline})
    results.sort(key=lambda r: -r["recall"])
    winner = results[0]
    print(f"\n  best overall: context_radius={winner['context_radius']} "
          f"Recall@5 {winner['recall']:.4f}")
    for row in results:
        print(f"    radius {row['context_radius']}: {row['recall']:.4f}")
    if output:
        write_json(output, {"best": winner["best"], "recall": winner["recall"],
                            "by_radius": results, "questions": limit,
                            "cached_depth": depth, "cached_passages": passages})
    return winner["best"]


def _rerank_pass(args, limit, depth, passages, radius):
    """One scoring pass at a fixed context radius, then coordinate ascent on the cache."""
    from .pipeline import Pipeline, final_order

    pipeline = Pipeline(args.store, args.lexical,
                        dense_directory=args.dense if (Path(args.dense) / "manifest.json").is_file() else None,
                        embedding_path=args.embedding_model,
                        reranker_path=args.reranker_model, device=args.device,
                        settings=dict(json.loads(args.settings) if args.settings else {},
                                      context_radius=radius))
    if not pipeline.rerankers:
        raise ValueError("rerank sweep needs at least one --reranker-model")
    records = load_questions(args.questions, labeled=True)
    items = list(records.items())[:limit]

    cached, started, exact, checked = [], time.time(), 0, 0
    for number, (_, record) in enumerate(items, 1):
        question = record["question"]
        fused, prior = pipeline.candidates(question)
        docs = [d for d, _ in fused[:depth]]
        cache = {}
        table = pipeline._rerank(pipeline.rerankers[0], question, docs, prior, cache,
                                passages=passages)
        # Spot-check the truncation assumption on the first few questions rather than
        # trusting it: recomputing a shallower selection must match the cached prefix.
        if number <= 5:
            for doc in docs[:10]:
                checked += 1
                shallow = pipeline._passages(doc, prior, 2)
                exact += int(shallow == cache[doc][:2])
        cached.append((record["answer"], fused[:depth], table))
        if number % 25 == 0:
            rate = number / max(1e-6, time.time() - started)
            print(f"  scored {number}/{len(items)} questions, "
                  f"eta {(len(items)-number)/max(rate,1e-9)/60:.1f} min", flush=True)
    base = dict(pipeline.settings)
    pipeline.close()
    print(f"  passage-prefix check: {exact}/{checked} shallower selections matched "
          f"the cached prefix", flush=True)

    def score(settings):
        rows = [(gold, final_order(fused, [table], settings)[:5])
                for gold, fused, table in cached]
        hits = [len(set(gold) & set(top)) / len(gold) for gold, top in rows]
        return sum(hits) / max(1, len(hits))

    best, history = dict(base), []
    best["passages"] = min(best["passages"], passages)
    best["candidates"] = min(best["candidates"], depth)
    current = score(best)
    print(f"\n  starting point (what shipped): Recall@5 {current:.4f}  "
          f"{ {k: best[k] for k in RERANK_GRID} }\n", flush=True)
    history.append({"settings": {k: best[k] for k in RERANK_GRID}, "recall": round(current, 4)})

    # Coordinate ascent: one knob at a time, two passes. The knobs interact (deeper
    # candidates change what rerank_blend has to overcome), so a single pass is not
    # enough and a full grid is 720 evaluations of no extra value.
    for cycle in range(2):
        for key, values in RERANK_GRID.items():
            keep, gain = best[key], current
            for value in values:
                if value == best[key] or (key == "candidates" and value > depth) \
                   or (key == "passages" and value > passages):
                    continue
                trial = dict(best, **{key: value})
                result = score(trial)
                history.append({"settings": {k: trial[k] for k in RERANK_GRID},
                                "recall": round(result, 4)})
                if result > gain:
                    keep, gain = value, result
            if gain > current:
                print(f"  cycle {cycle} {key}: {best[key]} -> {keep}  "
                      f"Recall@5 {current:.4f} -> {gain:.4f}", flush=True)
                best[key], current = keep, gain
    chosen = {k: best[k] for k in RERANK_GRID}
    print(f"  best at radius {radius}: Recall@5 {current:.4f}  {chosen}")
    print(f"  gain over what shipped: {current - history[0]['recall']:+.4f}")
    return chosen, round(current, 4), history[0]
