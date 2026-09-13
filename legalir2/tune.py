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
