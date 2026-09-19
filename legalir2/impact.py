"""Query-conditioned ranking with two evidence views and labelled-query retrieval.

Train a residual LambdaMART model on cached scores. Its pairwise gradients are
weighted by the change in macro Recall@5, the competition metric. Select rounds
and residual strength with grouped training folds, then evaluate validation once.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .cache import folds, normalise_question
from .compare import compare
from .h100 import digest, guard, validate_rows
from .lexical import tokens
from .question_memory import MEMORY_FEATURES, QuestionMemory
from .stack import feature_names
from .store import file_hash, read_json, source_records, write_json

SCHEMA = "impact-recall-v1"
ROUNDS = (40, 100, 200)
STRENGTHS = (.25, .5, 1.)
EXTRA_CANDIDATES = 10


def names():
    base = list(feature_names(2))
    result = [f"{view}_{name}" for view in ("retrieval", "document", "difference") for name in base]
    for view in ("retrieval", "document"):
        for name in ("rr_max", "rr_max_2", "fused_score", "lex_score", "den_score"):
            result += [f"{view}_{name}_{suffix}" for suffix in ("rr", "margin5", "margin6", "gap12")]
    return result + ["base_score", "base_rr", "base_margin5", "base_margin6", "base_gap12",
                     "view_score_difference", "memory_only", "query_words", "query_digits"] + list(MEMORY_FEATURES)


def context(values, docs):
    order = sorted(range(len(docs)), key=lambda i: (-values[i], docs[i]))
    ranks = np.empty(len(docs))
    ranks[order] = np.arange(1, len(docs) + 1)
    fifth = values[order[min(4, len(order) - 1)]]
    sixth = values[order[min(5, len(order) - 1)]]
    gap = values[order[0]] - values[order[min(1, len(order) - 1)]]
    return np.column_stack((1 / ranks, values - fifth, values - sixth, np.full(len(docs), gap)))


def features(old, new, weights, memory, known_docs):
    if any(old[k] != new[k] for k in ("id", "question", "gold", "docs")):
        raise ValueError("The two evidence views do not describe the same question/candidates")
    docs = list(old["docs"])
    a, b = np.asarray(old["features2"]), np.asarray(new["features2"])
    base = np.asarray(old["features1"]) @ weights
    blocks = [a, b, b - a]
    for matrix in (a, b):
        for col in (0, 12, 5, 7, 9):
            blocks.append(context(matrix[:, col], docs))
    base_context = context(base, docs)
    wordlist = tokens(old["question"])
    query = [np.log1p(len(wordlist)), sum(w.isdigit() for w in wordlist) / max(1, len(wordlist))]
    blocks += [base[:, None], base_context, ((b - a) @ weights)[:, None],
               np.zeros((len(docs), 1)), np.tile(query, (len(docs), 1))]
    matrix = np.column_stack(blocks)
    evidence, ranking = memory.search(old["question"], old["id"])
    extra = [d for d in ranking[:EXTRA_CANDIDATES] if d not in docs and d in known_docs]
    if extra:
        # Missing neural evidence is explicit NaN, never a fictitious low score.
        # Memory-only candidates start below the old fifth-place decision boundary.
        extra_base = np.sort(base)[-min(5, len(base))] - 2.
        tail = np.full((len(extra), matrix.shape[1]), np.nan)
        tail[:, -9] = extra_base
        tail[:, -3] = 1.
        tail[:, -2:] = query
        matrix = np.vstack((matrix, tail))
        base = np.concatenate((base, np.full(len(extra), extra_base)))
        docs += extra
    confidence = max((v[6] for v in evidence.values()), default=0.)
    absent = [0., 0., 0., 0., -confidence, 0., confidence]
    matrix = np.column_stack((matrix, [evidence.get(d, absent) for d in docs]))
    if matrix.shape != (len(docs), len(names())) or np.isinf(matrix).any():
        raise ValueError("Invalid impact feature matrix")
    return {"id": old["id"], "question": old["question"], "gold": old["gold"], "docs": docs,
            "matrix": matrix, "base": base, "candidate_ranks": old["candidate_ranks"],
            "original_count": len(old["docs"])}


def filtered_memory(records, heldout, known_docs):
    held_text = {normalise_question(r["question"]) for r in heldout.values()}
    result = {}
    for qid, record in records.items():
        if qid in heldout or normalise_question(record["question"]) in held_text:
            continue
        answers = [d for d in record["answer"] if d in known_docs]
        if answers:
            result[qid] = dict(record, answer=answers)
    return result


def load_view(source, mode, split, records):
    directory = source / mode / split
    marker = read_json(directory / "resume.json")
    if (marker["input_digest"] != digest(read_json(source / "inputs.json")) or
            marker["runtime"] != read_json(source / "runtime.json") or
            marker["questions"] != digest(records) or marker["mode"] != mode):
        raise ValueError(f"Cache input guard mismatch: {directory}")
    items, result = list(records.items()), []
    size = marker["shard_size"]
    expected = [directory / f"part-{i // size:05d}.json" for i in range(0, len(items), size)]
    if set(directory.glob("part-*.json")) != set(expected):
        raise ValueError(f"Incomplete cache: {directory}")
    for start, path in zip(range(0, len(items), size), expected):
        rows = read_json(path)
        validate_rows(rows, items[start:start + size], mode)
        result.extend(rows)
    return result


def make_rows(old, new, memory, weights, known_docs):
    if len(old) != len(new):
        raise ValueError("Evidence view lengths differ")
    return [features(a, b, weights, memory, known_docs) for a, b in zip(old, new)]


def predicted(row, scores):
    return [row["docs"][i] for i in sorted(range(len(scores)), key=lambda i: (-scores[i], row["docs"][i]))[:5]]


def evaluate(rows, corrections=None, strength=0.):
    diagnostics = {}
    offset = 0
    for row in rows:
        size = len(row["docs"])
        scores = row["base"].copy()
        if corrections is not None:
            scores += strength * corrections[offset:offset + size]
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite residual prediction")
        diagnostics[row["id"]] = {"question": row["question"], "relevant": row["gold"],
                                   "predicted": predicted(row, scores),
                                   "candidate_ranks": {g: row["docs"].index(g) + 1 if g in row["docs"] else None
                                                       for g in row["gold"]}}
        offset += size
    return diagnostics


def recall(diagnostics):
    return sum(len(set(r["relevant"]) & set(r["predicted"])) / len(r["relevant"])
               for r in diagnostics.values()) / len(diagnostics)


class RecallObjective:
    """Pairwise logistic lambdas, weighted by absolute delta macro Recall@5.

    Swapping a positive and negative matters only across the top-five boundary.
    A small (0.05) reciprocal-rank term keeps gradients useful within either side;
    missing positives still count in the denominator of multi-positive questions.
    """
    def __init__(self, rows, auxiliary=.05):
        self.slices, positive, negative, scale = [], [], [], []
        offset = 0
        for row in rows:
            docs, gold = row["docs"], set(row["gold"])
            if not gold:
                raise ValueError("Cannot fit unlabelled ranking queries")
            pos = [i + offset for i, d in enumerate(docs) if d in gold]
            neg = [i + offset for i, d in enumerate(docs) if d not in gold]
            for p in pos:
                for n in neg:
                    positive.append(p)
                    negative.append(n)
                    scale.append(1. / len(gold))
            self.slices.append((offset, offset + len(docs), docs))
            offset += len(docs)
        self.positive, self.negative = np.asarray(positive, dtype=int), np.asarray(negative, dtype=int)
        self.scale, self.auxiliary = np.asarray(scale), auxiliary

    def __call__(self, scores, dataset=None):
        ranks = np.empty(len(scores))
        for start, stop, docs in self.slices:
            order = sorted(range(stop - start), key=lambda i: (-scores[start + i], docs[i]))
            ranks[start + np.asarray(order)] = np.arange(1, stop - start + 1)
        p, n = self.positive, self.negative
        boundary = np.abs((ranks[p] <= 5).astype(float) - (ranks[n] <= 5).astype(float))
        delta = self.scale * (boundary + self.auxiliary * np.abs(1 / ranks[p] - 1 / ranks[n]))
        probability = 1 / (1 + np.exp(np.clip(scores[p] - scores[n], -50, 50)))
        gradient, hessian = np.zeros(len(scores)), np.zeros(len(scores))
        magnitude = delta * probability
        curvature = np.maximum(delta * probability * (1 - probability), 0.)
        np.add.at(gradient, p, -magnitude)
        np.add.at(gradient, n, magnitude)
        np.add.at(hessian, p, curvature)
        np.add.at(hessian, n, curvature)
        return gradient, hessian


def train(rows, rounds=200, threads=4):
    import lightgbm as lgb

    matrix = np.vstack([r["matrix"] for r in rows])
    labels = np.concatenate([[float(d in r["gold"]) for d in r["docs"]] for r in rows])
    base = np.concatenate([r["base"] for r in rows])
    data = lgb.Dataset(matrix, label=labels, group=[len(r["docs"]) for r in rows],
                       init_score=base, feature_name=names(), free_raw_data=True)
    params = {"objective": RecallObjective(rows), "metric": "None", "learning_rate": .035,
              "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 100,
              "min_sum_hessian_in_leaf": .1, "lambda_l2": 10., "max_bin": 63,
              "feature_fraction": .9, "seed": 2026, "num_threads": threads,
              "deterministic": True, "force_col_wise": True, "verbosity": -1}
    return lgb.train(params, data, num_boost_round=rounds)


def fit(args):
    import lightgbm as lgb

    started = time.time()
    inputs = read_json(args.source / "inputs.json")
    all_train, validation = read_json(args.train), read_json(args.validation)
    for path in (args.train, args.validation):
        key = "artifacts/splits/" + path.name
        if file_hash(path) != inputs["files"][key]:
            raise ValueError(f"Question file differs from original H100 run: {path}")
    calibration = {q: all_train[q] for q in inputs["train_ids"]}
    old = load_view(args.source, "retrieval", "train", calibration)
    new = load_view(args.source, "document", "train", calibration)
    old_val = load_view(args.source, "retrieval", "validation", validation)
    new_val = load_view(args.source, "document", "validation", validation)
    weights = np.asarray(read_json(args.baseline / "v9-stacker-ens.json")["weights"])
    known = {str(json.loads(raw)["id"]) for _, raw in source_records(args.corpus)}
    corpus_hash = file_hash(args.corpus)
    memory_records = filtered_memory(all_train, validation, known)
    assignments = folds(calibration, count=3, texts=calibration)
    signature = {"schema": SCHEMA, "source_inputs": digest(inputs), "lightgbm": lgb.__version__,
                 "rounds": ROUNDS, "strengths": STRENGTHS, "memory": digest(memory_records),
                 "corpus": corpus_hash, "baseline": weights.tolist(), "features": names(),
                 "code": {n: file_hash(Path(__file__).parent / n) for n in ("impact.py", "question_memory.py")},
                 "shards": {p.relative_to(args.source).as_posix(): file_hash(p)
                            for p in sorted(args.source.glob("*/*/part-*.json"))
                            if p.parent.name in ("train", "validation")}}
    # JSON canonicalisation keeps tuple settings stable across resume.
    guard(args.out / "inputs.json", json.loads(json.dumps(signature)))
    pooled = {(rounds, strength): {} for rounds in ROUNDS for strength in STRENGTHS}
    baseline_oof = {}
    for fold in range(3):
        heldout = {q: r for q, r in calibration.items() if assignments[q] == fold}
        memory = QuestionMemory(filtered_memory(memory_records, heldout, known))
        rows = make_rows(old, new, memory, weights, known)
        training = [r for r in rows if assignments[r["id"]] != fold]
        held = [r for r in rows if assignments[r["id"]] == fold]
        print(f"Fold {fold + 1}/3: fit {len(training)}, score {len(held)}; memory {len(memory.ids)}", flush=True)
        model = train(training, max(ROUNDS), args.threads)
        matrix = np.vstack([r["matrix"] for r in held])
        baseline_oof.update(evaluate(held))
        for rounds in ROUNDS:
            correction = model.predict(matrix, num_iteration=rounds, num_threads=args.threads)
            for strength in STRENGTHS:
                pooled[rounds, strength].update(evaluate(held, correction, strength))
    summary = [{"rounds": r, "strength": s, "recall": recall(diag)} for (r, s), diag in pooled.items()]
    # Validation is not consulted for rounds, shrinkage, features or memory cutoffs.
    best = max(summary, key=lambda r: (r["recall"], -r["strength"], -r["rounds"]))
    write_json(args.out / "cross-validation.json", {"baseline": recall(baseline_oof), "trials": summary,
               "chosen": best, "note": "Grouped meta-model CV; neural rerankers may have seen calibration questions."})
    print(f"Training-only choice: {best}; baseline={recall(baseline_oof):.6f}", flush=True)
    memory = QuestionMemory(memory_records)
    rows = make_rows(old, new, memory, weights, known)
    held = make_rows(old_val, new_val, memory, weights, known)
    model = train(rows, best["rounds"], args.threads)
    model_record = {"schema": SCHEMA, "features": names(), "weights": weights.tolist(),
                    "strength": best["strength"], "rounds": best["rounds"],
                    "trees": model.model_to_string(), "memory": memory_records, "known_docs": sorted(known),
                    "source_input_digest": digest(inputs), "runtime": read_json(args.source / "runtime.json"),
                    "input_digest": digest(read_json(args.out / "inputs.json"))}
    write_json(args.out / "ranker.json", model_record)
    correction = model.predict(np.vstack([r["matrix"] for r in held]), num_threads=args.threads)
    frozen, result = evaluate(held), evaluate(held, correction, best["strength"])
    write_json(args.out / "frozen.diagnostics.json", frozen)
    write_json(args.out / "impact.diagnostics.json", result)
    comparison = compare(args.out / "frozen.diagnostics.json", args.out / "impact.diagnostics.json",
                         label="Question-memory + two-view LambdaRecall vs frozen V9")
    importance = sorted(zip(names(), model.feature_importance(importance_type="gain")), key=lambda p: -p[1])
    write_json(args.out / "comparison.json", {"comparison": comparison, "cv": best,
               "feature_importance": [[n, float(v)] for n, v in importance],
               "promoted": best["recall"] > recall(baseline_oof) and comparison["ci"][0] > 0,
               "original_candidate_recall": sum(sum(g in r["docs"][:r["original_count"]] for g in r["gold"])/len(r["gold"]) for r in held)/len(held),
               "expanded_candidate_recall": sum(sum(g in r["docs"] for g in r["gold"])/len(r["gold"]) for r in held)/len(held),
               "seconds": time.time() - started})
    print(f"Finished CPU experiment in {(time.time() - started) / 60:.1f} minutes: {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/train.json"))
    parser.add_argument("--validation", type=Path, default=Path("artifacts/splits/validation.json"))
    parser.add_argument("--baseline", type=Path, default=Path("analysis/v11/investigation/baselines"))
    parser.add_argument("--corpus", type=Path, default=Path("analysis/v11/baseline/corpus.jsonl.gz"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fit(args)


if __name__ == "__main__":
    main()
