"""Mine chunk level positives and tiered hard negatives from the training split.

Only the organiser supplied `train.json` labels and the organiser supplied corpus
are used. Nothing is generated: a positive is the passage of a *labelled* gold
document that the current retriever scores highest, and a negative is a passage of
a document the retriever ranks highly but the labels do not list.

Two forms of label noise are handled explicitly:

* Retrieval misses. When no passage of a gold document appears in the candidate
  list, the gold document's own best passage is still taken as the positive, so
  the hardest questions are not silently dropped from training.
* Near duplicate documents. Vietnamese law is republished as consolidated and
  amending versions, so a "negative" whose text nearly repeats a positive is very
  often an equally correct answer that simply is not listed. Those are dropped
  rather than taught as negatives.
"""
import json
import random
import time
from pathlib import Path

import numpy as np

from .store import Store, read_json, write_json

TIERS = ((0, 10), (10, 30), (30, 100))


def token_set(text):
    return set(text.lower().split())


def jaccard(first, second):
    return len(first & second) / max(1, len(first | second))


def mine(store_path, lexical_directory, questions_path, output,
         dense_directory=None, embedding_path=None, device="cuda",
         positives=2, negatives=12, depth=100, duplicate_threshold=0.9, seed=2026,
         lexical_settings=None, limit=None):
    from .lexical import Lexical

    store = Store(store_path)
    lexical = Lexical(lexical_directory, lexical_settings)
    positions = {int(c): i for i, c in enumerate(lexical.unigram.chunk_ids)}
    dense = None
    if dense_directory:
        from .dense import DenseIndex
        dense = DenseIndex(dense_directory, store_path, embedding_path, device=device)
        dense_positions = {int(c): i for i, c in enumerate(dense.chunk_ids)}

    records = read_json(questions_path)
    items = list(records.items())[:limit] if limit else list(records.items())
    rng = random.Random(seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("w", encoding="utf-8")
    started, stats = time.time(), {"questions": 0, "missed": 0, "pruned": 0, "skipped": 0}
    for number, (qid, record) in enumerate(items, 1):
        question = record["question"]
        gold = [g for g in record["answer"] if g in store.doc_position]
        if not gold:
            stats["skipped"] += 1
            continue
        scores = lexical.chunk_scores(question)
        if dense:
            dense_scores = dense.scores(question)
            combined = np.zeros(len(scores), dtype="float32")
            for chunk, index in dense_positions.items():
                combined[positions[chunk]] = dense_scores[index]
            scores = _blend(scores, combined)
        picks = lexical.top_chunks(scores, lexical.settings["chunk_candidates"])
        ranked = lexical.documents(picks, scores)
        rank_of = {doc: rank for rank, (doc, _) in enumerate(ranked[:depth])}

        positive_chunks = []
        for doc in gold:
            chunk_ids = store.document_chunk_ids(doc)
            if not chunk_ids:
                continue
            values = [(float(scores[positions[c]]), c) for c in chunk_ids if c in positions]
            values.sort(key=lambda x: (-x[0], x[1]))
            positive_chunks.extend(c for _, c in values[:positives])
            if doc not in rank_of:
                stats["missed"] += 1
        if not positive_chunks:
            stats["skipped"] += 1
            continue
        positive_bags = [token_set(store.chunk_text(c, with_label=False)) for c in positive_chunks]

        tiered = {tier: [] for tier in TIERS}
        for doc, rank in rank_of.items():
            if doc in gold:
                continue
            for tier in TIERS:
                if tier[0] <= rank < tier[1]:
                    tiered[tier].append(doc)
                    break
        chosen, quota = [], _quota(negatives, len(TIERS))
        for tier, want in zip(TIERS, quota):
            pool = tiered[tier]
            rng.shuffle(pool)
            chosen.extend(pool[:want])
        leftovers = [d for tier in TIERS for d in tiered[tier] if d not in set(chosen)]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:max(0, negatives - len(chosen))])

        negative_chunks = []
        for doc in chosen:
            chunk_ids = [c for c in store.document_chunk_ids(doc) if c in positions]
            if not chunk_ids:
                continue
            best = max(chunk_ids, key=lambda c: (float(scores[positions[c]]), -c))
            bag = token_set(store.chunk_text(best, with_label=False))
            if any(jaccard(bag, other) > duplicate_threshold for other in positive_bags):
                stats["pruned"] += 1
                continue
            negative_chunks.append(best)
        if not negative_chunks:
            stats["skipped"] += 1
            continue
        stream.write(json.dumps({"id": qid, "question": question, "gold": gold,
                                 "positives": positive_chunks,
                                 "negatives": negative_chunks[:negatives],
                                 "gold_ranks": {d: rank_of.get(d) for d in gold}},
                                ensure_ascii=False) + "\n")
        stats["questions"] += 1
        if number % 200 == 0:
            print(f"  mined {number}/{len(items)} questions {time.time() - started:.0f}s",
                  flush=True)
    stream.close()
    if dense:
        dense.close()
    store.close()
    stats["seconds"] = round(time.time() - started)
    write_json(output.with_suffix(".stats.json"), stats)
    print(f"Mined {stats['questions']} groups: {stats}")
    return stats


def _blend(lexical_scores, dense_scores):
    """Rank free blend used only for mining, so one channel cannot dominate."""
    def unit(values):
        low, high = float(values.min()), float(values.max())
        return (values - low) / ((high - low) or 1.0)
    return unit(lexical_scores) + unit(dense_scores)


def _quota(total, buckets):
    base = total // buckets
    quota = [base] * buckets
    for index in range(total - base * buckets):
        quota[index] += 1
    return quota


def load_groups(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def teacher_scores(groups_path, store_path, teacher_path, output, device="cuda",
                   batch_size=8, max_length=1024, head=None):
    """Cache a large teacher reranker's scores for distillation.

    The teacher is only ever run offline to produce these numbers; it is not part of
    the deployed system and is not counted against the deployed parameter budget.
    """
    from .rerank import Reranker

    store = Store(store_path)
    groups = load_groups(groups_path)
    teacher = Reranker(teacher_path, device=device, head=head, max_length=max_length)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("w", encoding="utf-8")
    started = time.time()
    for number, group in enumerate(groups, 1):
        chunks = list(group["positives"]) + list(group["negatives"])
        texts = [store.chunk_text(c) for c in chunks]
        scores = teacher.score(group["question"], texts, batch_size, max_length)
        stream.write(json.dumps({"id": group["id"],
                                 "scores": {str(c): float(s) for c, s in zip(chunks, scores)}}) + "\n")
        if number % 100 == 0:
            rate = number / max(1e-6, time.time() - started)
            print(f"  teacher scored {number}/{len(groups)}, "
                  f"eta {(len(groups) - number) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    stream.close()
    teacher.close()
    store.close()
    print(f"Cached teacher scores for {len(groups)} groups in {time.time() - started:.0f}s")
