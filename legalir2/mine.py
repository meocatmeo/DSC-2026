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

from .store import Store, read_json, write_json

TIERS = ((0, 10), (10, 30), (30, 100))


def token_set(text):
    return set(text.lower().split())


def jaccard(first, second):
    return len(first & second) / max(1, len(first | second))


def mine(store_path, lexical_directory, questions_path, output,
         dense_directory=None, embedding_path=None, device="cuda",
         positives=2, negatives=12, depth=100, duplicate_threshold=0.9, seed=2026,
         lexical_settings=None, limit=None, settings=None, channel="fused"):
    """Mine training groups through the *inference* candidate path.

    The negatives a reranker learns to reject are only useful if they are the
    passages it will actually be asked to reject. Mining through its own private
    blend of the two channels - which is what this did originally - trains the model
    on one distribution and deploys it against another, and the cost is invisible:
    every number still looks reasonable, the reranker just quietly converts fewer of
    the candidates it is given. So the candidate list here comes from Pipeline,
    reranker unloaded, with the same tuned fusion settings prediction will use.

    `channel` says WHICH inference path the model being trained will actually face,
    and it is not the same answer for every model.

    A cross encoder reranks the fused shortlist, so "fused" is right for it. The
    bi-encoder does not: at inference it produces the dense channel's own ranking over
    the whole corpus, independently, and fusion happens afterwards. Training it against
    documents the FUSION ranked highly teaches it to separate a distribution it never
    sees - the same mismatch that cost this project 0.021 on the reranker, one model
    over. `channel="dense"` tiers negatives on the dense channel's own ranking, so the
    hard negatives are the documents the bi-encoder itself gets wrong.
    """
    from .pipeline import Pipeline

    pipeline = Pipeline(store_path, lexical_directory, dense_directory=dense_directory,
                        embedding_path=embedding_path, reranker_path=None, device=device,
                        settings=settings, lexical_settings=lexical_settings)
    store = pipeline.store
    lexical = pipeline.lexical
    positions = {int(c): i for i, c in enumerate(lexical.unigram.chunk_ids)}

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
        per_channel = {}
        fused, prior = pipeline.candidates(question, channels=per_channel)
        # Negatives are tiered on the ranking the trained model will itself produce;
        # positives and their passages still come from the fused path, because which
        # passage best represents a gold document is a different question from which
        # documents are hard negatives.
        source = per_channel.get(channel) or fused
        if channel != "fused" and not per_channel.get(channel):
            stats["empty_channel"] = stats.get("empty_channel", 0) + 1
        rank_of = {doc: rank for rank, (doc, _) in enumerate(source[:depth])}
        # Fallback only: a gold document fusion never surfaced has no prior, so its
        # best passage is chosen lexically rather than dropping the hardest questions.
        fallback = lexical.chunk_scores(question)

        def rank_chunks(doc):
            chunk_ids = [c for c in store.document_chunk_ids(doc) if c in positions]
            if not chunk_ids:
                return []
            if any(prior.get(c, 0.0) for c in chunk_ids):
                key = lambda c: (-prior.get(c, 0.0), c)
            else:
                key = lambda c: (-float(fallback[positions[c]]), c)
            return sorted(chunk_ids, key=key)

        # Positives are kept grouped BY gold document, not pooled into one flat list.
        # 8.9% of questions have two or more gold documents and the scorer gives partial
        # credit for each, but a flat list cannot express "these three are all correct" -
        # it can only be sampled from, which is what trains the model to rank one of them
        # first and forget the rest. The flat list is still emitted for compatibility.
        by_gold, positive_chunks = {}, []
        for doc in gold:
            picked = rank_chunks(doc)[:positives]
            if picked:
                by_gold[doc] = picked
                positive_chunks.extend(picked)
            if doc not in rank_of:
                stats["missed"] += 1
        stats.setdefault("multi_gold", 0)
        stats["multi_gold"] += int(len(by_gold) > 1)
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
            ordered = rank_chunks(doc)
            if not ordered:
                continue
            best = ordered[0]
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
                                 "positives_by_gold": {str(k): v for k, v in by_gold.items()},
                                 "negatives": negative_chunks[:negatives],
                                 "gold_ranks": {d: rank_of.get(d) for d in gold}},
                                ensure_ascii=False) + "\n")
        stats["questions"] += 1
        if number % 200 == 0:
            print(f"  mined {number}/{len(items)} questions {time.time() - started:.0f}s",
                  flush=True)
    stream.close()
    pipeline.close()
    stats["seconds"] = round(time.time() - started)
    stats["positives"] = positives
    stats["negatives"] = negatives
    stats["depth"] = depth
    stats["channel"] = channel
    stats["fusion"] = {k: pipeline.settings[k]
                       for k in ("lexical_weight", "dense_weight", "rrf_constant")}
    stats["dense"] = bool(dense_directory)
    write_json(output.with_suffix(".stats.json"), stats)
    print(f"Mined {stats['questions']} groups: {stats}")
    return stats


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
