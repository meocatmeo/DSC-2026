"""Cheap model selection probe: rank one gold passage against mined distractors.

Building a full dense index costs about an hour per checkpoint, which is far too
expensive a way to find out that a checkpoint is the wrong choice. This probe scores
a few hundred questions against a small fixed set of hard distractors instead, so
several approved checkpoints can be compared in minutes.

It measures *ranking* ability on realistic confusions, not end-to-end recall, so
treat it as a screen: use it to drop weak candidates, then build the full index for
the one or two survivors and measure properly with `predict --evaluate`.
"""
import time
from pathlib import Path

from .lexical import Lexical
from .predict import load_questions
from .store import Store, write_json


def groups(store, lexical, questions_path, limit, distractors):
    records = load_questions(questions_path, labeled=True)
    positions = {int(c): i for i, c in enumerate(lexical.unigram.chunk_ids)}
    built = []
    for qid, record in list(records.items())[:limit]:
        gold = [g for g in record["answer"] if g in store.doc_position]
        if not gold:
            continue
        scores = lexical.chunk_scores(record["question"])
        picks = lexical.top_chunks(scores, lexical.settings["chunk_candidates"])
        ranked = lexical.documents(picks, scores)
        positive = None
        for doc in gold:
            chunks = [c for c in store.document_chunk_ids(doc) if c in positions]
            if chunks:
                positive = max(chunks, key=lambda c: (float(scores[positions[c]]), -c))
                break
        if positive is None:
            continue
        negatives = []
        for doc, _ in ranked:
            if doc in gold or len(negatives) >= distractors:
                continue
            chunks = [c for c in store.document_chunk_ids(doc) if c in positions]
            if chunks:
                negatives.append(max(chunks, key=lambda c: (float(scores[positions[c]]), -c)))
        if len(negatives) < 4:
            continue
        built.append({"id": qid, "question": record["question"],
                      "positive": positive, "negatives": negatives})
    return built


def metrics(ranks, size):
    top1 = sum(1 for r in ranks if r == 1) / max(1, len(ranks))
    top3 = sum(1 for r in ranks if r <= 3) / max(1, len(ranks))
    mrr = sum(1.0 / r for r in ranks) / max(1, len(ranks))
    return {"questions": len(ranks), "group_size": size,
            "top1": round(top1, 4), "top3": round(top3, 4), "mrr": round(mrr, 4)}


def run(store_path, lexical_directory, questions_path, model_path, mode="embedding",
        device="cuda", limit=300, distractors=23, batch_size=16, max_length=768,
        output=None, lexical_settings=None):
    store = Store(store_path)
    lexical = Lexical(lexical_directory, lexical_settings)
    built = groups(store, lexical, questions_path, limit, distractors)
    if not built:
        raise ValueError("No probe groups could be built from these questions")
    started = time.time()
    ranks, sizes = [], set()

    if mode == "embedding":
        import numpy as np

        from .dense import Embedder
        model = Embedder(model_path, device=device)
        parameters = model.parameters
        for group in built:
            chunks = [group["positive"]] + group["negatives"]
            texts = [store.chunk_text(c) for c in chunks]
            passages = model.encode(texts, batch_size, max_length).astype("float32")
            query = model.encode([group["question"]], 1, 256, query=True).astype("float32")[0]
            scores = passages @ query
            order = np.argsort(-scores, kind="stable")
            ranks.append(int(np.where(order == 0)[0][0]) + 1)
            sizes.add(len(chunks))
    else:
        from .rerank import Reranker
        model = Reranker(model_path, device=device, max_length=max_length)
        parameters = model.parameters
        for group in built:
            chunks = [group["positive"]] + group["negatives"]
            texts = [store.chunk_text(c) for c in chunks]
            scores = model.score(group["question"], texts, batch_size, max_length)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            ranks.append(order.index(0) + 1)
            sizes.add(len(chunks))

    result = metrics(ranks, sorted(sizes)[-1] if sizes else 0)
    result.update(model=str(model_path), mode=mode, parameters=parameters,
                  head=getattr(model, "head", None), pooling=getattr(model, "pooling", None),
                  seconds=round(time.time() - started))
    model.close()
    store.close()
    print(f"{Path(model_path).name} [{mode}] top1={result['top1']} top3={result['top3']} "
          f"mrr={result['mrr']} over {result['questions']} groups of {result['group_size']} "
          f"({result['parameters']:,} parameters, {result['seconds']}s)")
    if output:
        write_json(output, result)
    return result
