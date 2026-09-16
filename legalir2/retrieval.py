"""Measure the candidate generator on its own terms, per channel.

legalir2 judged every change by final Recall@5, which is the right metric for the
system and the wrong one for this half of it. Reranking converts about 58% of the
headroom it is given, and that rate held across seven different interventions - so a
genuine 0.010 improvement in retrieval arrives as 0.006 of final recall, which is below
what 1,050 questions can resolve. Measured that way, a real gain reads as noise and gets
abandoned.

So this reports candidate recall for the lexical channel, the dense channel and the
fusion separately, with a paired comparison against a saved baseline. A retrieval change
is judged on whether gold reaches the pool and how high, and only afterwards on what the
reranker does with it.
"""
import json
import time
from pathlib import Path

from .store import write_json

DEPTHS = (1, 5, 10, 20, 50, 100, 200)


def evaluate(pipeline, questions, depths=DEPTHS, progress=100):
    """Per-channel candidate recall, plus the per-question ranks behind it.

    Recall here is the organisers' partial-credit formula - the fraction of a question's
    gold documents inside the top d - so it is directly comparable with the numbers the
    rest of the project reports, and a multi-gold question cannot be "solved" by finding
    one of its three answers.
    """
    channels = ("lexical", "dense", "fused")
    ranks = {name: {} for name in channels}
    items = list(questions.items())
    started = time.time()
    for number, (qid, record) in enumerate(items, 1):
        gold = list(record["answer"])
        found = {}
        pipeline.candidates(record["question"], channels=found)
        for name in channels:
            order = {doc: position + 1
                     for position, (doc, _) in enumerate(found.get(name) or [])}
            ranks[name][qid] = {g: order.get(g) for g in gold}
        if number % progress == 0:
            rate = number / max(1e-6, time.time() - started)
            print(f"  {number}/{len(items)} questions, {rate:.2f} q/s, "
                  f"eta {(len(items) - number) / max(rate, 1e-9) / 60:.1f} min",
                  flush=True)

    report = {}
    for name in channels:
        report[name] = {f"@{d}": _recall_at(ranks[name], d) for d in depths}
    return report, ranks


def _recall_at(ranks, depth):
    total = 0.0
    for per_gold in ranks.values():
        if not per_gold:
            continue
        hit = sum(1 for r in per_gold.values() if r is not None and r <= depth)
        total += hit / len(per_gold)
    return round(total / max(1, len(ranks)), 4)


def per_question(ranks, depth):
    """{question id: recall@depth} for one channel - the input a paired test needs."""
    out = {}
    for qid, per_gold in ranks.items():
        if not per_gold:
            continue
        hit = sum(1 for r in per_gold.values() if r is not None and r <= depth)
        out[qid] = hit / len(per_gold)
    return out


def report(path, ranks, label=""):
    write_json(path, {"label": label, "ranks": ranks})


def compare(before_path, after_path, channel="dense", depths=(5, 20, 50), label=""):
    """Paired comparison of two saved retrieval runs, per depth.

    Reported at several depths because they answer different questions. @50 is the
    ceiling the reranker can reach; @5 is the order it starts from; a change that lifts
    @50 without lifting @5 has bought headroom rather than spent it.
    """
    from .compare import bootstrap, sign_test

    before = json.loads(Path(before_path).read_text(encoding="utf-8"))["ranks"][channel]
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))["ranks"][channel]
    keys = sorted(set(before) & set(after))
    print(f"\n  {label or 'retrieval'} - {channel} channel, {len(keys)} questions")
    out = {}
    for depth in depths:
        first = per_question({k: before[k] for k in keys}, depth)
        second = per_question({k: after[k] for k in keys}, depth)
        differences = [second[k] - first[k] for k in keys if k in first and k in second]
        if not differences:
            continue
        mean = sum(differences) / len(differences)
        low, high = bootstrap(differences, resamples=4000)
        wins = sum(1 for d in differences if d > 0)
        losses = sum(1 for d in differences if d < 0)
        real = not (low <= 0 <= high)
        print(f"    @{depth:<4} {sum(first.values())/len(first):.4f} -> "
              f"{sum(second.values())/len(second):.4f}   {mean:+.4f} "
              f"[{low:+.4f}, {high:+.4f}]  {wins}w/{losses}l  "
              f"p={sign_test(wins, losses):.3f}  {'REAL' if real else '-'}")
        out[f"@{depth}"] = {"mean": mean, "low": low, "high": high, "real": real}

    # 58% is what seven measured interventions converted, not a guess. It is printed so
    # a retrieval gain can be translated into the currency the leaderboard pays in,
    # WITHOUT pretending the translation is exact.
    if "@50" in out and "@5" in out:
        expected = 0.58 * (out["@50"]["mean"] - out["@5"]["mean"]) + out["@5"]["mean"]
        print(f"\n    at the historical 58% conversion rate this is worth roughly "
              f"{expected:+.4f}")
        print(f"    of final Recall@5 - detection floor on 1,050 questions is 0.0094,")
        print(f"    so judge this stage HERE and not on the end-to-end number.")
    return out
