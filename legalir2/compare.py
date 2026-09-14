"""Paired comparison of two runs over the same questions.

Every decision left in this project is "is system B better than system A", and both are
always scored on the same 1050 validation questions. Comparing their means throws that
away: the standard error of a mean is ~0.0075 here, so differences smaller than about
0.015 are invisible, and most of the differences we care about are smaller than that.

A paired comparison is far more powerful. Question difficulty cancels, because the same
question is scored under both systems, so only the questions where they DISAGREE carry
information. Typically 40-80 of 1050 disagree, and the standard error of the paired
difference is several times smaller than that of either mean.

Reports the paired difference with a bootstrap interval, the win/loss split, and - since
recall here is often 0 or 1 - an exact sign test on the disagreements, which needs no
distributional assumption at all.
"""
import json
import math
import random
from pathlib import Path


def per_question(diagnostics):
    """{question id: recall}, using the organiser's partial-credit formula."""
    out = {}
    for key, record in diagnostics.items():
        gold = record["relevant"]
        predicted = record["predicted"][:5]
        out[key] = len(set(gold) & set(predicted)) / len(gold)
    return out


def sign_test(wins, losses):
    """Two-sided exact binomial p under 'a disagreement is a coin flip'.

    No normal approximation: with 40-odd disagreements the approximation is exactly
    where it is least trustworthy, and the exact sum is a few lines of arithmetic.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def bootstrap(differences, resamples=10000, seed=2026):
    rng = random.Random(seed)
    n = len(differences)
    means = []
    for _ in range(resamples):
        means.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def compare(before_path, after_path, label="", subset=None, resamples=10000):
    before = per_question(json.loads(Path(before_path).read_text(encoding="utf-8")))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))
    gold_size = {k: len(v["relevant"]) for k, v in after.items()}
    after = per_question(after)

    keys = sorted(set(before) & set(after))
    if subset == "single":
        keys = [k for k in keys if gold_size.get(k, 1) == 1]
    elif subset == "multi":
        keys = [k for k in keys if gold_size.get(k, 1) > 1]
    if not keys:
        raise ValueError("no questions in common")

    differences = [after[k] - before[k] for k in keys]
    wins = sum(1 for d in differences if d > 0)
    losses = sum(1 for d in differences if d < 0)
    mean_before = sum(before[k] for k in keys) / len(keys)
    mean_after = sum(after[k] for k in keys) / len(keys)
    delta = mean_after - mean_before
    low, high = bootstrap(differences, resamples)
    p = sign_test(wins, losses)

    name = label or f"{Path(after_path).name} vs {Path(before_path).name}"
    scope = {"single": " (single-gold only)", "multi": " (multi-gold only)"}.get(subset, "")
    print(f"\n  {name}{scope}")
    print(f"    questions compared   {len(keys)}")
    print(f"    before               {mean_before:.4f}")
    print(f"    after                {mean_after:.4f}")
    print(f"    paired difference    {delta:+.4f}   95% CI [{low:+.4f}, {high:+.4f}]")
    print(f"    disagreements        {wins} better, {losses} worse, "
          f"{len(keys) - wins - losses} identical")
    print(f"    sign test            p = {p:.3f}")
    if low <= 0 <= high:
        print(f"    VERDICT: not distinguishable from no change. Do not spend GPU hours")
        print(f"             chasing a difference this size - it is below what {len(keys)}")
        print(f"             questions can resolve.")
    elif delta > 0:
        print(f"    VERDICT: a real improvement.")
    else:
        print(f"    VERDICT: a real regression.")
    return {"questions": len(keys), "before": mean_before, "after": mean_after,
            "delta": delta, "ci": [low, high], "wins": wins, "losses": losses, "p": p}
