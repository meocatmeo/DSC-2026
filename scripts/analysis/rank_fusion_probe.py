"""Cross-fitted rank-fusion probe over precomputed orderings.

    python rank_fusion_probe.py --cache validation.compact.jsonl.gz --report out.json

THE POLICY

    score(d) = sum_m  w_m / (K + rank_m(d))          K = 60, ranks 1-based
    members  = {v9: W}  u  {g32_only, qwen_only, bundle_only: 1.0}
    ties     : ascending position in the v9 ordering
    output   : top 5

W is the ONLY free parameter. It is chosen on a fitting fold and scored on a
disjoint fold, never on the same questions.

WHY THE CROSS-FIT IS NOT OPTIONAL

Choosing W on the questions you then report is worth about +0.0023 of pure
selection bias here: the in-sample best is +0.0051 and the cross-fitted mean is
+0.0028. The first number is what an ad-hoc sweep produces and it is wrong.

Note also that "equal weights over all seven orderings" is NOT parameter-free.
Four of the seven (v9, control, treatment, treatment_without_bundle) are
near-copies of each other, differing on 3, 55 and 61 of 1,050 top-fives, so
equal weighting silently encodes W ~ 4. One vote per DISTINCT view - which is
genuinely parameter-free - scores 0.9435, i.e. 0.0029 WORSE than v9. The gain
lives entirely in the weight.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
import random
from pathlib import Path

K = 60
WEAK = ("g32_only", "qwen_only", "bundle_only")
BASE = "v9"
GRID = (1, 2, 3, 4, 5, 6, 8, 10, 14, 20)


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ordering(row, name):
    index = row["orders"][name]
    return [row["docs"][i] for i in index] if isinstance(index[0], int) else list(index)


def recall(row, selected):
    gold = set(row["gold"])
    return len(gold & set(selected)) / len(gold) if gold else 0.0


def fuse(row, weight, members=None):
    members = members or ([(BASE, weight)] + [(m, 1.0) for m in WEAK])
    score = collections.defaultdict(float)
    for name, w in members:
        for rank, doc in enumerate(ordering(row, name), 1):
            score[doc] += w / (K + rank)
    tie = {d: i for i, d in enumerate(ordering(row, BASE))}
    return [d for d, _ in sorted(score.items(),
                                 key=lambda x: (-x[1], tie.get(x[0], 10**6)))][:5]


def mean_recall(rows, weight):
    return sum(recall(r, fuse(r, weight)) for r in rows) / len(rows)


def baseline(rows):
    return sum(recall(r, ordering(r, BASE)[:5]) for r in rows) / len(rows)


def paired(rows, weight):
    """Per-question differences, plus a bootstrap interval and an exact sign test."""
    diffs = [recall(r, fuse(r, weight)) - recall(r, ordering(r, BASE)[:5]) for r in rows]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    rng = random.Random(2026)
    boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(10000))
    total = wins + losses
    p = (sum(math.comb(total, i) for i in range(min(wins, losses) + 1)) / 2 ** total * 2
         if total else 1.0)
    return {"mean": sum(diffs) / len(diffs), "ci": [boot[250], boot[9750]],
            "wins": wins, "losses": losses, "unchanged": len(diffs) - total,
            "sign_test_p": min(p, 1.0),
            "changed_sets": sum(1 for r in rows
                                if set(fuse(r, weight)) != set(ordering(r, BASE)[:5]))}


def crossfit(rows, folds=20):
    """Choose W on one half, score on the other. Both directions, many splits."""
    gains, chosen = [], collections.Counter()
    for seed in range(folds):
        index = list(range(len(rows)))
        random.Random(seed).shuffle(index)
        half = len(rows) // 2
        a = [rows[i] for i in index[:half]]
        b = [rows[i] for i in index[half:]]
        total = 0.0
        for fit, evaluate in ((a, b), (b, a)):
            w = max(GRID, key=lambda x: mean_recall(fit, x))
            chosen[w] += 1
            total += (mean_recall(evaluate, w) - baseline(evaluate)) * len(evaluate)
        gains.append(total / len(rows))
    gains.sort()
    return {"folds": folds, "mean": sum(gains) / len(gains), "median": gains[len(gains) // 2],
            "min": gains[0], "max": gains[-1], "weights_chosen": dict(sorted(chosen.items()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="compact jsonl(.gz) with docs/gold/orders")
    ap.add_argument("--report", help="write a JSON report here")
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--bar", type=float, default=0.0043,
                    help="minimum detectable effect; the gate this must clear")
    args = ap.parse_args()

    rows = load(args.cache)
    missing = [m for m in (BASE,) + WEAK if m not in rows[0]["orders"]]
    if missing:
        raise SystemExit(f"cache lacks required orderings: {missing}\n"
                         f"available: {sorted(rows[0]['orders'])}")

    base = baseline(rows)
    in_sample_w = max(GRID, key=lambda x: mean_recall(rows, x))
    in_sample = mean_recall(rows, in_sample_w) - base
    cf = crossfit(rows, args.folds)
    detail = paired(rows, in_sample_w)

    print(f"questions                {len(rows)}")
    print(f"{BASE} baseline recall@5  {base:.4f}")
    print(f"in-sample best W         {in_sample_w}  ->  {in_sample:+.4f}   (BIASED, do not quote)")
    print(f"cross-fitted mean gain   {cf['mean']:+.4f}   median {cf['median']:+.4f}"
          f"   range [{cf['min']:+.4f}, {cf['max']:+.4f}]")
    print(f"selection bias           {in_sample - cf['mean']:+.4f}")
    print(f"weights chosen per fold  {cf['weights_chosen']}")
    print(f"\nat the in-sample W, for reference only:")
    print(f"  changed sets {detail['changed_sets']}, {detail['wins']} better, "
          f"{detail['losses']} worse, sign test p={detail['sign_test_p']:.3f}")
    print(f"  95% CI [{detail['ci'][0]:+.4f}, {detail['ci'][1]:+.4f}]")

    cleared = cf["mean"] >= args.bar
    print(f"\nGATE  cross-fitted {cf['mean']:+.4f} vs bar {args.bar:+.4f}  ->  "
          f"{'CLEARED' if cleared else 'NOT CLEARED'}")
    if not cleared:
        print("  Do not submit. The in-sample number is not evidence.")

    if args.report:
        payload = {"cache": str(args.cache),
                   "cache_sha256": hashlib.sha256(Path(args.cache).read_bytes()).hexdigest(),
                   "policy": {"k": K, "base": BASE, "weak": list(WEAK), "grid": list(GRID),
                              "tie_break": "ascending position in the v9 ordering"},
                   "questions": len(rows), "baseline_recall": base,
                   "in_sample": {"weight": in_sample_w, "gain": in_sample, "biased": True},
                   "crossfit": cf, "paired_at_in_sample_weight": detail,
                   "bar": args.bar, "gate_cleared": cleared,
                   "limitation": "Reused labeled set. Gain is not a private-score forecast."}
        Path(args.report).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
