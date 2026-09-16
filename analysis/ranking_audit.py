"""Audit saved LegalIR results without loading a model or changing predictions.

Run from the repository root: python -m analysis.ranking_audit
"""
import argparse
import collections
import math
import unicodedata
from pathlib import Path

from legalir2.cache import folds
from legalir2.compare import sign_test
from legalir2.predict import metrics
from legalir2.store import Store, file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def recall(record):
    predicted, gold = record["predicted"], record["relevant"]
    if not gold or len(set(gold)) != len(gold):
        raise ValueError("gold must be nonempty and unique")
    if len(set(predicted)) != len(predicted):
        raise ValueError("predictions must be unique")
    return len(set(gold) & set(predicted)) / len(gold) if 0 < len(predicted) <= 5 else 0.0


def audit(results, train_path, validation_path, all_path, store_path):
    base_path = results / "v4-stacked.diagnostics.json"
    baseline = read_json(base_path)
    manifest = read_json(results / "submission-v4.run.json")
    validation_manifest = read_json(results / "v4-stacked.run.json")
    if manifest["settings"] != validation_manifest["settings"]:
        raise ValueError("submission and validation settings differ")
    if manifest["models"] != validation_manifest["models"]:
        raise ValueError("submission and validation model paths differ")
    train, validation, all_questions = map(read_json, (train_path, validation_path, all_path))
    if set(baseline) != set(validation) or set(train) & set(validation):
        raise ValueError("validation coverage mismatch or overlapping train/validation ids")
    for qid, row in baseline.items():
        if row["question"] != validation[qid]["question"] or set(row["relevant"]) != set(validation[qid]["answer"]):
            raise ValueError(f"question or gold differs: {qid}")
    depth = int(manifest["settings"]["candidates"])
    scores = {q: recall(r) for q, r in baseline.items()}
    mean = sum(scores.values()) / len(scores)
    official_shape = metrics({q: {"answer": r["predicted"]} for q, r in baseline.items()},
                             {q: r["relevant"] for q, r in baseline.items()})
    if not math.isclose(mean, official_shape["recall"], abs_tol=1e-12):
        raise ValueError("recall does not match pipeline scorer")
    recorded = read_json(results / "v4-stacked.metrics.json")
    if not math.isclose(mean, recorded["recall"], abs_tol=1e-12):
        raise ValueError("recomputed baseline does not match saved metrics")

    report = {
        "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in
                    (base_path, results / "submission-v4.run.json", train_path, validation_path)},
        "settings": manifest["settings"], "models": manifest["models"],
        "parameters": manifest["audit"], "validation_recall": mean,
        "candidate_recall": {}, "buckets": {}, "misses": [], "comparisons": {},
    }
    for k in (5, 20, 50, 100, 200):
        report["candidate_recall"][str(k)] = sum(
            sum(0 < (r["candidate_ranks"].get(g) or 10**9) <= k for g in r["relevant"])
            / len(r["relevant"]) for r in baseline.values()) / len(baseline)

    frequency = collections.Counter(g for r in train.values() for g in set(r["answer"]))
    store = Store(store_path)
    try:
        report["local_corpus_metadata_differences"] = {
            k: {"local": store.metadata.get(k), "baseline": manifest["store"].get(k)}
            for k in set(store.metadata) | set(manifest["store"])
            if store.metadata.get(k) != manifest["store"].get(k)
        }
        # Titles are annotations only. All scores and ranks come from saved results;
        # a differing corpus fingerprint prevents a reproduction claim, not this audit.
        for qid, row in baseline.items():
            for gold in row["relevant"]:
                if gold in row["predicted"]:
                    continue
                rank = row["candidate_ranks"].get(gold)
                report["misses"].append({
                    "question_id": qid, "question": row["question"],
                    "gold": gold, "gold_count": len(row["relevant"]),
                    "candidate_rank": rank, "was_reranked": rank is not None and rank <= depth,
                    "recall_mass": 1 / len(row["relevant"]),
                    "training_questions_with_gold": frequency[gold],
                    "gold_title": store.label(gold),
                    "predicted": [{"id": d, "title": store.label(d)} for d in row["predicted"]],
                })
    finally:
        store.close()

    for name, multi in (("single", False), ("multi", True)):
        keys = [q for q, r in baseline.items() if (len(r["relevant"]) > 1) == multi]
        misses = [r for r in report["misses"] if (r["gold_count"] > 1) == multi]
        report["buckets"][name] = {
            "questions": len(keys), "recall": sum(scores[q] for q in keys) / len(keys),
            "missed_gold_documents": len(misses),
            "missed_gold_inside_rerank_pool": sum(r["was_reranked"] for r in misses),
            "lost_macro_recall": sum(r["recall_mass"] for r in misses) / len(baseline),
            "recoverable_macro_recall_at_actual_depth": sum(
                r["recall_mass"] for r in misses if r["was_reranked"]) / len(baseline),
        }
    lost = sum(r["recall_mass"] for r in report["misses"]) / len(baseline)
    if not math.isclose(lost, 1 - mean, abs_tol=1e-12):
        raise ValueError("error decomposition does not account for all recall loss")

    for path in sorted(results.glob("*.diagnostics.json")):
        other = read_json(path)
        if set(other) != set(baseline):
            continue
        if any(set(other[q]["relevant"]) != set(baseline[q]["relevant"]) for q in baseline):
            raise ValueError(f"inconsistent gold in {path}")
        differences = [recall(other[q]) - scores[q] for q in baseline]
        n, delta = len(differences), sum(differences) / len(differences)
        wins, losses = sum(d > 0 for d in differences), sum(d < 0 for d in differences)
        variance = sum((d - delta)**2 for d in differences) / (n - 1)
        report["comparisons"][path.stem] = {
            "recall": mean + delta, "delta_vs_stacked": delta,
            "wins": wins, "losses": losses, "equal_recall": n - wins - losses,
            "ordered_top5_changed": sum(other[q]["predicted"] != baseline[q]["predicted"] for q in baseline),
            "top5_membership_changed": sum(set(other[q]["predicted"]) != set(baseline[q]["predicted"]) for q in baseline),
            "paired_standard_error": math.sqrt(variance / n),
            "sign_test_p": sign_test(wins, losses),
        }

    grouped = collections.defaultdict(list)
    for qid, row in all_questions.items():
        key = " ".join(unicodedata.normalize("NFC", row["question"]).lower().split())
        grouped[key].append(qid)
    duplicates = [ids for ids in grouped.values() if len(ids) > 1]
    assignment = folds(all_questions)
    report["current_id_based_crossfit"] = {
        "exact_duplicate_query_groups": len(duplicates),
        "groups_split_across_two_folds": [ids for ids in duplicates if len({assignment[q] for q in ids}) > 1],
    }
    report["limitations"] = [
        "Public 0.9493 is user-reported; public labels are unavailable.",
        "No phase-5/6 prediction files, cross-fit score caches or trained reranker weights are present locally.",
        "Candidate coverage is an oracle upper bound, not a forecast of achievable model quality.",
        "These are retrospective development diagnostics, not a new blind evaluation.",
        "Local corpus titles are auxiliary annotations; inspect corpus fingerprint differences before reproducing inference.",
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "_results" / "outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "v7-audit.json")
    args = parser.parse_args()
    report = audit(args.results, ROOT / "artifacts/splits/train.json",
                   ROOT / "artifacts/splits/validation.json", ROOT / "train.json",
                   ROOT / "artifacts2/store.sqlite")
    write_json(args.output, report)
    print(f"Validation Recall@5: {report['validation_recall']:.8f}")
    print(f"Actual rerank depth: {report['settings']['candidates']}")
    for name, bucket in report["buckets"].items():
        print(f"{name}: {bucket['missed_gold_inside_rerank_pool']} missed gold inside pool; "
              f"recoverable macro recall {bucket['recoverable_macro_recall_at_actual_depth']:.8f}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
