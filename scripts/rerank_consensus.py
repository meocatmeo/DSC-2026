"""Gated Soft-Borda Consensus Tournament (GSB-CT) Re-ranking for DSC-2026 LegalIR.

Operates purely on CPU using precomputed pointwise priors and pairwise comparison
probability matrices, without requiring any GPU inference or training.
"""
import argparse
import glob
import json
from pathlib import Path
import zipfile
import numpy as np

DEFAULT_W_BT = 0.15
DEFAULT_W_BORDA = 0.25
DEFAULT_P_THRESH = 0.52
DEFAULT_MAX_DEPTH = 10


def load_cached_records(cache_dir: Path):
    part_files = sorted(glob.glob(str(cache_dir / "part-*.json")))
    if not part_files:
        raise FileNotFoundError(f"No part-*.json files found in {cache_dir}")
    records = []
    for p in part_files:
        with open(p, "r", encoding="utf-8") as fp:
            records.extend(json.load(fp))
    return records


def run_gsb_ct(records,
               w_bt=DEFAULT_W_BT,
               w_borda=DEFAULT_W_BORDA,
               p_thresh=DEFAULT_P_THRESH,
               max_depth=DEFAULT_MAX_DEPTH):
    """Applies Gated Soft-Borda Consensus Tournament reranking to cached candidate records."""
    predictions = {}
    swap_stats = {"total_swaps": 0, "questions_changed": 0}

    for r in records:
        qid = str(r["id"])
        P = np.asarray(r["probabilities"], dtype=float)
        N = len(P)

        # 1. Linear Soft Borda Count (Global Win Rate)
        borda = (P.sum(axis=1) - 0.5) / (N - 1)
        borda_std = float(borda.std())
        borda_norm = (borda - borda.mean()) / (borda_std if borda_std > 1e-6 else 1.0)

        # 2. Multi-signal ensemble
        prior = np.asarray(r["prior"], dtype=float)
        ability = np.asarray(r["ability"], dtype=float)
        score = prior + w_bt * (ability - prior) + w_borda * borda_norm

        # 3. Anchor with V9 baseline top-5
        cur_top = list(r["baseline"][:5])
        eligible = r["docs"][:max_depth]
        ranked_docs = [d for _, d in sorted(zip(-score[:max_depth], eligible))]

        new_top = list(cur_top)
        swapped_in_q = 0

        # 4. Conservative Pairwise Swap Gating
        for cand in ranked_docs:
            if cand in new_top:
                continue
            cand_idx = r["docs"].index(cand)
            # Find weakest incumbent
            weakest = min(new_top, key=lambda d: score[r["docs"].index(d)])
            weakest_idx = r["docs"].index(weakest)

            # Requires score superiority AND direct pairwise head-to-head support
            if score[cand_idx] > score[weakest_idx] and P[cand_idx, weakest_idx] >= p_thresh:
                new_top.remove(weakest)
                new_top.append(cand)
                swap_stats["total_swaps"] += 1
                swapped_in_q += 1

        if swapped_in_q > 0:
            swap_stats["questions_changed"] += 1

        # Format answers: exactly 5 unique strings
        ans = [str(d) for d in new_top[:5]]
        if len(ans) != 5 or len(set(ans)) != 5:
            raise ValueError(f"Invalid answer set for question {qid}: {ans}")
        predictions[qid] = ans

    return predictions, swap_stats


def evaluate_metrics(predictions, ground_truth):
    """Calculates official competition Recall and Precision."""
    recalls = []
    precisions = []
    for qid, gold in ground_truth.items():
        qid_str = str(qid)
        pred = predictions.get(qid_str, [])
        gold_set = set(gold)
        pred_set = set(pred)
        if 0 < len(pred) <= 5:
            rec = len(pred_set & gold_set) / len(gold_set)
            prec = len(pred_set & gold_set) / len(pred)
        else:
            rec = 0.0
            prec = 0.0
        recalls.append(rec)
        precisions.append(prec)
    return float(np.mean(recalls)), float(np.mean(precisions))


def package_submission(predictions, official_questions_path: Path, output_dir: Path):
    """Creates the official submission.zip containing submission.json."""
    with open(official_questions_path, "r", encoding="utf-8") as fp:
        official_questions = json.load(fp)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "submission-consensus.json"
    zip_path = output_dir / "submission-consensus.zip"

    formatted = {}
    for qid in official_questions:
        qid_str = str(qid)
        if qid_str not in predictions:
            raise KeyError(f"Missing prediction for official question {qid_str}")
        formatted[qid_str] = {"answer": predictions[qid_str]}

    if len(formatted) != len(official_questions):
        raise ValueError(f"Count mismatch: expected {len(official_questions)}, got {len(formatted)}")

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(formatted, fp, ensure_ascii=False, indent=2)

    # Pack into zip containing only submission.json
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(formatted, ensure_ascii=False, indent=2))

    print(f"Submission packaged: {zip_path} ({zip_path.stat().st_size:,} bytes)")
    return json_path, zip_path


def main():
    parser = argparse.ArgumentParser(description="GSB-CT Re-ranking for DSC-2026")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache/duo"), help="Path to duo cache")
    parser.add_argument("--questions", type=Path, default=Path("data/private-official.json"), help="Official questions JSON")
    parser.add_argument("--out", type=Path, default=Path("submissions/private/20260920-consensus"), help="Output directory")
    parser.add_argument("--val-cache", type=Path, default=Path("artifacts/validation/duo"), help="Validation cache")
    parser.add_argument("--w-bt", type=float, default=DEFAULT_W_BT)
    parser.add_argument("--w-borda", type=float, default=DEFAULT_W_BORDA)
    parser.add_argument("--p-thresh", type=float, default=DEFAULT_P_THRESH)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    args = parser.parse_args()

    # If validation cache exists, run validation check
    if args.val_cache.exists():
        val_records = load_cached_records(args.val_cache)
        val_truth = {str(r["id"]): r["gold"] for r in val_records}
        val_preds, val_stats = run_gsb_ct(val_records, args.w_bt, args.w_borda, args.p_thresh, args.max_depth)
        val_rec, val_prec = evaluate_metrics(val_preds, val_truth)
        
        # Compute baseline recall
        base_preds = {str(r["id"]): list(r["baseline"][:5]) for r in val_records}
        base_rec, base_prec = evaluate_metrics(base_preds, val_truth)
        
        print(f"=== Validation Benchmark ({len(val_records)} questions) ===")
        print(f"V9 Baseline Recall: {base_rec:.6f} | Precision: {base_prec:.6f}")
        print(f"GSB-CT Recall:      {val_rec:.6f} | Precision: {val_prec:.6f}")
        print(f"Delta vs Baseline:  {val_rec - base_rec:+.6f}")
        print(f"Swaps applied:      {val_stats['total_swaps']} across {val_stats['questions_changed']} questions")

    # Run on target questions (e.g. private test)
    print(f"\n=== Processing Target ({args.cache}) ===")
    records = load_cached_records(args.cache)
    print(f"Loaded {len(records)} records.")
    predictions, stats = run_gsb_ct(records, args.w_bt, args.w_borda, args.p_thresh, args.max_depth)
    print(f"Generated predictions for {len(predictions)} questions.")
    print(f"Applied {stats['total_swaps']} candidate swaps across {stats['questions_changed']} questions.")

    json_path, zip_path = package_submission(predictions, args.questions, args.out)

    # Write summary manifest
    manifest = {
        "method": "Gated Soft-Borda Consensus Tournament (GSB-CT)",
        "parameters": {
            "w_bt": args.w_bt,
            "w_borda": args.w_borda,
            "p_thresh": args.p_thresh,
            "max_depth": args.max_depth,
        },
        "questions_count": len(predictions),
        "total_swaps": stats["total_swaps"],
        "questions_changed": stats["questions_changed"],
        "submission_json": str(json_path),
        "submission_zip": str(zip_path),
    }
    with open(args.out / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print("Manifest saved.")


if __name__ == "__main__":
    main()
