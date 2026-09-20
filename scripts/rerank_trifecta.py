"""Trifecta Consensus Tournament (Trifecta-CT) Re-ranking for DSC-2026 LegalIR.

Combines:
1. Cross-Question Memory Borrowing (Idea #1) from 7,000 labelled training questions
2. Corpus BM25F + Bigrams Retrieval (Idea #2) from 341k legal chunks
3. PageRank Markov Stationary Distribution on Pairwise Graph (Idea #4) from 16x16 P matrices
4. Gated Pairwise Swap Tournament with conservative head-to-head gating
5. High-confidence External Candidate Injection for low-confidence incumbent tails

Operates 100% on CPU without requiring any GPU or additional training.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import glob
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legalir2.lexical import Lexical
from legalir2.question_memory import QuestionMemory

DEFAULT_W_BT = 0.15
DEFAULT_W_BORDA = 0.25
DEFAULT_W_PR = 0.08
DEFAULT_W_MEM = 0.10
DEFAULT_W_BM = 0.00
DEFAULT_P_THRESH = 0.52
DEFAULT_MAX_DEPTH = 14


def load_cached_records(cache_dir: Path):
    part_files = sorted(glob.glob(str(cache_dir / "part-*.json")))
    if not part_files:
        raise FileNotFoundError(f"No part-*.json files found in {cache_dir}")
    records = []
    for p in part_files:
        with open(p, "r", encoding="utf-8") as fp:
            records.extend(json.load(fp))
    return records


def pagerank_score(P, alpha=0.85, max_iter=50):
    """Computes PageRank stationary distribution over the pairwise comparison matrix P."""
    N = len(P)
    col_sums = P.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    M = P / col_sums[np.newaxis, :]
    v = np.ones(N) / N
    for _ in range(max_iter):
        v = alpha * (M @ v) + (1.0 - alpha) / N
    return v


def zscore(arr):
    std = float(arr.std())
    return (arr - arr.mean()) / (std if std > 1e-6 else 1.0)


def extract_retrieval_signals(records, qm: QuestionMemory, lex: Lexical, max_workers=4, cache_file: Path = None):
    """Parallel extraction of Question Memory and BM25 scores for all records, with optional disk cache."""
    if cache_file and cache_file.exists():
        print(f"Loading cached retrieval signals from {cache_file}...", flush=True)
        with open(cache_file, "r", encoding="utf-8") as fp:
            return json.load(fp)

    print(f"Extracting external signals for {len(records)} questions ({max_workers} threads)...", flush=True)
    t0 = time.time()

    def process_one(r):
        q = r["question"]
        # Question memory search
        ev, qm_rank = qm.search(q, neighbors=20)
        # BM25 corpus search
        bm_docs, _ = lex.search(q, limit=100)
        return {
            "id": str(r["id"]),
            "qm_ev": {d: (float(ev[d][1]), float(ev[d][0])) for d in qm_rank[:20]},
            "qm_rank": qm_rank[:10],
            "bm_dict": dict(bm_docs[:30]),
            "bm_rank": [d[0] for d in bm_docs[:10]]
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(process_one, records))

    lookup = {item["id"]: item for item in results}
    print(f"Extracted signals in {time.time() - t0:.2f}s.", flush=True)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as fp:
            json.dump(lookup, fp)
        print(f"Saved retrieval signals to {cache_file}.")

    return lookup


def run_trifecta(records,
                 retrieval_signals,
                 w_bt=DEFAULT_W_BT,
                 w_borda=DEFAULT_W_BORDA,
                 w_pr=DEFAULT_W_PR,
                 w_mem=DEFAULT_W_MEM,
                 w_bm=DEFAULT_W_BM,
                 p_thresh=DEFAULT_P_THRESH,
                 max_depth=DEFAULT_MAX_DEPTH,
                 enable_external_injection=True,
                 ext_min_sim=0.70,
                 ext_weakest_max_prior=-0.3):
    """Runs Trifecta-CT multi-signal reranking with gated swaps and external injection."""
    predictions = {}
    stats = {
        "total_internal_swaps": 0,
        "questions_internal_swapped": 0,
        "total_external_injections": 0
    }

    for r in records:
        qid = str(r["id"])
        P = np.asarray(r["probabilities"], dtype=float)
        N = len(P)

        # 1. Soft Borda count
        borda = (P.sum(axis=1) - 0.5) / (N - 1)
        borda_norm = zscore(borda)

        # 2. PageRank Stationary Centrality
        pr = pagerank_score(P)
        pr_norm = zscore(pr)

        # 3. External retrieval signals
        sig = retrieval_signals.get(qid, {})
        qm_ev = sig.get("qm_ev", {})
        bm_dict = sig.get("bm_dict", {})

        qm_scores = np.array([qm_ev.get(d, (0.0, 0.0))[0] for d in r["docs"]], dtype=float)
        qm_norm = zscore(qm_scores) if qm_scores.max() > 0 else np.zeros(N)

        bm_scores = np.array([bm_dict.get(d, 0.0) for d in r["docs"]], dtype=float)
        bm_norm = zscore(bm_scores) if bm_scores.max() > 0 else np.zeros(N)

        # 4. Multi-signal ensemble
        prior = np.asarray(r["prior"], dtype=float)
        ability = np.asarray(r["ability"], dtype=float)
        score = (prior
                 + w_bt * (ability - prior)
                 + w_borda * borda_norm
                 + w_pr * pr_norm
                 + w_mem * qm_norm
                 + w_bm * bm_norm)

        # 5. Anchor with V9 baseline Top-5
        cur_top = list(r["baseline"][:5])
        eligible = r["docs"][:max_depth]
        ranked_docs = [d for _, d in sorted(zip(-score[:max_depth], eligible))]

        new_top = list(cur_top)
        swapped_in_q = 0

        # 6. Tầng 1: Gated Internal Tournament Swap
        for cand in ranked_docs:
            if cand in new_top:
                continue
            cand_idx = r["docs"].index(cand)
            weakest = min(new_top, key=lambda d: score[r["docs"].index(d)])
            weakest_idx = r["docs"].index(weakest)

            # Score superiority AND direct head-to-head support
            if score[cand_idx] > score[weakest_idx] and P[cand_idx, weakest_idx] >= p_thresh:
                new_top.remove(weakest)
                new_top.append(cand)
                stats["total_internal_swaps"] += 1
                swapped_in_q += 1

        if swapped_in_q > 0:
            stats["questions_internal_swapped"] += 1

        # 7. Tầng 2: High-Confidence External Injection (Only for very weak tail items)
        if enable_external_injection:
            weakest_doc = min(new_top, key=lambda d: score[r["docs"].index(d)])
            weakest_idx = r["docs"].index(weakest_doc)
            weakest_prior = prior[weakest_idx]
            weakest_score = score[weakest_idx]

            # Only consider replacing if weakest is genuinely low-confidence
            if weakest_prior <= ext_weakest_max_prior and weakest_score < score.mean():
                qm_rank = sig.get("qm_rank", [])
                bm_rank = sig.get("bm_rank", [])
                
                best_ext = None
                best_ext_sim = 0.0
                for d in qm_rank[:5]:
                    if d not in r["docs"] and d not in new_top:
                        mass, sim = qm_ev.get(d, (0.0, 0.0))
                        if sim >= ext_min_sim and sim > best_ext_sim:
                            if d in bm_dict or sim >= 0.80:
                                best_ext = d
                                best_ext_sim = sim

                if best_ext is not None:
                    new_top.remove(weakest_doc)
                    new_top.append(best_ext)
                    stats["total_external_injections"] += 1

        # Format answers: exactly 5 unique strings
        ans = [str(d) for d in new_top[:5]]
        if len(ans) != 5 or len(set(ans)) != 5:
            raise ValueError(f"Invalid answer set for question {qid}: {ans}")
        predictions[qid] = ans

    return predictions, stats


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
    json_path = output_dir / "submission-trifecta.json"
    zip_path = output_dir / "submission-trifecta.zip"

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

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(formatted, ensure_ascii=False, indent=2))

    print(f"Submission packaged: {zip_path} ({zip_path.stat().st_size:,} bytes)")
    return json_path, zip_path


def main():
    parser = argparse.ArgumentParser(description="Trifecta-CT Re-ranking for DSC-2026")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache/duo"))
    parser.add_argument("--questions", type=Path, default=Path("data/private-official.json"))
    parser.add_argument("--train", type=Path, default=Path("data/train.json"))
    parser.add_argument("--lexical", type=Path, default=Path("indexes/lexical"))
    parser.add_argument("--val-cache", type=Path, default=Path("artifacts/validation/duo"))
    parser.add_argument("--out", type=Path, default=Path("submissions/private/20260920-trifecta"))
    parser.add_argument("--w-bt", type=float, default=DEFAULT_W_BT)
    parser.add_argument("--w-borda", type=float, default=DEFAULT_W_BORDA)
    parser.add_argument("--w-pr", type=float, default=DEFAULT_W_PR)
    parser.add_argument("--w-mem", type=float, default=DEFAULT_W_MEM)
    parser.add_argument("--w-bm", type=float, default=DEFAULT_W_BM)
    parser.add_argument("--p-thresh", type=float, default=DEFAULT_P_THRESH)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--disable-ext", action="store_true", help="Disable external injection")
    parser.add_argument("--ext-min-sim", type=float, default=0.70)
    parser.add_argument("--ext-weakest-max-prior", type=float, default=-0.3)
    parser.add_argument("--signal-cache-dir", type=Path, default=Path("artifacts/cache/signals"), help="Cache dir for signals")
    parser.add_argument("--val-only", action="store_true", help="Only run validation benchmark and exit")
    args = parser.parse_args()

    args.signal_cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Question Memory and BM25
    print("=== Initializing Multi-Signal Engines ===")
    t0 = time.time()
    with open(args.train, "r", encoding="utf-8") as fp:
        train_data = json.load(fp)
    qm = QuestionMemory(train_data)
    print(f"Loaded QuestionMemory from {args.train} ({len(train_data)} records) in {time.time() - t0:.2f}s")

    t1 = time.time()
    lex = Lexical(args.lexical, {"chunk_candidates": 100})
    print(f"Loaded Lexical BM25 from {args.lexical} ({lex.unigram.chunks} chunks) in {time.time() - t1:.2f}s")

    # 2. Run Validation Benchmark
    if args.val_cache.exists():
        print(f"\n=== Validation Benchmark ({args.val_cache}) ===")
        val_records = load_cached_records(args.val_cache)
        val_truth = {str(r["id"]): r["gold"] for r in val_records}
        val_signal_file = args.signal_cache_dir / "val_signals.json"
        val_signals = extract_retrieval_signals(val_records, qm, lex, cache_file=val_signal_file)

        val_preds, val_stats = run_trifecta(
            val_records, val_signals,
            w_bt=args.w_bt, w_borda=args.w_borda, w_pr=args.w_pr,
            w_mem=args.w_mem, w_bm=args.w_bm,
            p_thresh=args.p_thresh, max_depth=args.max_depth,
            enable_external_injection=not args.disable_ext,
            ext_min_sim=args.ext_min_sim,
            ext_weakest_max_prior=args.ext_weakest_max_prior
        )
        val_rec, val_prec = evaluate_metrics(val_preds, val_truth)

        base_preds = {str(r["id"]): list(r["baseline"][:5]) for r in val_records}
        base_rec, base_prec = evaluate_metrics(base_preds, val_truth)

        regressions = 0
        improvements = 0
        for qid, gold in val_truth.items():
            base_hit = len(set(base_preds[qid]) & set(gold))
            trif_hit = len(set(val_preds[qid]) & set(gold))
            if trif_hit > base_hit:
                improvements += 1
            elif trif_hit < base_hit:
                regressions += 1

        print(f"V9 Baseline Recall:     {base_rec:.6f} | Precision: {base_prec:.6f}")
        print(f"Trifecta-CT Recall:     {val_rec:.6f} | Precision: {val_prec:.6f}")
        print(f"Delta vs Baseline:      {val_rec - base_rec:+.6f}")
        print(f"Questions improved:     {improvements} | Regressions: {regressions}")
        print(f"Internal swaps:         {val_stats['total_internal_swaps']} across {val_stats['questions_internal_swapped']} questions")
        print(f"External injections:    {val_stats['total_external_injections']}")

        if args.val_only:
            print("\nValidation only mode requested. Exiting.")
            return

    # 3. Run on Target (Private Test)
    print(f"\n=== Processing Target ({args.cache}) ===")
    records = load_cached_records(args.cache)
    target_signal_file = args.signal_cache_dir / "target_signals.json"
    target_signals = extract_retrieval_signals(records, qm, lex, cache_file=target_signal_file)

    predictions, stats = run_trifecta(
        records, target_signals,
        w_bt=args.w_bt, w_borda=args.w_borda, w_pr=args.w_pr,
        w_mem=args.w_mem, w_bm=args.w_bm,
        p_thresh=args.p_thresh, max_depth=args.max_depth,
        enable_external_injection=not args.disable_ext,
        ext_min_sim=args.ext_min_sim,
        ext_weakest_max_prior=args.ext_weakest_max_prior
    )
    print(f"Generated predictions for {len(predictions)} questions.")
    print(f"Internal swaps: {stats['total_internal_swaps']} across {stats['questions_internal_swapped']} questions.")
    print(f"External injections: {stats['total_external_injections']} questions.")

    json_path, zip_path = package_submission(predictions, args.questions, args.out)

    manifest = {
        "method": "Trifecta Consensus Tournament (Trifecta-CT)",
        "parameters": {
            "w_bt": args.w_bt,
            "w_borda": args.w_borda,
            "w_pr": args.w_pr,
            "w_mem": args.w_mem,
            "w_bm": args.w_bm,
            "p_thresh": args.p_thresh,
            "max_depth": args.max_depth,
            "enable_external_injection": not args.disable_ext,
            "ext_min_sim": args.ext_min_sim,
            "ext_weakest_max_prior": args.ext_weakest_max_prior
        },
        "questions_count": len(predictions),
        "internal_swaps": stats["total_internal_swaps"],
        "questions_swapped": stats["questions_internal_swapped"],
        "external_injections": stats["total_external_injections"],
        "submission_json": str(json_path),
        "submission_zip": str(zip_path),
    }
    with open(args.out / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Manifest saved to {args.out / 'manifest.json'}.")


if __name__ == "__main__":
    main()
