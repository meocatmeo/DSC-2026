"""Guarded H100 experiment: frozen V9, refit control, rank fix, and new passages."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .cache import normalise_question
from .compare import compare
from .pipeline import Pipeline, aggregate_passages
from .predict import load_questions, pack
from .stack import Stacker, build_features, feature_names, fit
from .store import Store, file_hash, read_json, write_json
from .v11_preflight import dense_ready, model_ready

SCHEMA = "ranking-h100-v1"
MODELS = ("embedding", "reranker-g32", "partner_qwen")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def guard(path, signature):
    """A run ID names immutable inputs, never a request to overwrite old results."""
    if path.exists():
        if read_json(path) != signature:
            raise ValueError(f"Inputs changed for {path}; use a new RUN_ID")
    else:
        # Completed shards without their guard cannot establish what was scored.
        if path.parent.exists() and any(path.parent.glob("part-*.json")):
            raise ValueError(f"Checkpoint shards have no input guard: {path.parent}")
        write_json(path, signature)


def questions(args):
    train = load_questions(args.train, labeled=True)
    validation = load_questions(args.validation, labeled=True)
    held = {normalise_question(r["question"]) for r in validation.values()}
    selected, seen = {}, set()
    # Match the first LIMIT calibration records, then remove duplicate/held-out text.
    for qid, row in list(train.items())[:args.limit]:
        text = normalise_question(row["question"])
        if qid not in validation and text not in held and text not in seen:
            selected[qid] = row
            seen.add(text)
    if not selected:
        raise ValueError("No training questions after split/duplicate checks")
    return selected, validation


def prepare(args):
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise ValueError("This launcher requires a visible Hopper GPU (H100/H200, SM90)")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 support is required")
    if transformers.__version__ != "4.51.3":
        raise ValueError("Use the isolated Transformers 4.51.3 H100 environment")
    baseline = read_json(args.baseline / "v9-ensemble.run.json")
    saved = read_json(args.baseline / "v9-ensemble.diagnostics.json")
    train, validation = questions(args)
    if set(saved) != set(validation):
        raise ValueError("Validation IDs differ from the frozen V9 baseline")
    for qid, record in validation.items():
        if (record["question"] != saved[qid]["question"] or
                record["answer"] != saved[qid]["relevant"]):
            raise ValueError(f"Validation content differs from V9: {qid}")
    frozen = Stacker.load(args.baseline / "v9-stacker-ens.json")
    if frozen.version != 1 or frozen.names != feature_names(2):
        raise ValueError("Expected the original 20-feature V9 stacker")
    store = Store(args.index / "store.sqlite")
    try:
        for key in ("corpus_sha256", "chunks", "documents", "chunk_words", "overlap_words"):
            if store.metadata[key] != baseline["store"][key]:
                raise ValueError(f"Store differs from V9: {key}")
        ids = np.fromiter((r[0] for r in store.db.execute("SELECT id FROM chunks ORDER BY id")),
                          dtype=np.int64)
        for name in ("unigram", "bigram"):
            with np.load(args.index / "lexical" / f"{name}.npz", allow_pickle=True) as data:
                if not np.array_equal(ids, data["chunk_ids"]):
                    raise ValueError(f"{name} chunk IDs differ from store")
                expected_docs = np.asarray(store.doc_ids)
                if not np.array_equal(data["doc_ids"], expected_docs):
                    raise ValueError(f"{name} document order differs from store")
    finally:
        store.close()
    for name in MODELS:
        model_ready(args.models / name)
    dense_ready(args.index / "dense", args.index / "store.sqlite", args.models / "embedding")

    print("Hashing code, models and indexes for safe resume (one sequential read).", flush=True)
    files = list(Path(__file__).parent.glob("*.py"))
    files += [args.train, args.validation, args.public]
    files += list(args.baseline.glob("*.json"))
    files += [args.index / "store.sqlite"]
    files += list((args.index / "lexical").glob("*.npz"))
    files += list((args.index / "dense").glob("*.npy"))
    files += list((args.index / "dense").glob("*.json"))
    for name in MODELS:
        files += [p for p in (args.models / name).iterdir() if p.is_file()]
    hashes = {str(p.relative_to(args.root)): file_hash(p) for p in sorted(set(files))}
    packages = {}
    for name in ("torch", "transformers", "numpy", "tokenizers", "sentencepiece", "protobuf",
                 "flash-attn", "causal-conv1d"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    signature = {"schema": SCHEMA, "files": hashes, "packages": packages,
                 "python": platform.python_version(), "cuda": torch.version.cuda,
                 "gpu": torch.cuda.get_device_name(0), "limit": args.limit,
                 "shard_size": args.shard_size, "benchmark_questions": args.benchmark_questions,
                 "flash_available": os.environ.get("H100_FLASH_AVAILABLE", "0"),
                 "baseline_tolerance": args.baseline_tolerance, "train_ids": list(train)}
    guard(args.out / "inputs.json", signature)
    print(f"Preflight passed: {len(train)} calibration / {len(validation)} validation questions.", flush=True)


def pipeline(args, runtime, mode):
    baseline = read_json(args.baseline / "v9-ensemble.run.json")
    settings = dict(baseline["settings"], stacker=None, passage_selection=mode,
                    rerank_batch=runtime["batch"])
    if settings["stage2_candidates"] != settings["candidates"]:
        raise ValueError("Both rerankers must score the full candidate pool")
    os.environ["LEGALIR_CAUSAL_ATTN"] = runtime["attention"]
    model = Pipeline(args.index / "store.sqlite", args.index / "lexical",
                     dense_directory=args.index / "dense", embedding_path=args.models / "embedding",
                     reranker_path=[args.models / n for n in MODELS[1:]], device="cuda",
                     settings=settings, lexical_settings=baseline["lexical"])
    if model.audit["components"] != baseline["audit"]["components"]:
        model.close()
        raise ValueError("Loaded model parameter counts differ from V9")
    return model


def order(row, weights, version):
    matrix = np.asarray(row[f"features{version}"], dtype=float)
    scores = matrix @ np.asarray(weights)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite ranking score")
    return [d for _, d in sorted(zip(-scores, row["docs"]))][:5]


def score_row(model, qid, record, mode):
    channels = {}
    fused, prior = model.candidates(record["question"], channels)
    docs = [d for d, _ in fused[:model.settings["candidates"]]]
    passages = {}
    first = model._rerank(model.rerankers[0], record["question"], docs, prior, passages)
    # Match Pipeline.search's second-reranker batching order, including at depth 50.
    ranked = sorted(first, key=lambda d: (-aggregate_passages(first[d], model.settings["aggregate"],
                                                             model.settings["aggregate_weight"]), d))
    second = model._rerank(model.rerankers[1], record["question"], ranked, prior, passages)
    for table in (first, second):
        if not all(np.isfinite(values).all() for values in table.values()):
            raise ValueError(f"Non-finite reranker scores for {qid}")
    gold = record.get("answer", [])
    row = {"id": qid, "question": record["question"], "gold": gold, "docs": docs,
           "candidate_ranks": {g: next((i for i, (d, _) in enumerate(fused, 1) if d == g), None)
                               for g in gold},
           "features2": build_features(docs, [first, second], channels).tolist()}
    if mode == "retrieval":
        row["features1"] = build_features(docs, [first, second], channels, version=1).tolist()
    return row


def benchmark(args):
    import torch
    inputs = read_json(args.out / "inputs.json")
    marker = digest(inputs)
    target = args.out / "runtime.json"
    if target.exists():
        if read_json(target)["input_digest"] != marker:
            raise ValueError("Benchmark inputs changed")
        print("Reusing the frozen runtime choice.", flush=True)
        return
    train, _ = questions(args)
    sample = list(train.items())[:args.benchmark_questions]
    weights = read_json(args.baseline / "v9-stacker-ens.json")["weights"]
    trials = [{"attention": "sdpa", "batch": 8}, {"attention": "sdpa", "batch": 32}]
    if os.environ.get("H100_FLASH_AVAILABLE") == "1":
        trials += [{"attention": "flash_attention_2", "batch": 16},
                   {"attention": "flash_attention_2", "batch": 32}]
    results, reference = [], None
    for trial in trials:
        print(f"Warmup and benchmark: {trial}", flush=True)
        model = None
        try:
            model = pipeline(args, trial, "retrieval")
            score_row(model, *sample[0], "retrieval")
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = [order(score_row(model, q, r, "retrieval"), weights, 1) for q, r in sample]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if reference is None:
                reference = predictions
            result = dict(trial, seconds=seconds, exact_predictions=predictions == reference)
            results.append(result)
            print(result, flush=True)
        except (RuntimeError, ImportError, ValueError) as error:
            if reference is None:
                raise
            results.append(dict(trial, error=str(error), exact_predictions=False))
            print(f"Optional accelerator rejected: {error}", flush=True)
        finally:
            if model is not None:
                model.close()
    safe = [r for r in results if r["exact_predictions"]]
    fastest = min(safe, key=lambda r: r["seconds"])
    # Small timing differences should not change the backend/batch composition.
    chosen = fastest if fastest["seconds"] < safe[0]["seconds"] * .9 else safe[0]
    write_json(target, {"input_digest": marker, "attention": chosen["attention"],
                        "batch": chosen["batch"], "trials": results})


def validate_rows(rows, items, mode):
    if len(rows) != len(items):
        raise ValueError("Incomplete checkpoint shard")
    for row, (qid, record) in zip(rows, items):
        if (row["id"] != qid or row["question"] != record["question"] or
                row["gold"] != record.get("answer", [])):
            raise ValueError("Checkpoint question differs")
        docs = row["docs"]
        if not docs or len(docs) != len(set(docs)):
            raise ValueError("Invalid candidate documents")
        for version in ((1, 2) if mode == "retrieval" else (2,)):
            matrix = np.asarray(row[f"features{version}"])
            if matrix.shape != (len(docs), 20) or not np.isfinite(matrix).all():
                raise ValueError("Invalid checkpoint features")


def collect(args, records, split, mode, runtime):
    directory = args.out / mode / split
    signature = {"schema": SCHEMA, "input_digest": digest(read_json(args.out / "inputs.json")),
                 "runtime": runtime, "mode": mode, "questions": digest(records),
                 "shard_size": args.shard_size}
    guard(directory / "resume.json", signature)
    items, all_rows, model = list(records.items()), [], None
    try:
        for start in range(0, len(items), args.shard_size):
            part = items[start:start + args.shard_size]
            path = directory / f"part-{start // args.shard_size:05d}.json"
            rows = None
            if path.exists():
                try:
                    rows = read_json(path)
                    validate_rows(rows, part, mode)
                except (ValueError, KeyError, TypeError):
                    print(f"Rescoring incomplete/corrupt shard {path}", flush=True)
                    rows = None
            if rows is None:
                if model is None:
                    model = pipeline(args, runtime, mode)
                tick = time.perf_counter()
                rows = [score_row(model, q, r, mode) for q, r in part]
                validate_rows(rows, part, mode)
                write_json(path, rows)
                print(f"{mode}/{split}: committed {start + len(part)}/{len(items)} "
                      f"({time.perf_counter() - tick:.1f}s/shard)", flush=True)
            all_rows.extend(rows)
    finally:
        if model is not None:
            model.close()
    return all_rows


def diagnostics(rows, weights, version):
    return {r["id"]: {"question": r["question"], "relevant": r["gold"],
                       "candidate_ranks": r["candidate_ranks"],
                       "predicted": order(r, weights, version)} for r in rows}


def recall(diag):
    return sum(len(set(r["relevant"]) & set(r["predicted"])) / len(r["relevant"])
               for r in diag.values()) / len(diag)


def candidate_recall(diag, depth=50):
    return sum(sum(rank is not None and rank <= depth for rank in r["candidate_ranks"].values()) /
               len(r["relevant"]) for r in diag.values()) / len(diag)


def fit_arm(args, rows, name, version):
    target = args.out / f"stacker-{name}.json"
    fitted_rows = [(np.asarray(r[f"features{version}"]),
                    np.asarray([float(d in r["gold"]) for d in r["docs"]])) for r in rows]
    weights, history = fit(fitted_rows)
    write_json(target, {"weights": weights.tolist(), "features": list(feature_names(2)),
                        "feature_version": version, "questions": len(rows), "history": history,
                        "input_digest": digest(read_json(args.out / "inputs.json"))})
    return weights


def choose(comparisons, scores):
    # A positive mean alone is not an automatic promotion after repeated tuning.
    eligible = [name for name in ("ranks", "document")
                if all(comparisons[f"{name}_vs_{base}"]["ci"][0] > 0
                       for base in ("frozen", "control"))]
    return max(eligible, key=lambda name: scores[name]) if eligible else "frozen"


def run(args):
    runtime = read_json(args.out / "runtime.json")
    if runtime["input_digest"] != digest(read_json(args.out / "inputs.json")):
        raise ValueError("Runtime/input mismatch")
    train, validation = questions(args)
    original = read_json(args.baseline / "v9-stacker-ens.json")["weights"]
    # Validate the restored system before spending the calibration GPU pass.
    val_old = collect(args, validation, "validation", "retrieval", runtime)
    frozen = diagnostics(val_old, original, 1)
    saved = read_json(args.baseline / "v9-ensemble.diagnostics.json")
    control_check = {"recall": recall(frozen), "saved_recall": recall(saved),
                     "candidate_recall50": candidate_recall(frozen),
                     "saved_candidate_recall50": candidate_recall(saved),
                     "changed_sets": sum(set(frozen[q]["predicted"]) != set(saved[q]["predicted"])
                                         for q in saved)}
    write_json(args.out / "baseline-check.json", control_check)
    print(f"Baseline replay: {control_check}", flush=True)
    if (abs(control_check["recall"] - control_check["saved_recall"]) > args.baseline_tolerance or
            abs(control_check["candidate_recall50"] - control_check["saved_candidate_recall50"]) > .001):
        raise ValueError("Baseline drift exceeds the gate; inspect baseline-check.json before a new run")
    train_old = collect(args, train, "train", "retrieval", runtime)
    weights = {"frozen": original,
               "control": fit_arm(args, train_old, "control", 1),
               "ranks": fit_arm(args, train_old, "ranks", 2)}
    del train_old
    train_new = collect(args, train, "train", "document", runtime)
    weights["document"] = fit_arm(args, train_new, "document", 2)
    del train_new
    val_new = collect(args, validation, "validation", "document", runtime)
    for before, after in zip(val_old, val_new):
        if before["id"] != after["id"] or before["docs"] != after["docs"]:
            raise ValueError("Passage selection changed the candidate pool")
    scores = {}
    for name in weights:
        rows = val_new if name == "document" else val_old
        version = 1 if name in ("frozen", "control") else 2
        diag = diagnostics(rows, weights[name], version)
        write_json(args.out / f"{name}.diagnostics.json", diag)
        scores[name] = recall(diag)
    comparisons = {}
    for treatment, base in (("control", "frozen"), ("ranks", "control"), ("ranks", "frozen"),
                            ("document", "ranks"), ("document", "control"), ("document", "frozen")):
        comparisons[f"{treatment}_vs_{base}"] = compare(
            args.out / f"{base}.diagnostics.json", args.out / f"{treatment}.diagnostics.json",
            label=f"{treatment} vs {base}")
    winner = choose(comparisons, scores)
    write_json(args.out / "comparison.json", {"recall": scores, "comparisons": comparisons})
    write_json(args.out / "selection.json", {"selected": winner,
               "rule": "Positive paired 95% lower bound versus both frozen and refit control; otherwise keep V9.",
               "exploratory": True, "calibration_may_overlap_reranker_training": True,
               "input_digest": digest(read_json(args.out / "inputs.json")),
               "runtime_digest": digest(runtime),
               "comparison_sha256": file_hash(args.out / "comparison.json"),
               "stackers": {name: file_hash(args.out / f"stacker-{name}.json")
                            for name in ("control", "ranks", "document")}})
    print(f"Validation finished; selected {winner}. Ready for stage submit.", flush=True)


def submit(args):
    selection = read_json(args.out / "selection.json")
    runtime = read_json(args.out / "runtime.json")
    if (selection["input_digest"] != digest(read_json(args.out / "inputs.json")) or
            selection["runtime_digest"] != digest(runtime) or
            runtime["input_digest"] != selection["input_digest"] or
            selection["comparison_sha256"] != file_hash(args.out / "comparison.json")):
        raise ValueError("Submission inputs or selection changed; rerun validation with a new RUN_ID")
    for name, expected in selection["stackers"].items():
        if file_hash(args.out / f"stacker-{name}.json") != expected:
            raise ValueError(f"Selected experiment weights changed: {name}")
    comparison = read_json(args.out / "comparison.json")
    winner = choose(comparison["comparisons"], comparison["recall"])
    if winner != selection["selected"]:
        raise ValueError("Selection does not match the validation promotion rule")
    if winner == "frozen":
        predictions = read_json(args.baseline / "submission-v9.json")
        print("No treatment cleared the gate. Keeping the original V9 public answers.", flush=True)
    else:
        public = load_questions(args.public)
        mode = "document" if winner == "document" else "retrieval"
        rows = collect(args, public, "public", mode, runtime)
        weights = Stacker.load(args.out / f"stacker-{winner}.json")
        if weights.version != 2:
            raise ValueError("Treatment requires version 2 stacker features")
        predictions = {r["id"]: {"answer": order(r, weights.weights, 2)} for r in rows}
    write_json(args.out / "submission.json", predictions)
    store = Store(args.index / "store.sqlite")
    temporary = args.out / "submission.zip.tmp"
    try:
        pack(args.out / "submission.json", store, args.public, temporary)
        temporary.replace(args.out / "submission.zip")
    finally:
        store.close()
    print(f"Finished: {winner}; {args.out / 'submission.zip'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "benchmark", "run", "submit"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("artifacts-ranking-h100"))
    parser.add_argument("--baseline", type=Path, default=Path("analysis/ranking-h100/baseline"))
    parser.add_argument("--models", type=Path, default=Path("models2"))
    parser.add_argument("--train", type=Path, default=Path("artifacts/splits/train.json"))
    parser.add_argument("--validation", type=Path, default=Path("artifacts/splits/validation.json"))
    parser.add_argument("--public", type=Path, default=Path("public-official.json"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--benchmark-questions", type=int, default=6)
    parser.add_argument("--baseline-tolerance", type=float, default=.002)
    args = parser.parse_args()
    args.root = Path.cwd().resolve()
    for name in ("out", "index", "baseline", "models", "train", "validation", "public"):
        setattr(args, name, getattr(args, name).resolve())
    if min(args.limit, args.shard_size, args.benchmark_questions) < 1 or args.baseline_tolerance < 0:
        parser.error("Limits must be positive and baseline tolerance nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "benchmark": benchmark, "run": run, "submit": submit}[args.stage](args)


if __name__ == "__main__":
    main()
