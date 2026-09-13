"""Command-line entry point: py -m legalir --help."""

import argparse
import json
import platform
import time
import urllib.request
from pathlib import Path

from .data import connect, file_hash, load_questions, prepare, read_json, split_questions, write_json
from .submission import evaluate, pack, validate

ROOT = Path(__file__).resolve().parents[1]


def download_models(directory):
    from .neural import EMBEDDING, RERANKER

    results = {}
    for role, repo in (("embedding", EMBEDDING), ("reranker", RERANKER)):
        target = Path(directory) / role
        target.mkdir(parents=True, exist_ok=True)
        # Only static artifact downloads, never hosted inference or external data.
        request = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/config.json")
        with urllib.request.urlopen(request, timeout=120) as response:
            revision = response.headers.get("x-repo-commit")
        if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise RuntimeError("Could not resolve an immutable model revision")
        previous = read_json(target / "provenance.json") if (target / "provenance.json").exists() else {}
        hashes = {}
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "README.md"):
            path = target / name
            if previous.get("revision") == revision and path.exists() and previous.get("sha256", {}).get(name) == file_hash(path):
                hashes[name] = previous["sha256"][name]
                continue
            print(f"Downloading {repo}@{revision}: {name}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=120) as response:
                with temporary.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            temporary.replace(path)
            hashes[name] = file_hash(path)
        provenance = {"repo": repo, "revision": revision, "license": "apache-2.0", "sha256": hashes}
        write_json(target / "provenance.json", provenance)
        results[role] = provenance
    return results


def document_ids(database):
    db = connect(database)
    try:
        return {r[0] for r in db.execute("SELECT id FROM documents")}
    finally:
        db.close()


def predict(args):
    from .retrieval import Retriever

    questions = load_questions(args.questions)
    if args.limit:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        questions = dict(list(questions.items())[:args.limit])
    if args.mode == "bm25" and args.rerank:
        print("Using BM25 candidate retrieval plus neural reranking")
    started = time.perf_counter()
    engine = Retriever(args.database, dense=args.dense if args.mode in ("hybrid", "dense") else None,
                       embedding=args.models / "embedding",
                       reranker=args.models / "reranker" if args.rerank else None,
                       device=args.device, candidates=args.candidates, chunk_candidates=args.chunk_candidates,
                       passages=args.passages, batch_size=args.batch_size, legal_weight=args.legal_weight,
                       dense_only=args.mode == "dense", context_radius=args.context_radius,
                       reranker_max_length=args.reranker_max_length)
    predictions = {}
    diagnostics = {}
    try:
        for index, (key, record) in enumerate(questions.items(), 1):
            predictions[key] = {"answer": engine.search(record["question"], args.top_k)}
            if args.evaluate:
                answers = record.get("answer")
                if not isinstance(answers, list) or not answers:
                    raise ValueError("Evaluation requires nonempty ground-truth labels")
                ranks = {doc: i for i, doc in enumerate(engine.last_candidates, 1)}
                diagnostics[key] = {"question": record["question"], "relevant": answers,
                                    "candidate_ranks": {doc: ranks.get(doc) for doc in answers},
                                    "predicted": predictions[key]["answer"]}
            if index % 25 == 0:
                print(f"Predicted {index}/{len(questions)} questions", flush=True)
        validate(predictions, questions, set(engine.documents))
        write_json(args.output, predictions)
        report = {**engine.manifest, "questions_sha256": file_hash(args.questions),
                  "prediction_sha256": file_hash(args.output), "top_k": args.top_k,
                  "limit": args.limit, "seconds": time.perf_counter() - started,
                  "python": platform.python_version(), "platform": platform.platform()}
        if args.evaluate:
            metrics = evaluate(predictions, questions)
            write_json(args.output.with_suffix(".metrics.json"), metrics)
            report["metrics"] = {k: v for k, v in metrics.items() if k != "per_question"}
            report["candidate_recall"] = sum(
                sum(rank is not None for rank in row["candidate_ranks"].values()) / len(row["relevant"])
                for row in diagnostics.values()) / len(diagnostics)
            write_json(args.output.with_suffix(".diagnostics.json"), diagnostics)
        write_json(args.output.with_suffix(".run.json"), report)
        return report
    finally:
        engine.close()


def parser():
    cli = argparse.ArgumentParser(description="UIT DSC Task 1: local legal document retrieval")
    commands = cli.add_subparsers(dest="command", required=True)
    prepare_cli = commands.add_parser("prepare", help="Build chunked SQLite BM25 index")
    prepare_cli.add_argument("--corpus", type=Path, default=ROOT / "selected-contexts")
    prepare_cli.add_argument("--database", type=Path, default=ROOT / "artifacts/corpus.sqlite")
    prepare_cli.add_argument("--chunk-words", type=int, default=180)
    prepare_cli.add_argument("--overlap-words", type=int, default=40)
    split_cli = commands.add_parser("split", help="Create reproducible grouped training holdout")
    split_cli.add_argument("--source", type=Path, default=ROOT / "train.json")
    split_cli.add_argument("--output", type=Path, default=ROOT / "artifacts/splits")
    split_cli.add_argument("--fraction", type=float, default=0.15)
    split_cli.add_argument("--seed", type=int, default=2026)
    download_cli = commands.add_parser("download", help="Download model artifacts, pin revisions and hash files")
    download_cli.add_argument("--models", type=Path, default=ROOT / "models")
    index_cli = commands.add_parser("index-dense", help="Embed corpus chunks on local hardware")
    index_cli.add_argument("--database", type=Path, default=ROOT / "artifacts/corpus.sqlite")
    index_cli.add_argument("--models", type=Path, default=ROOT / "models")
    index_cli.add_argument("--output", type=Path, default=ROOT / "artifacts/dense")
    index_cli.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    index_cli.add_argument("--batch-size", type=int, default=8)
    index_cli.add_argument("--max-length", type=int, default=1024)
    predict_cli = commands.add_parser("predict", help="Predict in relevance order; optionally score labeled input")
    predict_cli.add_argument("--questions", type=Path, required=True)
    predict_cli.add_argument("--output", type=Path, required=True)
    predict_cli.add_argument("--database", type=Path, default=ROOT / "artifacts/corpus.sqlite")
    predict_cli.add_argument("--mode", choices=["bm25", "hybrid", "dense"], default="bm25")
    predict_cli.add_argument("--dense", type=Path, default=ROOT / "artifacts/dense")
    predict_cli.add_argument("--models", type=Path, default=ROOT / "models")
    predict_cli.add_argument("--rerank", action="store_true")
    predict_cli.add_argument("--legal-weight", type=float, default=0.0,
                             help="Experimental extractive phrase evidence weight, 0..1; default preserves baseline")
    predict_cli.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    predict_cli.add_argument("--candidates", type=int, default=60)
    predict_cli.add_argument("--chunk-candidates", type=int, default=600)
    predict_cli.add_argument("--passages", type=int, default=2)
    predict_cli.add_argument("--context-radius", type=int, choices=[0, 1, 2], default=0,
                             help="Add same-document neighboring chunks around each reranker anchor")
    predict_cli.add_argument("--reranker-max-length", type=int, default=1536)
    predict_cli.add_argument("--batch-size", type=int, default=4)
    predict_cli.add_argument("--top-k", type=int, choices=range(1, 6), default=5)
    predict_cli.add_argument("--limit", type=int, help="Smoke test only; incomplete output cannot be submitted")
    predict_cli.add_argument("--evaluate", action="store_true")
    for name in ("validate", "pack"):
        sub = commands.add_parser(name)
        sub.add_argument("--predictions", type=Path, required=True)
        sub.add_argument("--questions", type=Path, default=ROOT / "public-official.json")
        sub.add_argument("--database", type=Path, default=ROOT / "artifacts/corpus.sqlite")
        if name == "pack":
            sub.add_argument("--output", type=Path, default=ROOT / "outputs/submission.zip")
    score_cli = commands.add_parser("evaluate")
    score_cli.add_argument("--predictions", type=Path, required=True)
    score_cli.add_argument("--truth", type=Path, required=True)
    score_cli.add_argument("--output", type=Path, required=True)
    audit_cli = commands.add_parser("audit-models", help="Verify hashes and count all parameters in both models")
    audit_cli.add_argument("--models", type=Path, default=ROOT / "models")
    audit_cli.add_argument("--output", type=Path, default=ROOT / "artifacts/model-audit.json")
    return cli


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare(args.corpus, args.database, args.chunk_words, args.overlap_words)
    elif args.command == "split":
        result = split_questions(args.source, args.output, args.fraction, args.seed)
    elif args.command == "download":
        result = download_models(args.models)
    elif args.command == "index-dense":
        from .retrieval import build_dense
        if args.batch_size < 1 or args.max_length < 32:
            raise ValueError("Invalid embedding batch size or sequence length")
        result = build_dense(args.database, args.models / "embedding", args.output,
                             args.device, args.batch_size, args.max_length)
    elif args.command == "predict":
        result = predict(args)
    elif args.command in ("validate", "pack"):
        questions = load_questions(args.questions)
        ids = document_ids(args.database)
        result = (pack(args.predictions, args.output, questions, ids) if args.command == "pack"
                  else validate(read_json(args.predictions), questions, ids))
    elif args.command == "evaluate":
        result = evaluate(read_json(args.predictions), load_questions(args.truth, labeled=True))
        write_json(args.output, result)
        result = {k: v for k, v in result.items() if k != "per_question"}
    else:
        from .neural import LocalModel
        result = {"models": {}, "total_parameters": 0}
        for role in ("embedding", "reranker"):
            path = args.models / role
            provenance = read_json(path / "provenance.json")
            for name, expected in provenance["sha256"].items():
                if file_hash(path / name) != expected:
                    raise ValueError(f"Model artifact changed: {path / name}")
            model = LocalModel(path, causal=role == "reranker", device="cpu")
            result["models"][role] = {**provenance, "parameters": model.parameters}
            result["total_parameters"] += model.parameters
            model.close()
        if result["total_parameters"] >= 4_000_000_000:
            raise ValueError("System violates parameter budget")
        result["strictly_under_4b"] = True
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()