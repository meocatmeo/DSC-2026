"""Command line entry point for the LegalIR v2 pipeline.

    python -m legalir2 <command> [options]

Commands, in the order a full run uses them:

    build-store     parse the corpus into article level chunks
    index-lexical   build the BM25F and bigram posting files
    download        fetch approved checkpoints and record provenance
    verify-models   re-check stored file hashes
    probe           screen candidate checkpoints in minutes, before indexing
    index-dense     embed every chunk with the bi-encoder
    mine            mine chunk positives and tiered hard negatives from train labels
    teacher         cache a larger reranker's scores for distillation (offline only)
    train-bi        fine tune the bi-encoder
    train-ce        fine tune the cross encoder reranker
    predict         run retrieval over a question file
    tune            sweep lexical / fusion settings (on training questions)
    audit           report the deployed parameter count
    validate        check a prediction file against the submission rules
    pack            write submission.zip
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "artifacts2" / "store.sqlite"
DEFAULT_LEXICAL = ROOT / "artifacts2" / "lexical"
DEFAULT_DENSE = ROOT / "artifacts2" / "dense"
DEFAULT_MODELS = ROOT / "models2"

APPROVED = {
    "embedding": "AITeamVN/Vietnamese_Embedding_v2",
    "reranker": "AITeamVN/Vietnamese_Reranker",
    "teacher": "Qwen/Qwen3-Reranker-4B",
}


def paths(args):
    models = args.reranker_model
    models = [models] if isinstance(models, str) else list(models)
    return {"store": Path(args.store), "lexical": Path(args.lexical),
            "dense": Path(args.dense), "embedding": Path(args.embedding_model),
            "rerankers": [Path(m) for m in models]}


def add_common(parser):
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--lexical", default=str(DEFAULT_LEXICAL))
    parser.add_argument("--dense", default=str(DEFAULT_DENSE))
    parser.add_argument("--embedding-model", default=str(DEFAULT_MODELS / "embedding"))
    parser.add_argument("--reranker-model", nargs="+", default=[str(DEFAULT_MODELS / "reranker")],
                        help="one checkpoint, or several to ensemble; with several the "
                             "first is the wide pass and the rest rescore its best "
                             "--stage2-candidates documents")
    parser.add_argument("--device", default="cuda")


def pipeline_from(args, use_dense=True, use_reranker=True, settings=None):
    from .pipeline import Pipeline
    p = paths(args)
    dense = str(p["dense"]) if use_dense and (p["dense"] / "manifest.json").is_file() else None
    reranker = [str(m) for m in p["rerankers"] if m.is_dir()] if use_reranker else []
    missing = [str(m) for m in p["rerankers"] if use_reranker and not m.is_dir()]
    if missing:
        print(f"note: reranker checkpoint(s) not found: {', '.join(missing)}", file=sys.stderr)
    if use_dense and not dense:
        print("note: no dense index found, running lexical only", file=sys.stderr)
    if use_reranker and not reranker:
        print("note: no reranker checkpoint found, returning fusion order", file=sys.stderr)
    lexical_settings = json.loads(args.lexical_settings) if getattr(args, "lexical_settings", None) else None
    extra = json.loads(args.settings) if getattr(args, "settings", None) else None
    return Pipeline(str(p["store"]), str(p["lexical"]), dense, str(p["embedding"]),
                    reranker, device=args.device,
                    settings=dict(settings or {}, **(extra or {})),
                    lexical_settings=lexical_settings)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="legalir2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-store")
    p.add_argument("--corpus", default=str(ROOT / "selected-contexts"))
    p.add_argument("--store", default=str(DEFAULT_STORE))
    p.add_argument("--chunk-words", type=int, default=320)
    p.add_argument("--overlap-words", type=int, default=64)

    p = sub.add_parser("index-lexical")
    p.add_argument("--store", default=str(DEFAULT_STORE))
    p.add_argument("--lexical", default=str(DEFAULT_LEXICAL))
    p.add_argument("--no-bigram", action="store_true")

    p = sub.add_parser("download")
    p.add_argument("--which", nargs="*", default=["embedding", "reranker"],
                   choices=sorted(APPROVED))
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--revision", default="main")
    p.add_argument("--repo", default=None, help="override the repository id")

    p = sub.add_parser("verify-models")
    p.add_argument("--models", default=str(DEFAULT_MODELS))

    p = sub.add_parser("index-dense")
    add_common(p)
    p.add_argument("--batch-size", type=int, default=16)
    # 320 whitespace words plus the label is comfortably under 768 XLM-R tokens.
    p.add_argument("--max-length", type=int, default=768)

    p = sub.add_parser("probe")
    add_common(p)
    p.add_argument("--model", required=True, help="checkpoint directory to screen")
    p.add_argument("--mode", choices=("embedding", "reranker"), default="embedding")
    p.add_argument("--questions", default=str(ROOT / "artifacts" / "splits" / "train.json"))
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--distractors", type=int, default=23)
    p.add_argument("--min-distractors", type=int, default=4,
                   help="skip a question with fewer distractors than this")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--output", default=None)

    p = sub.add_parser("mine")
    add_common(p)
    p.add_argument("--questions", default=str(ROOT / "artifacts" / "splits" / "train.json"))
    p.add_argument("--output", default=str(ROOT / "artifacts2" / "groups" / "train.jsonl"))
    p.add_argument("--negatives", type=int, default=12)
    p.add_argument("--positives", type=int, default=2)
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lexical-settings", default=None)

    p = sub.add_parser("teacher")
    add_common(p)
    p.add_argument("--groups", default=str(ROOT / "artifacts2" / "groups" / "train.jsonl"))
    p.add_argument("--teacher-model", default=str(DEFAULT_MODELS / "teacher"))
    p.add_argument("--output", default=str(ROOT / "artifacts2" / "groups" / "teacher.jsonl"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024)

    p = sub.add_parser("train-bi")
    add_common(p)
    p.add_argument("--groups", default=str(ROOT / "artifacts2" / "groups" / "train.jsonl"))
    p.add_argument("--output", default=str(DEFAULT_MODELS / "embedding-tuned"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--negatives", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--passage-length", type=int, default=512)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--max-groups", type=int, default=None)

    p = sub.add_parser("train-ce")
    add_common(p)
    p.add_argument("--groups", default=str(ROOT / "artifacts2" / "groups" / "train.jsonl"))
    p.add_argument("--output", default=str(DEFAULT_MODELS / "reranker-tuned"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--groups-per-step", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--teacher-scores", default=None)
    p.add_argument("--distill-weight", type=float, default=0.4)
    p.add_argument("--max-groups", type=int, default=None)

    p = sub.add_parser("predict")
    add_common(p)
    p.add_argument("--questions", default=str(ROOT / "artifacts" / "splits" / "validation.json"))
    p.add_argument("--output", default=str(ROOT / "outputs" / "v2-validation.json"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--settings", default=None, help="JSON overrides for pipeline settings")
    p.add_argument("--lexical-settings", default=None, help="JSON overrides for lexical settings")

    p = sub.add_parser("tune")
    add_common(p)
    # Tuning uses training questions so the validation split stays an unbiased check.
    p.add_argument("--questions", default=str(ROOT / "artifacts" / "splits" / "train.json"))
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--output", default=str(ROOT / "outputs" / "v2-tuning.json"))

    p = sub.add_parser("audit")
    add_common(p)

    p = sub.add_parser("validate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--questions", default=str(ROOT / "public-official.json"))

    p = sub.add_parser("pack")
    p.add_argument("--predictions", required=True)
    p.add_argument("--questions", default=str(ROOT / "public-official.json"))
    p.add_argument("--store", default=str(DEFAULT_STORE))
    p.add_argument("--output", default=str(ROOT / "outputs" / "submission.zip"))

    args = parser.parse_args(argv)
    return dispatch(args)


def dispatch(args):
    if args.command == "build-store":
        from .store import build
        build(args.corpus, args.store, args.chunk_words, args.overlap_words)
        return 0

    if args.command == "index-lexical":
        from .lexical import build
        print(build(args.store, args.lexical, bigram=not args.no_bigram))
        return 0

    if args.command == "download":
        from .download import download
        for which in args.which:
            repo = args.repo or APPROVED[which]
            target = Path(args.models) / which
            print(f"Downloading {repo} to {target}")
            record = download(repo, target, args.revision)
            print(f"  commit {record['commit']}, {len(record['files'])} files")
        return 0

    if args.command == "verify-models":
        from .download import verify
        for directory in sorted(Path(args.models).iterdir()):
            if (directory / "provenance.json").is_file():
                record = verify(directory)
                print(f"{directory.name}: {record['repository']} @ {record['commit'][:12]} ok")
        return 0

    if args.command == "index-dense":
        from .dense import build
        print(build(args.store, args.embedding_model, args.dense, args.device,
                    args.batch_size, args.max_length))
        return 0

    if args.command == "probe":
        from .probe import run
        run(args.store, args.lexical, args.questions, args.model, mode=args.mode,
            device=args.device, limit=args.limit, distractors=args.distractors,
            batch_size=args.batch_size, max_length=args.max_length, output=args.output,
            min_distractors=args.min_distractors)
        return 0

    if args.command == "mine":
        from .mine import mine
        dense = None if args.no_dense else (
            args.dense if (Path(args.dense) / "manifest.json").is_file() else None)
        mine(args.store, args.lexical, args.questions, args.output,
             dense_directory=dense, embedding_path=args.embedding_model, device=args.device,
             positives=args.positives, negatives=args.negatives, limit=args.limit,
             lexical_settings=json.loads(args.lexical_settings) if args.lexical_settings else None)
        return 0

    if args.command == "teacher":
        from .mine import teacher_scores
        teacher_scores(args.groups, args.store, args.teacher_model, args.output,
                       device=args.device, batch_size=args.batch_size,
                       max_length=args.max_length)
        return 0

    if args.command == "train-bi":
        from .train_bi import train
        train(args.store, args.groups, args.embedding_model, args.output, device=args.device,
              epochs=args.epochs, batch_size=args.batch_size, negatives=args.negatives,
              learning_rate=args.learning_rate, passage_length=args.passage_length,
              accumulate=args.accumulate, max_groups=args.max_groups)
        return 0

    if args.command == "train-ce":
        from .train_ce import train
        base = args.reranker_model[0] if isinstance(args.reranker_model, list) else args.reranker_model
        train(args.store, args.groups, base, args.output, device=args.device,
              epochs=args.epochs, group_size=args.group_size,
              groups_per_step=args.groups_per_step, learning_rate=args.learning_rate,
              max_length=args.max_length, accumulate=args.accumulate,
              teacher_path=args.teacher_scores, distill_weight=args.distill_weight,
              max_groups=args.max_groups)
        return 0

    if args.command == "predict":
        from .predict import run
        pipeline = pipeline_from(args, use_dense=not args.no_dense,
                                 use_reranker=not args.no_rerank)
        try:
            run(pipeline, args.questions, args.output, limit=args.limit,
                evaluate=args.evaluate)
        finally:
            pipeline.close()
        return 0

    if args.command == "tune":
        from .tune import sweep
        sweep(args, limit=args.limit, output=args.output)
        return 0

    if args.command == "audit":
        pipeline = pipeline_from(args)
        try:
            print(json.dumps(pipeline.audit, indent=1))
        finally:
            pipeline.close()
        return 0

    if args.command == "validate":
        from .predict import validate
        print(validate(args.predictions, args.questions))
        return 0

    if args.command == "pack":
        from .predict import pack
        from .store import Store
        store = Store(args.store)
        try:
            pack(args.predictions, store, args.questions, args.output)
        finally:
            store.close()
        return 0

    raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
