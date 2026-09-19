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
    # 2B, yes/no-logit cross encoder, on the approved sheet. Deployed budget with the
    # 0.6B embedder is 2.6B of 4B, or 3.2B ensembled with reranker-g32.
    "reranker2b": "infgrad/Prism-Qwen3.5-Reranker-2B",
    # Ensemble partners. The shipped reranker is XLM-R lineage (AITeamVN fine-tunes
    # bge-reranker-v2-m3), so a second XLM-R model would fail on the same documents and
    # add nothing. These are deliberately from other families: a causal decoder, and two
    # PhoBERT-lineage Vietnamese rerankers with a different tokenizer entirely. Error
    # decorrelation is the whole point - capacity on its own measured at -0.0038.
    "partner_qwen": "Qwen/Qwen3-Reranker-0.6B",
    "partner_pho": "itdainb/PhoRanker",
    "partner_viranker": "namdp-ptit/ViRanker",
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
    p.add_argument("--corpus", default=str(ROOT / "data" / "selected-contexts"))
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
    p.add_argument("--depth", type=int, default=100,
                   help="candidate depth used to mine negatives; set this to the "
                        "inference rerank depth when training a cutoff-aware loss")
    p.add_argument("--channel", choices=("fused", "dense", "lexical"), default="fused",
                   help="which ranking to draw hard negatives from. 'fused' for a cross "
                        "encoder, which reranks the fused shortlist. 'dense' for the "
                        "bi-encoder, which at inference produces the dense ranking over "
                        "the whole corpus on its own - training it against the fusion's "
                        "hard cases is the mismatch that cost 0.021 on the reranker")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fold", type=int, default=None,
                   help="mine only this fold's questions, so a model trained on it can "
                        "score the other fold without having seen those questions")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--lexical-settings", default=None)
    p.add_argument("--settings", default=None,
                   help="fusion settings JSON, the same string passed to predict. Mining "
                        "and prediction must agree or the reranker trains on a candidate "
                        "distribution it never sees at inference")

    p = sub.add_parser("cache", help="freeze candidates and reranker scores for offline tuning")
    add_common(p)
    p.add_argument("--questions", default=str(ROOT / "artifacts" / "splits" / "train.json"))
    p.add_argument("--output", required=True)
    p.add_argument("--depth", type=int, default=200)
    p.add_argument("--rerank-depth", type=int, default=100)
    p.add_argument("--cache-passages", type=int, default=3)
    p.add_argument("--fold", type=int, default=None,
                   help="score only questions in this fold. Pair with a reranker trained "
                        "on the OTHER fold, or the cache measures memorisation")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--settings", default=None)

    p = sub.add_parser("cv", help="evaluate or fit against frozen caches - no GPU")
    p.add_argument("--cache", nargs="+", required=True)
    p.add_argument("--settings", default=None)
    p.add_argument("--stacker", default=None)
    p.add_argument("--fit-stacker", default=None,
                   help="fit the learned fusion from these caches and write it here. "
                        "The reported score is OUT OF FOLD; the written weights are "
                        "fitted on everything, because more data is better for the "
                        "artefact that ships and worthless for judging it")
    p.add_argument("--stacker-folds", type=int, default=5,
                   help="folds used to score --fit-stacker out of sample")
    p.add_argument("--questions", default=None,
                   help="the labelled question file, so folds can be keyed on question "
                        "TEXT. Without it, the same question under two ids lands in two "
                        "folds and leaks across the split")
    p.add_argument("--against", default=None,
                   help="second settings JSON; reports a PAIRED comparison against it")
    p.add_argument("--against-stacker", default=None)
    p.add_argument("--label", default="")
    p.add_argument("--search", action="store_true",
                   help="coordinate ascent over the cache, adopting a setting only when "
                        "its paired interval excludes zero")
    p.add_argument("--write-settings", default=None,
                   help="where to write the chosen settings")

    p = sub.add_parser("retrieval", help="candidate recall per channel - judge the "
                                         "candidate generator on its own terms")
    add_common(p)
    p.add_argument("--questions", default=str(ROOT / "artifacts" / "splits" / "validation.json"))
    p.add_argument("--output", required=True)
    p.add_argument("--settings", default=None)
    p.add_argument("--lexical-settings", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--label", default="")
    p.add_argument("--against", default=None,
                   help="a previous retrieval report; prints a PAIRED comparison")
    p.add_argument("--channel", default="dense", choices=("lexical", "dense", "fused"))

    p = sub.add_parser("compare", help="paired comparison of two runs' diagnostics")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--subset", choices=("all", "single", "multi"), default="all")

    p = sub.add_parser("stack", help="fit the learned rank fusion")
    add_common(p)
    p.add_argument("--questions", default=str(ROOT / "artifacts" / "splits" / "train.json"),
                   help="TRAINING questions. Fitting on validation would tune the "
                        "combination on the same questions used to judge it")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--depth", type=int, default=None,
                   help="candidates per question; defaults to the pipeline's setting")
    p.add_argument("--cache", default=str(ROOT / "artifacts2" / "groups" / "stack.jsonl"))
    p.add_argument("--output", default=str(ROOT / "outputs" / "v4-stacker.json"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--settings", default=None)

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
    p.add_argument("--teacher-scores", default=None,
                   help="teacher score cache from the `teacher` command. Adds a KL term "
                        "against the teacher's distribution over each group")
    p.add_argument("--distill-weight", type=float, default=0.4)
    p.add_argument("--lora-rank", type=int, default=0,
                   help="0 trains every weight. >0 attaches LoRA adapters and merges "
                        "them on save, so the deployed checkpoint is ordinary. Required "
                        "for a 2B model on anything under ~40 GB")
    p.add_argument("--lora-alpha", type=int, default=None, help="defaults to 2x rank")
    p.add_argument("--max-positives", type=int, default=1,
                   help="gold documents treated as positives in one group. 1 reproduces "
                        "the single-positive objective exactly; >1 splits the target mass "
                        "across a multi-gold question's documents, which is what the "
                        "partial-credit scorer actually rewards")
    p.add_argument("--document-loss", action="store_true",
                   help="train the sequence reranker at document level: aggregate all "
                        "mined passages for each gold document and weight pairwise "
                        "updates by the Recall@5 cutoff. Use with the shipped 0.6B "
                        "AITeamVN reranker; incompatible with teacher scores.")
    p.add_argument("--distill-temperature", type=float, default=1.0,
                   help="softens both distributions before the KL term; >1 transfers "
                        "more of the teacher's ordering among the negatives")
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--no-checkpointing", action="store_true",
                   help="trade VRAM for speed. Gradient checkpointing recomputes each "
                        "layer's forward during the backward pass, so a step costs about "
                        "four forwards instead of three - roughly 25%% of wall clock. "
                        "With LoRA the optimiser state is tiny, so on a large card the "
                        "memory it saves is memory you were not going to run out of. "
                        "If it OOMs, drop the flag rather than the group size.")

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
    p.add_argument("--stage", choices=("lexical", "fusion", "rerank"), default="lexical",
                   help="lexical: BM25F and bigram settings (no GPU). "
                        "fusion: how the lexical and dense channels combine (needs a dense index)")
    p.add_argument("--target-depth", type=int, default=100,
                   help="fusion stage: the candidate depth to maximise recall at")
    p.add_argument("--settings", default=None, help="JSON overrides for pipeline settings")
    p.add_argument("--context-radius", type=int, nargs="+", default=[0, 1],
                   help="rerank stage: context radii to try. Each value needs its own "
                        "scoring pass because it changes the passage text, so this "
                        "multiplies the sweep's GPU cost by how many you list")
    p.add_argument("--cache-passages", type=int, default=4,
                   help="rerank stage: passages per document to score and cache. The "
                        "sweep cannot test a passages setting above this")
    p.add_argument("--output", default=str(ROOT / "outputs" / "v2-tuning.json"))

    p = sub.add_parser("audit")
    add_common(p)

    p = sub.add_parser("validate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--questions", default=str(ROOT / "data" / "public-official.json"))

    p = sub.add_parser("pack")
    p.add_argument("--predictions", required=True)
    p.add_argument("--questions", default=str(ROOT / "data" / "public-official.json"))
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
                if record.get("derived"):
                    base = record.get("derived_from", {})
                    print(f"{directory.name}: fine tuned from "
                          f"{base.get('repository', base.get('path', 'unknown'))} "
                          f"(our weights, nothing to hash-verify)")
                else:
                    print(f"{directory.name}: {record['repository']} @ "
                          f"{record['commit'][:12]} ok")
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
        questions = args.questions
        if args.fold is not None:
            from .cache import folds
            from .predict import load_questions
            from .store import write_json as _write
            records = load_questions(questions, labeled=True)
            assignment = folds(list(records), args.folds, texts=records)
            kept = {q: v for q, v in records.items() if assignment[q] == args.fold}
            questions = f"{Path(args.output).with_suffix('')}.questions.json"
            _write(questions, kept)
            print(f"  fold {args.fold} of {args.folds}: {len(kept)} of {len(records)} questions")
        dense = None if args.no_dense else (
            args.dense if (Path(args.dense) / "manifest.json").is_file() else None)
        mine(args.store, args.lexical, questions, args.output,
             dense_directory=dense, embedding_path=args.embedding_model, device=args.device,
             positives=args.positives, negatives=args.negatives, limit=args.limit,
             depth=args.depth, channel=args.channel,
             settings=json.loads(args.settings) if args.settings else None,
             lexical_settings=json.loads(args.lexical_settings) if args.lexical_settings else None)
        return 0

    if args.command == "cache":
        from .cache import build, folds
        from .predict import load_questions
        questions = load_questions(args.questions, labeled=True)
        if args.fold is not None:
            assignment = folds(list(questions), args.folds, texts=questions)
            questions = {q: v for q, v in questions.items() if assignment[q] == args.fold}
            print(f"  fold {args.fold} of {args.folds}: {len(questions)} questions")
        pipeline = pipeline_from(args)
        try:
            build(pipeline, questions, args.output, depth=args.depth,
                  rerank_depth=args.rerank_depth, passages=args.cache_passages)
        finally:
            pipeline.close()
        return 0

    if args.command == "cv":
        from .cache import Cache, summarise
        from .compare import bootstrap, sign_test
        from .stack import Stacker, fit
        from .store import write_json
        caches = [Cache(c) for c in args.cache]
        sizes, total = {}, 0
        for c in caches:
            for row, qid in c.rows():
                sizes[qid] = int((c.arrays["gold"][row] >= 0).sum())
            total += len(c)
        print(f"  {total} questions across {len(caches)} cache(s)")

        if args.fit_stacker:
            from .cache import duplicate_groups, folds
            from .stack import FEATURES, FEATURE_VERSION, Stacker
            rows = [r for c in caches for r in c.features()]
            identifiers = [qid for qid, _, _ in rows]

            texts = None
            if args.questions:
                from .predict import load_questions
                texts = load_questions(args.questions, labeled=True)
                shared = duplicate_groups(texts)
                copies = sum(len(v) for v in shared.values())
                print(f"  {len(shared)} question texts appear more than once "
                      f"({copies} ids); every copy is pinned to one fold")
            else:
                print("  no --questions given, so folds are keyed on question id.")
                print("  Duplicate question texts under different ids will straddle the")
                print("  split and the out-of-fold number below will be optimistic.")

            assignment = folds(identifiers, args.stacker_folds, texts=texts)
            base_settings = json.loads(args.settings) if args.settings else {}

            # The honest number: every question scored by a stacker fitted without it.
            print(f"  fitting {args.stacker_folds} fold stackers over {len(rows)} questions")
            held_out = {}
            for k in range(args.stacker_folds):
                inside = [(m, y) for qid, m, y in rows if assignment[qid] == k]
                outside = [(m, y) for qid, m, y in rows if assignment[qid] != k]
                if not inside or not outside:
                    continue
                weights, _ = fit(outside, verbose=False)
                subset = {qid for qid in identifiers if assignment[qid] == k}
                for c in caches:
                    held_out.update(c.evaluate(base_settings, stacker=Stacker(weights),
                                               subset=subset))
                print(f"    fold {k}: fitted on {len(outside)}, scored {len(subset)}")
            summarise(held_out, sizes, "learned fusion, OUT OF FOLD")

            # The artefact that ships is fitted on everything - more data is better for
            # the deployed weights. Its in-sample score is not evidence, so it is the
            # out-of-fold number above that decides whether to ship it at all.
            weights, history = fit([(m, y) for _, m, y in rows])
            write_json(args.fit_stacker,
                       {"features": list(FEATURES), "weights": weights.tolist(),
                        "feature_version": FEATURE_VERSION,
                        "questions": len(rows), "loss": history[-1],
                        "out_of_fold": {"folds": args.stacker_folds,
                                        "questions": len(held_out),
                                        "recall": round(
                                            sum(held_out.values()) / max(1, len(held_out)), 4),
                                        "grouped_by_question_text": bool(texts)}})
            print(f"  wrote {args.fit_stacker} (weights fitted on all {len(rows)})")
            args.stacker = args.fit_stacker

        def run(settings_json, stacker_path):
            settings = json.loads(settings_json) if settings_json else {}
            stacker = Stacker.load(stacker_path) if stacker_path else None
            out = {}
            for c in caches:
                out.update(c.evaluate(settings, stacker=stacker))
            return out

        if args.search:
            from .cache import search
            stacker = Stacker.load(args.stacker) if args.stacker else None
            limits = {"candidates": min(c.arrays["rr_ids"].shape[1] for c in caches),
                      "passages": min(c.passages for c in caches)}
            from .pipeline import DEFAULTS
            base = dict(DEFAULTS, **(json.loads(args.settings) if args.settings else {}))
            chosen, per_question, mean = search(caches, base, stacker=stacker, limits=limits)
            if args.write_settings:
                keep = {k: chosen[k] for k in ("candidates", "passages", "aggregate",
                                               "aggregate_weight", "rerank_blend")}
                if args.stacker:
                    keep["stacker"] = args.stacker
                write_json(args.write_settings, {"best": keep, "recall": round(mean, 4),
                                                 "questions": len(per_question)})
                print(f"  wrote {args.write_settings}")
            return 0

        after = run(args.settings, args.stacker)
        label = args.label or "settings"
        if args.fit_stacker and args.stacker == args.fit_stacker:
            label += " (IN SAMPLE - the stacker was fitted on these questions)"
        summarise(after, sizes, label)
        if args.against is not None or args.against_stacker is not None:
            before = run(args.against, args.against_stacker)
            keys = sorted(set(before) & set(after))
            differences = [after[k] - before[k] for k in keys]
            wins = sum(1 for d in differences if d > 0)
            losses = sum(1 for d in differences if d < 0)
            delta = sum(differences) / len(differences)
            low, high = bootstrap(differences)
            print(f"\n  PAIRED against the baseline, over {len(keys)} questions")
            print(f"    difference {delta:+.4f}   95% CI [{low:+.4f}, {high:+.4f}]")
            print(f"    {wins} better, {losses} worse, {len(keys)-wins-losses} identical"
                  f"   sign test p = {sign_test(wins, losses):.4f}")
            print(f"    {'REAL' if not (low <= 0 <= high) else 'not distinguishable from no change'}")
        return 0

    if args.command == "retrieval":
        from .predict import load_questions
        from .retrieval import compare as retrieval_compare, evaluate, report
        from .store import write_json
        questions = load_questions(args.questions, labeled=True)
        if args.limit:
            questions = dict(list(questions.items())[:args.limit])
        pipeline = pipeline_from(args, use_reranker=False)
        try:
            summary, ranks = evaluate(pipeline, questions)
        finally:
            pipeline.close()
        report(args.output, ranks, args.label)
        print(f"\n  candidate recall over {len(questions)} questions")
        for name, row in summary.items():
            print(f"    {name:<8} " + "  ".join(f"@{d.lstrip('@')}={v:.4f}"
                                                for d, v in row.items()))
        write_json(Path(args.output).with_suffix(".summary.json"), summary)
        if args.against:
            # Every channel, from the one evaluation pass. Comparing them one at a time
            # would mean re-running the whole retrieval to look at a different column of
            # the same result - the ranks are already in the report.
            for name in ("lexical", "dense", "fused"):
                retrieval_compare(args.against, args.output, channel=name,
                                  label=args.label)
        return 0

    if args.command == "compare":
        from .compare import compare
        for subset in (("all", "single", "multi") if args.subset == "all" else (args.subset,)):
            compare(args.before, args.after, args.label,
                    None if subset == "all" else subset)
        return 0

    if args.command == "stack":
        from .stack import collect, train
        if not Path(args.cache).is_file():
            collect(args, limit=args.limit, output=args.cache, depth=args.depth)
        else:
            print(f"  reusing cached features at {args.cache}")
        train(args.cache, args.output, epochs=args.epochs)
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
              distill_temperature=args.distill_temperature,
              max_positives=args.max_positives,
              lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
              checkpointing=not args.no_checkpointing,
              max_groups=args.max_groups, document_loss=args.document_loss)
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
        if args.stage == "rerank":
            from .tune import rerank_sweep
            rerank_sweep(args, limit=args.limit, output=args.output,
                         depth=args.target_depth, passages=args.cache_passages,
                         radii=tuple(args.context_radius))
        elif args.stage == "fusion":
            from .tune import fusion_sweep
            fusion_sweep(args, limit=args.limit, output=args.output, target=args.target_depth)
        else:
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
