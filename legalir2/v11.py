"""GPU V11 experiment: python -m legalir2.v11 {benchmark,collect,fit,evaluate}."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .evidence import bundle_from_store, SCHEMA
from .stack import build_features, feature_names, fit
from .store import read_json, write_json
from .v11_compat import compatible_signature

# The only keys `fit`, `evaluate` and `pack_public` read. The rest of a cached row -
# bundles, baseline_passages, passage_scores, fused, channels - is provenance the plan
# requires on disk and nothing requires in memory. A full row parses to ~5.9 MB, so
# retaining whole rows costs ~11.7 GB at 2,000 questions; filtering costs ~0.2 GB.
ANALYTIC_KEYS = ('id', 'question', 'gold', 'docs', 'features', 'bundle_scores',
                 'frozen_prediction', 'seconds')

# Modules whose behaviour changes a cached row. Editing an unrelated file - compare.py,
# mine.py, a training script - must not invalidate hours of committed checkpoints.
SCORING_MODULES = ('v11.py', 'evidence.py', 'rerank.py', 'pipeline.py', 'stack.py',
                   'models.py', 'store.py', 'dense.py', 'lexical.py', 'fuse.py')


def normalized(text):
    return ' '.join(text.split()).casefold().rstrip('?.!:;, ')


def canonical(value):
    """JSON round trip, so a signature compares equal to its own reloaded form.

    `list(dict.items())` yields tuples; JSON has no tuple, so reading the guard back
    gives lists and a naive `!=` is always true. That made every resume raise
    'Resume inputs/configuration changed', which is exactly the state resume exists
    to survive.
    """
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))


def matrix(row, treatment):
    original = np.asarray(row['features'], dtype=float)
    if not treatment:
        return original
    scores = np.asarray(row['bundle_scores'], dtype=float)
    span = np.ptp(scores)
    scores = (scores - scores.min()) / span if span > 1e-12 else np.zeros_like(scores)
    return np.column_stack([original, scores])


def stream_cache(path, keys=ANALYTIC_KEYS):
    """Yield one filtered row at a time. Never builds a list of whole rows."""
    keep = set(keys)
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                yield {k: v for k, v in row.items() if k in keep}


def load_cache(path, keys=ANALYTIC_KEYS):
    return list(stream_cache(path, keys))


def checkpoint_row(directory, row):
    """Commit one gzip member atomically, then record it in the append-only index.

    A crash between the two leaves a row on disk that the index does not mention; it
    is simply rescored, costing one question. The reverse cannot happen, because the
    index line is only written after `replace` has returned.
    """
    name = hashlib.sha256(str(row['id']).encode()).hexdigest() + '.json.gz'
    target = directory / name
    temporary = target.with_suffix(target.suffix + '.partial')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
            stream.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(target)
    entry = json.dumps({'id': row['id'], 'file': name, 'seconds': row['seconds'],
                        'question': row['question'], 'gold': row['gold'],
                        'bytes': target.stat().st_size},
                       ensure_ascii=False)
    with (directory / 'index.jsonl').open('a', encoding='utf-8') as stream:
        # Separate a previous torn append from this new, complete record.
        stream.write('\n' + entry + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_index(directory, expected):
    """Committed rows, from the index alone - no row is decompressed or parsed.

    Reading 2,000 committed rows back to recover a filename and a duration meant
    parsing ~4.4 GB of JSON, twice per split. The index is ~300 KB.
    """
    path = directory / 'index.jsonl'
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from a kill mid-append; that row is rescored
        qid = entry['id']
        name = hashlib.sha256(str(qid).encode()).hexdigest() + '.json.gz'
        if entry['file'] != name:
            raise ValueError('Invalid checkpoint filename')
        target = directory / name
        if qid in expected and target.is_file():
            if 'question' in entry and (entry['question'] != expected[qid]['question'] or
                                       entry['gold'] != expected[qid].get('answer', [])):
                raise ValueError('Invalid checkpoint question/labels')
            if 'bytes' in entry and entry['bytes'] != target.stat().st_size:
                continue  # incomplete/corrupt local or restored file: rescore it
            done[entry['id']] = (target, entry['seconds'])  # a later line supersedes
    return done


def prepare_resume(directory, signature, items):
    directory.mkdir(parents=True, exist_ok=True)
    guard = directory / 'resume.json'
    reference = canonical(signature)
    if guard.exists():
        if compatible_signature(read_json(guard)) != compatible_signature(reference):
            raise ValueError('Resume inputs/configuration changed; use a new output directory')
    elif any(directory.glob('*.json.gz')):
        raise ValueError('Checkpoint rows have no resume metadata')
    else:
        write_json(guard, reference)
    # Per-row question/gold checks are redundant: the guard above already compares the
    # full item list, the model and index hashes, and the scoring modules.
    return read_index(directory, dict(items))


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def collect(args):
    from .pipeline import Pipeline, final_order
    run = read_json(args.baseline / 'v9-ensemble.run.json')
    settings = dict(run['settings'], stacker=str(args.baseline / 'v9-stacker-ens.json'))
    settings.setdefault('passage_selection', 'retrieval')  # frozen V9 evidence
    if settings['stage2_candidates'] != settings['candidates']:
        raise ValueError('This V11 variant requires both rerankers to score every candidate')
    # Require requested components; never silently fall back to lexical-only.
    for path in [args.store, args.dense / 'manifest.json', args.embedding / 'config.json',
                 args.first / 'config.json', args.qwen / 'config.json']:
        if not path.exists():
            raise FileNotFoundError(path)
    questions = read_json(args.questions)
    evaluation = read_json(args.validation)
    if args.stage in ('benchmark', 'collect') and args.split == 'train':
        held = {normalized(v['question']) for v in evaluation.values()}
        questions = {k: v for k, v in questions.items()
                     if k not in evaluation and normalized(v['question']) not in held}
    items = list(questions.items())[:args.limit] if args.limit else list(questions.items())
    if not items:
        raise ValueError('No questions to score')
    output = args.output / (args.split + '.jsonl.gz')
    meta_path = args.output / (args.split + '.manifest.json')
    args.output.mkdir(parents=True, exist_ok=True)
    print(f'{args.split}: checking cache inputs', flush=True)
    code = Path(__file__).parent
    signature = {'schema': SCHEMA, 'settings': settings, 'lexical': run['lexical'],
                 'items': items, 'store': hash_file(args.store),
                 'dense': hash_file(args.dense / 'manifest.json'),
                 'stacker': hash_file(args.baseline / 'v9-stacker-ens.json'),
                 'code': {name: hash_file(code / name) for name in SCORING_MODULES},
                 'models': [{p.name: hash_file(p) for p in model.glob('*.json')}
                            for model in [args.embedding, args.first, args.qwen]]}
    if meta_path.exists() and output.exists():
        metadata = read_json(meta_path)
        if 'resume_signature' in metadata:
            if compatible_signature(metadata['resume_signature']) != compatible_signature(canonical(signature)):
                raise ValueError('Completed cache inputs/configuration changed; use a new output directory')
            if metadata['cache_bytes'] != output.stat().st_size:
                raise ValueError('Completed cache size differs from manifest; restore the complete cache')
        else:
            # Older completed releases deleted the full guard. Validate every input
            # recorded by that format; never silently skip a changed LIMIT/split.
            from .store import Store
            from .models import checkpoint_identity
            store = Store(args.store)
            try:
                identities = [checkpoint_identity(read_json(p / 'provenance.json')
                              if (p / 'provenance.json').exists() else {'path': str(p)})
                              for p in [args.first, args.qwen]]
                checks = {'schema': SCHEMA, 'settings': settings, 'lexical': run['lexical'],
                          'count': len(items), 'split': args.split, 'store': store.metadata,
                          'questions_sha256': hash_file(args.questions), 'model_identities': identities}
                if any(metadata.get(k) != canonical(v) for k, v in checks.items()):
                    raise ValueError('Completed legacy cache inputs changed; use a new output directory')
            finally:
                store.close()
            print('Legacy cache: recorded inputs match; original code/weight hashes are unavailable.', flush=True)
        print(f'{args.split}: already completed', flush=True)
        return
    checkpoint_dir = args.output / (args.split + '.rows')
    done = prepare_resume(checkpoint_dir, signature, items)
    print(f'{args.split}: resuming {len(done)}/{len(items)} committed questions', flush=True)
    pipeline = Pipeline(args.store, args.lexical, args.dense, args.embedding,
                        [args.first, args.qwen], device=args.device,
                        settings=settings, lexical_settings=run['lexical'])
    started = time.perf_counter()
    count = len(done)
    try:
        qwen = pipeline.rerankers[1]
        if qwen.head != 'causal':
            raise ValueError('Second reranker must be Qwen causal')
        if qwen.context_limit and qwen.context_limit < 2048:
            raise ValueError('Qwen context limit below 2048')
        torch = qwen.local.torch
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for qid, row in items:
            if qid in done:
                continue
            tick = time.perf_counter()
            query = row['question']
            channels = {}
            fused, prior = pipeline.candidates(query, channels)
            docs = [d for d, _ in fused[:settings['candidates']]]
            cache, tables = {}, []
            for model in pipeline.rerankers:
                tables.append(pipeline._rerank(model, query, docs, prior, cache))
            frozen = final_order(fused, tables, settings, settings['reranker_weights'],
                                 channels, pipeline.stacker)[:5]
            bundles = [bundle_from_store(pipeline.store, doc, query, prior,
                                         qwen.local.tokenizer, qwen.template) for doc in docs]
            scores = qwen.score_bundles(bundles, args.bundle_batch)
            payload = {'id': qid, 'question': query, 'gold': row.get('answer', []),
                       'docs': docs, 'fused': fused, 'channels': channels,
                       'features': build_features(docs, tables, channels,
                                                  version=getattr(pipeline.stacker, 'version', 1)).tolist(),
                       'passage_scores': tables, 'baseline_passages': cache,
                       'bundle_scores': scores, 'bundles': bundles,
                       'frozen_prediction': frozen, 'seconds': time.perf_counter() - tick}
            checkpoint_row(checkpoint_dir, payload)
            count += 1
            print(f'{count}/{len(items)} {payload["seconds"]:.2f}s/question', flush=True)
        done = read_index(checkpoint_dir, dict(items))
        missing = [qid for qid, _ in items if qid not in done]
        if missing:
            raise ValueError(f'{len(missing)} questions never committed, first: {missing[0]}')
        print(f'{args.split}: assembling {len(items)} checkpoint files', flush=True)
        temporary = output.with_suffix(output.suffix + '.partial')
        # Concatenated independent gzip members are readable as one gzip stream.
        with temporary.open('wb') as stream:
            for qid, _ in items:
                with done[qid][0].open('rb') as member:
                    shutil.copyfileobj(member, stream)
        temporary.replace(output)
        seconds = sum(value[1] for value in done.values())
        metadata = pipeline.manifest()
        metadata.update(schema=SCHEMA, feature_names=list(feature_names(2)),
                        bundle_feature='qwen_bundle_score_v1', questions=str(args.questions),
                        questions_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),
                        split=args.split, count=count, seconds=seconds,
                        seconds_per_question=seconds / count,
                        projected_1050_seconds=1050 * seconds / count,
                        gpu_peak_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                        corpus_verification_gate=False, baseline_corpus=run['store'],
                        exploratory_calibration='Training overlap with g32 is possible; not an untouched test.',
                        model_identities=[r.local.identity for r in pipeline.rerankers],
                        resume_signature=canonical(signature), cache_bytes=output.stat().st_size)
        write_json(meta_path, metadata)
        # The rows are now inside `output`; keeping them doubles ~2.5 GB on disk and
        # leaves thousands of small files for every later rclone pass to re-walk.
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print(json.dumps({k: metadata[k] for k in ['count', 'seconds', 'projected_1050_seconds', 'gpu_peak_bytes']}))
    finally:
        pipeline.close()


def fit_pair(args):
    source = args.output / 'train.jsonl.gz'
    metadata = read_json(args.output / 'train.manifest.json')
    print('fit: reading compact training features from cache', flush=True)
    rows = load_cache(source)
    held = read_json(args.validation)
    held_text = {normalized(v['question']) for v in held.values()}
    if any(r['id'] in held or normalized(r['question']) in held_text for r in rows):
        raise ValueError('Evaluation questions found in calibration cache')
    seen, unique = set(), []
    for row in rows:
        key = normalized(row['question'])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    fitted = {'schema': SCHEMA, 'exploratory': True,
              'note': 'Calibration excludes development labels but may overlap g32 training.',
              'training_manifest': metadata, 'calibration_ids': [r['id'] for r in unique],
              'calibration_texts': sorted(seen)}
    for name, treatment in [('control', False), ('treatment', True)]:
        data = [(matrix(r, treatment), np.asarray([float(d in r['gold']) for d in r['docs']])) for r in unique]
        weights, history = fit(data, epochs=300, learning_rate=0.15, l2=1e-4, seed=2026)
        fitted[name] = {'weights': weights.tolist(), 'history': history,
                        'features': list(feature_names(2)) + (['qwen_bundle_score_v1'] if treatment else [])}
    write_json(args.output / 'fusion.json', fitted)


def evaluate(args):
    from .compare import compare
    fitted = read_json(args.output / 'fusion.json')
    metadata = read_json(args.output / 'validation.manifest.json')
    train_meta = fitted['training_manifest']
    for key in ['schema', 'settings', 'lexical', 'store', 'model_identities']:
        if metadata[key] != train_meta[key]:
            raise ValueError(f'Train/evaluation cache mismatch: {key}')
    rows = load_cache(args.output / 'validation.jsonl.gz')
    truth = read_json(args.validation)
    if {r['id'] for r in rows} != set(truth) or len(rows) != len(truth):
        raise ValueError('Evaluation cache must cover the complete validation set')
    if any(r['id'] in fitted['calibration_ids'] or normalized(r['question']) in fitted['calibration_texts'] for r in rows):
        raise ValueError('Calibration/evaluation overlap')
    for name in ['frozen', 'control', 'treatment']:
        diagnostics, predictions = {}, {}
        for row in rows:
            if row['question'] != truth[row['id']]['question'] or row['gold'] != truth[row['id']]['answer']:
                raise ValueError('Evaluation labels differ from the declared split')
            if name == 'frozen':
                selected = row['frozen_prediction']
            else:
                features = matrix(row, name == 'treatment')
                scores = features @ np.asarray(fitted[name]['weights'])
                selected = [d for _, d in sorted(zip(-scores, row['docs']))][:5]
            predictions[row['id']] = {'answer': selected}
            diagnostics[row['id']] = {'question': row['question'], 'relevant': row['gold'], 'predicted': selected}
        write_json(args.output / (name + '.json'), predictions)
        write_json(args.output / (name + '.diagnostics.json'), diagnostics)
    reports = {}
    for before, after in [('frozen', 'control'), ('control', 'treatment'), ('frozen', 'treatment')]:
        reports[f'{after}_vs_{before}'] = compare(args.output / (before + '.diagnostics.json'),
                                               args.output / (after + '.diagnostics.json'))
    write_json(args.output / 'comparison.json', reports)


def pack_public(args):
    """Apply the fitted treatment to public scores; never fit on public data."""
    from .predict import pack
    from .store import Store
    fitted = read_json(args.output / 'fusion.json')
    metadata = read_json(args.output / 'public.manifest.json')
    for key in ['schema', 'settings', 'lexical', 'store', 'model_identities']:
        if metadata[key] != fitted['training_manifest'][key]:
            raise ValueError(f'Train/public cache mismatch: {key}')
    names = list(feature_names(2)) + ['qwen_bundle_score_v1']
    if fitted['treatment']['features'] != names:
        raise ValueError('Unexpected treatment feature schema')
    weights = np.asarray(fitted['treatment']['weights'], dtype=float)
    if weights.shape != (21,) or not np.isfinite(weights).all():
        raise ValueError('Invalid treatment weights')
    questions = read_json(args.public)
    predictions = {}
    # Stream public cache rather than retaining bundle inputs for every question.
    for row in stream_cache(args.output / 'public.jsonl.gz'):
        qid = row['id']
        if qid in predictions or qid not in questions or row['question'] != questions[qid]['question']:
            raise ValueError('Invalid public question coverage')
        scores = matrix(row, True) @ weights
        if not np.isfinite(scores).all() or len(set(row['docs'])) != len(row['docs']):
            raise ValueError('Invalid public candidate scores')
        answer = [d for _, d in sorted(zip(-scores, row['docs']))][:5]
        if len(answer) != 5:
            raise ValueError('Public prediction must contain five unique documents')
        predictions[qid] = {'answer': answer}
    if set(predictions) != set(questions):
        raise ValueError('Public cache does not cover all questions')
    target = args.output / 'submission-v11.json'
    write_json(target, predictions)
    store = Store(args.store)
    try:
        pack(target, store, args.public, args.output / 'submission-v11.zip')
    finally:
        store.close()
    write_json(args.output / 'submission-v11.run.json', {
        'variant': 'treatment', 'exploratory': True, 'public_manifest': metadata,
        'fusion_sha256': hash_file(args.output / 'fusion.json'),
        'questions': len(predictions), 'prediction_sha256': hash_file(target),
        'zip_sha256': hash_file(args.output / 'submission-v11.zip')})


def submit(args):
    if not (args.output / 'fusion.json').exists():
        raise FileNotFoundError('Run fit before submit: fusion.json is missing')
    args.questions, args.split, args.limit = args.public, 'public', None
    collect(args)
    pack_public(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['benchmark', 'collect', 'fit', 'evaluate', 'submit'])
    p.add_argument('--baseline', type=Path, default=Path('analysis/v11/baseline'))
    p.add_argument('--output', type=Path, default=Path('outputs/v11'))
    p.add_argument('--store', type=Path, default=Path('artifacts2/store.sqlite'))
    p.add_argument('--lexical', type=Path, default=Path('artifacts2/lexical'))
    p.add_argument('--dense', type=Path, default=Path('artifacts2/dense'))
    p.add_argument('--embedding', type=Path, default=Path('models2/embedding'))
    p.add_argument('--first', type=Path, default=Path('models2/reranker-g32'))
    p.add_argument('--qwen', type=Path, default=Path('models2/partner_qwen'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--questions', type=Path, default=Path('artifacts/splits/train.json'))
    p.add_argument('--public', type=Path, default=Path('public-official.json'))
    p.add_argument('--validation', type=Path, default=Path('artifacts/splits/validation.json'))
    p.add_argument('--split', choices=['train', 'validation'], default='train')
    p.add_argument('--limit', type=int)
    p.add_argument('--bundle-batch', type=int, default=2)
    args = p.parse_args()
    if args.stage == 'benchmark':
        args.limit = args.limit or 10
        args.split = 'train'
        args.output = args.output / 'benchmark'
    if args.stage in ('benchmark', 'collect'):
        collect(args)
    elif args.stage == 'fit':
        fit_pair(args)
    elif args.stage == 'submit':
        submit(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
