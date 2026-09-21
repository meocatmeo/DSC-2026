"""Collect top-100 BF16 scores, fit fusion, expand duo, validate, and submit."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from . import deep_fusion as fusion, duo, fulltext_prior as prior, private_infer as private
from .duo_evidence import Evidence
from .evidence import build_bundle
from .h100 import digest, guard, pipeline
from .models import Local, audit
from .question_memory import QuestionMemory
from .store import file_hash, read_json, write_json

SCHEMA = 'deep-pool-100-v1'
POLICIES = ('baseline', 'deep')


def baseline_args(args):
    return SimpleNamespace(**dict(vars(args), out=args.out/'baseline'))


def compressed(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def datasets(args):
    return compressed(args.deep_assets/'training.json.gz')


def records(args, split):
    return datasets(args)['records'] if split == 'train' else prior.records(args, split)


def v9_rows(args, split):
    if split == 'train':
        return datasets(args)['v9']
    if split != 'private':
        return read_json(args.dataset/f'v9-{split}.json')
    base = baseline_args(args)
    return [r for p in sorted((args.source/'cache/v9').glob('part-*.json')) for r in prior.checked_source(base, p)]


def base_rows(args, split):
    if split == 'development':
        return compressed(args.deep_assets/'development-baseline.json.gz')
    if split == 'train':
        return []
    return [r for _, rows in prior.bases(baseline_args(args), split) for r in rows]


def identity(args):
    base = prior.manifest(baseline_args(args))
    asset = read_json(args.deep_assets/'manifest.json')
    for name, expected in asset['files'].items():
        if file_hash(args.deep_assets/name) != expected:
            raise ValueError('Deep-pool asset changed: '+name)
    return {'schema': SCHEMA, 'baseline': digest(base), 'assets': asset,
        'code': {n: file_hash(Path(__file__).with_name(n)) for n in ('deep_pool.py', 'deep_fusion.py')},
        'depth': fusion.DEPTH, 'extra': fusion.EXTRA, 'features': fusion.FEATURES,
        'batch': args.batch, 'precision': 'bfloat16', 'quantization': None,
        'attention': 'flash_attention_2', 'policies': list(POLICIES), 'evidence': 'fulltext-v2'}


def manifest(args):
    value = read_json(args.out/'inputs.json')
    if value != identity(args):
        raise ValueError('Deep-pool identity changed; use a separate run ID')
    return value


def validate_split(training, development, validation):
    held = dict(development, **validation)
    prior.validate_memory(training, held)
    if not training or set(development) & set(validation):
        raise ValueError('Invalid training/development/validation split')


def prepare(args):
    prior.prepare(baseline_args(args))
    marker = identity(args)
    original = read_json(args.duo_run/'inputs.json')
    asset = marker['assets']
    if asset['duo_inputs'] != digest(original):
        raise ValueError('Training assets belong to another model')
    training = datasets(args)
    dev, val = records(args, 'development'), records(args, 'validation')
    validate_split(training['records'], dev, val)
    if any(q not in original['memory'] or r != original['memory'][q] for q, r in training['records'].items()):
        raise ValueError('Fusion training is not from the frozen training partition')
    known = set(original['known_docs'])
    private.validate_v9(training['v9'], list(training['records'].items()), known)
    prior.validate_base(base_rows(args, 'development'), list(dev.items()), known)
    guard(args.out/'inputs.json', marker)
    components = prior.manifest(baseline_args(args))['assets']['parameter_audit']['components']
    write_json(args.out/'audit.json', audit(dict(components, deep_fusion=len(fusion.FEATURES))))
    print(f"Prepared deep pool: {len(training['records'])} training, {len(dev)} development; no quantization.", flush=True)


def require_bf16(local):
    torch = local.torch
    if (getattr(local.model, 'is_quantized', False) or
            any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in local.model.parameters())):
        raise ValueError('Unquantized BF16 weights required')


def validate_trace(trace):
    for t in trace['documents'].values():
        if (len(t['input_ids']) > t['budget'] or not t['input_ids'] or
                hashlib.sha256(json.dumps(t['input_ids']).encode()).hexdigest() != t['sha256']):
            raise ValueError('Full-text evidence token trace changed')


def validate_raw(rows, items, old, known, weights, preserved=None):
    if len(rows) != len(items):
        raise ValueError('Incomplete deep score shard')
    for row, (qid, record) in zip(rows, items):
        baseline = old[qid]
        if (row['id'] != qid or row['question'] != record['question'] or row['gold'] != record['answer']
                or row['v9_input'] != digest(baseline)):
            raise ValueError('Deep row identity changed')
        if row['docs'] != fusion.candidate_pool(row['retrieved'], row['retained'], known):
            raise ValueError('Deep candidate selection changed')
        if preserved is not None and row['retained'] != preserved[qid]['docs']:
            raise ValueError('Original duo pool was not retained')
        expected_v9 = dict(zip(baseline['docs'], (np.asarray(baseline['features1'])@weights).tolist()))
        if row['v9'] != expected_v9:
            raise ValueError('Frozen V9 scores changed')
        for channel in ('fused', 'lexical', 'dense'):
            values = row['channels'][channel]
            if len({d for d, _ in values}) != len(values) or any(d not in known or not np.isfinite(s) for d, s in values):
                raise ValueError('Invalid retrieval channel')
        if row['retrieved'] != row['channels']['fused'][:fusion.DEPTH]:
            raise ValueError('Saved top-100 differs from fusion channel')
        matrix = fusion.features(row['docs'], row['tables'], row['channels'], row['v9'])
        if not np.array_equal(matrix, np.asarray(row['features'])):
            raise ValueError('Raw scores do not reproduce fusion features')
        if set(row['evidence']['documents']) != set(row['docs']):
            raise ValueError('Incomplete full-text evidence')
        if preserved is not None and any(row['evidence']['documents'][d] != trace
                for d, trace in preserved[qid]['evidence']['documents'].items()):
            raise ValueError('Original duo passages changed')
        validate_trace(row['evidence'])


def score_raw(args, model, evidence, memory, qid, record, old, original, weights, known):
    channels = {}
    fused, _ = model.candidates(record['question'], channels)
    # Some documents can be appended only as a retrieval fallback. Record the
    # actual fused output used here; it must always match the selected top 100.
    channels['fused'] = fused
    if original is None:
        retained, base = duo.pool(old, weights, memory, known)
        retained, _ = evidence.expand(record['question'], retained, base, known)
    else:
        retained = list(original['docs'])
    docs = fusion.candidate_pool(fused, retained, known)
    # Frozen Evidence.encode appends to its trace; expand used to reset it.
    # Held-out pools are already known, so reset explicitly for every query.
    evidence.trace = {'documents': {}}
    evidence.encode(record['question'], [d for d in docs if original is None or d not in original['docs']])
    traces = deepcopy(evidence.trace)
    if original is not None:
        # Use exactly the passages that produced the saved duo baseline.
        traces['documents'].update(deepcopy(original['evidence']['documents']))
    texts, owners = [], []
    for doc in docs:
        trace = traces['documents'][doc]
        pieces = [p['text'] for p in trace['packed']] or ['']
        for text in pieces:
            texts.append(trace['label']+'\n'+text)
            owners.append(doc)
    first = model.rerankers[0]
    values = first.score(record['question'], texts, batch_size=args.batch, max_length=first.max_length)
    if len(values) != len(texts):
        raise ValueError('Incomplete first-reranker scores')
    table = {d: [] for d in docs}
    for doc, score in zip(owners, values):
        table[doc].append(float(score))
    second = model.rerankers[1]
    bundles = [build_bundle(record['question'], traces['documents'][d]['label'],
        [{'id': p['rowid'], 'text': p['text'], 'score': -i} for i, p in enumerate(traces['documents'][d]['packed'])],
        second.local.tokenizer, second.template, max_tokens=2048, max_passages=3) for d in docs]
    values = second.score_bundles(bundles, batch_size=args.batch)
    tables = [table, {d: [float(s)] for d, s in zip(docs, values)}]
    v9 = dict(zip(old['docs'], (np.asarray(old['features1'])@weights).tolist()))
    return {'id': qid, 'question': record['question'], 'gold': record['answer'], 'docs': docs,
        'retained': retained, 'retrieved': fused[:fusion.DEPTH], 'channels': channels, 'tables': tables,
        'v9': v9, 'v9_input': digest(old), 'features': fusion.features(docs, tables, channels, v9).tolist(),
        'evidence': traces}


def collect(args, split):
    marker, questions = manifest(args), records(args, split)
    directory = args.out/'cache/deep'/split
    guard(directory/'resume.json', {'inputs': digest(marker), 'split': split})
    signature = read_json(args.duo_run/'inputs.json')
    weights, known = np.asarray(signature['weights']), set(signature['known_docs'])
    old = {r['id']: r for r in v9_rows(args, split)}
    preserved = {r['id']: r for r in base_rows(args, split)} if split != 'train' else None
    model = evidence = memory = None
    started, computed = time.time(), 0
    try:
        for start, items in private.shard_items(questions, 5):
            path = directory/f'part-{start//5:05d}.json'
            if path.exists():
                validate_raw(read_json(path), items, old, known, weights, preserved)
            else:
                if model is None:
                    model = pipeline(args, {'attention': 'flash_attention_2', 'batch': args.batch}, 'retrieval')
                    for local in [model.dense.embedder.local]+[r.local for r in model.rerankers]:
                        require_bf16(local)
                    codec = duo.PairTokenizer(model.rerankers[1].local.tokenizer, 2048)
                    evidence = Evidence(args.index, codec, args.evidence_index)
                    memory = QuestionMemory(signature['memory'])
                rows = [score_raw(args, model, evidence, memory, q, r, old[q],
                    preserved[q] if preserved is not None else None, weights, known) for q, r in items]
                validate_raw(rows, items, old, known, weights, preserved)
                write_json(path, rows)
                computed += len(rows)
            private.progress('deep-pool/'+split, start+len(items), len(questions), computed, started)
    finally:
        if evidence is not None:
            evidence.close()
        if model is not None:
            model.close()


def raw_parts(args, split):
    marker, questions = manifest(args), records(args, split)
    directory = args.out/'cache/deep'/split
    if read_json(directory/'resume.json') != {'inputs': digest(marker), 'split': split}:
        raise ValueError('Deep score cache identity differs')
    signature = read_json(args.duo_run/'inputs.json')
    old = {r['id']: r for r in v9_rows(args, split)}
    preserved = {r['id']: r for r in base_rows(args, split)} if split != 'train' else None
    for start, items in private.shard_items(questions, 5):
        path = directory/f'part-{start//5:05d}.json'
        rows = read_json(path)
        validate_raw(rows, items, old, set(signature['known_docs']), np.asarray(signature['weights']), preserved)
        yield path, rows


def training_rows(args):
    return [{k: r[k] for k in ('id', 'question', 'gold', 'docs', 'features')}
            for _, part in raw_parts(args, 'train') for r in part]


def fit(args):
    rows = training_rows(args)
    if (args.out/'fusion.json').exists():
        saved = fitted(args)
        if saved['training'] != digest(rows):
            raise ValueError('Saved fusion belongs to different training scores')
        print('Reusing the verified training-only fusion model.', flush=True)
        return
    record = fusion.train(rows)
    record.update(inputs=digest(manifest(args)), training=digest(rows), training_ids=[r['id'] for r in rows],
        limitation='Frozen neural rerankers may have seen training questions; evaluate on development/validation.')
    guard(args.out/'fusion.json', record)
    print('Trained depth-100 fusion on', len(rows), 'training questions only.', flush=True)


def fitted(args):
    value = read_json(args.out/'fusion.json')
    if (value['inputs'] != digest(manifest(args)) or value['training_ids'] != list(records(args, 'train'))
            or value['features'] != fusion.FEATURES or len(value['weights']) != len(fusion.FEATURES)
            or not np.isfinite(value['weights']).all()):
        raise ValueError('Fusion identity differs')
    # Full raw-score checks run at fit and before calibration; later stages use
    # the selected immutable fusion file hash in their cache/selection guards.
    return value


def expand_probabilities(local, codec, query, encoded, original, docs, batch):
    n = len(original['docs'])
    edges = fusion.measured_edges(n, len(docs))
    if docs[:n] != original['docs']:
        raise ValueError('Expanded tournament reordered original documents')
    probability = np.full((len(docs), len(docs)), .5)
    probability[:n, :n] = original['probabilities']
    examples = [(i, j) for a, b in edges for i, j in ((a, b), (b, a))]
    scores = []
    for start in range(0, len(examples), batch):
        ids = [codec.pair(query, encoded[docs[i]], encoded[docs[j]]) for i, j in examples[start:start+batch]]
        inputs = codec.tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt')
        with local.torch.inference_mode():
            logits = local.model(**{k: v.to(local.device) for k, v in inputs.items()}, logits_to_keep=1).logits[:, -1, codec.labels].float()
            scores.extend(logits.softmax(dim=-1)[:, 0].cpu().tolist())
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite new comparisons')
    for k, (i, j) in enumerate(edges):
        probability[i, j] = (scores[2*k]+1-scores[2*k+1])/2
        probability[j, i] = 1-probability[i, j]
    return probability


def validate_comparisons(rows, raw, original, model):
    if len(rows) != len(raw):
        raise ValueError('Incomplete expanded tournament shard')
    for row, source in zip(rows, raw):
        old = original[source['id']]
        docs, expected = fusion.shortlist(source, old, model)
        if (row['id'] != source['id'] or row['question'] != source['question'] or row['gold'] != source['gold']
                or row['raw'] != digest(source) or row['base'] != digest(old)
                or row['docs'] != docs or not np.array_equal(row['prior'], expected)
                or row['baseline'] != prior.predict(old, prior.BASELINE)):
            raise ValueError('Expanded tournament inputs differ')
        n = len(old['docs'])
        if not np.array_equal(np.asarray(row['probabilities'])[:n, :n], old['probabilities']):
            raise ValueError('Original comparisons changed')
        probability, ability = np.asarray(row['probabilities']), np.asarray(row['ability'])
        if (probability.shape != (len(docs), len(docs)) or ability.shape != (len(docs),)
                or not np.isfinite(probability).all() or not np.isfinite(ability).all()
                or (probability < 0).any() or (probability > 1).any()
                or not np.allclose(probability+probability.T, 1., atol=1e-6, rtol=0)):
            raise ValueError('Invalid expanded comparison matrix')
        private.validate_comparison_fit(row)


def compare(args, split):
    model, marker = fitted(args), manifest(args)
    original = {r['id']: r for r in base_rows(args, split)}
    directory = args.out/'cache/comparisons'/split
    guard(directory/'resume.json', {'inputs': digest(marker), 'fusion': digest(model), 'split': split})
    local = codec = None
    started, computed, count = time.time(), 0, 0
    try:
        for path, raw in raw_parts(args, split):
            target = directory/path.name
            if target.exists():
                validate_comparisons(read_json(target), raw, original, model)
            else:
                if local is None:
                    import torch
                    os.environ['LEGALIR_CAUSAL_ATTN'] = 'flash_attention_2'
                    local = Local(args.duo_run/'model', kind='causal', device='cuda', dtype=torch.bfloat16)
                    require_bf16(local)
                    codec = duo.PairTokenizer(local.tokenizer, 2048)
                rows = []
                for source in raw:
                    old = original[source['id']]
                    docs, initial = fusion.shortlist(source, old, model)
                    encoded = {d: [source['evidence']['documents'][d]['input_ids']] for d in docs}
                    probability = expand_probabilities(local, codec, source['question'], encoded, old, docs, args.batch)
                    ability = prior.fit(initial, probability)
                    rows.append({'id': source['id'], 'question': source['question'], 'gold': source['gold'],
                        'docs': docs, 'prior': initial.tolist(), 'ability': ability.tolist(),
                        'probabilities': probability.tolist(), 'raw': digest(source), 'base': digest(old),
                        'baseline': prior.predict(old, prior.BASELINE)})
                validate_comparisons(rows, raw, original, model)
                write_json(target, rows)
                computed += len(rows)
            count += len(raw)
            private.progress('deep-comparisons/'+split, count, len(original), computed, started)
    finally:
        if local is not None:
            local.close()


def compared(args, split):
    model, marker = fitted(args), manifest(args)
    directory = args.out/'cache/comparisons'/split
    if read_json(directory/'resume.json') != {'inputs': digest(marker), 'fusion': digest(model), 'split': split}:
        raise ValueError('Expanded tournament cache differs')
    original, result = {r['id']: r for r in base_rows(args, split)}, []
    for path, raw in raw_parts(args, split):
        rows = read_json(directory/path.name)
        validate_comparisons(rows, raw, original, model)
        result.extend(rows)
    return result


def recall(rows, policy):
    if not rows or any(not r['gold'] for r in rows):
        raise ValueError('Labeled evaluation rows required')
    return np.asarray([len(set(fusion.predict(r, policy)) & set(r['gold']))/len(set(r['gold'])) for r in rows])


def coverage(args, split, comparisons):
    by_id = {r['id']: r for r in comparisons}
    original = {r['id']: r for r in base_rows(args, split)}
    levels = {k: [] for k in ('fused_top50', 'fused_top100', 'scored_pool', 'original_duo_pool', 'expanded_duo_pool')}
    for _, part in raw_parts(args, split):
        for row in part:
            gold = set(row['gold'])
            if not gold:
                raise ValueError('Coverage diagnostics require labeled questions')
            retrieved = [d for d, _ in row['retrieved']]
            pools = (retrieved[:50], retrieved, row['docs'], original[row['id']]['docs'], by_id[row['id']]['docs'])
            for key, docs in zip(levels, pools):
                levels[key].append(len(gold & set(docs))/len(gold))
    if any(len(values) != len(comparisons) for values in levels.values()):
        raise ValueError('Incomplete candidate coverage diagnostic')
    return {k: float(np.mean(v)) for k, v in levels.items()}


def calibrate(args):
    model = fitted(args)
    if model['training'] != digest(training_rows(args)):
        raise ValueError('Fusion training cache changed')
    rows = compared(args, 'development')
    trials = [{'policy': p, 'recall': float(recall(rows, p).mean())} for p in POLICIES]
    selected = max(trials, key=lambda r: (r['recall'], r['policy'] == 'baseline'))['policy']
    candidates = coverage(args, 'development', rows)
    write_json(args.out/'calibration.json', {'inputs': digest(manifest(args)), 'fusion': digest(model),
        'scores': digest(rows), 'trials': trials, 'selected': selected,
        'candidate_coverage': candidates})
    print('Development choice:', selected, trials, flush=True)
    print('Candidate coverage:', candidates, flush=True)


def choice(args):
    value = read_json(args.out/'calibration.json')
    rows = compared(args, 'development')
    if (value['inputs'] != digest(manifest(args)) or value['fusion'] != digest(fitted(args))
            or value['scores'] != digest(rows)
            or value['trials'] != [{'policy': p, 'recall': float(recall(rows, p).mean())} for p in POLICIES]):
        raise ValueError('Calibration inputs changed')
    expected = max(value['trials'], key=lambda r: (r['recall'], r['policy'] == 'baseline'))['policy']
    if value['selected'] != expected or expected not in POLICIES:
        raise ValueError('Calibration selection differs')
    return value


def decision(rows):
    before, after = recall(rows, 'baseline'), recall(rows, 'deep')
    delta = after-before
    rng = np.random.default_rng(2026)
    ci = np.quantile([rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)], [.025, .975])
    return {'baseline_recall': float(before.mean()), 'candidate_recall': float(after.mean()),
        'delta': float(delta.mean()), 'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum()),
        'ci95': ci.tolist(), 'deploy': bool(delta.mean() > 1e-12)}


def validate(args):
    selected = choice(args)
    report = {'inputs': digest(manifest(args)), 'calibration': digest(selected),
        'limitation': 'Historically reused validation; not a guarantee of a private gain.'}
    if selected['selected'] == 'baseline':
        report.update(evaluated=False, deploy=False, scores=None, delta=None)
    else:
        rows = compared(args, 'validation')
        report.update(decision(rows), evaluated=True, scores=digest(rows),
                      candidate_coverage=coverage(args, 'validation', rows))
    write_json(args.out/'validation.json', report)
    print('Validation:', report, flush=True)


def deployment(args):
    selected, report = choice(args), read_json(args.out/'validation.json')
    if report['inputs'] != digest(manifest(args)) or report['calibration'] != digest(selected):
        raise ValueError('Validation input identity changed')
    if report['evaluated']:
        rows = compared(args, 'validation')
        if selected['selected'] != 'deep' or report['scores'] != digest(rows) or any(report[k] != v for k, v in decision(rows).items()):
            raise ValueError('Validation result changed')
    elif selected['selected'] != 'baseline' or report['deploy'] or report['scores'] is not None:
        raise ValueError('Only baseline can skip validation')
    return report


def submit(args):
    report, marker = deployment(args), prior.manifest(baseline_args(args))
    answers = {}
    for name in ('v9', 'duo'):
        path = args.source/f'submission-private-{name}.zip'
        if file_hash(path) != marker['source_files'][path.name] or (name == 'duo' and file_hash(path) != prior.SCORED_ZIP):
            raise ValueError('Scored fallback archive changed')
        with zipfile.ZipFile(path) as archive:
            answers[name] = json.loads(archive.read('submission.json'))
    if report['deploy']:
        answers['duo'] = {r['id']: {'answer': fusion.predict(r, 'deep')} for r in compared(args, 'private')}
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    stamp = dict(marker['assets'], parameter_audit=read_json(args.out/'audit.json'),
        evidence={'policy': SCHEMA, 'deployed': report['deploy'], 'fusion': digest(fitted(args))})
    for name in ('v9', 'duo'):
        private.write_submission(args, answers[name], name, known, stamp)
    write_json(args.out/'submission-choice.json', {'candidate_deployed': report['deploy'], 'validation': report,
        'fallback_private_score_user_reported': .9510, 'candidate_private_score': None,
        'precision': 'bfloat16', 'quantization': None})


def run_split(args, split):
    collect(args, split)
    compare(args, split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'training', 'fit', 'development', 'calibrate', 'validation', 'validate', 'private', 'submit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('outputs/private/private-duo-fulltext-v2-4090-fast1'))
    parser.add_argument('--duo-run', type=Path, default=Path('outputs/duo/duo-20260918T110105Z'))
    parser.add_argument('--index', type=Path, default=Path('artifacts-ranking-h100'))
    parser.add_argument('--evidence-index', type=Path, default=Path('artifacts-ranking-h100/duo-fulltext-v2.sqlite'))
    parser.add_argument('--baseline', type=Path, default=Path('analysis/private/baseline'))
    parser.add_argument('--models', type=Path, default=Path('models2'))
    parser.add_argument('--questions', type=Path, default=Path('data/private-official.json'))
    parser.add_argument('--dataset', type=Path, default=Path('analysis/fulltext-prior'))
    parser.add_argument('--deep-assets', type=Path, default=Path('analysis/deep-pool'))
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1 or args.out.resolve() in (args.source.resolve(), args.duo_run.resolve()):
        parser.error('Use a positive batch and separate output directory')
    if args.stage == 'training':
        collect(args, 'train')
    elif args.stage == 'development':
        run_split(args, 'development')
    elif args.stage == 'validation':
        if choice(args)['selected'] != 'baseline':
            prior.collect_base(baseline_args(args), 'validation')
            run_split(args, 'validation')
        else:
            print('Development retained baseline; skip validation GPU scoring.', flush=True)
    elif args.stage == 'private':
        if deployment(args)['deploy']:
            run_split(args, 'private')
        else:
            print('No validated gain; retain the scored private answers.', flush=True)
    else:
        {'prepare': prepare, 'fit': fit, 'calibrate': calibrate, 'validate': validate, 'submit': submit}[args.stage](args)


if __name__ == '__main__':
    main()
