"""Calibrate full-text relevance priors with the frozen duo comparisons.

Development selects one policy; validation gates deployment. Private questions
never fit a weight. Existing private comparisons are reused under hash guards.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from . import duo, private_infer as private
from .cache import normalise_question
from .consensus import rerank
from .duo_evidence import Evidence
from .evidence import build_bundle
from .h100 import digest, guard
from .models import Local
from .question_memory import QuestionMemory
from .rerank import Reranker
from .store import file_hash, read_json, write_json

SCHEMA = 'fulltext-prior-consensus-v1'
BASELINE = {'alpha': 0., 'strength': .25}
POLICIES = [BASELINE] + [dict(alpha=a, strength=s)
    for a in (.25, .5, .75, 1.) for s in (.25, .5, 1.)]
POLICIES += [dict(alpha=a, strength=.15, consensus=c)
    for a in (0., .5, 1.) for c in ('borda', 'graph')]
SCORED_ZIP = '40caa611cfdbf4f853296fdfef93774b117fffc4c339d591690080a59e0a6d57'


def fit(prior, probability):
    """Fit with an explicit prior, preserving its scale after interpolation."""
    from scipy.optimize import minimize
    prior, probability = np.asarray(prior), np.asarray(probability)
    n = len(prior)
    if (prior.shape != (n,) or n < 5 or probability.shape != (n, n)
            or not np.isfinite(prior).all() or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability+probability.T, 1, atol=1e-6)):
        raise ValueError('Invalid full-text prior tournament')
    first, second = np.triu_indices(n, 1)
    def objective(value):
        diff = value[first]-value[second]
        loss = np.sum(np.logaddexp(0, diff)-probability[first, second]*diff)
        loss += .5*np.square(value-prior).sum()
        estimate = 1/(1+np.exp(-np.clip(value[:, None]-value[None, :], -30, 30)))
        return loss, (estimate-probability).sum(axis=1)+value-prior
    result = minimize(objective, prior.copy(), jac=True, method='BFGS', options={'gtol': 1e-7})
    gradient = objective(result.x)[1]
    if not np.isfinite(result.x).all() or np.max(np.abs(gradient)) > 1e-5:
        raise ValueError('Full-text prior fit did not converge')
    return result.x


def predict(row, policy):
    if policy not in POLICIES:
        raise ValueError('Unregistered full-text policy')
    prior, ability = np.array(row['prior']), np.array(row['ability'])
    if policy['alpha']:
        relevance = np.asarray(row['relevance'], dtype=float)
        if relevance.shape != prior.shape or not np.isfinite(relevance).all():
            raise ValueError('Missing or invalid full-text scores')
        relevance = (relevance-relevance.mean())/max(float(relevance.std()), .5)
        prior = (1-policy['alpha'])*prior + policy['alpha']*relevance
        ability = fit(prior, row['probabilities'])
    score = prior+policy['strength']*(ability-prior)
    if policy.get('consensus'):
        return rerank(row['docs'], score, row['probabilities'],
                      predict(row, BASELINE), policy['consensus'])
    return [d for _, d in sorted(zip(-score, row['docs']))][:5]


def recall_vector(rows, policy):
    if any(not r['gold'] for r in rows):
        raise ValueError('Calibration and validation require labeled records')
    return np.array([len(set(predict(r, policy)) & set(r['gold']))/len(set(r['gold'])) for r in rows])


def select(rows):
    trials = [{'policy': p, 'recall': float(recall_vector(rows, p).mean())} for p in POLICIES]
    best = max(trials, key=lambda r: (r['recall'], r['policy'] == BASELINE,
                                     -r['policy']['alpha'], -r['policy']['strength']))
    return {'selected': best['policy'], 'trials': trials}


def decision(rows, policy):
    before, after = recall_vector(rows, BASELINE), recall_vector(rows, policy)
    diff = after-before
    rng = np.random.default_rng(2026)
    means = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(5000)]
    return {'baseline_recall': float(before.mean()), 'candidate_recall': float(after.mean()),
        'delta': float(diff.mean()), 'wins': int((diff > 0).sum()), 'losses': int((diff < 0).sum()),
        'ci95': np.quantile(means, [.025, .975]).tolist(),
        'deploy': bool(policy != BASELINE and diff.mean() > 1e-12),
        'limitation': 'Historically reused validation; positive delta does not guarantee a private improvement.'}


def code_identity():
    return {name: file_hash(Path(__file__).with_name(name)) for name in (
        'fulltext_prior.py', 'consensus.py', 'evidence.py', 'private_infer.py', 'duo_evidence.py')}


def validate_memory(memory, heldout):
    if (set(memory) & set(heldout) or
            {normalise_question(r['question']) for r in memory.values()} &
            {normalise_question(r['question']) for r in heldout.values()}):
        raise ValueError('Held-out labels leaked into question memory')


def manifest(args):
    value = read_json(args.out/'inputs.json')
    if value['schema'] != SCHEMA or value['code'] != code_identity() or value['batch'] != args.batch:
        raise ValueError('Run identity changed; use a separate run ID')
    for name, expected in value['datasets'].items():
        if file_hash(args.dataset/name) != expected:
            raise ValueError('Labeled dataset changed')
    return value


def records(args, split):
    return private.question_records(args.questions) if split == 'private' else read_json(args.dataset/f'{split}.json')


def source_paths(args, split):
    return sorted((args.source/'cache/duo').glob('part-*.json')) if split == 'private' else []


def checked_source(args, path):
    saved = manifest(args)['source_files']
    name = path.relative_to(args.source).as_posix()
    if name not in saved or file_hash(path) != saved[name]:
        raise ValueError('Source-run file changed: '+name)
    return read_json(path)


def validate_base(rows, items, known):
    duo.validate_scores(rows, items)
    for row in rows:
        private.validate_comparison_fit(row)
        if not set(row['docs']) <= known:
            raise ValueError('Unknown corpus candidate')
        traces = row['evidence']['documents']
        if set(traces) != set(row['docs']):
            raise ValueError('Incomplete full-text evidence')
        for trace in traces.values():
            import hashlib
            if (len(trace['input_ids']) > trace['budget'] or
                hashlib.sha256(json.dumps(trace['input_ids']).encode()).hexdigest() != trace['sha256']):
                raise ValueError('Corrupt full-text token trace')


def prepare(args):
    options = SimpleNamespace(**vars(args))
    options.out = args.out/'asset-check'
    options.infer_batch, options.shard_size = args.batch, 10
    options.duo_policy = 'fulltext-v2'
    _, signature, assets = private.preflight(options)
    previous = read_json(args.source/'inputs.json')
    for key in ('duo_inputs', 'duo_model', 'calibration', 'questions', 'question_ids', 'assets'):
        if previous[key] != assets[key]:
            raise ValueError('Incompatible scored private source: '+key)
    if any(previous['evidence'][key] != assets['evidence'][key]
           for key in ('policy', 'metadata', 'code')) or previous['strength'] != .25:
        raise ValueError('Source is not the frozen full-text baseline')
    if file_hash(args.source/'submission-private-duo.zip') != SCORED_ZIP:
        raise ValueError('Source ZIP is not the scored 0.9510 baseline')
    dev, val = records(args, 'development'), records(args, 'validation')
    if set(dev) != set(signature['development']):
        raise ValueError('Development membership differs from frozen split')
    if set(dev) & set(val) or {normalise_question(r['question']) for r in dev.values()} & {
            normalise_question(r['question']) for r in val.values()}:
        raise ValueError('Development/validation question leakage')
    validate_memory(signature['memory'], dict(dev, **val))
    datasets = {name: file_hash(args.dataset/name) for name in (
        'development.json', 'validation.json', 'v9-development.json', 'v9-validation.json')}
    for split, items in [('development', dev), ('validation', val)]:
        private.validate_v9(read_json(args.dataset/f'v9-{split}.json'), list(items.items()), set(signature['known_docs']))
    # Validate the complete source caches and reproduce the actually scored ZIP.
    questions = records(args, 'private')
    old = private.read_v9(SimpleNamespace(out=args.source, shard_size=previous['shard_size']), questions, signature, previous)
    with zipfile.ZipFile(args.source/'submission-private-duo.zip') as archive:
        predicted = json.loads(archive.read('submission.json'))
    count = 0
    for start, items in private.shard_items(questions, previous['shard_size']):
        part = read_json(args.source/'cache/duo'/f'part-{start//previous["shard_size"]:05d}.json')
        private.validate_duo(part, items, set(signature['known_docs']), old[start:start+len(items)], signature['weights'])
        validate_base(part, items, set(signature['known_docs']))
        for row in part:
            if predict(row, BASELINE) != predicted[row['id']]['answer']:
                raise ValueError('Source comparisons differ from the scored ZIP')
        count += len(part)
    if count != len(questions) or set(predicted) != set(questions):
        raise ValueError('Incomplete source comparisons')
    paths = [args.source/'inputs.json', args.source/'submission-private-duo.zip', args.source/'submission-private-v9.zip']
    paths += list((args.source/'cache/duo').glob('part-*.json'))
    paths += list((args.source/'cache/v9').glob('part-*.json'))
    marker = {'schema': SCHEMA, 'code': code_identity(), 'batch': args.batch,
        'datasets': datasets, 'assets': assets, 'policies': POLICIES,
        'source_files': {p.relative_to(args.source).as_posix(): file_hash(p) for p in paths}}
    guard(args.out/'inputs.json', marker)
    write_json(args.out/'audit.json', assets['parameter_audit'])
    print('Prepared: exact scored source and labeled splits verified; no new model parameters.', flush=True)


def collect_base(args, split):
    marker = manifest(args)
    signature = read_json(args.duo_run/'inputs.json')
    questions = records(args, split)
    old = {r['id']: r for r in read_json(args.dataset/f'v9-{split}.json')}
    directory = args.out/'cache/base'/split
    guard(directory/'resume.json', {'inputs': digest(marker), 'split': split})
    known, weights = set(signature['known_docs']), np.array(signature['weights'])
    model = evidence = memory = codec = None
    started, computed = time.time(), 0
    try:
        def compute(items):
            nonlocal model, evidence, memory, codec
            if model is None:
                os.environ['LEGALIR_CAUSAL_ATTN'] = signature['attention']
                model = Local(args.duo_run/'model', kind='causal', device='cuda')
                codec = duo.PairTokenizer(model.tokenizer, signature['max_length'])
                evidence = Evidence(args.index, codec, args.evidence_index)
                memory = QuestionMemory(signature['memory'])
            rows = []
            for qid, record in items:
                docs, base = duo.pool(old[qid], weights, memory, known)
                docs, base = evidence.expand(record['question'], docs, base, known)
                probability = duo.compare_documents(model, codec, record['question'],
                    evidence.encode(record['question'], docs), docs, args.batch)
                base = np.asarray(base, dtype=float)
                prior = (base-base.mean())/max(float(base.std()), .5)
                ability = fit(prior, probability)
                rows.append({'id': qid, 'question': record['question'], 'gold': record['answer'],
                    'docs': docs, 'prior': prior.tolist(), 'ability': ability.tolist(),
                    'probabilities': probability.tolist(), 'evidence': deepcopy(evidence.trace)})
            return rows
        for start, items in private.shard_items(questions, 10):
            _, fresh = private.cache_row(directory/f'part-{start//10:05d}.json', items,
                lambda r, i: validate_base(r, i, known), compute)
            computed += len(items) if fresh else 0
            private.progress('fulltext-comparisons/'+split, start+len(items), len(questions), computed, started)
    finally:
        if evidence is not None:
            evidence.close()
        if model is not None:
            model.close()


def bases(args, split):
    marker, questions = manifest(args), records(args, split)
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    directory = args.source/'cache/duo' if split == 'private' else args.out/'cache/base'/split
    if split != 'private' and read_json(directory/'resume.json') != {'inputs': digest(marker), 'split': split}:
        raise ValueError('Full-text comparison cache identity changed')
    source_size = marker['assets']['shard_size'] if split != 'private' else read_json(args.source/'inputs.json')['shard_size']
    for start, items in private.shard_items(questions, source_size):
        path = directory/f'part-{start//source_size:05d}.json'
        rows = checked_source(args, path) if split == 'private' else read_json(path)
        validate_base(rows, items, known)
        yield path, rows


def bundles(row, model):
    result = []
    for doc in row['docs']:
        trace = row['evidence']['documents'][doc]
        chunks = [{'id': p['rowid'], 'text': p['text'], 'score': -i}
                  for i, p in enumerate(trace['packed'])]
        result.append(build_bundle(row['question'], trace['label'], chunks,
            model.local.tokenizer, model.template, max_tokens=2048, max_passages=3))
    return result


def validate_pointwise(rows, base):
    if len(rows) != len(base):
        raise ValueError('Incomplete relevance score shard')
    for row, original in zip(rows, base):
        if row['id'] != original['id'] or row['docs'] != original['docs'] or row['base'] != digest(original):
            raise ValueError('Relevance scores belong to another comparison row')
        scores = np.asarray(row['relevance'])
        if scores.shape != (len(original['docs']),) or not np.isfinite(scores).all():
            raise ValueError('Invalid relevance scores')


def collect_pointwise(args, split):
    marker = manifest(args)
    directory = args.out/'cache/relevance'/split
    guard(directory/'resume.json', {'inputs': digest(marker), 'split': split})
    model, count, computed, started = None, 0, 0, time.time()
    try:
        for path, base in bases(args, split):
            target = directory/path.name
            if target.exists():
                validate_pointwise(read_json(target), base)
            else:
                if model is None:
                    os.environ['LEGALIR_CAUSAL_ATTN'] = 'flash_attention_2'
                    model = Reranker(args.models/'partner_qwen', max_length=2048)
                result = []
                for row in base:
                    packed = bundles(row, model)
                    scores = np.asarray(model.score_bundles(packed, batch_size=args.batch))
                    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                        raise ValueError('Invalid GPU relevance probabilities')
                    scores = np.clip(scores, 1e-6, 1-1e-6)
                    result.append({'id': row['id'], 'docs': row['docs'], 'base': digest(row),
                        'relevance': (np.log(scores)-np.log1p(-scores)).tolist(),
                        'bundles': [{'doc': d, 'sha256': b['input_sha256'], 'tokens': b['total_tokens'],
                                     'truncated': b['truncated']} for d, b in zip(row['docs'], packed)]})
                validate_pointwise(result, base)
                write_json(target, result)
                computed += len(base)
            count += len(base)
            private.progress('fulltext-relevance/'+split, count, len(records(args, split)), computed, started)
    finally:
        if model is not None:
            model.close()


def scored(args, split):
    marker = manifest(args)
    directory = args.out/'cache/relevance'/split
    if read_json(directory/'resume.json') != {'inputs': digest(marker), 'split': split}:
        raise ValueError('Relevance cache identity changed')
    result = []
    for path, base in bases(args, split):
        rows = read_json(directory/path.name)
        validate_pointwise(rows, base)
        for row, score in zip(base, rows):
            result.append({k: v for k, v in dict(row, relevance=score['relevance']).items() if k != 'evidence'})
    return result


def calibrate(args):
    rows = scored(args, 'development')
    report = dict(select(rows), inputs=digest(manifest(args)), scores=digest(rows))
    write_json(args.out/'calibration.json', report)
    print('Development choice:', report, flush=True)


def choice(args):
    value = read_json(args.out/'calibration.json')
    if value['inputs'] != digest(manifest(args)) or value['selected'] not in POLICIES:
        raise ValueError('Calibration identity changed')
    if value['scores'] != digest(scored(args, 'development')):
        raise ValueError('Development scores changed after calibration')
    return value


def validate(args):
    chosen = choice(args)
    if chosen['selected'] == BASELINE:
        write_json(args.out/'validation.json', {'deploy': False, 'evaluated': False,
            'reason': 'Development selected baseline; validation/private GPU scoring skipped.',
            'policy': BASELINE, 'inputs': digest(manifest(args)), 'calibration': digest(chosen),
            'scores': None, 'delta': None})
        print('Development retained baseline; no validation GPU run needed.', flush=True)
        return
    rows = scored(args, 'validation')
    report = dict(decision(rows, chosen['selected']), policy=chosen['selected'],
        evaluated=True, inputs=digest(manifest(args)), calibration=digest(chosen), scores=digest(rows))
    write_json(args.out/'validation.json', report)
    print('Validation:', report, flush=True)


def deployment(args):
    chosen, report = choice(args), read_json(args.out/'validation.json')
    if (report['inputs'] != digest(manifest(args)) or report['calibration'] != digest(chosen)
            or report['policy'] != chosen['selected']):
        raise ValueError('Validation identity changed')
    if not report['evaluated']:
        if chosen['selected'] != BASELINE or report['deploy'] or report['scores'] is not None:
            raise ValueError('Only the baseline can skip validation')
        return report
    if report['scores'] != digest(scored(args, 'validation')):
        raise ValueError('Validation scores changed')
    if report['deploy'] != bool(chosen['selected'] != BASELINE and report['delta'] > 1e-12):
        raise ValueError('Invalid deployment decision')
    return report


def private_scores(args):
    if deployment(args)['deploy']:
        collect_pointwise(args, 'private')
    else:
        print('No validation gain: skip private GPU scoring and retain the 0.9510 baseline.', flush=True)


def submit(args):
    report, marker = deployment(args), manifest(args)
    predictions = {}
    if report['deploy']:
        predictions = {r['id']: {'answer': predict(r, report['policy'])} for r in scored(args, 'private')}
    else:
        path = args.source/'submission-private-duo.zip'
        if file_hash(path) != SCORED_ZIP:
            raise ValueError('Scored fallback changed')
        with zipfile.ZipFile(path) as archive:
            predictions = json.loads(archive.read('submission.json'))
    v9path = args.source/'submission-private-v9.zip'
    if file_hash(v9path) != marker['source_files'][v9path.name]:
        raise ValueError('V9 fallback changed')
    with zipfile.ZipFile(v9path) as archive:
        v9 = json.loads(archive.read('submission.json'))
    stamp = dict(marker['assets'], strength=report['policy']['strength'],
        evidence={'policy': SCHEMA, 'selected': report['policy'], 'deployed': report['deploy']})
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    private.write_submission(args, v9, 'v9', known, stamp)
    private.write_submission(args, predictions, 'duo', known, stamp)
    write_json(args.out/'submission-choice.json', {'candidate_deployed': report['deploy'],
        'validation': report, 'fallback_private_score_user_reported': .9510,
        'candidate_private_score': None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'development', 'calibrate', 'validation', 'validate', 'private', 'submit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('outputs/private/private-duo-fulltext-v2-4090-fast1'))
    parser.add_argument('--duo-run', type=Path, default=Path('outputs/duo/duo-20260918T110105Z'))
    parser.add_argument('--index', type=Path, default=Path('artifacts-ranking-h100'))
    parser.add_argument('--evidence-index', type=Path, default=Path('artifacts-ranking-h100/duo-fulltext-v2.sqlite'))
    parser.add_argument('--baseline', type=Path, default=Path('analysis/private/baseline'))
    parser.add_argument('--models', type=Path, default=Path('models2'))
    parser.add_argument('--questions', type=Path, default=Path('data/private-official.json'))
    parser.add_argument('--dataset', type=Path, default=Path('analysis/fulltext-prior'))
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1 or args.out.resolve() == args.source.resolve():
        parser.error('Use a positive batch and a new output directory')
    if args.stage in ('development', 'validation'):
        if args.stage == 'validation' and choice(args)['selected'] == BASELINE:
            print('Development retained baseline; skipping validation GPU scoring.', flush=True)
        else:
            collect_base(args, args.stage)
            collect_pointwise(args, args.stage)
    else:
        {'prepare': prepare, 'calibrate': calibrate, 'validate': validate,
         'private': private_scores, 'submit': submit}[args.stage](args)


if __name__ == '__main__':
    main()
