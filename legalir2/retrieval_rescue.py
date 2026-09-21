"""BF16 focused-evidence challenges with development selection and validation.

Only compare contenders to the frozen top five. No fabricated comparisons,
pointwise reweighting, quantization, or automatic citation promotions.
"""
import argparse
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from . import duo, fulltext_prior as prior, private_infer as private
from .focused_retrieval import FocusedEvidence
from .h100 import digest, guard
from .models import Local
from .related_documents import Relations
from .store import file_hash, read_json, write_json

SCHEMA = 'focused-citation-rescue-v1'
HEAD, EXTRA = 8, 8
POLICIES = [{'scope': 'baseline', 'threshold': 1.}] + [
    {'scope': scope, 'threshold': threshold}
    for scope in ('focused', 'citations') for threshold in (.75, .9)]
BASELINE = POLICIES[0]


def baseline_args(args):
    return SimpleNamespace(**dict(vars(args), out=args.out/'baseline'))


def identity(args):
    base = prior.manifest(baseline_args(args))
    asset = read_json(args.rescue_assets/'manifest.json')
    for name, expected in asset['files'].items():
        if file_hash(args.rescue_assets/name) != expected:
            raise ValueError('Rescue asset changed: '+name)
    return {'schema': SCHEMA, 'baseline_inputs': digest(base), 'assets': asset,
        'code': {name: file_hash(Path(__file__).with_name(name)) for name in (
            'retrieval_rescue.py', 'focused_retrieval.py', 'related_documents.py')},
        'batch': args.batch, 'head': HEAD, 'extra': EXTRA, 'policies': POLICIES,
        'precision': 'bfloat16', 'quantization': None}


def manifest(args):
    value = read_json(args.out/'inputs.json')
    if value != identity(args):
        raise ValueError('Rescue identity changed; use a new run ID')
    return value


def graph(args):
    return Relations(read_json(args.rescue_assets/'citation-graph.json'), {})


def development_base(args):
    with gzip.open(args.rescue_assets/'development-baseline.json.gz', 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def prepare(args):
    prior.prepare(baseline_args(args))
    value = identity(args)
    source = prior.manifest(baseline_args(args))
    asset = value['assets']
    expected = source['assets']['assets']['artifacts-ranking-h100/store.sqlite']
    if (asset['store_sha256'] != expected or asset['duo_inputs'] != source['assets']['duo_inputs']
            or asset['corpus_sha256'] != source['assets']['evidence']['metadata']['corpus_sha256']):
        raise ValueError('Rescue graph or development cache belongs to another corpus/model')
    items = list(prior.records(args, 'development').items())
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    if graph(args).known != known:
        raise ValueError('Citation graph document identities differ')
    rows = development_base(args)
    prior.validate_base(rows, items, known)
    if abs(float(prior.recall_vector(rows, prior.BASELINE).mean())-asset['development_recall']) > 1e-12:
        raise ValueError('Saved development baseline did not replay')
    guard(args.out/'inputs.json', value)
    write_json(args.out/'audit.json', source['assets']['parameter_audit'])
    print('Prepared: reused full-text development baseline; BF16 only, no quantization.', flush=True)


def base_parts(args, split):
    if split == 'development':
        rows = development_base(args)
        items = list(prior.records(args, split).items())
        for start in range(0, len(rows), 10):
            yield f'part-{start//10:05d}.json', rows[start:start+10], items[start:start+10]
    else:
        questions = prior.records(args, split)
        for path, rows in prior.bases(baseline_args(args), split):
            yield path.name, rows, [(r['id'], questions[r['id']]) for r in rows]


def plan(row, relations):
    original = np.asarray(row['prior'])
    score = original+.25*(np.asarray(row['ability'])-original)
    ranked = [doc for _, doc in sorted(zip(-score, row['docs']))]
    anchor, focused = ranked[:5], ranked[:HEAD]
    proposals = relations.propose(anchor, row['docs'], row['evidence']['documents'], limit=EXTRA, mode='citations')
    return {'baseline': anchor, 'focused_count': len(focused),
            'docs': focused+[p['doc'] for p in proposals], 'proposals': proposals}


def compare_anchors(local, codec, query, evidence, docs, batch):
    """Both A/B orders for every measured edge; unknown edges are not created."""
    torch = local.torch
    edges = [(i, j) for i in range(len(docs)) for j in range(min(i, 5))]
    examples = [(i, j) for a, b in edges for i, j in ((a, b), (b, a))]
    scores = []
    for start in range(0, len(examples), batch):
        ids = [codec.pair(query, evidence[docs[i]], evidence[docs[j]])
               for i, j in examples[start:start+batch]]
        inputs = codec.tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt')
        with torch.inference_mode():
            logits = local.model(**{k: v.to(local.device) for k, v in inputs.items()}, logits_to_keep=1).logits[:, -1, codec.labels].float()
            scores.extend(logits.softmax(dim=-1)[:, 0].cpu().tolist())
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite focused comparison scores')
    wins = np.full((len(docs), 5), np.nan)
    np.fill_diagonal(wins[:5], .5)
    for k, (i, j) in enumerate(edges):
        value = (scores[2*k]+1-scores[2*k+1])/2
        wins[i, j] = value
        if i < 5:
            wins[j, i] = 1-value
    return wins


def validate_scores(rows, base, relations):
    if len(rows) != len(base):
        raise ValueError('Incomplete rescue shard')
    for row, old in zip(rows, base):
        expected = plan(old, relations)
        if (row['id'] != old['id'] or row['base'] != digest(old)
                or row['question'] != old['question'] or row['gold'] != old['gold']
                or any(row[key] != value for key, value in expected.items())):
            raise ValueError('Rescue scores belong to different inputs or candidates')
        wins = np.asarray(row['wins'])
        if (wins.shape != (len(row['docs']), 5) or not np.isfinite(wins).all()
                or (wins < 0).any() or (wins > 1).any()
                or not np.allclose(wins[:5]+wins[:5].T, 1, atol=1e-6, rtol=0)):
            raise ValueError('Invalid anchor comparison scores')
        traces = row['evidence']['documents']
        if set(traces) != set(row['docs']):
            raise ValueError('Incomplete focused evidence')
        for trace in traces.values():
            import hashlib
            if (len(trace['input_ids']) > trace['budget'] or
                    hashlib.sha256(json.dumps(trace['input_ids']).encode()).hexdigest() != trace['sha256']):
                raise ValueError('Corrupt focused evidence token trace')


def collect(args, split):
    marker, relations = manifest(args), graph(args)
    directory = args.out/'cache/rescue'/split
    guard(directory/'resume.json', {'inputs': digest(marker), 'split': split})
    model = evidence = codec = None
    started, computed, count = time.time(), 0, 0
    total = len(prior.records(args, split))
    try:
        def compute(base):
            nonlocal model, evidence, codec
            if model is None:
                os.environ['LEGALIR_CAUSAL_ATTN'] = 'flash_attention_2'
                import torch
                model = Local(args.duo_run/'model', kind='causal', device='cuda', dtype=torch.bfloat16)
                if (getattr(model.model, 'is_quantized', False) or
                        any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in model.model.parameters())):
                    raise ValueError('This run requires unquantized BF16 model weights')
                codec = duo.PairTokenizer(model.tokenizer, 2048)
                evidence = FocusedEvidence(args.index, codec, args.evidence_index)
            result = []
            for row in base:
                proposed = plan(row, relations)
                encoded = evidence.encode(row['question'], proposed['docs'])
                wins = compare_anchors(model, codec, row['question'], encoded, proposed['docs'], args.batch)
                result.append(dict(proposed, id=row['id'], question=row['question'], gold=row['gold'],
                    base=digest(row), wins=wins.tolist(), evidence=deepcopy(evidence.trace)))
            return result
        for name, base, items in base_parts(args, split):
            path = directory/name
            if path.exists():
                validate_scores(read_json(path), base, relations)
            else:
                result = compute(base)
                validate_scores(result, base, relations)
                write_json(path, result)
                computed += len(base)
            count += len(base)
            private.progress('focused-rescue/'+split, count, total, computed, started)
    finally:
        if evidence is not None:
            evidence.close()
        if model is not None:
            model.close()


def scores(args, split):
    marker, relations = manifest(args), graph(args)
    directory = args.out/'cache/rescue'/split
    if read_json(directory/'resume.json') != {'inputs': digest(marker), 'split': split}:
        raise ValueError('Rescue cache identity changed')
    result = []
    for name, base, _ in base_parts(args, split):
        rows = read_json(directory/name)
        validate_scores(rows, base, relations)
        result.extend({k: v for k, v in row.items() if k != 'evidence'} for row in rows)
    return result


def predict(row, policy):
    if policy not in POLICIES:
        raise ValueError('Unknown rescue policy')
    answer = list(row['baseline'])
    if policy == BASELINE:
        return answer
    count = row['focused_count'] if policy['scope'] == 'focused' else len(row['docs'])
    wins = np.asarray(row['wins'])
    quality = wins.mean(axis=1)
    # At most one replacement, only at ranks four/five. The threshold and
    # agreement with the whole anchor must support it; citations alone cannot.
    proposals = [(quality[c]-quality[v], wins[c, v], row['docs'][c], v)
        for c in range(5, count) for v in (3, 4)
        if wins[c, v] >= policy['threshold'] and quality[c] > quality[v]+.05]
    if proposals:
        _, _, candidate, victim = max(proposals, key=lambda p: (p[0], p[1], p[2], p[3]))
        answer[victim] = candidate
    return answer


def recalls(rows, policy):
    if not rows or any(not row['gold'] for row in rows):
        raise ValueError('Labeled rows required for calibration/validation')
    return np.asarray([len(set(predict(row, policy)) & set(row['gold']))/len(set(row['gold'])) for row in rows])


def calibrate(args):
    rows = scores(args, 'development')
    trials = [{'policy': p, 'recall': float(recalls(rows, p).mean())} for p in POLICIES]
    best = max(trials, key=lambda t: (t['recall'], t['policy'] == BASELINE,
        t['policy']['threshold'], t['policy']['scope'] == 'focused'))
    value = {'selected': best['policy'], 'trials': trials, 'inputs': digest(manifest(args)), 'scores': digest(rows)}
    write_json(args.out/'calibration.json', value)
    print('Development choice:', value, flush=True)


def choice(args):
    value = read_json(args.out/'calibration.json')
    if (value['inputs'] != digest(manifest(args)) or value['selected'] not in POLICIES
            or value['scores'] != digest(scores(args, 'development'))):
        raise ValueError('Calibration identity or scores changed')
    return value


def decision(rows, policy):
    before, after = recalls(rows, BASELINE), recalls(rows, policy)
    diff = after-before
    rng = np.random.default_rng(2026)
    ci = np.quantile([rng.choice(diff, len(diff), replace=True).mean() for _ in range(5000)], [.025, .975])
    return {'baseline_recall': float(before.mean()), 'candidate_recall': float(after.mean()),
            'delta': float(diff.mean()), 'wins': int((diff > 0).sum()), 'losses': int((diff < 0).sum()),
            'ci95': ci.tolist(), 'deploy': bool(policy != BASELINE and diff.mean() > 1e-12)}


def validate(args):
    selected = choice(args)
    report = {'inputs': digest(manifest(args)), 'calibration': digest(selected), 'policy': selected['selected'],
              'limitation': 'Historically reused validation; gain does not guarantee private improvement.'}
    if selected['selected'] == BASELINE:
        report.update(evaluated=False, deploy=False, scores=None, delta=None,
                      reason='Development retained baseline; skip validation/private GPU scoring.')
    else:
        rows = scores(args, 'validation')
        report.update(decision(rows, selected['selected']), evaluated=True, scores=digest(rows))
    write_json(args.out/'validation.json', report)
    print('Validation:', report, flush=True)


def deployment(args):
    selected, report = choice(args), read_json(args.out/'validation.json')
    if (report['inputs'] != digest(manifest(args)) or report['calibration'] != digest(selected)
            or report['policy'] != selected['selected']):
        raise ValueError('Validation identity changed')
    if report['evaluated']:
        rows = scores(args, 'validation')
        if report['scores'] != digest(rows) or any(report[k] != v for k, v in decision(rows, selected['selected']).items()):
            raise ValueError('Validation results changed')
    elif selected['selected'] != BASELINE or report['deploy'] or report['scores'] is not None:
        raise ValueError('Only the baseline may skip validation')
    return report


def submit(args):
    report = deployment(args)
    marker = prior.manifest(baseline_args(args))
    if report['deploy']:
        answers = {row['id']: {'answer': predict(row, report['policy'])} for row in scores(args, 'private')}
    else:
        if file_hash(args.source/'submission-private-duo.zip') != prior.SCORED_ZIP:
            raise ValueError('Scored fallback ZIP changed')
        with zipfile.ZipFile(args.source/'submission-private-duo.zip') as archive:
            answers = json.loads(archive.read('submission.json'))
    v9path = args.source/'submission-private-v9.zip'
    if file_hash(v9path) != marker['source_files'][v9path.name]:
        raise ValueError('V9 fallback ZIP changed')
    with zipfile.ZipFile(v9path) as archive:
        v9 = json.loads(archive.read('submission.json'))
    known = graph(args).known
    stamp = dict(marker['assets'], evidence={'policy': SCHEMA, 'selected': report['policy'], 'deployed': report['deploy']})
    private.write_submission(args, v9, 'v9', known, stamp)
    private.write_submission(args, answers, 'duo', known, stamp)
    write_json(args.out/'submission-choice.json', {'candidate_deployed': report['deploy'], 'validation': report,
        'fallback_private_score_user_reported': .9510, 'candidate_private_score': None,
        'precision': 'bfloat16', 'quantization': None})


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
    parser.add_argument('--rescue-assets', type=Path, default=Path('analysis/retrieval-rescue'))
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1 or args.out.resolve() in (args.source.resolve(), args.duo_run.resolve()):
        parser.error('Use a positive batch and a separate output directory')
    if args.stage == 'development':
        collect(args, 'development')
    elif args.stage == 'validation':
        if choice(args)['selected'] == BASELINE:
            print('Development retained baseline; skipping validation GPU scoring.', flush=True)
        else:
            prior.collect_base(baseline_args(args), 'validation')
            collect(args, 'validation')
    elif args.stage == 'private':
        if deployment(args)['deploy']:
            collect(args, 'private')
        else:
            print('No validated gain; retain the scored private answers.', flush=True)
    else:
        {'prepare': prepare, 'calibrate': calibrate, 'validate': validate, 'submit': submit}[args.stage](args)


if __name__ == '__main__':
    main()
