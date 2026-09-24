"""Prepare, score and evaluate a fixed development-only passage-order ablation.

No training, policy search, private inference, or submission stage. Preparation
reuses verified deep-run passage scores for the original baseline pool only.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import os
import time

import numpy as np

from . import duo, duo_agreement, fulltext_prior, passage_order
from .h100 import digest, guard
from .labeled_gate import summarize
from .private_infer import validate_comparison_fit
from .store import read_json, write_json, file_hash

SCHEMA = 'passage-order-development-v1'
ARMS = ('original-duo', 'retrained-control')
PACKAGES = {'torch': '2.7.1+cu128', 'transformers': '4.51.3', 'tokenizers': '0.21.4',
            'flash-attn': '2.8.3', 'numpy': '2.2.6', 'scipy': '1.15.3'}


def code_identity():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('passage_order.py', 'passage_order_run.py', 'duo.py', 'duo_agreement.py',
             'fulltext_prior.py', 'models.py', 'private_infer.py', 'labeled_gate.py',
             'compare.py', 'store.py', 'h100.py')}


def compact(row):
    return {k: deepcopy(row[k]) for k in ('id', 'question', 'gold', 'docs', 'prior',
                                         'ability', 'probabilities')}


def verify_checkpoint(directory, expected=None):
    record = read_json(directory/'duo.json')
    if record['max_length'] != 2048 or (expected is not None and record != expected):
        raise ValueError('Unexpected comparator checkpoint')
    if not record['files'] or 'model.safetensors' not in record['files']:
        raise ValueError('Incomplete checkpoint file inventory')
    for name, expected_hash in record['files'].items():
        if Path(name).name != name or file_hash(directory/name) != expected_hash:
            raise ValueError('Checkpoint file changed: '+name)
    return record


def prepare(args):
    from transformers import AutoTokenizer
    root = args.workspace
    source = root/'reports/append-only-deep-v1'
    audit = root/'reports/retrain-private-audit/labeled'
    bases = root/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    checkpoints = {arm: verify_checkpoint(args.models/arm) for arm in ARMS}
    # Both comparator checkpoints must interpret every pre-encoded input equally.
    tokenizer_names = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                       'added_tokens.json', 'special_tokens_map.json')
    for name in tokenizer_names:
        if checkpoints[ARMS[0]]['files'].get(name) != checkpoints[ARMS[1]]['files'].get(name):
            raise ValueError('Comparator tokenizers differ: '+name)
    codec = duo.PairTokenizer(AutoTokenizer.from_pretrained(args.models/ARMS[0],
        local_files_only=True, padding_side='left'), 2048)
    calibration = read_json(source/'source/calibration.json')
    expanded = [r for p in sorted((source/'source/cache/comparisons/development').glob('part-*.json'))
                for r in read_json(p)]
    if len(expanded) != 429 or digest(expanded) != calibration['scores']:
        raise ValueError('Expanded comparison cache does not match calibration')
    by_id = {r['id']: r for r in expanded}
    if len(by_id) != 429:
        raise ValueError('Duplicate expanded question IDs')
    original, retrained, old_evidence = [], [], []
    for path in sorted(bases.glob('part-*.json')):
        original.extend(read_json(path))
        retrained.extend(read_json(audit/'cache/control/development'/path.name))
        old_evidence.extend(read_json(audit/'cache/old-evidence-control/development'/path.name))
    if any(len(rows) != 429 for rows in (original, retrained, old_evidence)) or digest(
            [original, old_evidence, retrained]) != read_json(audit/'screen.json')['scores']:
        raise ValueError('Original development source identity differs')
    plan = {'schema': SCHEMA, 'policy': passage_order.POLICY, 'agreement': duo_agreement.POLICY,
        'code': code_identity(), 'models': checkpoints, 'source_scores': calibration['scores'],
        'baseline_scores': read_json(audit/'screen.json')['scores'], 'questions': 429,
        'attention': 'flash_attention_2', 'dtype': 'bfloat16', 'batch': args.batch,
        'packages': PACKAGES, 'replay_tolerance': 1e-4,
        'limitation': 'Previously examined development; no independent holdout or private score claim.'}
    guard(args.out/'plan.json', plan)
    base_by_id = {r['id']: (r, n) for r, n in zip(original, retrained)}
    seen = set(); files = {}; changed_docs = total_docs = changed_questions = edges = all_edges = 0
    for path in sorted((source/'raw').glob('part-*.json')):
        for raw in read_json(path):
            qid = raw['id']
            if qid in seen or qid not in base_by_id or digest(raw) != by_id[qid]['raw']:
                raise ValueError('Duplicated, unexpected or changed raw passage-score row')
            seen.add(qid)
            b, n = base_by_id[qid]
            if by_id[qid]['base'] != digest(b) or n['base'] != digest(b) or any(
                    b[k] != n[k] or b[k] != raw[k] for k in ('id', 'question', 'gold')):
                raise ValueError('Baseline and passage-score identities differ')
            if any(b[k] != n[k] for k in ('docs', 'prior')) or raw['retained'] != b['docs']:
                raise ValueError('Baseline pool or prior differs')
            for row in (b, n):
                duo_agreement.scores(row); validate_comparison_fit(row)
            before, after, ordering = {}, {}, {}
            for doc in b['docs']:
                trace = b['evidence']['documents'][doc]
                if trace != raw['evidence']['documents'][doc]:
                    raise ValueError('Saved passage scores refer to different evidence')
                reordered = passage_order.reorder(trace, raw['tables'][0][doc], codec)
                before[doc], after[doc] = trace['input_ids'], reordered['input_ids']
                ordering[doc] = reordered['ordering']
            measured = passage_order.changed_edges(b['docs'], before, after)
            item = {'original': compact(b), 'control': compact(n), 'before': before, 'after': after,
                    'ordering': ordering, 'raw': digest(raw), 'base': digest(b),
                    'edges': [list(pair) for pair in measured]}
            filename = f'questions/{len(files):05d}.json'
            guard(args.out/filename, item)
            files[filename] = file_hash(args.out/filename)
            changed = sum(before[d] != after[d] for d in b['docs'])
            changed_docs += changed; total_docs += len(b['docs']); changed_questions += bool(changed)
            edges += len(measured); all_edges += len(b['docs'])*(len(b['docs'])-1)//2
        print(f'Prepared {len(files)}/429 questions.', flush=True)
    if seen != set(base_by_id):
        raise ValueError('All 429 development raw score rows are required')
    report = {'questions': len(seen), 'documents': total_docs, 'changed_documents': changed_docs,
              'changed_questions': changed_questions, 'pairs_to_rescore_per_model': edges,
              'pairs_total_per_model': all_edges, 'pairs_reused_per_model': all_edges-edges,
              'new_gpu_predictions': False, 'private_inference': False}
    guard(args.out/'prepared.json', {'plan': digest(plan), 'files': files, 'report': report})
    print(report, flush=True)


def load(args):
    plan, prepared = (read_json(args.out/name) for name in ('plan.json', 'prepared.json'))
    if (plan['schema'] != SCHEMA or plan['policy'] != passage_order.POLICY
            or plan['agreement'] != duo_agreement.POLICY or plan['code'] != code_identity()
            or plan['packages'] != PACKAGES
            or prepared['plan'] != digest(plan) or len(prepared['files']) != plan['questions']):
        raise ValueError('Prepared experiment identity differs; use a separate run')
    items = []; seen = set()
    for filename, expected in prepared['files'].items():
        if Path(filename).is_absolute() or '..' in Path(filename).parts:
            raise ValueError('Unsafe prepared path')
        if file_hash(args.out/filename) != expected:
            raise ValueError('Prepared question changed: '+filename)
        row = read_json(args.out/filename)
        if row['original']['id'] in seen:
            raise ValueError('Duplicate prepared question')
        seen.add(row['original']['id'])
        expected_edges = passage_order.changed_edges(row['original']['docs'], row['before'], row['after'])
        if row['edges'] != [list(pair) for pair in expected_edges]:
            raise ValueError('Changed-edge inventory differs')
        items.append((Path(filename).name, row))
    return plan, items


def validate_result(result, item, arm):
    source = item['original' if arm == ARMS[0] else 'control']
    if result['source'] != digest(item) or any(result[k] != source[k] for k in
            ('id', 'question', 'gold', 'docs', 'prior')):
        raise ValueError('Scored result identity differs')
    duo_agreement.scores(result); validate_comparison_fit(result)
    changed = {tuple(edge) for edge in item['edges']}
    for i in range(len(source['docs'])):
        for j in range(i, len(source['docs'])):
            if (i, j) not in changed:
                for a, b in ((i, j), (j, i)):
                    if result['probabilities'][a][b] != source['probabilities'][a][b]:
                        raise ValueError('An unchanged comparison was modified')


def score(args):
    from importlib.metadata import version
    from .models import Local
    plan, items = load(args)
    runtime = {name: version(name) for name in plan['packages']}
    if runtime != plan['packages']:
        raise RuntimeError(f'Runtime differs from the frozen scoring environment: {runtime}')
    for arm in ARMS:
        verify_checkpoint(args.models/arm, plan['models'][arm])
    if args.batch != plan['batch']:
        raise ValueError('Inference batch differs from frozen plan')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This experiment requires a BF16-capable CUDA GPU')
    os.environ['LEGALIR_CAUSAL_ATTN'] = plan['attention']
    for arm in ARMS:
        cache = args.out/'scores'/arm
        guard(cache/'resume.json', {'plan': digest(plan), 'model': digest(plan['models'][arm])})
        local = None; computed = 0; started = time.monotonic()
        try:
            for number, (name, item) in enumerate(items, 1):
                if args.max_questions and number > args.max_questions:
                    break
                target = cache/name
                if target.exists():
                    result = read_json(target)
                else:
                    original = item['original' if arm == ARMS[0] else 'control']
                    result = deepcopy(original)
                    if item['edges']:
                        if local is None:
                            local = Local(args.models/arm, kind='causal', device='cuda', dtype=torch.bfloat16)
                            if getattr(local.model, 'is_quantized', False) or any(
                                    p.is_floating_point() and p.dtype != torch.bfloat16 for p in local.model.parameters()):
                                raise ValueError('Unquantized BF16 required')
                            codec = duo.PairTokenizer(local.tokenizer, 2048)
                            # Check an unmodified question on this actual device before
                            # mixing saved comparisons with newly measured edges.
                            sample = items[0][1]
                            reference = sample['original' if arm == ARMS[0] else 'control']
                            repeated = duo.compare_documents(local, codec, reference['question'],
                                {d: [ids] for d, ids in sample['before'].items()}, reference['docs'], args.batch)
                            error = float(np.max(np.abs(repeated-np.asarray(reference['probabilities']))))
                            replayed = deepcopy(reference)
                            replayed.update(probabilities=repeated.tolist(),
                                ability=fulltext_prior.fit(reference['prior'], repeated).tolist())
                            if error > plan['replay_tolerance'] or fulltext_prior.predict(replayed,
                                    fulltext_prior.BASELINE) != fulltext_prior.predict(reference, fulltext_prior.BASELINE):
                                raise ValueError(f'Unmodified control replay differs: max probability error={error}')
                            write_json(cache/'replay.json', {'plan': digest(plan), 'model': digest(plan['models'][arm]),
                                'question': reference['id'], 'maximum_probability_error': error,
                                'packages': runtime, 'gpu': torch.cuda.get_device_name(0),
                                'baseline_predictions_identical': True})
                        probability = passage_order.rescore(local, codec, original, item['after'],
                            [tuple(edge) for edge in item['edges']], args.batch)
                        result.update(probabilities=probability.tolist(),
                            ability=fulltext_prior.fit(original['prior'], probability).tolist())
                    result['source'] = digest(item)
                    validate_result(result, item, arm)
                    write_json(target, result); computed += 1
                validate_result(result, item, arm)
                if number % 10 == 0 or number == len(items):
                    print(f'passage-order/{arm}: {number}/{len(items)}; computed={computed}; '
                          f'elapsed={(time.monotonic()-started)/60:.1f} min', flush=True)
        finally:
            if local is not None:
                local.close()


def evaluate(args):
    plan, items = load(args)
    results = []; changes = []; fingerprints = {}
    for name, item in items:
        scored = []
        for arm in ARMS:
            cache = args.out/'scores'/arm
            if read_json(cache/'resume.json') != {'plan': digest(plan), 'model': digest(plan['models'][arm])}:
                raise ValueError('Scored experiment marker differs')
            replay = read_json(cache/'replay.json')
            if (replay['plan'] != digest(plan) or replay['model'] != digest(plan['models'][arm])
                    or not replay['baseline_predictions_identical']
                    or not 0 <= replay['maximum_probability_error'] <= plan['replay_tolerance']):
                raise ValueError('Unmodified control replay has not passed')
            row = read_json(cache/name); validate_result(row, item, arm); scored.append(row)
            fingerprints[f'{arm}/{name}'] = file_hash(cache/name)
        before, _ = duo_agreement.predict(item['original'], item['control'])
        after, _ = duo_agreement.predict(*scored)
        gold = set(item['original']['gold'])
        if not gold:
            raise ValueError('Development labels required')
        first, second = (Fraction(len(gold & set(pred)), len(gold)) for pred in (before, after))
        results.append((first, second, set(before) != set(after)))
        if set(before) != set(after):
            changes.append({'id': item['original']['id'], 'before': before, 'after': after,
                            'gold': sorted(gold), 'delta': float(second-first)})
    report = {'plan': digest(plan), 'scores': digest(fingerprints), 'development': summarize(results),
              'limitation': plan['limitation'], 'deploy': False, 'private_inference': False,
              'meaning': 'Evidence-order ablation only. No automatic promotion or submission.'}
    write_json(args.out/'evaluation.json', report); write_json(args.out/'changes.json', changes)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'score', 'evaluate'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=Path('.'))
    parser.add_argument('--models', type=Path,
        default=Path('releases/handoff/20260922-best-09523-v1/model-vault'))
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--max-questions', type=int, default=0, help='Score smoke test; evaluate still requires all 429.')
    args = parser.parse_args()
    if args.batch < 1 or args.max_questions < 0:
        parser.error('Batch must be positive; max-questions cannot be negative')
    frozen = (args.workspace/'releases/handoff').resolve()
    if args.out.resolve().is_relative_to(frozen) or args.out.resolve().is_relative_to(args.models.resolve()):
        parser.error('Use a separate experimental output directory')
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
