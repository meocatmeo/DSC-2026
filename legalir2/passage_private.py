"""User-requested private passage-order inference, using the frozen 0.98 rule."""
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from . import agreement_097, deadline_agreement, duo, fulltext_prior, passage_order
from . import passage_order_run as development
from .h100 import digest, guard
from .private_infer import write_submission
from .store import file_hash, read_json, write_json

SCHEMA = 'passage-order-private-strict-v1'
BEST_SHA = 'd8ea76dd918331fbd21067dfdaed31106d8f99de2d9b852e26c34792c389cae8'
CURRENT_BEST_SHA = '34167af0e86aa5478992320319f319c45130d3ea894fb5b161deae661a4b4fc6'
ARMS = development.ARMS


def code_identity():
    result = development.code_identity()
    for name in ('passage_private.py', 'deadline_agreement.py', 'agreement_097.py'):
        result[name] = file_hash(Path(__file__).with_name(name))
    return result


def load(out):
    plan, prepared = read_json(out/'plan.json'), read_json(out/'prepared.json')
    if (plan['schema'] != SCHEMA or plan['policy'] != passage_order.POLICY
            or plan['agreement'] != deadline_agreement.POLICIES['fifth-strict']
            or plan['primary_agreement'] != agreement_097.POLICY
            or plan['code'] != code_identity() or plan['packages'] != development.PACKAGES
            or plan['questions'] != 2080 or prepared['plan'] != digest(plan)
            or len(prepared['files']) != 2080):
        raise ValueError('Private prepared experiment identity differs')
    for name, expected in plan['data_files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(out/name) != expected:
            raise ValueError('Private data changed: '+name)
    return plan, prepared


def items(out, prepared):
    seen = set()
    for name, expected in prepared['files'].items():
        if not name.startswith('questions/') or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe prepared question filename')
        if file_hash(out/name) != expected:
            raise ValueError('Prepared question changed: '+name)
        row = read_json(out/name)
        if row['original']['id'] in seen:
            raise ValueError('Duplicated private question')
        seen.add(row['original']['id'])
        edges = passage_order.changed_edges(row['original']['docs'], row['before'], row['after'])
        if row['edges'] != [list(e) for e in edges]:
            raise ValueError('Changed-edge inventory differs')
        yield Path(name).name, row


def replay(local, codec, sample, arm, plan, batch):
    reference = sample['original' if arm == ARMS[0] else 'control']
    probabilities = duo.compare_documents(local, codec, reference['question'],
        {d: [ids] for d, ids in sample['before'].items()}, reference['docs'], batch)
    error = float(np.max(np.abs(probabilities-np.asarray(reference['probabilities']))))
    result = dict(reference, probabilities=probabilities.tolist(),
                  ability=fulltext_prior.fit(reference['prior'], probabilities).tolist())
    if error > plan['replay_tolerance'] or fulltext_prior.predict(result, fulltext_prior.BASELINE) != fulltext_prior.predict(reference, fulltext_prior.BASELINE):
        raise ValueError(f'Unchanged private control replay differs: probability error={error}')
    return {'plan': digest(plan), 'model': digest(plan['models'][arm]), 'question': reference['id'],
            'maximum_probability_error': error, 'baseline_predictions_identical': True}


def score(args):
    from .models import Local
    import torch
    plan, prepared = load(args.out)
    runtime = {name: version(name) for name in plan['packages']}
    if runtime != plan['packages'] or args.batch != plan['batch']:
        raise ValueError('Runtime or batch differs from frozen plan')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Unquantized BF16 CUDA GPU required')
    for arm in ARMS:
        development.verify_checkpoint(args.models/arm, plan['models'][arm])
    os.environ['LEGALIR_CAUSAL_ATTN'] = plan['attention']
    sample = next(items(args.out, prepared))[1]
    for arm in ARMS:
        cache = args.out/'scores'/arm
        guard(cache/'resume.json', {'plan': digest(plan), 'model': digest(plan['models'][arm])})
        local = None; computed = 0; started = time.monotonic()
        try:
            for number, (name, item) in enumerate(items(args.out, prepared), 1):
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
                            report = replay(local, codec, sample, arm, plan, args.batch)
                            report.update(packages=runtime, gpu=torch.cuda.get_device_name(0))
                            write_json(cache/'replay.json', report)
                        probability = passage_order.rescore(local, codec, original, item['after'],
                            [tuple(edge) for edge in item['edges']], args.batch)
                        result.update(probabilities=probability.tolist(),
                            ability=fulltext_prior.fit(original['prior'], probability).tolist())
                    result['source'] = digest(item)
                    development.validate_result(result, item, arm)
                    write_json(target, result); computed += 1
                development.validate_result(result, item, arm)
                if number % 10 == 0 or number == plan['questions']:
                    seconds = (time.monotonic()-started)/max(1, computed)
                    print(f'private/passage-order/{arm}: {number}/{plan["questions"]}; '
                          f'{seconds:.2f}s/question; current-arm eta={(plan["questions"]-number)*seconds/60:.1f} min', flush=True)
        finally:
            if local is not None:
                local.close()


def scored(out, plan, prepared):
    for arm in ARMS:
        cache = out/'scores'/arm
        marker = {'plan': digest(plan), 'model': digest(plan['models'][arm])}
        if read_json(cache/'resume.json') != marker:
            raise ValueError('Foreign comparator score cache')
        check = read_json(cache/'replay.json')
        if (any(check[k] != v for k, v in marker.items()) or check['baseline_predictions_identical'] is not True
                or not 0 <= check['maximum_probability_error'] <= plan['replay_tolerance']):
            raise ValueError('Missing or failed unchanged-input replay')
    for name, item in items(out, prepared):
        rows = []
        for arm in ARMS:
            row = read_json(out/'scores'/arm/name)
            development.validate_result(row, item, arm); rows.append(row)
        yield name, item, rows


def submit(args):
    plan, prepared = load(args.out)
    if file_hash(args.out/'data/best-09525.zip') != BEST_SHA:
        raise ValueError('Best reference ZIP differs')
    with zipfile.ZipFile(args.out/'data/best-09525.zip') as archive:
        best = json.loads(archive.read('submission.json'))
    if file_hash(args.out/'data/best-09533.zip') != CURRENT_BEST_SHA:
        raise ValueError('Current 0.9533 reference ZIP differs')
    with zipfile.ZipFile(args.out/'data/best-09533.zip') as archive:
        current_best = json.loads(archive.read('submission.json'))
    questions = read_json(args.out/'data/questions.json'); known = set(read_json(args.out/'data/known-docs.json'))
    predictions = {}; primary_predictions = {}; changes = []; primary_changes = []; fingerprints = {}
    for name, item, rows in scored(args.out, plan, prepared):
        qid = item['original']['id']
        before, _ = deadline_agreement.predict(item['original'], item['control'], 'fifth-strict')
        if qid in predictions or before != best[qid]['answer'] or item['original']['question'] != questions[qid]['question']:
            raise ValueError('Private question or best prediction replay differs')
        answer, trace = deadline_agreement.predict(*rows, 'fifth-strict')
        predictions[qid] = {'answer': answer}
        current, _ = agreement_097.predict(item['original'], item['control'])
        if current != current_best[qid]['answer']:
            raise ValueError('Current 0.9533 predictions do not replay')
        primary, primary_trace = agreement_097.predict(*rows)
        primary_predictions[qid] = {'answer': primary}
        if set(current) != set(primary):
            primary_changes.append({'id': qid, 'before': current, 'after': primary, 'trace': primary_trace})
        if set(before) != set(answer):
            changes.append({'id': qid, 'before': before, 'after': answer, 'trace': trace})
        for arm in ARMS:
            fingerprints[f'{arm}/{name}'] = file_hash(args.out/'scores'/arm/name)
    if len(predictions) != 2080 or set(predictions) != set(questions):
        raise ValueError('Complete 2080 predictions required; partial submission refused')
    args.questions = args.out/'data/questions.json'
    write_submission(args, predictions, 'passage-order-098', known,
        {'evidence': {'policy': passage_order.POLICY}, 'plan': digest(plan), 'agreement': plan['agreement']})
    write_submission(args, primary_predictions, 'passage-order', known,
        {'evidence': {'policy': passage_order.POLICY}, 'plan': digest(plan), 'agreement': plan['primary_agreement']})
    choice = {'plan': digest(plan), 'scores': digest(fingerprints), 'questions': 2080,
              'zip_sha256': file_hash(args.out/'submission-private-passage-order.zip'),
              'primary_threshold': .97, 'reference_score': .9533,
              'changed_sets_vs_09533': len(primary_changes),
              'alternate_zip_sha256': file_hash(args.out/'submission-private-passage-order-098.zip'),
              'changed_sets_vs_09525': len(changes), 'private_score': None,
              'user_requested_unvalidated_submission': True, 'promotion_gate_passed': False,
              'validation_evaluated': False, 'private_inference_performed': True}
    write_json(args.out/'changes-098.json', changes); write_json(args.out/'changes.json', primary_changes)
    write_json(args.out/'submission-choice.json', choice)
    print(choice, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'score', 'submit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=Path('models-passage-order'))
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--max-questions', type=int, default=0)
    args = parser.parse_args()
    if args.batch < 1 or args.max_questions < 0:
        parser.error('Invalid batch or question count')
    if args.stage == 'check':
        plan, prepared = load(args.out)
        print('Verified', sum(1 for _ in items(args.out, prepared)), 'private questions.')
    else:
        globals()[args.stage](args)


if __name__ == '__main__':
    main()
