"""One fixed CPU combination; verify cache provenance before evaluating or packing."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import agreement_097, duo_agreement, passage_combination as combination
from legalir2 import passage_order_run as development, passage_private
from legalir2.h100 import digest, guard
from legalir2.labeled_gate import summarize
from legalir2.private_infer import write_submission
from legalir2.store import file_hash, read_json, write_json

OUT = ROOT/'reports/passage-combination-v1'
BEST_SHA = '34167af0e86aa5478992320319f319c45130d3ea894fb5b161deae661a4b4fc6'
SOURCES = {
    'development': {'inputs': 'reports/passage-order-v2',
        'scores': 'reports/passage-order-gpu-completed/scores', 'questions': 429,
        'plan': '990c3ab51386ef3671d69ec60d4808db51c8342dd09dc91a40e403a3124c2e06',
        'scores_digest': 'a8d6600587beb35fce6d3cca80d327fd2433591339dc9840edb5ae5b2693e78b'},
    'private': {'inputs': 'reports/passage-private-v2',
        'scores': 'reports/passage-private-gpu-completed/scores', 'questions': 2080,
        'plan': 'dc873a0f7d3f2a385dd39e462f5ee0f79d35fb1736fd6eebfa3b5470023f43c3',
        'scores_digest': '6064681d2210a2f7ea5c3a5b5e393a13252ee67a97b1b7f2446995d55e131b60'}}


def freeze():
    marker = {'schema': 'passage-combination-cpu-v1', 'policy': combination.POLICY,
        'sources': SOURCES, 'reference_zip_sha256': BEST_SHA,
        'code': {name: file_hash(ROOT/'legalir2'/name) for name in
                 ('passage_combination.py', 'agreement_097.py', 'duo_agreement.py',
                  'labeled_gate.py', 'compare.py')},
        'script': file_hash(__file__), 'parameter_sweep': False, 'private_labels_used': False,
        'limitation': 'One fixed policy after leaderboard feedback; development is previously examined, not independent.'}
    guard(OUT/'plan.json', marker)
    return marker


def verified_rows(split):
    source = SOURCES[split]; inputs = ROOT/source['inputs']; scores = ROOT/source['scores']
    plan, prepared = read_json(inputs/'plan.json'), read_json(inputs/'prepared.json')
    if (digest(plan) != source['plan'] or prepared['plan'] != source['plan']
            or len(prepared['files']) != source['questions'] or plan['questions'] != source['questions']):
        raise ValueError('Foreign or incomplete prepared split')
    if split == 'private':
        passage_private.load(inputs)
    elif plan['code'] != development.code_identity():
        raise ValueError('Development code identity changed')
    fingerprints = {}
    names = {Path(name).name for name in prepared['files']}
    if len(names) != source['questions']:
        raise ValueError('Duplicate score filename')
    for arm in development.ARMS:
        cache = scores/arm
        marker = {'plan': source['plan'], 'model': digest(plan['models'][arm])}
        if read_json(cache/'resume.json') != marker:
            raise ValueError('Foreign score cache')
        replay = read_json(cache/'replay.json')
        if (any(replay[k] != v for k, v in marker.items())
                or replay['baseline_predictions_identical'] is not True
                or not 0 <= replay['maximum_probability_error'] <= plan['replay_tolerance']):
            raise ValueError('Unmodified-input GPU replay failed')
        if {p.name for p in cache.glob('*.json')} != names | {'resume.json', 'replay.json'}:
            raise ValueError('Missing or extra comparator scores')
        for name in sorted(names):
            fingerprints[f'{arm}/{name}'] = file_hash(cache/name)
    if digest(fingerprints) != source['scores_digest']:
        raise ValueError('Score hashes differ from completed GPU receipt')
    for name, item in passage_private.items(inputs, prepared):
        rows = []
        for arm in development.ARMS:
            path = scores/arm/name
            if file_hash(path) != fingerprints[f'{arm}/{name}']:
                raise ValueError('Score changed during evaluation')
            row = read_json(path); development.validate_result(row, item, arm); rows.append(row)
        yield item, rows


def evaluate():
    marker = freeze(); measured = []; changes = []; rejected = Counter(); eligible = 0
    for item, rows in verified_rows('development'):
        before, _ = agreement_097.predict(item['original'], item['control'])
        after, trace = combination.predict(item['original'], item['control'], *rows)
        gold = set(item['original']['gold'])
        if not gold: raise ValueError('Missing development labels')
        if after[:4] != before[:4]: raise ValueError('Protected prefix changed')
        first, second = (Fraction(len(gold & set(answer)), len(gold)) for answer in (before, after))
        measured.append((first, second, set(before) != set(after)))
        rejected.update(trace['rejected']); eligible += trace['eligible']
        if before != after:
            changes.append({'id': item['original']['id'], 'before': before, 'after': after,
                            'gold': sorted(gold), 'delta': float(second-first), 'trace': trace})
    if len(measured) != 429: raise ValueError('Incomplete development comparison')
    result = summarize(measured)
    report = {'plan': digest(marker), 'development': result, 'rejected_challengers': dict(rejected),
              'eligible_challengers': eligible, 'validation_evaluated': False,
              'promotion_gate_passed': False, 'limitation': marker['limitation']}
    write_json(OUT/'development.json', report); write_json(OUT/'development-changes.json', changes)
    print(report, flush=True)


def zip_predictions(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['submission.json'] or archive.testzip() is not None:
            raise ValueError('Unexpected or corrupt submission ZIP')
        return json.loads(archive.read('submission.json'))


def package():
    marker = freeze(); report = read_json(OUT/'development.json')
    if report['plan'] != digest(marker): raise ValueError('Development policy changed')
    # A tie may be packaged for inspection, but is not evidence of improvement.
    if report['development']['delta'] < 0:
        raise ValueError('Development regressed; do not package another speculative submission')
    best_path = ROOT/'submissions/ready/submission-best-0.9533.zip'
    if file_hash(best_path) != BEST_SHA: raise ValueError('Preserved winner differs')
    best = zip_predictions(best_path)
    receipt = read_json(ROOT/'reports/passage-private-submission-review/submission-choice.json')
    if receipt['plan'] != SOURCES['private']['plan'] or receipt['scores'] != SOURCES['private']['scores_digest']:
        raise ValueError('Private GPU receipt differs')
    passage_path = ROOT/'reports/passage-private-gpu-completed/submission-private-passage-order.zip'
    if file_hash(passage_path) != receipt['zip_sha256']: raise ValueError('GPU ZIP differs')
    passage = zip_predictions(passage_path)
    inputs = ROOT/SOURCES['private']['inputs']
    questions = read_json(inputs/'data/questions.json'); known = set(read_json(inputs/'data/known-docs.json'))
    answers = {}; changes = []; comparisons = []; rejected = Counter()
    for number, (item, rows) in enumerate(verified_rows('private'), 1):
        base, control = item['original'], item['control']; qid = base['id']
        before, _ = agreement_097.predict(base, control)
        gpu, _ = agreement_097.predict(*rows)
        if (qid in answers or before != best[qid]['answer'] or gpu != passage[qid]['answer']
                or base['question'] != questions[qid]['question']):
            raise ValueError('Current winner or GPU submission does not replay')
        after, trace = combination.predict(base, control, *rows)
        if after[:4] != before[:4] or len(set(after)) != 5 or not set(after) <= known:
            raise ValueError('Protected answers or output membership changed')
        answers[qid] = {'answer': after}; rejected.update(trace['rejected'])
        comparisons.append({'id': qid, 'baseline': base, 'control': control,
                            'ordered_original': rows[0], 'ordered_control': rows[1]})
        if before != after: changes.append({'id': qid, 'before': before, 'after': after, 'trace': trace})
        if number % 200 == 0: print('Verified CPU combination:', number, '/ 2080', flush=True)
    if len(answers) != 2080 or set(answers) != set(questions) or set(best) != set(questions):
        raise ValueError('Incomplete predictions')
    duplicate_of = [path.name for path in (ROOT/'submissions/ready').glob('*.zip')
                    if path.name != 'submission-test-passage-combination.zip'
                    and answers == zip_predictions(path)]
    out = ROOT/'submissions/private/passage-combination-v1'
    guard(out/'plan.json', marker)
    write_json(out/'changes.json', changes)
    review = {'plan': digest(marker), 'questions': 2080, 'changed_sets_vs_09533': len(changes),
              'first_four_preserved': True, 'private_score': None,
              'rejected_challengers': dict(rejected), 'duplicate_of': duplicate_of,
              'development': report['development'], 'promotion_gate_passed': False,
              'limitation': marker['limitation'], 'gpu_rerun_performed': False}
    if not changes or duplicate_of:
        review['status'] = 'No distinct submission; keep remaining attempts.'
    else:
        write_submission(SimpleNamespace(out=out, questions=inputs/'data/questions.json'), answers,
                         'passage-combination', known, {'evidence': combination.POLICY, 'source': digest(marker)})
        path = out/'submission-private-passage-combination.zip'
        if zip_predictions(path) != answers: raise ValueError('Packed prediction mismatch')
        ready = ROOT/'submissions/ready/submission-test-passage-combination.zip'
        if ready.exists() and file_hash(ready) != file_hash(path): raise ValueError('Ready ZIP already differs')
        if not ready.exists(): ready.write_bytes(path.read_bytes())
        review.update(status='Experimental candidate; not submitted or promoted.', zip_sha256=file_hash(path))
        # Small verified score records permit CPU replay without full passage inputs.
        write_json(out/'recovery-scores.json', comparisons)
        review['recovery_scores_sha256'] = file_hash(out/'recovery-scores.json')
    write_json(out/'review.json', review); print(review, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('evaluate', 'package'))
    args = parser.parse_args()
    globals()[args.stage]()
