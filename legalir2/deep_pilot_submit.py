"""User-requested private inference from the completed pilot, without validation."""
import argparse
import json
from pathlib import Path
import zipfile

from . import deep_pool as deep
from .h100 import digest, guard
from .store import file_hash, read_json, write_json

PILOT = 'private-deep-pilot-4090-v1'
INPUTS = '15817574110d6a82026fbc0ec3d2cd501620ef0994c91b08f384b55a19263783'
FUSION = '7a2cf9fba9b857e9ec2834ab80146ddc024530955bf19321ceaed510f046879c'
SCORES = '9a3d19c77f106d4da7d089767a4b39b922001af17fa7c7fcd6c537184b421eb7'


def identity(args):
    selected, model = deep.choice(args), deep.fitted(args)
    if (digest(deep.manifest(args)) != INPUTS or digest(model) != FUSION
            or selected['scores'] != SCORES or selected['selected'] != 'deep'):
        raise ValueError('Expected the exact completed 256-question pilot and its development gain')
    return {'schema': 'deep-pilot-direct-submit-v1', 'source_pilot': PILOT,
        'inputs': INPUTS, 'fusion': FUSION, 'calibration': digest(selected),
        'code': file_hash(Path(__file__)), 'validation_evaluated': False,
        'selection_basis': 'development only; user requested private inference without validation',
        'candidate_private_score': None}


def prepared(args):
    value = identity(args)
    if read_json(args.out/'direct-submit.json') != value:
        raise ValueError('Direct-submission identity changed')
    return value


def prepare(args):
    # Check all restored assets and the scored source. Never refit or recalibrate.
    deep.prepare(args)
    model = deep.fitted(args)
    if model['training'] != digest(deep.training_rows(args)):
        raise ValueError('Pilot training scores changed')
    guard(args.out/'direct-submit.json', identity(args))
    print('Verified completed pilot; skipping training, development and validation GPU work.', flush=True)


def submit(args):
    approval = prepared(args)
    marker = deep.prior.manifest(deep.baseline_args(args))
    rows = deep.compared(args, 'private')
    questions = deep.records(args, 'private')
    if (len(rows) != len(questions) or len({r['id'] for r in rows}) != len(rows)
            or {r['id'] for r in rows} != set(questions)):
        raise ValueError('Incomplete or duplicate private predictions')
    answers = {}
    for name in ('v9', 'duo'):
        path = args.source/f'submission-private-{name}.zip'
        if file_hash(path) != marker['source_files'][path.name] or (name == 'duo' and file_hash(path) != deep.prior.SCORED_ZIP):
            raise ValueError('Scored source archive changed')
        with zipfile.ZipFile(path) as archive:
            answers[name] = json.loads(archive.read('submission.json'))
    candidate = {r['id']: {'answer': deep.fusion.predict(r, 'deep')} for r in rows}
    changed = sum(set(candidate[q]['answer']) != set(answers['duo'][q]['answer']) for q in candidate)
    answers['duo'] = candidate
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    stamp = dict(marker['assets'], parameter_audit=read_json(args.out/'audit.json'),
        evidence=dict(approval, policy='deep-pilot-direct-submit-v1', deployed=True))
    for name in ('v9', 'duo'):
        deep.private.write_submission(args, answers[name], name, known, stamp)
    write_json(args.out/'submission-choice.json', dict(approval, candidate_deployed=True,
        private_inference_performed=True, questions=len(candidate), changed_answer_sets=changed,
        private_scores=digest(rows), fallback_private_score_user_reported=.9510,
        precision='bfloat16', quantization=None))
    print(f'NEW DUO SUBMISSION: {changed}/{len(candidate)} answer sets changed; V9 unchanged. Validation was skipped.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'private', 'submit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
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
    pilot_out = Path('outputs/private')/PILOT
    if args.batch != 4 or args.out.resolve() in (args.source.resolve(), args.duo_run.resolve(), pilot_out.resolve()):
        parser.error('Keep pilot batch=4 and use a separate output directory')
    if args.stage == 'private':
        prepared(args)
        deep.run_split(args, 'private')
    else:
        {'prepare': prepare, 'submit': submit}[args.stage](args)


if __name__ == '__main__':
    main()
