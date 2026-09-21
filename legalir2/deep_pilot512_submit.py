"""User-requested private inference from the completed 512-question pilot, without validation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

from . import deep_pool as deep
from .h100 import digest, guard
from .store import file_hash, read_json, write_json

PILOT = 'private-deep-pilot512-4090-v1'
INPUTS = '96e3ad91f570c8345b1d8a629c4c94ae73fef015bbeee9cc8ea7ddab7aed1f80'
FUSION = '84787ad63ad219de43be5b28b80ec32556105da5a186aa1c3099b8cc694285d7'
SCORES = '1b3480789ed71db590ec4e9bcfed430ced472fe7450c3de5fa14a7ffb272150b'
REUSE_INPUTS = '15817574110d6a82026fbc0ec3d2cd501620ef0994c91b08f384b55a19263783'


def identity(args):
    selected, model = deep.choice(args), deep.fitted(args)
    if (digest(deep.manifest(args)) != INPUTS or digest(model) != FUSION
            or selected['scores'] != SCORES or selected['selected'] != 'deep'):
        raise ValueError('Expected the exact completed 512-question pilot and its development gain')
    return {'schema': 'deep-pilot512-direct-submit-v1', 'source_pilot': PILOT,
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


def import_private(args):
    """Raw reranker scores do not depend on the learned fusion weights."""
    old = SimpleNamespace(**dict(vars(args), out=args.reuse_out, deep_assets=Path('analysis/deep-pool')))
    source, target = deep.manifest(old), deep.manifest(args)
    if digest(source) != REUSE_INPUTS:
        raise ValueError('Private reuse source is not the frozen 256-question pilot')
    if {k: v for k, v in source.items() if k != 'assets'} != {k: v for k, v in target.items() if k != 'assets'}:
        raise ValueError('Private raw-score settings differ; cannot reuse')
    source_dir = args.reuse_out/'cache/deep/private'
    source_parts = list(source_dir.glob('part-*.json'))
    if (source_dir/'resume.json').exists():
        if read_json(source_dir/'resume.json') != {'inputs': REUSE_INPUTS, 'split': 'private'}:
            raise ValueError('Private reuse cache identity differs')
    elif source_parts:
        raise ValueError('Private reuse shards have no identity marker')
    directory = args.out/'cache/deep/private'
    guard(directory/'resume.json', {'inputs': digest(target), 'split': 'private'})
    signature = read_json(args.duo_run/'inputs.json')
    v9 = {r['id']: r for r in deep.v9_rows(args, 'private')}
    original = {r['id']: r for r in deep.base_rows(args, 'private')}
    questions = deep.records(args, 'private')
    expected_names, reused = set(), 0
    for start, items in deep.private.shard_items(questions, 5):
        name = f'part-{start//5:05d}.json'
        expected_names.add(name)
        path = source_dir/name
        if not path.exists():
            continue
        rows = read_json(path)
        deep.validate_raw(rows, items, v9, set(signature['known_docs']), np.asarray(signature['weights']), original)
        destination = directory/name
        if destination.exists():
            # A resumed 512 run may have computed this shard itself before the
            # matching 256 shard reached cloud backup. Keep its valid scores.
            deep.validate_raw(read_json(destination), items, v9, set(signature['known_docs']), np.asarray(signature['weights']), original)
        else:
            write_json(destination, rows)
            reused += len(rows)
    if any(p.name not in expected_names for p in source_parts):
        raise ValueError('Unexpected private reuse shard')
    write_json(args.out/'private-reuse.json', {'source_inputs': REUSE_INPUTS,
        'source_shards_available': len(source_parts), 'newly_imported_questions': reused,
        'comparison_reuse': False, 'limitation': 'Only completed uploaded raw shards can be reused.'})
    print(f'Imported {reused} private raw scores from the 256 run. Only missing raw scores will be computed.', flush=True)


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
        evidence=dict(approval, policy='deep-pilot512-direct-submit-v1', deployed=True))
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
    parser.add_argument('--deep-assets', type=Path, default=Path('analysis/deep-pool512'))
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--reuse-out', type=Path, default=Path('outputs/reuse/deep-pilot256-private'))
    args = parser.parse_args()
    pilot_out = Path('outputs/private')/PILOT
    if args.batch != 4 or args.out.resolve() in (args.source.resolve(), args.duo_run.resolve(), pilot_out.resolve(),
            args.reuse_out.resolve(), Path('outputs/private/private-deep-pilot-submit-4090-v1').resolve(),
            Path('outputs/private/private-deep-pilot-4090-v1').resolve()):
        parser.error('Keep pilot batch=4 and use a separate output directory')
    if args.stage == 'private':
        prepared(args)
        import_private(args)
        deep.run_split(args, 'private')
    else:
        {'prepare': prepare, 'submit': submit}[args.stage](args)


if __name__ == '__main__':
    main()
