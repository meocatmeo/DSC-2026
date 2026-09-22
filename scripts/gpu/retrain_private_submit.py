"""User-requested private inference from the saved rebuilt-evidence comparator."""
import argparse
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

# This additive tool lives outside the immutable training release.
sys.path.insert(0, str(Path.cwd()))
from legalir2 import fulltext_retrain as retrain
from legalir2.h100 import digest, guard
from legalir2.store import file_hash, read_json, write_json


def options(out):
    return SimpleNamespace(out=out, batch=4, training=Path('analysis/fulltext-training'),
        dataset=Path('analysis/fulltext-prior'), source=Path('outputs/private/private-duo-fulltext-v2-4090-fast1'),
        duo_run=Path('outputs/duo/duo-20260918T110105Z'), models=Path('models2'),
        index=Path('artifacts-ranking-h100'), evidence_index=Path('artifacts-ranking-h100/duo-fulltext-v2.sqlite'),
        baseline=Path('analysis/private/baseline'), questions=Path('data/private-official.json'))


def copy_verified(source, target):
    expected = file_hash(source)
    if target.exists():
        if file_hash(target) != expected:
            raise ValueError('Existing private-run input differs: '+str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.tmp')
    shutil.copyfile(source, temporary)
    if file_hash(temporary) != expected:
        raise ValueError('Copy checksum mismatch: '+str(source))
    temporary.replace(target)


def prepare(source_out, out):
    if source_out.resolve() == out.resolve() or source_out.resolve() in out.resolve().parents:
        raise ValueError('Use a separate output directory outside the training run')
    original, args = options(source_out), options(out)
    marker = retrain.manifest(original)
    directory, signature = retrain.arm_inputs(original, 'control')
    model = retrain.duo.verify_model(directory/'model', signature)
    source_gate = read_json(source_out/'gate.json')
    if source_gate.get('inputs') != digest(marker):
        raise ValueError('Original gate does not match the training run')
    files = [source_out/'inputs.json', source_out/'reference/inputs.json', directory/'inputs.json']
    files += [p for p in (directory/'model').rglob('*') if p.is_file()]
    if (directory/'audit.json').is_file():
        files.append(directory/'audit.json')
    for path in files:
        copy_verified(path, out/path.relative_to(source_out))
    guard(out/'source-gate.json', source_gate)
    guard(out/'request.json', {'schema': 'retrain-private-request-v1', 'inputs': digest(marker),
        'model': digest(model), 'arm': 'control', 'strength': .25,
        'validation_requested': False, 'user_requested_submission': True,
        'tool_sha256': file_hash(Path(__file__)), 'source_gate': digest(source_gate)})
    retrain.manifest(args)
    retrain.duo.verify_model(out/'arms/control/model', signature)
    print('Saved control checkpoint verified. No training or validation will run.', flush=True)
    return args


def submit(args):
    marker = retrain.manifest(args)
    request = read_json(args.out/'request.json')
    directory, signature = retrain.arm_inputs(args, 'control')
    if (request['inputs'] != digest(marker) or request['model'] != digest(retrain.duo.verify_model(directory/'model', signature))
            or request['tool_sha256'] != file_hash(Path(__file__)) or request['user_requested_submission'] is not True):
        raise ValueError('Private inference request changed')
    # Read only fully verified private shards. Never call the development screen or gate.
    rows = retrain.collect(args, 'private', 'control', compute=False)
    predictions = {r['id']: {'answer': retrain.prior.predict(r, retrain.prior.BASELINE)} for r in rows}
    decision = {'inputs': digest(marker), 'deploy': True, 'decision': 'user-requested-unvalidated',
        'promotion_gate_passed': False, 'validation_evaluated': False,
        'request': digest(request), 'reason': 'User explicitly requested private submission without validation.'}
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    retrain.private.write_submission(args, predictions, 'duo', known,
        {'strength': .25, 'evidence': {'policy': retrain.SCHEMA, 'arm': 'control',
                                    'deployed': True, 'validation_evaluated': False}})
    write_json(args.out/'gate.json', decision)
    write_json(args.out/'submission-choice.json', {'candidate_deployed': True,
        'validation_evaluated': False, 'promotion_gate_passed': False,
        'baseline_private_score_user_reported': .9510, 'candidate_private_score': None,
        'zip_sha256': file_hash(args.out/'submission-private-duo.zip'), 'gate': decision})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'private', 'submit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-out', type=Path, default=Path('outputs/retrain/duo-fulltext-retrain-4090-v1'))
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.source_out, args.out)
    elif args.stage == 'private':
        retrain.collect(options(args.out), 'private', 'control')
    else:
        submit(options(args.out))


if __name__ == '__main__':
    main()
