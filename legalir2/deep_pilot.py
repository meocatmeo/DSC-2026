"""Finish the development-only deep-pool pilot without private inference."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from . import deep_pool as deep
from .cache import normalise_question
from .h100 import digest
from .store import file_hash, read_json, write_json


def training_subset(records, count=256):
    """Fixed selection by question text, independent of labels or scores."""
    if len(records) < count or count < 1:
        raise ValueError('Invalid pilot training size')
    def key(q):
        text = 'deep-pilot-256-v1\n'+normalise_question(records[q]['question'])
        return hashlib.sha256(text.encode()).hexdigest(), q
    selected = sorted(records, key=key)[:count]
    return {q: records[q] for q in selected}


def submit(args):
    selected = deep.choice(args)
    marker = deep.prior.manifest(deep.baseline_args(args))
    model = deep.fitted(args)
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    report = {'schema': 'deep-pool-pilot-v1', 'calibration': selected,
        'training_questions': model['questions'], 'candidate_deployed': False,
        'validation_evaluated': False, 'private_inference_performed': False,
        'next_step': 'Review development results before paying for validation/private inference.',
        'fallback_private_score_user_reported': .9510, 'candidate_private_score': None}
    stamp = dict(marker['assets'], parameter_audit=read_json(args.out/'audit.json'),
        evidence={'policy': 'deep-pool-pilot-v1', 'deployed': False, 'fusion': digest(model)})
    for name in ('v9', 'duo'):
        path = args.source/f'submission-private-{name}.zip'
        if file_hash(path) != marker['source_files'][path.name] or (name == 'duo' and file_hash(path) != deep.prior.SCORED_ZIP):
            raise ValueError('Scored fallback archive changed')
        with zipfile.ZipFile(path) as archive:
            answers = json.loads(archive.read('submission.json'))
        deep.private.write_submission(args, answers, name, known, stamp)
    write_json(args.out/'submission-choice.json', report)
    write_json(args.out/'pilot-report.json', report)
    print('Pilot complete. ZIPs contain unchanged baseline answers; review calibration.json.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('submit',))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--duo-run', type=Path, default=Path('outputs/duo/duo-20260918T110105Z'))
    parser.add_argument('--dataset', type=Path, default=Path('analysis/fulltext-prior'))
    parser.add_argument('--deep-assets', type=Path, default=Path('analysis/deep-pool'))
    parser.add_argument('--questions', type=Path, default=Path('data/private-official.json'))
    args = parser.parse_args()
    if args.batch < 1 or args.out.resolve() in (args.source.resolve(), args.duo_run.resolve()):
        parser.error('Use a positive batch and separate output directory')
    submit(args)


if __name__ == '__main__':
    main()
