"""Create a separate checked 4090 experiment from the preserved scored release."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import duo, h100
from legalir2.impact import load_view
from legalir2.store import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-unpublished', action='store_true')
    args = parser.parse_args()
    base = ROOT/'releases/private/20260919-duo-fulltext-v2-4090-fast1/private-inference-overlay.zip'
    if hashlib.sha256(base.read_bytes()).hexdigest() != '5d21497066f0ac3e6806fdc56f03ddb85084e58cb41091eada5fd6c070ccdd9f':
        raise ValueError('Scored release differs')
    with zipfile.ZipFile(base) as archive:
        payload = {name: archive.read(name) for name in archive.namelist() if name != 'private-files.sha256'}
    run = ROOT/'analysis/impact/duo-20260918T110105Z'
    signature = read_json(run/'inputs.json')
    all_questions = read_json(ROOT/'data/train.json')
    source = ROOT/'analysis/ranking-h100/results/20260918T013034Z'
    train = {q: all_questions[q] for q in signature['source_inputs']['train_ids']}
    training_rows = load_view(source, 'retrieval', 'train', train)
    # Take the established validation membership from its guarded comparator cache.
    validation_parts = [r for p in sorted((run/'scores/validation').glob('part-*.json')) for r in read_json(p)]
    validation = {r['id']: all_questions[r['id']] for r in validation_parts}
    duo.validate_scores(validation_parts, list(validation.items()))
    if h100.digest(validation) != read_json(run/'scores/validation/resume.json')['questions']:
        raise ValueError('Validation membership/labels changed')
    validation_rows = load_view(source, 'retrieval', 'validation', validation)
    development = {q: all_questions[q] for q in signature['development']}
    from legalir2.cache import normalise_question
    if ({normalise_question(r['question']) for r in development.values()} &
            {normalise_question(r['question']) for r in validation.values()}):
        raise ValueError('Development/validation text overlap')
    by_id = {r['id']: r for r in training_rows}
    datasets = {'development': (development, [by_id[q] for q in development]),
                'validation': (validation, validation_rows)}
    for split, (records, rows) in datasets.items():
        for name, value in [(split, records), ('v9-'+split, rows)]:
            payload['analysis/fulltext-prior/'+name+'.json'] = json.dumps(value, ensure_ascii=False).encode()
    names = ['legalir2/fulltext_prior.py', 'legalir2/consensus.py', 'legalir2/private_infer.py', 'legalir2/evidence.py',
             'scripts/gpu/gpu-fulltext-prior-4090.sh', 'scripts/gpu/vast_stop_after.py',
             'tests2/test_fulltext_prior.py', 'tests2/test_fulltext_prior_shell.py', 'tests2/test_consensus.py',
             'docs/runbooks/RUNBOOK-fulltext-prior-4090.md', 'docs/state/THONG-INTEGRATION.md']
    for name in names:
        payload[name] = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
    # Frozen models' inference modules must retain their original bytes.
    for name, expected in signature['code'].items():
        if hashlib.sha256(payload['legalir2/'+name]).hexdigest() != expected:
            raise ValueError('Frozen comparator source changed')
    payload['private-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    output = ROOT/'releases/private/20260920-fulltext-prior-4090-v1/private-inference-overlay.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(payload):
            raise ValueError('Invalid release ZIP')
        for name, data in payload.items():
            if archive.read(name) != data:
                raise ValueError('Release bytes changed')
    if output.exists() and output.read_bytes() != temporary.read_bytes() and (
            not args.replace_unpublished or (output.parent/'published.json').exists()):
        raise ValueError('Release already exists; use a new version')
    temporary.replace(output)
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{sha}  {output.name}\n', encoding='utf-8')
    print(f'Verified {len(payload)} files; {len(development)} development, {len(validation)} validation questions.')
    print(output, sha)


if __name__ == '__main__':
    main()
