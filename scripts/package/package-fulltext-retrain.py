"""Build a self-contained matched-training release while retaining frozen inference bytes."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.store import read_json
from legalir2.h100 import digest
from legalir2.fulltext_retrain import verify_data
from types import SimpleNamespace
from release_dependencies import complete_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-unpublished', action='store_true')
    parser.add_argument('--release-id', default='20260921-fulltext-retrain-4090-v1')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.release_id):
        parser.error('Invalid release ID')
    training = ROOT/'reports/duo-fulltext-training/20260921-v3'
    dataset = ROOT/'reports/private/fulltext-prior-release-check/dataset'
    verify_data(SimpleNamespace(training=training, dataset=dataset))
    base = ROOT/'releases/private/20260920-fulltext-prior-4090-v1/private-inference-overlay.zip'
    if hashlib.sha256(base.read_bytes()).hexdigest() != '8326ceb744e0fe885a80c55502d7b748615046ef64fc00a3efd4d4497858d4cb':
        raise ValueError('Pinned base release changed')
    with zipfile.ZipFile(base) as archive:
        payload = {n: archive.read(n) for n in archive.namelist() if n != 'private-files.sha256'}
    for name in ('inputs.json', 'ready.json', 'verified.json', 'audit.json', 'evidence-review.json',
                 'old-evidence-control.json.gz', 'control.json.gz', 'hard-negatives.json.gz'):
        payload['analysis/fulltext-training/'+name] = (training/name).read_bytes()
    for name in ('development.json', 'validation.json', 'v9-development.json', 'v9-validation.json'):
        payload['analysis/fulltext-prior/'+name] = (dataset/name).read_bytes()
    sys.path.insert(0, str(ROOT/'scripts/analysis'))
    from audit_prior_development import load, SOURCE
    rows = [{k: v for k, v in row.items() if k != 'relevance'} for row in load()]
    source_marker = read_json(SOURCE/'inputs.json')
    files = {}
    for path in sorted((SOURCE/'cache/base/development').glob('part-*.json')):
        data = path.read_bytes()
        payload['analysis/fulltext-training/development-cache/'+path.name] = data
        files[path.name] = hashlib.sha256(data).hexdigest()
    payload['analysis/fulltext-training/development-cache/seed.json'] = json.dumps({
        'source': digest(source_marker), 'assets': source_marker['assets'],
        'datasets': source_marker['datasets'], 'files': files, 'rows': digest(rows)}, ensure_ascii=False).encode()
    names = ('legalir2/fulltext_retrain.py', 'legalir2/fulltext_training.py', 'legalir2/labeled_gate.py',
             'scripts/gpu/gpu-fulltext-retrain-4090.sh', 'scripts/gpu/gpu-duo-runtime.sh',
             'tests2/test_fulltext_retrain.py', 'tests2/test_fulltext_retrain_shell.py',
             'tests2/test_fulltext_training.py', 'tests2/test_labeled_gate.py',
             'docs/runbooks/RUNBOOK-fulltext-retrain-4090.md', 'docs/development/DUO-FULLTEXT-TRAINING.md')
    for name in names:
        payload[name] = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
    print('Added training dependencies:', complete_imports(payload, ROOT), flush=True)
    signature = read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')
    for name, expected in signature['code'].items():
        if hashlib.sha256(payload['legalir2/'+name]).hexdigest() != expected:
            raise ValueError('Frozen comparator code changed: '+name)
    for name, expected in signature['source_inputs']['files'].items():
        if name.startswith('legalir2/') and name in payload and hashlib.sha256(payload[name]).hexdigest() != expected:
            raise ValueError('Frozen V9 code changed: '+name)
    payload['private-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    directory = ROOT/'releases/retrain'/args.release_id
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/'fulltext-retrain.zip'
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() or set(archive.namelist()) != set(payload):
            raise ValueError('Invalid release ZIP')
        for name, data in payload.items():
            if archive.read(name) != data:
                raise ValueError('Release bytes differ')
    if output.exists() and output.read_bytes() != temporary.read_bytes() and (
            not args.replace_unpublished or (directory/'published.json').exists()):
        raise ValueError('Use a new release version; published releases are immutable')
    temporary.replace(output)
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{sha}  {output.name}\n', encoding='utf-8')
    start = (ROOT/'scripts/gpu/start-fulltext-retrain-4090.sh').read_text().replace('@ZIP_SHA256@', sha)
    start = start.replace('REMOTE=gdrive:DSC-2026-results/releases/retrain/20260921-fulltext-retrain-4090-v1',
                          'REMOTE=gdrive:DSC-2026-results/releases/retrain/'+args.release_id)
    (directory/'start.sh').write_text(start, encoding='utf-8', newline='\n')
    (directory/'start.sh.sha256').write_text(hashlib.sha256(start.encode()).hexdigest()+'  start.sh\n', encoding='utf-8')
    print(f'Verified {len(payload)} files, {output.stat().st_size/2**20:.1f} MiB: {output}\nSHA256 {sha}')


if __name__ == '__main__':
    main()
