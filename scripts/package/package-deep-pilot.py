"""Create an immutable development-only pilot from the published deep release."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.deep_pilot import training_subset
from legalir2.deep_pool import validate_split
from legalir2.store import file_hash


def main():
    source = ROOT/'releases/private/20260920-deep-pool-4090-v1/private-inference-overlay.zip'
    if file_hash(source) != '3f54c78515e40c861baa8a71e2654651563335ba9ddffd9ef9590881df15e82a':
        raise ValueError('Published source changed')
    with zipfile.ZipFile(source) as archive:
        payload = {n: archive.read(n) for n in archive.namelist() if n != 'private-files.sha256'}
        for line in archive.read('private-files.sha256').decode().splitlines():
            expected, name = line.split('  ', 1)
            if hashlib.sha256(payload[name]).hexdigest() != expected:
                raise ValueError('Source manifest mismatch')
    prefix = 'analysis/deep-pool/'
    original = json.loads(gzip.decompress(payload[prefix+'training.json.gz']))
    train = training_subset(original['records'])
    dev = json.loads(payload['analysis/fulltext-prior/development.json'])
    val = json.loads(payload['analysis/fulltext-prior/validation.json'])
    validate_split(train, dev, val)
    by_id = {r['id']: r for r in original['v9']}
    training = {'records': train, 'v9': [by_id[q] for q in train]}
    payload[prefix+'training.json.gz'] = gzip.compress(json.dumps(training, ensure_ascii=False).encode(), mtime=0)
    assets = json.loads(payload[prefix+'manifest.json'])
    assets.update(training_questions=len(train), training_ids=list(train),
        sampling={'method': 'SHA256 of salted normalized question text, then ID',
                  'salt': 'deep-pilot-256-v1', 'source_questions': len(original['records']),
                  'selection_uses_labels_or_scores': False})
    assets['files']['training.json.gz'] = hashlib.sha256(payload[prefix+'training.json.gz']).hexdigest()
    payload[prefix+'manifest.json'] = json.dumps(assets, ensure_ascii=False).encode()
    for name in ('legalir2/deep_pilot.py', 'scripts/gpu/gpu-deep-pilot-4090.sh',
                 'tests2/test_deep_pilot.py', 'tests2/test_deep_pool_shell.py',
                 'docs/runbooks/RUNBOOK-deep-pilot-4090.md'):
        payload[name] = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
    payload['private-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    target = ROOT/'releases/private/20260920-deep-pilot-4090-v1/private-inference-overlay.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('Do not overwrite a release')
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() or any(archive.read(n) != v for n, v in payload.items()):
            raise ValueError('Archive verification failed')
    temporary.replace(target)
    target.with_suffix('.zip.sha256').write_text(f'{file_hash(target)}  {target.name}\n', encoding='utf-8')
    print(f'Verified {len(train)} training; {len(dev)} development; validation/private not run.')
    print(target, file_hash(target))


if __name__ == '__main__':
    main()
