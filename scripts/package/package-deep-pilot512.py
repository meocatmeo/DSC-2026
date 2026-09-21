"""Add a nested 512-question training capsule to an existing pilot workspace."""
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


def payload_from_release():
    source = ROOT/'releases/private/20260920-deep-pool-4090-v1/private-inference-overlay.zip'
    if hashlib.sha256(source.read_bytes()).hexdigest() != '3f54c78515e40c861baa8a71e2654651563335ba9ddffd9ef9590881df15e82a':
        raise ValueError('Published source changed')
    with zipfile.ZipFile(source) as archive:
        all_training = json.loads(gzip.decompress(archive.read('analysis/deep-pool/training.json.gz')))
        train = training_subset(all_training['records'], 512)
        dev = json.loads(archive.read('analysis/fulltext-prior/development.json'))
        val = json.loads(archive.read('analysis/fulltext-prior/validation.json'))
        validate_split(train, dev, val)
        old = {r['id']: r for r in all_training['v9']}
        training = {'records': train, 'v9': [old[q] for q in train]}
        prefix = 'analysis/deep-pool512/'
        payload = {prefix+'training.json.gz': gzip.compress(json.dumps(training, ensure_ascii=False).encode(), mtime=0),
            prefix+'development-baseline.json.gz': archive.read('analysis/deep-pool/development-baseline.json.gz')}
        assets = json.loads(archive.read('analysis/deep-pool/manifest.json'))
        assets.update(training_questions=512, training_ids=list(train),
            sampling={'method': 'nested SHA256 sample of normalized question text; labels/scores unused',
                      'salt': 'deep-pilot-256-v1', 'source_questions': len(all_training['records'])})
        assets['files'] = {Path(n).name: hashlib.sha256(data).hexdigest() for n, data in payload.items()}
        payload[prefix+'manifest.json'] = json.dumps(assets, ensure_ascii=False).encode()
    pilot = ROOT/'releases/private/20260920-deep-pilot-4090-v1/private-inference-overlay.zip'
    if hashlib.sha256(pilot.read_bytes()).hexdigest() != '7312f4209336ffbfc5d8a7f2f04a8e7fb9a89a5c3c24f7a4d574618903ef9486':
        raise ValueError('Published pilot changed')
    with zipfile.ZipFile(pilot) as archive:
        before = json.loads(gzip.decompress(archive.read('analysis/deep-pool/training.json.gz')))
        if list(train)[:256] != list(before['records']) or training['v9'][:256] != before['v9']:
            raise ValueError('512 training capsule is not a nested extension of the pilot')
        existing = set(archive.namelist())
    for name in ('legalir2/deep_pilot512.py', 'scripts/gpu/gpu-deep-pilot512-4090.sh',
                 'tests2/test_deep_pilot512.py', 'docs/runbooks/RUNBOOK-deep-pilot512-4090.md'):
        payload[name] = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
    if set(payload) & existing:
        raise ValueError('Do not overwrite any published pilot files')
    return payload


def main():
    payload = payload_from_release()
    payload['pilot512-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    target = ROOT/'releases/private/20260920-deep-pilot512-4090-v1/pilot512-overlay.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('Never overwrite a release')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() or any(archive.read(n) != data for n, data in payload.items()):
            raise ValueError('Archive verification failed')
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix('.zip.sha256').write_text(f'{sha}  {target.name}\n', encoding='utf-8')
    print(target, sha)


if __name__ == '__main__':
    main()
