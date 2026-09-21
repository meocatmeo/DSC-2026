"""Package the separate BF16 top-100 experiment with frozen training assets."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import deep_pool
from legalir2.h100 import digest
from legalir2.impact import load_view
from legalir2.store import file_hash, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-unpublished', action='store_true')
    args = parser.parse_args()
    source = ROOT/'releases/private/20260920-retrieval-rescue-4090-v1/private-inference-overlay.zip'
    if file_hash(source) != '361e91286cbd1983e79d8876835eb8e5268cde6d64163c6d1e058ececc600424':
        raise ValueError('Published source release changed')
    with zipfile.ZipFile(source) as archive:
        payload = {n: archive.read(n) for n in archive.namelist() if n != 'private-files.sha256'}
        for line in archive.read('private-files.sha256').decode().splitlines():
            expected, name = line.split('  ', 1)
            if hashlib.sha256(payload[name]).hexdigest() != expected:
                raise ValueError('Source payload changed')
    signature = read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')
    all_records = read_json(ROOT/'data/train.json')
    original_records = {q: all_records[q] for q in signature['source_inputs']['train_ids']}
    original = load_view(ROOT/'analysis/ranking-h100/results/20260918T013034Z', 'retrieval', 'train', original_records)
    development = json.loads(payload['analysis/fulltext-prior/development.json'])
    validation = json.loads(payload['analysis/fulltext-prior/validation.json'])
    train = {q: signature['memory'][q] for q in original_records if q in signature['memory']}
    deep_pool.validate_split(train, development, validation)
    old = {r['id']: r for r in original}
    training = {'records': train, 'v9': [old[q] for q in train]}
    asset_dir = 'analysis/deep-pool/'
    payload[asset_dir+'training.json.gz'] = gzip.compress(json.dumps(training, ensure_ascii=False).encode(), mtime=0)
    payload[asset_dir+'development-baseline.json.gz'] = payload['analysis/retrieval-rescue/development-baseline.json.gz']
    assets = {'schema': 'deep-pool-assets-v1', 'duo_inputs': digest(signature), 'training_questions': len(train),
        'training_ids': list(train), 'training_source': digest(signature['source_inputs']),
        'files': {name: hashlib.sha256(payload[asset_dir+name]).hexdigest()
            for name in ('training.json.gz', 'development-baseline.json.gz')}}
    payload[asset_dir+'manifest.json'] = json.dumps(assets, ensure_ascii=False).encode()
    for name in ('legalir2/deep_fusion.py', 'legalir2/deep_pool.py',
            'scripts/gpu/gpu-deep-pool-4090.sh', 'tests2/test_deep_pool.py', 'tests2/test_deep_pool_shell.py',
            'docs/runbooks/RUNBOOK-deep-pool-4090.md'):
        payload[name] = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
    for name, expected in signature['code'].items():
        if hashlib.sha256(payload['legalir2/'+name]).hexdigest() != expected:
            raise ValueError('Frozen comparator code changed')
    payload['private-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    target = ROOT/'releases/private/20260920-deep-pool-4090-v1/private-inference-overlay.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (not args.replace_unpublished or (target.parent/'published.json').exists()):
        raise ValueError('Do not overwrite a published release')
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() or any(archive.read(n) != v for n, v in payload.items()):
            raise ValueError('Archive verification failed')
    temporary.replace(target)
    target.with_suffix('.zip.sha256').write_text(f'{file_hash(target)}  {target.name}\n', encoding='utf-8')
    print(f'Verified {len(payload)} files; {len(train)} training, {len(development)} development, {len(validation)} validation.')
    print(target, file_hash(target))


if __name__ == '__main__':
    main()
