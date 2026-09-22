"""Build an additive repair for the immutable retraining release."""
import hashlib
import json
from pathlib import Path
import zipfile

from release_dependencies import complete_imports

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA = 'b0dbcf9ff954133d55aa554c53f8117a8b74ad72bb5f0814a7742611356d9d0d'


def main():
    archive = ROOT/'releases/retrain/20260921-fulltext-retrain-4090-v1/fulltext-retrain.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Published base archive changed')
    with zipfile.ZipFile(archive) as bundle:
        payload = {name: bundle.read(name) for name in bundle.namelist() if name.endswith('.py')}
        base_manifest = bundle.read('private-files.sha256')
    added = complete_imports(payload, ROOT)
    if added != ['legalir2/mine.py', 'legalir2/train_bi.py', 'legalir2/train_ce.py']:
        raise ValueError('Unexpected missing dependencies: '+str(added))
    repair = {name: payload[name] for name in added}
    repair['repair_retrain_dependencies.py'] = (ROOT/'scripts/gpu/repair_retrain_dependencies.py').read_bytes().replace(b'\r\n', b'\n')
    manifest = {'schema': 'retrain-missing-imports-v1', 'base_archive_sha256': BASE_SHA,
                'base_manifest_sha256': hashlib.sha256(base_manifest).hexdigest(),
                'files': {name: hashlib.sha256(payload[name]).hexdigest() for name in added},
                'purpose': 'Add missing training imports; preserve frozen source, prepared inputs and checkpoints.'}
    repair['repair-manifest.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    repair['files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(repair.items())).encode()
    destination = ROOT/'releases/tools/20260921-retrain-training-deps-v1'
    for name, data in repair.items():
        target = destination/name
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Refusing to change existing repair release: '+name)
    for name, data in repair.items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(destination)
    print('Files:', len(repair), 'Bytes:', sum(map(len, repair.values())))


if __name__ == '__main__':
    main()
