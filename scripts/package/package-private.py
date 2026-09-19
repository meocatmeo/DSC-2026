"""Build a checked private-inference overlay without changing trained module bytes."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[2]
    signature = json.loads((root/'analysis/impact/duo-20260918T110105Z/inputs.json').read_text(encoding='utf-8'))
    import sys
    sys.path.insert(0, str(root))
    from legalir2.private_infer import SOURCE_MODULES
    payloads = {}
    names = ['legalir2/private_infer.py', 'legalir2/private_restore.py', 'legalir2/download.py',
             'scripts/gpu/gpu-private-h100.sh', 'scripts/gpu/gpu-private-setup.sh',
             'config/requirements-private-h100.txt', 'scripts/gpu/gpu-duo-runtime.sh',
             'scripts/gpu/gpu-ranking-h100-runtime.sh', 'tests2/__init__.py', 'tests2/test_private_infer.py',
             'tests2/test_private_restore.py', 'tests2/test_layout.py',
             'docs/runbooks/RUNBOOK-private.md', 'data/private-official.json']
    pins = dict(line.split('==') for line in (root/'config/requirements-private-h100.txt').read_text().splitlines()
                if line and not line.startswith('#'))
    expected = {k:v for k,v in dict(signature['source_inputs']['packages'], **signature['packages']).items()
                if v is not None}
    if pins != expected:
        raise ValueError('Private environment pins differ from the trained run')
    frozen = {'legalir2/'+n:signature['source_inputs']['files']['legalir2/'+n]
              for n in (*SOURCE_MODULES, '__init__.py')}
    frozen.update({'legalir2/'+n:h for n,h in signature['code'].items()})
    for name, expected in frozen.items():
        data = (root/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Frozen module differs: ' + name)
        payloads[name] = data
    for name in names:
        data = (root/name).read_bytes()
        # Preserve the official question-file bytes. Bash/text source uses LF.
        payloads[name] = data if name.endswith('.json') else data.replace(b'\r\n', b'\n')
    for name in ['v9-ensemble.run.json', 'v9-stacker-ens.json']:
        data = (root/'analysis/v11/investigation/baselines'/name).read_bytes()
        key = 'analysis/ranking-h100/baseline/' + name
        if hashlib.sha256(data).hexdigest() != signature['source_inputs']['files'][key]:
            raise ValueError('Frozen baseline differs: ' + name)
        payloads['analysis/private/baseline/'+name] = data
    output = root/'releases/private/20260919-layout-v2/private-inference-overlay.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name,data in sorted(payloads.items()))
    payloads['private-files.sha256'] = manifest.encode()
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(payloads)
        for name, data in payloads.items():
            assert archive.read(name) == data
        assert len(json.loads(archive.read('data/private-official.json'))) == 2080
    temporary.replace(output)
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{sha}  {output.name}\n', encoding='utf-8')
    print(f'Verified {len(payloads)} files: {output} ({output.stat().st_size:,} bytes)')
    print(sha)


if __name__ == '__main__':
    main()
