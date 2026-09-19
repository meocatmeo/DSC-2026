"""Build a checked private-inference overlay without changing trained module bytes."""
import hashlib
import argparse
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fulltext', action='store_true')
    parser.add_argument('--gpu', choices=('h100', '4090'), default='h100')
    parser.add_argument('--fast-evidence', action='store_true')
    args = parser.parse_args()
    if args.gpu == '4090' and not args.fulltext:
        parser.error('The 4090 release requires --fulltext')
    if args.fast_evidence and (not args.fulltext or args.gpu != '4090'):
        parser.error('--fast-evidence requires --fulltext --gpu 4090')
    root = Path(__file__).resolve().parents[2]
    signature = json.loads((root/'analysis/impact/duo-20260918T110105Z/inputs.json').read_text(encoding='utf-8'))
    import sys
    sys.path.insert(0, str(root))
    from legalir2.private_infer import SOURCE_MODULES
    payloads = {}
    names = ['legalir2/private_infer.py', 'legalir2/private_restore.py', 'legalir2/download.py',
             'legalir2/private_gpu.py',
             'scripts/gpu/gpu-private-h100.sh', 'scripts/gpu/gpu-private-setup.sh',
             'config/requirements-private-h100.txt', 'scripts/gpu/gpu-duo-runtime.sh',
             'scripts/gpu/gpu-ranking-h100-runtime.sh', 'tests2/__init__.py', 'tests2/test_private_infer.py',
             'tests2/test_private_restore.py', 'tests2/test_layout.py',
             'docs/runbooks/RUNBOOK-private.md', 'data/private-official.json']
    if args.fulltext:
        names += ['legalir2/duo_evidence.py', 'scripts/gpu/gpu-duo-fulltext.sh',
                  'docs/runbooks/RUNBOOK-duo-fulltext.md', 'tests2/test_duo_evidence.py',
                  'tests2/test_duo.py']
        payloads['analysis/private/corpus.jsonl.gz'] = (root/'analysis/v11/baseline/corpus.jsonl.gz').read_bytes()
    if args.gpu == '4090':
        names += ['scripts/gpu/gpu-duo-4090.sh', 'docs/runbooks/RUNBOOK-duo-4090.md',
                  'tests2/test_private_gpu.py']
    if args.fast_evidence:
        names += ['scripts/gpu/gpu-duo-4090-fast.sh', 'tests2/test_duo_overlap.py',
                  'docs/runbooks/RUNBOOK-duo-4090-fast.md']
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
    release = '20260919-duo-fulltext-v2' if args.fulltext else '20260919-layout-v2'
    if args.gpu == '4090':
        release += '-4090'
    if args.fast_evidence:
        release += '-fast1'
    output = root/'releases/private'/release/'private-inference-overlay.zip'
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
