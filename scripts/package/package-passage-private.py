"""Package the user-requested private submission experiment, preserving frozen code bytes."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.passage_private import load, items
from legalir2.store import read_json, write_json, file_hash

RELEASE = '20260923-passage-private-4090-v2'


def main():
    source = ROOT/'reports/passage-private-v2'
    plan, prepared = load(source)
    count = sum(1 for _ in items(source, prepared))
    directory = ROOT/'releases/passage-order'/RELEASE
    directory.mkdir(parents=True, exist_ok=True)
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted((ROOT/'legalir2').glob('*.py'))}
    names = ('config/requirements-private-h100.txt', 'scripts/gpu/gpu-private-setup.sh',
        'scripts/gpu/gpu-duo-runtime.sh', 'scripts/gpu/gpu-ranking-h100-runtime.sh',
        'scripts/gpu/gpu-passage-private-4090.sh', 'tests2/__init__.py',
        'tests2/test_passage_order.py', 'tests2/test_passage_order_run.py', 'tests2/test_duo_agreement.py', 'tests2/test_agreement_097.py',
        'tests2/test_passage_private_shell.py', 'scripts/gpu/vast_stop_passage_private.py', 'tests2/test_passage_private.py', 'tests2/test_vast_stop_passage_private.py', 'scripts/gpu/start-passage-private-4090.sh',
        'docs/development/PASSAGE-ORDER-EXPERIMENT.md', 'docs/runbooks/RUNBOOK-passage-private-4090.md')
    for name in names:
        data = (ROOT/name).read_bytes()
        payload[name] = data.replace(b'\r\n', b'\n') if name.endswith('.sh') else data
    prepared = read_json(source/'prepared.json')
    for name in ('plan.json', 'prepared.json', *plan['data_files'], *prepared['files']):
        payload['experiment/passage-private-v1/'+name] = (source/name).read_bytes()
    for name, expected in plan['code'].items():
        if hashlib.sha256(payload['legalir2/'+name]).hexdigest() != expected:
            raise ValueError('Packaged code differs from the prepared experiment: '+name)
    payload['private-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    output = directory/'passage-private.zip'
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() or set(archive.namelist()) != set(payload):
            raise ValueError('Incomplete release ZIP')
        for name, data in payload.items():
            if archive.read(name) != data:
                raise ValueError('Release bytes differ: '+name)
    if output.exists() and file_hash(output) != file_hash(temporary):
        raise ValueError('Release is immutable; choose a new version')
    temporary.replace(output)
    sha = file_hash(output)
    output.with_suffix('.zip.sha256').write_text(f'{sha}  {output.name}\n', encoding='utf8')
    start = (ROOT/'scripts/gpu/start-passage-private-4090.sh').read_text().replace('@ZIP_SHA256@', sha)
    (directory/'start.sh').write_text(start, encoding='utf8', newline='\n')
    (directory/'start.sh.sha256').write_text(file_hash(directory/'start.sh')+'  start.sh\n', encoding='utf8')
    write_json(directory/'release.json', {'release': RELEASE, 'files': len(payload), 'questions': count,
        'archive_sha256': sha, 'archive_bytes': output.stat().st_size,
        'start_sha256': file_hash(directory/'start.sh'), 'gpu_tested': False,
        'purpose': '2080 private questions; unvalidated experimental submission with 0.97 primary and 0.98 alternative policies',
        'remote': 'gdrive:DSC-2026-results/releases/passage-order/'+RELEASE})
    print(read_json(directory/'release.json'))


if __name__ == '__main__':
    main()
