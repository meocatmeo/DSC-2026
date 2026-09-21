"""Package only additive files for the completed pilot's private continuation."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import deep_pilot512_submit as direct
from legalir2.h100 import digest


def main():
    saved = ROOT/'reports/private/deep-pilot512-completed'
    actual = {n: json.loads((saved/(n+'.json')).read_text(encoding='utf-8'))
              for n in ('inputs', 'fusion', 'calibration')}
    if (digest(actual['inputs']) != direct.INPUTS or digest(actual['fusion']) != direct.FUSION
            or actual['calibration']['scores'] != direct.SCORES or actual['calibration']['selected'] != 'deep'):
        raise ValueError('Continuation constants differ from completed cloud pilot')
    original = ROOT/'releases/private/20260920-deep-pilot-4090-v1/private-inference-overlay.zip'
    if hashlib.sha256(original.read_bytes()).hexdigest() != '7312f4209336ffbfc5d8a7f2f04a8e7fb9a89a5c3c24f7a4d574618903ef9486':
        raise ValueError('Published pilot changed')
    names = ('legalir2/deep_pilot512_submit.py', 'scripts/gpu/gpu-deep-pilot512-submit-4090.sh',
             'tests2/test_deep_pilot512_submit.py', 'docs/runbooks/RUNBOOK-deep-pilot512-submit-4090.md')
    with zipfile.ZipFile(original) as archive:
        if set(names) & set(archive.namelist()):
            raise ValueError('Continuation must not overwrite any pilot files')
    extension = ROOT/'releases/private/20260920-deep-pilot512-4090-v1/pilot512-overlay.zip'
    if hashlib.sha256(extension.read_bytes()).hexdigest() != '626faa69d69ebdd0b7652a3792fab4e7891dc5f8a0d563c4f7368e8d8105cd09':
        raise ValueError('Published 512 extension changed')
    with zipfile.ZipFile(extension) as archive:
        if set(names) & set(archive.namelist()):
            raise ValueError('Continuation must not overwrite 512 training assets')
    payload = {n: (ROOT/n).read_bytes().replace(b'\r\n', b'\n') for n in names}
    payload['pilot512-submit-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    target = ROOT/'releases/private/20260921-deep-pilot512-submit-4090-v1/pilot512-submit-overlay.zip'
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
