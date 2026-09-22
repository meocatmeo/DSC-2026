"""Verify that the independent source ZIP recovers the exact scored submission."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
RELEASE=ROOT/'releases/handoff/20260922-best-09523-v1'
TARGET=ROOT/'reports/best-09523-recovery/source-archive'


def main():
    locked=json.loads((RELEASE/'release-lock.json').read_text(encoding='utf-8'))
    source=RELEASE/'source-code.zip'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=locked['files']['source-code.zip']:
        raise ValueError('Source bundle differs')
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() or len(set(archive.namelist()))!=len(archive.namelist()):
            raise ValueError('Invalid archive')
        for name in archive.namelist():
            target=(TARGET/name).resolve()
            if not target.is_relative_to(TARGET.resolve()):
                raise ValueError('Unsafe archive path')
            data=archive.read(name)
            if target.exists() and target.read_bytes()!=data:
                raise ValueError('Refuse overwrite of different extracted source')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    subprocess.run([sys.executable,str(TARGET/'recovery/recover.py'),
                    '--out',str(TARGET/'recomputed')],check=True,cwd=TARGET)
    report=json.loads((TARGET/'recomputed/recovery-report.json').read_text())
    if report['zip_sha256']!=locked['best_zip_sha256']:
        raise ValueError('Extracted source recovery differs')
    print('Independent source archive recovery passed.',flush=True)


if __name__=='__main__':
    main()
