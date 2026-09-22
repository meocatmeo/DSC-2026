"""Finalize the best recovery release without accepting changed existing locks."""
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legalir2.store import file_hash,read_json

OUT=ROOT/'releases/handoff/20260922-best-09523-v1'


def put(path,data):
    if path.exists() and path.read_bytes()!=data:
        raise ValueError('Release file is already frozen: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def include(path):
    return path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc'


def main():
    if not read_json(OUT/'verification/model-assets.json')['complete']:
        raise ValueError('Missing model assets')
    recovery=read_json(OUT/'verification/docker-recovery-report.json')
    if not recovery['exact_zip_match']:
        raise ValueError('Docker recovery not verified')
    put(OUT/'README.md',(ROOT/'docs/runbooks/RUNBOOK-best-09523-handoff.md').read_bytes())
    put(OUT/'source/packaging/lock_best_release.py',Path(__file__).read_bytes())
    for name in ('test_best_recovery.py','test_duo_agreement.py'):
        put(OUT/'source/tests2'/name,(ROOT/'tests2'/name).read_bytes())
    for path in OUT.rglob('*'):
        if not include(path):
            continue
        relative=path.relative_to(OUT)
        if set(relative.parts)&{'.secrets','.git','.ssh'} or path.name in {'rclone.conf','.env','id_rsa','id_ed25519'} or path.suffix=='.conf':
            raise ValueError('Credential path in release: '+str(relative))
        if path.suffix in {'.json','.py','.md','.txt','.sh'} and path.stat().st_size<2**20:
            value=path.read_text(encoding='utf-8',errors='strict')
            if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"(?:access_token|refresh_token)"\s*:\s*"[^"\s]{20,}"',value):
                raise ValueError('Possible credential content: '+str(relative))
    # Upstream source archive is allowlisted too; reject credential filenames.
    with zipfile.ZipFile(OUT/'upstream/fulltext-retrain.zip') as archive:
        for name in archive.namelist():
            parts=set(Path(name).parts)
            if parts&{'.secrets','.git','.ssh'} or name.endswith(('.conf','/.env')):
                raise ValueError('Credential filename in original source archive')
    bundle_names={'Dockerfile','.dockerignore','requirements-recovery.txt','README.md'}
    folders={'source','recovery','provenance','verification','wheels'}
    sources=[p for p in sorted(OUT.rglob('*')) if include(p) and
             (p.relative_to(OUT).parts[0] in folders or p.relative_to(OUT).as_posix() in bundle_names)]
    target=OUT/'source-code.zip'
    temporary=OUT/'source-code.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            info=zipfile.ZipInfo(path.relative_to(OUT).as_posix(),(2026,9,22,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() or len(archive.namelist())!=len(sources):
            raise ValueError('Invalid source archive')
        for path in sources:
            if hashlib.sha256(archive.read(path.relative_to(OUT).as_posix())).hexdigest()!=file_hash(path):
                raise ValueError('Archived source differs')
    if target.exists():
        if file_hash(target)!=file_hash(temporary):
            raise ValueError('Source archive already frozen')
        temporary.unlink()
    else:
        temporary.replace(target)
    files={p.relative_to(OUT).as_posix():file_hash(p) for p in sorted(OUT.rglob('*'))
           if include(p) and p.name not in {'RELEASE.sha256','release-lock.json'}}
    lock={'schema':'dsc-best-handoff-lock-v1','version':'20260922-best-09523-v1','reported_score':.9523,
          'best_zip_sha256':recovery['zip_sha256'],'image_id':'sha256:84d031593e47671cf346f12c9d61684188c9eb9aee63261ea1c3682516fc7a27',
          'docker_offline_recovery_passed':True,'docker_export_load_passed':True,
          'fresh_gpu_inference_passed':False,'organizer_acceptance_confirmed':False,
          'credential_scan':'Explicit source allowlists and credential filename/key-pattern checks passed; no rclone config packaged.',
          'retention':'Preserve this version. Create a new handoff only after another result actually exceeds 0.9523.',
          'files':files}
    put(OUT/'release-lock.json',json.dumps(lock,ensure_ascii=False,indent=1).encode())
    files['release-lock.json']=file_hash(OUT/'release-lock.json')
    put(OUT/'RELEASE.sha256',''.join(f'{value}  {name}\n' for name,value in sorted(files.items())).encode())
    print(json.dumps({'files':len(files),'bytes':sum((OUT/n).stat().st_size for n in files),
                      'source_zip':file_hash(target),'image_tar':file_hash(OUT/'dsc-best-09523-recovery-v1.docker.tar'),
                      'release_manifest':file_hash(OUT/'RELEASE.sha256')},indent=2),flush=True)


if __name__=='__main__':
    main()
