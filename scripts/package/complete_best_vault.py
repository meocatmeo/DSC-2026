"""Finish the best-result vault with exact evidence, raw scores and source tools."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legalir2.store import read_json,file_hash,write_json

OUT=ROOT/'releases/handoff/20260922-best-09523-v1'


def copy(source,target,expected=None):
    expected=expected or file_hash(source)
    if file_hash(source)!=expected:
        raise ValueError('Source checksum differs: '+str(source))
    if target.exists():
        if file_hash(target)!=expected:
            raise ValueError('Immutable target differs: '+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    if file_hash(target)!=expected:
        raise ValueError('Copy checksum differs')


def main():
    baseline=read_json(OUT/'provenance/baseline/inputs.json')
    # Do not silently replace the historical index with a local rebuild.
    # Preserve the rebuild under an explicit name and report the mismatch.
    from legalir2.duo_evidence import index_metadata
    local=ROOT/'artifacts2/frozen-20260918/duo-fulltext-v2.sqlite'
    if index_metadata(local)!=baseline['evidence']['metadata']:
        raise ValueError('Rebuilt evidence corpus metadata differs')
    actual=file_hash(local)
    copy(local,OUT/'upstream/rebuilt-evidence-windows.sqlite')
    write_json(OUT/'verification/evidence-index.json',{
        'historical_sha256':baseline['evidence']['index'],'preserved_rebuild_sha256':actual,
        'metadata_matches':True,'byte_identical':actual==baseline['evidence']['index'],
        'exact_original_token_traces':'upstream/full-score-caches.zip',
        'limitation':'The historical SQLite bytes are not available locally. Rebuild equivalence needs verification; exact score recovery does not use this index.'})
    copy(ROOT/'analysis/v11/baseline/corpus.jsonl.gz',OUT/'upstream/corpus.jsonl.gz')
    audit=ROOT/'reports/retrain-private-audit'
    hashes=read_json(audit/'source-hashes.json')
    target=OUT/'upstream/full-score-caches.zip'
    if not target.exists():
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
            for name,expected in sorted(hashes.items()):
                for role,path in (('baseline',audit/'baseline/cache/duo'/name),
                                  ('candidate',audit/'candidate/cache/control/private'/name)):
                    if file_hash(path)!=expected[role]:
                        raise ValueError('Source score cache changed')
                    info=zipfile.ZipInfo(role+'/'+name,(2026,9,22,0,0,0))
                    info.compress_type=zipfile.ZIP_DEFLATED
                    archive.writestr(info,path.read_bytes())
    with zipfile.ZipFile(target) as archive:
        if len(archive.namelist())!=2*len(hashes) or archive.testzip():
            raise ValueError('Invalid raw-score backup')
        for name,expected in hashes.items():
            for role in ('baseline','candidate'):
                if hashlib.sha256(archive.read(role+'/'+name)).hexdigest()!=expected[role]:
                    raise ValueError('Archived score differs')
    for name in ('calibration.json','selection.json'):
        copy(ROOT/'analysis/impact/duo-20260918T110105Z'/name,OUT/'provenance/original-duo'/name)
    for name in ('complete_best_vault.py','verify_best_assets.py'):
        copy(ROOT/'scripts/package'/name,OUT/'source/packaging'/name)
    copy(ROOT/'reports/best-09523-recovery/docker/recovery-report.json',OUT/'verification/docker-recovery-report.json')
    report=read_json(OUT/'verification/docker-recovery-report.json')
    if not report['exact_zip_match'] or report['zip_sha256']!='37eb1aedbb06620f429c72c6c02de299d5452e628faa44aff816e255c45e2d3f':
        raise ValueError('Docker recovery has not passed')
    print('Full evidence, original raw score traces, checkpoints and Docker recovery receipt preserved.',flush=True)


if __name__=='__main__':
    main()
