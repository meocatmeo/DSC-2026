"""Create a separate, checksum-locked recovery/source handoff for private 0.9523."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/analysis'))
from legalir2.store import file_hash,read_json,write_json
from legalir2.h100 import digest
from legalir2 import duo_agreement
from audit_retrained_private import compare,predictions

OUT=ROOT/'releases/handoff/20260922-best-09523-v1'
BEST=ROOT/'submissions/private/20260922-duo-agreement-v1'
HASH='37eb1aedbb06620f429c72c6c02de299d5452e628faa44aff816e255c45e2d3f'


def put(path,data):
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError('Immutable release differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)


def encoded(value):
    return json.dumps(value,ensure_ascii=False,indent=1).encode('utf-8')


def main():
    archive=BEST/'submission-private-agreement.zip'
    if file_hash(archive)!=HASH:
        raise ValueError('Best ZIP differs')
    best=predictions(archive);audit=ROOT/'reports/retrain-private-audit'
    hashes=read_json(audit/'source-hashes.json')
    marker=read_json(BEST/'inputs.json')
    if digest(hashes)!=marker['source_hashes'] or file_hash(duo_agreement.__file__)!=marker['policy']['code']:
        raise ValueError('Best rule or sources differ')
    questions=ROOT/'data/private-official.json'
    if file_hash(questions)!=marker['questions_sha256']:
        raise ValueError('Official questions differ')
    context=OUT/'recovery'
    put(context/'recover.py',(ROOT/'scripts/package/best_recovery_runtime.py').read_bytes())
    put(context/'legalir2/__init__.py',b'')
    put(context/'legalir2/duo_agreement.py',Path(duo_agreement.__file__).read_bytes())
    put(context/'data/questions.json',questions.read_bytes())
    signature=read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')
    put(context/'data/known-docs.json',encoded(signature['known_docs']))
    shards=[];seen=set()
    keys=('id','question','docs','prior','ability','probabilities')
    for name,expected in sorted(hashes.items()):
        first=audit/'baseline/cache/duo'/name;second=audit/'candidate/cache/control/private'/name
        if file_hash(first)!=expected['baseline'] or file_hash(second)!=expected['candidate']:
            raise ValueError('Changed comparator cache')
        base,new=read_json(first),read_json(second)
        if len(base)!=len(new):
            raise ValueError('Incomplete paired cache')
        compact=[]
        for b,n in zip(base,new):
            compare(b,n)
            answer,_=duo_agreement.predict(b,n)
            qid=b['id']
            if qid in seen or answer!=best[qid]['answer']:
                raise ValueError('Best answer replay mismatch')
            seen.add(qid)
            compact.append({'baseline':{k:b[k] for k in keys},'candidate':{k:n[k] for k in keys}})
        target='data/scores/'+name
        put(context/target,encoded(compact));shards.append(target)
    if seen!=set(best) or len(seen)!=2080:
        raise ValueError('Incomplete recovery sources')
    with zipfile.ZipFile(archive) as saved:
        entry=saved.getinfo('submission.json')
        payload_sha=hashlib.sha256(saved.read('submission.json')).hexdigest()
        zipmeta={k:getattr(entry,k) for k in ('date_time','compress_type','create_system','create_version','extract_version','external_attr','internal_attr','flag_bits')}
    manifest={'schema':'best-private-09523-recovery-v1','reported_private_score':.9523,
              'mode':'saved-score recovery; not fresh model inference','best_zip_sha256':HASH,
              'policy':duo_agreement.POLICY,'score_sources':hashes,'score_shards':shards,
              'questions_sha256':file_hash(questions),'submission_json_sha256':payload_sha,'zip_metadata':zipmeta,
              'files':{p.relative_to(context).as_posix():file_hash(p) for p in sorted(context.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    put(context/'manifest.json',encoded(manifest))
    for name in ('submission-private-agreement.zip','inputs.json','review.json','leaderboard-result.json'):
        put(OUT/'reference'/name,(BEST/name).read_bytes())
    # Source snapshots come from the published immutable release, never the
    # evolving working tree. Explicit allowlists exclude credentials and logs.
    published=ROOT/'releases/retrain/20260921-fulltext-retrain-4090-v1/fulltext-retrain.zip'
    if file_hash(published)!='b0dbcf9ff954133d55aa554c53f8117a8b74ad72bb5f0814a7742611356d9d0d':
        raise ValueError('Frozen training release differs')
    put(OUT/'upstream/fulltext-retrain.zip',published.read_bytes())
    with zipfile.ZipFile(published) as source:
        for name in source.namelist():
            if name.startswith('legalir2/') and name.endswith('.py') or name.startswith('config/') and name.endswith('.txt'):
                if '..' in Path(name).parts:
                    raise ValueError('Unsafe source path')
                put(OUT/'source/frozen-training'/name,source.read(name))
    repair=ROOT/'releases/tools/20260922-retrain-private-submit-v3'
    for name in ('retrain_private_submit.py','retrain_private_resume.py','verify_retrain_restore.py','files.sha256','README.md'):
        put(OUT/'source/inference-repair'/name,(repair/name).read_bytes())
    for name in ('duo_agreement.py',):
        put(OUT/'source/agreement/legalir2'/name,(ROOT/'legalir2'/name).read_bytes())
    for name in ('submit_duo_agreement.py','audit_retrained_private.py'):
        put(OUT/'source/agreement/scripts/analysis'/name,(ROOT/'scripts/analysis'/name).read_bytes())
    for name in ('best_recovery_runtime.py','freeze_best.py'):
        put(OUT/'source/packaging'/name,(ROOT/'scripts/package'/name).read_bytes())
    for relative in ('candidate/inputs.json','candidate/request.json','candidate/submission-choice.json',
                     'baseline/inputs.json','source-hashes.json'):
        put(OUT/'provenance'/relative,(audit/relative).read_bytes())
    put(OUT/'provenance/duo-inputs.json',encoded(signature))
    for relative in ('arms/control/inputs.json','inputs.json','reference/inputs.json','gate.json'):
        put(OUT/'provenance/retraining'/relative,(ROOT/'reports/retrain-cloud-restore-check'/relative).read_bytes())
    print(json.dumps({'release':str(OUT),'questions_replayed':len(seen),'score_shards':len(shards),
                      'best_zip':HASH,'mode':'source freeze and offline score recovery'},indent=2),flush=True)


if __name__=='__main__':
    main()
