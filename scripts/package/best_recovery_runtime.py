"""Offline recovery of the frozen private submission from saved comparator scores.

This is score-cache replay, NOT fresh retrieval/model inference. Refuse new or
modified questions rather than pretending to infer their answers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from legalir2.duo_agreement import predict


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique)


def recover(root,out,questions=None):
    manifest=read(root/'manifest.json')
    for name,expected in manifest['files'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root.resolve()) or sha(target)!=expected:
            raise ValueError('Release checksum mismatch: '+name)
    questions_path=questions or root/'data/questions.json'
    if sha(questions_path)!=manifest['questions_sha256']:
        raise ValueError('This recovery image accepts only the frozen official private questions; fresh inference is a separate workflow')
    questions=read(questions_path);known=set(read(root/'data/known-docs.json'))
    answers={};changes=0
    for name in manifest['score_shards']:
        for pair in read(root/name):
            base,candidate=pair['baseline'],pair['candidate']
            qid=base['id']
            if qid in answers or qid not in questions or base['question']!=questions[qid]['question']:
                raise ValueError('Question identity differs')
            answer,trace=predict(base,candidate)
            if len(answer)!=5 or len(set(answer))!=5 or not set(answer)<=known:
                raise ValueError('Invalid answer')
            answers[qid]={'answer':answer};changes+=trace['replaced']
    if set(answers)!=set(questions) or len(answers)!=2080 or changes!=306:
        raise ValueError('Incomplete or different recovery')
    payload=json.dumps(answers,ensure_ascii=False).encode('utf-8')
    payload_sha=hashlib.sha256(payload).hexdigest()
    if payload_sha!=manifest['submission_json_sha256']:
        raise ValueError('Recomputed submission payload differs from the scored result')
    out.mkdir(parents=True,exist_ok=True)
    output=out/'submission-private-agreement.zip'
    if output.exists() and sha(output)!=manifest['best_zip_sha256']:
        raise ValueError('Refuse to overwrite a different output ZIP')
    info=zipfile.ZipInfo('submission.json',tuple(manifest['zip_metadata']['date_time']))
    for key,value in manifest['zip_metadata'].items():
        if key!='date_time':
            setattr(info,key,value)
    temporary=out/'submission-private-agreement.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info,payload)
    if sha(temporary)!=manifest['best_zip_sha256']:
        raise ValueError('Answer payload matches but ZIP bytes differ; inspect Python/zlib versions')
    temporary.replace(output)
    (out/'submission.json').write_bytes(payload)
    report={'questions':2080,'replaced_fifths':changes,'answers_recomputed':True,
            'payload_sha256':payload_sha,'zip_sha256':sha(output),'exact_zip_match':True,
            'mode':'offline score-cache recovery','fresh_model_inference':False,
            'python':sys.version,'numpy':__import__('numpy').__version__}
    (out/'recovery-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/output'))
    parser.add_argument('--questions',type=Path)
    args=parser.parse_args()
    recover(Path(__file__).resolve().parent,args.out,args.questions)
