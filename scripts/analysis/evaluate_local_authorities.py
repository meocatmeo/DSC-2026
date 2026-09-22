"""CPU development coverage only for one fixed local-authority proposal policy."""
from contextlib import closing
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legalir2 import local_authorities, duo_agreement
from legalir2.h100 import digest,guard
from legalir2.store import read_json,write_json,file_hash,source_records
from audit_retrained_private import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('passages','documents','raw'),default='passages')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    out=args.out or ROOT/f'reports/local-authorities-v3-{args.scope}'
    source=ROOT/'reports/retrain-private-audit/labeled/cache'
    base_dir=ROOT/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    store=ROOT/'artifacts2/frozen-20260918/store.sqlite'
    signature=read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')
    store_hash=file_hash(store)
    if store_hash!=signature['source_inputs']['files']['artifacts-ranking-h100/store.sqlite']:
        raise ValueError('Frozen metadata differs')
    with closing(sqlite3.connect(store.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        documents={doc:dict(title=title or '',number=number or '',subject=subject or '')
                   for doc,title,number,subject in db.execute('select id,title,number,subject from documents')}
        corpus_hash=json.loads(db.execute("select value from metadata where key='corpus'").fetchone()[0])['corpus_sha256']
    model=local_authorities.Authorities(documents)
    plan={'policy':local_authorities.POLICY,'code':file_hash(local_authorities.__file__),
          'anchor_code':file_hash(duo_agreement.__file__),'store':store_hash,
          'evaluator':file_hash(__file__),'scope':args.scope,
          'corpus_source_bytes':corpus_hash,
          'split':'existing development only','private_data_used':False,'validation_evaluated':False,
          'parameter_search':False,'outputs':'candidate proposals and coverage; no answer changes'}
    guard(out/'plan.json',plan)
    base_all,candidate_all,old_all=[],[],[]
    proposals,changes,pairs,ceilings=[],[],[],[]
    bodies={}
    if args.scope=='documents':
        with closing(sqlite3.connect(store.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            for doc,body in db.execute('select document_id,body from chunks order by document_id,ord'):
                bodies.setdefault(doc,{'packed':[]})['packed'].append({'text':body})
    elif args.scope=='raw':
        source_hash=hashlib.sha256()
        for name,raw in source_records(ROOT/'analysis/v11/baseline/corpus.jsonl.gz'):
            source_hash.update(name.encode()+b'\0'+raw+b'\0')
            record=json.loads(raw.decode('utf-8')); doc=str(record['id'])
            if doc in bodies or doc not in documents:
                raise ValueError('Invalid raw corpus identity')
            bodies[doc]={'packed':[{'text':record.get('passage','')}]}
        if source_hash.hexdigest()!=corpus_hash or set(bodies)!=set(documents):
            raise ValueError('Raw corpus differs from frozen store source')
    started=time.monotonic()
    for path in sorted(base_dir.glob('part-*.json')):
        base=read_json(path); new=read_json(source/'control/development'/path.name); old=read_json(source/'old-evidence-control/development'/path.name)
        if not len(base)==len(new)==len(old):
            raise ValueError('Incomplete source shard')
        for b,n in zip(base,new):
            compare(b,n)
            anchor,_=duo_agreement.predict(b,n)
            traces=bodies if args.scope!='passages' else b['evidence']['documents']
            linked=model.candidates(b['question'],anchor,b['docs'],traces)
            extra=linked[:local_authorities.POLICY['maximum_additions']]
            gold=set(b['gold']); present=gold&set(b['docs']); recovered=gold&{p['doc'] for p in extra}
            pairs.append((Fraction(len(present),len(gold)),Fraction(len(present|recovered),len(gold))))
            ceilings.append(Fraction(len(gold&(set(b['docs'])|{p['doc'] for p in linked})),len(gold)))
            proposals.append({'id':b['id'],'anchor':anchor,'proposals':extra})
            if recovered:
                changes.append({'id':b['id'],'question':b['question'],'recovered':sorted(recovered),
                                'proposals':[p for p in extra if p['doc'] in recovered]})
        base_all.extend(base);candidate_all.extend(new);old_all.extend(old)
    if len(pairs)!=429 or digest([base_all,old_all,candidate_all])!=read_json(ROOT/'reports/retrain-private-audit/labeled/screen.json')['scores']:
        raise ValueError('Development cache identity differs')
    total=sum(len(r['proposals']) for r in proposals)
    report={'plan':digest(plan),'questions':429,'baseline_pool_coverage':float(sum(a for a,b in pairs)/429),
            'expanded_pool_coverage':float(sum(b for a,b in pairs)/429),
            'coverage_delta':float(sum(b-a for a,b in pairs)/429),
            'unbounded_link_coverage_ceiling':float(sum(ceilings)/429),
            'added_candidates':total,'questions_with_additions':sum(bool(r['proposals']) for r in proposals),
            'gold_instances_recovered':sum(len(r['recovered']) for r in changes),'recoveries':changes,
            'number_identities':len(model.numbers),'law_name_aliases':len(model.aliases),
            'elapsed_seconds':time.monotonic()-started,'final_recall_measured':False,'submission_generated':False,
            'limitation':'Development diagnostic; new candidates have no comparator scores. Not a private-score gain.'}
    write_json(out/'proposals.json',proposals);write_json(out/'report.json',report)
    print({k:v for k,v in report.items() if k!='recoveries'},flush=True)
    print('Recoveries:',[(r['id'],r['recovered']) for r in changes],flush=True)


if __name__=='__main__':
    main()
