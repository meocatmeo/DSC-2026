"""Fixed CPU supervised retrieval experiment; development diagnostics only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from fractions import Fraction
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legalir2 import learned_question_retrieval as learner,duo_agreement
from legalir2.h100 import digest,guard
from legalir2.store import read_json,write_json,file_hash
from audit_retrained_private import compare


def main():
    out=ROOT/'reports/learned-question-retrieval-v2'
    signature=read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')
    records=signature['memory']
    dataset=ROOT/'reports/private/fulltext-prior-release-check/dataset'
    held={};splits={}
    for split in ('development','validation'):
        splits[split]=read_json(dataset/(split+'.json'))
        held.update(splits[split])
    plan={'policy':learner.POLICY,'code':file_hash(learner.__file__),'evaluator':file_hash(__file__),
          'training':digest(records),'training_count':len(records),'labels_from':'frozen training memory only',
          'heldout_questions':{k:digest({qid:r['question'] for qid,r in v.items()}) for k,v in splits.items()},
          'evaluation':'429 reused development only','validation_scored':False,'private_scored':False,
          'parameter_search':False,'outputs':'candidate coverage; no submission or ranking deployment'}
    guard(out/'plan.json',plan)
    started=time.monotonic()
    print('Fitting supervised question-to-document retriever on',len(records),'training questions.',flush=True)
    model=learner.LearnedQuestions(records,signature['known_docs'],held,[r['question'] for r in held.values()])
    print('Fit complete:',len(model.documents),'document labels;',model.cleaning,round(time.monotonic()-started,1),'seconds.',flush=True)
    audit=ROOT/'reports/retrain-private-audit/labeled/cache'
    directory=ROOT/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    all_base,all_new,all_old=[],[],[]
    totals={'learned':[],'nearest_neighbor':[]};proposals=[];recoveries=[];available=[]
    for path in sorted(directory.glob('part-*.json')):
        base=read_json(path);new=read_json(audit/'control/development'/path.name);old=read_json(audit/'old-evidence-control/development'/path.name)
        if not len(base)==len(new)==len(old):
            raise ValueError('Incomplete evaluation shard')
        for b,n in zip(base,new):
            compare(b,n)
            extra=model.propose(b['question'],b['docs'],b['id'])
            evidence,ranking=model.memory.search(b['question'],b['id'])
            memory=[d for d in ranking if d not in b['docs']][:4]
            gold=set(b['gold']);present=gold&set(b['docs'])
            before=Fraction(len(present),len(gold))
            for key,docs in (('learned',[r['doc'] for r in extra]),('nearest_neighbor',memory)):
                totals[key].append((before,Fraction(len(gold&(set(b['docs'])|set(docs))),len(gold))))
            recovered=gold&{r['doc'] for r in extra}
            if recovered:
                recoveries.append({'id':b['id'],'question':b['question'],'recovered':sorted(recovered),
                                   'proposals':extra,'existing_memory_recovers':sorted(gold&set(memory))})
            current,_=duo_agreement.predict(b,n)
            values=model.scores(b['question'],b['id'])
            order=sorted(values,key=lambda d:(-values[d],d))
            for doc in gold-set(current):
                available.append({'id':b['id'],'doc':doc,'in_pool':doc in b['docs'],
                    'training_support':model.label_counts[doc],'learned_rank':order.index(doc)+1 if doc in order else None,
                    'learned_score':values.get(doc),'current_fifth':current[4],
                    'fifth_learned_score':values.get(current[4])})
            proposals.append({'id':b['id'],'proposals':extra})
        all_base.extend(base);all_new.extend(new);all_old.extend(old)
    if len(all_base)!=429 or digest([all_base,all_old,all_new])!=read_json(ROOT/'reports/retrain-private-audit/labeled/screen.json')['scores']:
        raise ValueError('Development cache changed')
    report={'plan':digest(plan),'questions':429,'training_questions':len(records),'training_document_labels':len(model.documents),
            'training_cleanup':model.cleaning,
            'coverage':{k:{'before':float(sum(a for a,b in rows)/429),'after':float(sum(b for a,b in rows)/429),
                           'delta':float(sum(b-a for a,b in rows)/429)} for k,rows in totals.items()},
            'learned_gold_instances_recovered':sum(len(r['recovered']) for r in recoveries),
            'candidate_additions':sum(len(r['proposals']) for r in proposals),'recoveries':recoveries,
            'elapsed_seconds':time.monotonic()-started,'final_recall_measured':False,
            'warning':'Historically reused development; coverage is not final recall. No private labels, no validation scoring, no submission.'}
    write_json(out/'report.json',report);write_json(out/'proposals.json',proposals);write_json(out/'misses.json',available)
    print({k:v for k,v in report.items() if k!='recoveries'},flush=True)
    print('Recoveries:',[(r['id'],r['recovered']) for r in recoveries],flush=True)


if __name__=='__main__':
    main()
