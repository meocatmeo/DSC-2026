"""Evaluate three fixed CPU alternatives and package separate experimental ZIPs."""
import argparse
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legalir2 import deadline_agreement as model, duo_agreement, fulltext_prior
from legalir2.h100 import digest,guard
from legalir2.labeled_gate import summarize
from legalir2.private_infer import write_submission
from legalir2.store import file_hash,read_json,write_json
from audit_retrained_private import compare,predictions

OUT=ROOT/'reports/deadline-agreement-v1'
AUDIT=ROOT/'reports/retrain-private-audit'
BEST=ROOT/'submissions/private/20260922-duo-agreement-v1/submission-private-agreement.zip'
BEST_HASH='37eb1aedbb06620f429c72c6c02de299d5452e628faa44aff816e255c45e2d3f'


def require(condition,message):
    if not condition:
        raise ValueError(message)


def plan():
    value={'policies':model.POLICIES,'code':file_hash(model.__file__),
           'anchor_code':file_hash(duo_agreement.__file__),'evaluator':file_hash(__file__),
           'best_zip':BEST_HASH,'development_questions':429,'validation_evaluated':False,
           'private_labels_used':False,'cpu_only':True,
           'selection_bias':'Three variants on historically reused development; intervals are descriptive, not selection-adjusted.',
           'authorization':'User accepts speculative submissions with four attempts and one day remaining; previous coverage gate is not required.'}
    guard(OUT/'plan.json',value)
    return value


def evaluate():
    frozen=plan()
    base_dir=ROOT/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    all_base,all_old,all_new=[],[],[]
    pairs={name:[] for name in model.POLICIES}; changes={name:[] for name in model.POLICIES}
    for path in sorted(base_dir.glob('part-*.json')):
        base=read_json(path)
        old=read_json(AUDIT/'labeled/cache/old-evidence-control/development'/path.name)
        new=read_json(AUDIT/'labeled/cache/control/development'/path.name)
        require(len(base)==len(old)==len(new),'Incomplete development shard')
        for b,n in zip(base,new):
            compare(b,n)
            current,_=duo_agreement.predict(b,n)
            gold=set(b['gold'])
            before=Fraction(len(gold&set(current)),len(gold))
            for name in model.POLICIES:
                answer,trace=model.predict(b,n,name)
                after=Fraction(len(gold&set(answer)),len(gold))
                pairs[name].append((before,after,set(answer)!=set(current)))
                if answer!=current:
                    changes[name].append({'id':b['id'],'before':current,'after':answer,
                                          'delta':float(after-before),'gold':sorted(gold),'trace':trace})
        all_base.extend(base); all_old.extend(old); all_new.extend(new)
    require(len(all_base)==429 and digest([all_base,all_old,all_new])==read_json(AUDIT/'labeled/screen.json')['scores'],
            'Development score identity differs')
    result={'plan':digest(frozen),'reference':'Current agreement rule, user-reported private 0.9523',
            'development':{name:summarize(rows) for name,rows in pairs.items()},
            'warning':'Exploratory reused development, not a private-score estimate or deployment gate.'}
    write_json(OUT/'development.json',result);write_json(OUT/'development-changes.json',changes)
    print(result,flush=True)


def package():
    frozen=plan(); evaluated=read_json(OUT/'development.json')
    require(evaluated['plan']==digest(frozen),'Evaluation differs')
    require(file_hash(BEST)==BEST_HASH,'Best submission changed')
    best=predictions(BEST)
    baseline_zip=AUDIT/'baseline/submission-private-duo.zip'
    retrained_zip=AUDIT/'candidate/submission-private-duo.zip'
    require(file_hash(baseline_zip)==fulltext_prior.SCORED_ZIP,'Baseline ZIP changed')
    require(file_hash(retrained_zip)==read_json(AUDIT/'candidate/submission-choice.json')['zip_sha256'],'Retrained ZIP changed')
    before,retrained=predictions(baseline_zip),predictions(retrained_zip)
    inputs=read_json(AUDIT/'candidate/inputs.json');request=read_json(AUDIT/'candidate/request.json')
    require(read_json(AUDIT/'candidate/cache/control/private/resume.json')==
            {'inputs':digest(inputs),'model':request['model'],'split':'private'},'Resume provenance differs')
    question_path=ROOT/'data/private-official.json';questions=read_json(question_path)
    require(set(best)==set(before)==set(retrained)==set(questions) and len(best)==2080,'Question membership differs')
    known=set(read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')['known_docs'])
    hashes=read_json(AUDIT/'source-hashes.json')
    reference=read_json(ROOT/'reports/retrain-cloud-restore-check/reference/inputs.json')
    results={name:{} for name in model.POLICIES};changes={name:[] for name in model.POLICIES}
    for key,expected in sorted(reference['source_files'].items()):
        if not key.startswith('cache/duo/part-'):
            continue
        path=AUDIT/'baseline'/key;newpath=AUDIT/'candidate/cache/control/private'/path.name
        require(file_hash(path)==expected==hashes[path.name]['baseline'],'Baseline shard changed')
        require(file_hash(newpath)==hashes[path.name]['candidate'],'Candidate shard changed')
        base,new=read_json(path),read_json(newpath)
        require(len(base)==len(new),'Incomplete private shard')
        for b,n in zip(base,new):
            audited=compare(b,n);qid=b['id']
            require(qid in questions and b['question']==questions[qid]['question'] and set(b['docs'])<=known,'Private identity differs')
            require(audited['old']==before[qid]['answer'] and audited['new']==retrained[qid]['answer'],'Source ZIP replay differs')
            current,_=duo_agreement.predict(b,n)
            require(current==best[qid]['answer'],'Best submission replay differs')
            for name in model.POLICIES:
                require(qid not in results[name],'Duplicate question')
                answer,trace=model.predict(b,n,name)
                require(len(answer)==len(set(answer))==5 and set(answer)<=known,'Invalid final answer')
                require(len(set(answer)-set(current))<=1,'More than one change versus best')
                results[name][qid]={'answer':answer}
                if answer!=current:
                    changes[name].append({'id':qid,'before':current,'after':answer,'trace':trace})
    summary={}
    root=ROOT/'submissions/private/20260922-deadline-agreement-v1'
    marker={'plan':digest(frozen),'sources':digest(hashes),'questions':file_hash(question_path),
            'best_zip':BEST_HASH,'private_labels_used':False,'validation_evaluated':False}
    guard(root/'inputs.json',marker)
    for name,result in results.items():
        require(set(result)==set(questions),'Incomplete predictions')
        out=root/name
        write_submission(SimpleNamespace(out=out,questions=question_path),result,name,known,
                         {'evidence':{'policy':name},'source':marker})
        archive=out/f'submission-private-{name}.zip'
        require(predictions(archive)==result,'Final ZIP replay differs')
        report={'policy':model.POLICIES[name],'changed_sets_vs_09523':len(changes[name]),
                'questions':len(result),'zip_sha256':file_hash(archive),
                'development_vs_agreement':evaluated['development'][name],
                'private_score':None,'status':'Speculative candidate; no score increase established.'}
        write_json(out/'review.json',report);write_json(out/'changes.json',changes[name])
        (out/(archive.name+'.sha256')).write_text(file_hash(archive)+'  '+archive.name+'\n')
        summary[name]=report
    summary['pairwise_changed_sets']={a+' vs '+b:sum(set(results[a][q]['answer'])!=set(results[b][q]['answer']) for q in questions)
        for i,a in enumerate(model.POLICIES) for b in list(model.POLICIES)[i+1:]}
    write_json(root/'summary.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('evaluate','package'))
    args=parser.parse_args()
    {'evaluate':evaluate,'package':package}[args.stage]()
