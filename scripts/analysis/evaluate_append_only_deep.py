"""Evaluate append-only deep candidates against the frozen agreement baseline."""
from fractions import Fraction
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legalir2 import append_only_deep as model,duo_agreement
from legalir2.h100 import guard,digest
from legalir2.store import read_json,write_json,file_hash
from legalir2.labeled_gate import summarize
from audit_retrained_private import compare


def main():
    out=ROOT/'reports/append-only-deep-v1'
    source=out/'source';directory=source/'cache/comparisons/development'
    calibration=read_json(source/'calibration.json')
    if calibration!=read_json(ROOT/'reports/private/deep-pilot512-completed/calibration.json'):
        raise ValueError('Unexpected source experiment')
    if digest(read_json(source/'inputs.json'))!=calibration['inputs'] or digest(read_json(source/'fusion.json'))!=calibration['fusion']:
        raise ValueError('Deep run identity differs')
    plan={'policy':model.POLICY,'code':file_hash(model.__file__),'evaluator':file_hash(__file__),
          'anchor_code':file_hash(duo_agreement.__file__),'expanded_scores':calibration['scores'],
          'setting_search':False,'validation_evaluated':False,'private_evaluated':False,
          'frozen_before_reading_comparison_rows':True}
    guard(out/'plan.json',plan)
    expanded=[r for path in sorted(directory.glob('part-*.json')) for r in read_json(path)]
    if len(expanded)!=429 or digest(expanded)!=calibration['scores']:
        raise ValueError('Incomplete or changed expanded comparisons')
    by_id={r['id']:r for r in expanded}
    if len(by_id)!=429:
        raise ValueError('Duplicate expanded questions')
    audit=ROOT/'reports/retrain-private-audit/labeled'
    bases=ROOT/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    all_base,all_new,all_old=[],[],[];pairs=[];changes=[];new_gold=[];promoted_gold=[]
    for path in sorted(bases.glob('part-*.json')):
        base=read_json(path);new=read_json(audit/'cache/control/development'/path.name)
        old=read_json(audit/'cache/old-evidence-control/development'/path.name)
        if len(base)!=len(new) or len(base)!=len(old):
            raise ValueError('Incomplete original paired shard')
        for b,n in zip(base,new):
            compare(b,n);e=by_id[b['id']]
            if b['gold']!=e['gold']:
                raise ValueError('Expanded labels differ')
            current,_=duo_agreement.predict(b,n);answer,trace=model.predict(b,n,e)
            gold=set(b['gold']);before=Fraction(len(gold&set(current)),len(gold));after=Fraction(len(gold&set(answer)),len(gold))
            pairs.append((before,after,set(current)!=set(answer)))
            recovered=gold&(set(e['docs'])-set(b['docs']))
            if recovered:
                new_gold.append({'id':b['id'],'docs':sorted(recovered)})
            if trace['replaced']:
                changes.append({'id':b['id'],'question':b['question'],'before':current,'after':answer,
                                'gold':sorted(gold),'delta':float(after-before),'trace':trace})
                if trace['added'] in recovered:
                    promoted_gold.append({'id':b['id'],'doc':trace['added']})
        all_base.extend(base);all_new.extend(new);all_old.extend(old)
    if len(all_base)!=429 or digest([all_base,all_old,all_new])!=read_json(audit/'screen.json')['scores']:
        raise ValueError('Original comparison identity differs')
    report={'plan':digest(plan),'development':summarize(pairs),'expanded_gold':new_gold,
            'new_gold_promoted':promoted_gold,'no_gpu_compute':True,'final_recall_measured':True,
            'private_score':None,'limitation':'Fixed policy on reused development. Neither independent holdout nor private score guarantee.'}
    write_json(out/'evaluation.json',report);write_json(out/'changes.json',changes)
    print(report,flush=True)


if __name__=='__main__':
    main()
