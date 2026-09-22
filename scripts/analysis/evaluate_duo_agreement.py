"""Fixed CPU-only fifth-slot rescue; development diagnostic, no private submission."""
from fractions import Fraction
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from legalir2 import duo, duo_agreement, fulltext_prior
from legalir2.h100 import digest
from legalir2.labeled_gate import summarize
from legalir2.store import read_json, write_json, file_hash
from legalir2.question_memory import QuestionMemory
from audit_retrained_private import compare, order


def main():
    directory = ROOT/'reports/retrain-private-audit'
    out = ROOT/'reports/duo-agreement-v1'
    # Freeze one general rule before scoring; no threshold sweep or best-of selection.
    plan = {'policy': duo_agreement.POLICY, 'code': file_hash(duo_agreement.__file__),
            'evaluation': 'existing 429 development only', 'validation_evaluated': False,
            'private_evaluated': False, 'parameter_search': False}
    if (out/'plan.json').exists() and read_json(out/'plan.json') != plan:
        raise ValueError('Use a new experiment directory for a changed policy')
    write_json(out/'plan.json', plan)
    base_dir = ROOT/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    base_all, candidate_all, control_all, pairs, retrained_pairs, changes = [], [], [], [], [], []
    for path in sorted(base_dir.glob('part-*.json')):
        base = read_json(path)
        candidate = read_json(directory/'labeled/cache/control/development'/path.name)
        control = read_json(directory/'labeled/cache/old-evidence-control/development'/path.name)
        if not len(base) == len(candidate) == len(control):
            raise ValueError('Incomplete score shard')
        for b, n in zip(base, candidate):
            compare(b, n)
            gold = set(b['gold'])
            if not gold:
                raise ValueError('Development labels required for reporting only')
            before = fulltext_prior.predict(b, fulltext_prior.BASELINE)
            after, trace = duo_agreement.predict(b, n)
            old, new = Fraction(len(gold&set(before)), len(gold)), Fraction(len(gold&set(after)), len(gold))
            pairs.append((old,new,set(before)!=set(after)))
            retrained = fulltext_prior.predict(n, fulltext_prior.BASELINE)
            retrained_pairs.append((Fraction(len(gold&set(retrained)),len(gold)),new,set(retrained)!=set(after)))
            if before != after:
                changes.append(dict(id=b['id'], question=b['question'], before=before, after=after,
                                    delta=float(new-old), gold=sorted(gold), trace=trace))
        base_all.extend(base); candidate_all.extend(candidate); control_all.extend(control)
    screen = read_json(directory/'labeled/screen.json')
    if len(base_all) != 429 or digest([base_all,control_all,candidate_all]) != screen['scores']:
        raise ValueError('Development cache identity differs')
    report = {'plan':plan, 'development':summarize(pairs),
              'versus_retrained_development': summarize(retrained_pairs), 'changes':changes,
              'deployment_authorized_by_evaluation': False,
              'limitation':'Historically inspected development only. No validation/private score or guarantee.'}
    write_json(out/'evaluation.json',report)
    # Identify why the five V9-present gold documents never entered the comparator.
    signature = read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')
    memory = QuestionMemory(signature['memory'])
    v9 = {r['id']:r for r in read_json(ROOT/'reports/private/fulltext-prior-release-check/dataset/v9-development.json')}
    bases = {r['id']:r for r in base_all}
    admissions = []
    for miss in read_json(directory/'labeled-admission-misses.json'):
        if miss['cached_v9_position'] is None:
            continue
        qid,doc = miss['id'],miss['gold_doc']
        row = v9[qid]
        s = np.asarray(row['features1'])@np.asarray(signature['weights'])
        ranked = sorted(range(len(s)), key=lambda i:(-s[i],row['docs'][i]))
        true_rank = next(i+1 for i,j in enumerate(ranked) if row['docs'][j]==doc)
        evidence, ranking = memory.search(row['question'],qid)
        pool, _ = duo.pool(row,signature['weights'],memory,set(signature['known_docs']))
        if not set(pool)<=set(bases[qid]['docs']):
            raise ValueError('Replayed original pool not contained in recorded expanded pool')
        admissions.append(dict(miss, question=row['question'], weighted_v9_rank=true_rank,
            comparator_depth=duo.DEPTH, in_original_pool=doc in pool,
            memory_rank=ranking.index(doc)+1 if doc in ranking else None,
            memory_similarity=evidence.get(doc,[0.])[0],
            phrase_additions=bases[qid]['evidence']['additions']))
    write_json(out/'admission-diagnostics.json',admissions)
    print(report['development'],flush=True)
    print('Changed questions:',[(r['id'],r['delta']) for r in changes],flush=True)
    print('Admission ranks:',[(r['id'],r['weighted_v9_rank'],r['memory_rank'],r['memory_similarity']) for r in admissions])


if __name__ == '__main__':
    main()
