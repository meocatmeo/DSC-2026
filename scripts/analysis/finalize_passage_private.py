"""Freeze already verified private evidence with the newly scored 0.97 reference."""
from pathlib import Path
import sys
import zipfile
import json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import agreement_097, passage_private
from legalir2.store import file_hash, read_json
from legalir2.h100 import digest, guard


def main():
    source = ROOT/'reports/passage-private-v1'; out = ROOT/'reports/passage-private-v2'
    plan, prepared = read_json(source/'plan.json'), read_json(source/'prepared.json')
    if prepared['plan'] != digest(plan) or len(prepared['files']) != 2080:
        raise ValueError('Private evidence preparation incomplete')
    parent_plan = digest(plan)
    best = ROOT/'submissions/ready/submission-best-0.9533.zip'
    if file_hash(best) != passage_private.CURRENT_BEST_SHA:
        raise ValueError('0.9533 ZIP changed')
    with zipfile.ZipFile(best) as archive:
        answers = json.loads(archive.read('submission.json'))
    inventory = dict(plan['data_files'], **prepared['files'])
    for name, expected in inventory.items():
        if file_hash(source/name) != expected:
            raise ValueError('Prepared evidence changed: '+name)
        target = out/name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_hash(target) != expected:
            raise ValueError('Finalized output differs: '+name)
        if not target.exists(): target.write_bytes((source/name).read_bytes())
        if name in prepared['files']:
            row = read_json(target)
            predicted, _ = agreement_097.predict(row['original'], row['control'])
            if predicted != answers[row['original']['id']]['answer']:
                raise ValueError('Current-best replay failed')
    path = out/'data/best-09533.zip'
    if path.exists() and file_hash(path) != passage_private.CURRENT_BEST_SHA:
        raise ValueError('Current-best reference changed')
    if not path.exists(): path.write_bytes(best.read_bytes())
    plan.update(code=passage_private.code_identity(), primary_agreement=agreement_097.POLICY,
                preparation_parent_plan=parent_plan)
    plan['data_files']['data/best-09533.zip'] = file_hash(path)
    guard(out/'plan.json', plan)
    prepared.update(plan=digest(plan)); prepared['report']['best_09533_replayed'] = True
    guard(out/'prepared.json', prepared)
    loaded, ready = passage_private.load(out)
    print('Verified private inputs:', sum(1 for _ in passage_private.items(out, ready)), 'plan:', digest(loaded))


if __name__ == '__main__': main()
