"""Evaluate and package one predeclared 0.99 fifth-slot alternative to 0.9525."""
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import strict_tail, deadline_agreement, duo_agreement
from legalir2.h100 import digest, guard
from legalir2.store import read_json, write_json, file_hash
from legalir2.labeled_gate import summarize
from legalir2.private_infer import write_submission
from audit_retrained_private import compare

BEST_HASH = 'd8ea76dd918331fbd21067dfdaed31106d8f99de2d9b852e26c34792c389cae8'


def main():
    out = ROOT/'submissions/private/20260923-fifth-099-v1'
    audit = ROOT/'reports/retrain-private-audit/labeled'
    frozen = ROOT/'releases/handoff/20260923-best-09525-v1/recovery'
    manifest = read_json(frozen/'manifest.json')
    plan = {'policy': strict_tail.POLICY, 'code': file_hash(strict_tail.__file__),
        'parent_code': file_hash(deadline_agreement.__file__), 'anchor_code': file_hash(duo_agreement.__file__),
        'script': file_hash(__file__), 'reference_zip': BEST_HASH,
        'source_manifest': file_hash(frozen/'manifest.json'),
        'development_scores': read_json(audit/'screen.json')['scores'],
        'private_labels_used': False, 'validation_evaluated': False, 'parameter_sweep': False,
        'limitation': 'One speculative threshold chosen after user-reported private scores; development is reused.'}
    guard(out/'plan.json', plan)
    bases = ROOT/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    all_base, all_old, all_new = [], [], []; measured = []; development_changes = []
    for path in sorted(bases.glob('part-*.json')):
        base = read_json(path); new = read_json(audit/'cache/control/development'/path.name)
        old = read_json(audit/'cache/old-evidence-control/development'/path.name)
        if not len(base) == len(new) == len(old):
            raise ValueError('Incomplete development shard')
        for b, n in zip(base, new):
            compare(b, n)
            before, _ = deadline_agreement.predict(b, n, 'fifth-strict')
            after, trace = strict_tail.predict(b, n)
            gold = set(b['gold'])
            a, z = (Fraction(len(gold & set(p)), len(gold)) for p in (before, after))
            measured.append((a, z, set(before) != set(after)))
            if before != after:
                development_changes.append({'id': b['id'], 'before': before, 'after': after,
                                            'delta': float(z-a), 'trace': trace})
        all_base.extend(base); all_old.extend(old); all_new.extend(new)
    if len(measured) != 429 or digest([all_base, all_old, all_new]) != plan['development_scores']:
        raise ValueError('Development source identity changed')
    development = summarize(measured)
    write_json(out/'development.json', development)
    write_json(out/'development-changes.json', development_changes)
    print('Development:', development, flush=True)
    best_path = ROOT/'submissions/ready/submission-best-0.9525.zip'
    if file_hash(best_path) != BEST_HASH or manifest['best_zip_sha256'] != BEST_HASH:
        raise ValueError('Current best differs')
    with zipfile.ZipFile(best_path) as archive:
        best = json.loads(archive.read('submission.json'))
    for name, expected in manifest['files'].items():
        if file_hash(frozen/name) != expected:
            raise ValueError('Frozen recovery source changed: '+name)
    questions = read_json(frozen/'data/questions.json'); known = set(read_json(frozen/'data/known-docs.json'))
    answers = {}; changes = []
    for name in manifest['score_shards']:
        for pair in read_json(frozen/name):
            base, candidate = pair['baseline'], pair['candidate']; qid = base['id']
            before, _ = deadline_agreement.predict(base, candidate, 'fifth-strict')
            if qid in answers or before != best[qid]['answer'] or base['question'] != questions[qid]['question']:
                raise ValueError('Current best does not replay or question differs')
            after, trace = strict_tail.predict(base, candidate)
            if after[:4] != before[:4] or len(set(after)) != 5 or not set(after) <= known:
                raise ValueError('Protected answers or output membership changed')
            answers[qid] = {'answer': after}
            if after != before:
                changes.append({'id': qid, 'before': before, 'after': after, 'trace': trace})
    if len(answers) != 2080 or set(answers) != set(questions) or not changes:
        raise ValueError('Incomplete or duplicate-of-best candidate')
    args = SimpleNamespace(out=out, questions=frozen/'data/questions.json')
    write_submission(args, answers, 'fifth-099', known, {'evidence': {'policy': strict_tail.POLICY},
                     'source': digest(plan), 'reference_private_score': .9525})
    path = out/'submission-private-fifth-099.zip'
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['submission.json'] or archive.testzip() or json.loads(archive.read('submission.json')) != answers:
            raise ValueError('Packed predictions do not match')
    ready = ROOT/'submissions/ready/submission-test-fifth-099.zip'
    if ready.exists() and ready.read_bytes() != path.read_bytes():
        raise ValueError('Ready candidate already differs')
    if not ready.exists():
        ready.write_bytes(path.read_bytes())
    write_json(out/'changes.json', changes)
    review = {'policy': strict_tail.POLICY, 'private_score': None,
        'reference_private_score_user_reported': .9525, 'changed_sets': len(changes),
        'questions': len(answers), 'first_four_preserved': True, 'development': development,
        'zip_sha256': file_hash(path), 'limitation': plan['limitation']}
    write_json(out/'review.json', review)
    print(review, flush=True)


if __name__ == '__main__':
    main()
