"""Apply the unchanged development-tested rule to cached private scores on CPU."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import duo_agreement, fulltext_prior as prior
from legalir2.h100 import digest, guard
from legalir2.private_infer import write_submission
from legalir2.store import read_json, write_json, file_hash
from audit_retrained_private import compare, predictions


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'submissions/private/20260922-duo-agreement-v1')
    args = parser.parse_args()
    source = ROOT/'reports/retrain-private-audit'
    plan = read_json(ROOT/'reports/duo-agreement-v1/plan.json')
    require(plan['policy'] == duo_agreement.POLICY and plan['code'] == file_hash(duo_agreement.__file__),
            'The fixed evaluated policy changed')
    reference = read_json(ROOT/'reports/retrain-cloud-restore-check/reference/inputs.json')
    baseline_zip = source/'baseline/submission-private-duo.zip'
    candidate_zip = source/'candidate/submission-private-duo.zip'
    choice = read_json(source/'candidate/submission-choice.json')
    inputs = read_json(source/'candidate/inputs.json')
    request = read_json(source/'candidate/request.json')
    require(file_hash(baseline_zip) == prior.SCORED_ZIP, 'Scored baseline ZIP differs')
    require(file_hash(candidate_zip) == choice['zip_sha256'], 'Retrained ZIP differs')
    require(read_json(source/'candidate/cache/control/private/resume.json') ==
            {'inputs': digest(inputs), 'model': request['model'], 'split': 'private'}, 'Candidate cache identity differs')
    questions_path = ROOT/'data/private-official.json'
    questions = read_json(questions_path)
    before, retrained = predictions(baseline_zip), predictions(candidate_zip)
    require(set(before) == set(retrained) == set(questions) and len(questions) == 2080, 'Question membership differs')
    known = set(read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')['known_docs'])
    saved_hashes = read_json(source/'source-hashes.json')
    marker = {'schema': 'duo-agreement-private-v1', 'policy': plan,
              'baseline_zip': file_hash(baseline_zip), 'retrained_zip': file_hash(candidate_zip),
              'source_hashes': digest(saved_hashes), 'questions_sha256': file_hash(questions_path),
              'validation_evaluated': False, 'private_labels_used': False,
              'user_requested_submission': True, 'cpu_only': True}
    guard(args.out/'inputs.json', marker)
    result, changes = {}, []
    for name, expected in sorted(reference['source_files'].items()):
        if not name.startswith('cache/duo/part-'):
            continue
        path = source/'baseline'/name
        target = source/'candidate/cache/control/private'/path.name
        require(file_hash(path) == expected == saved_hashes[path.name]['baseline'], 'Baseline shard differs: '+name)
        require(file_hash(target) == saved_hashes[path.name]['candidate'], 'Candidate shard differs: '+name)
        base, candidate = read_json(path), read_json(target)
        require(len(base) == len(candidate), 'Incomplete paired shard')
        for b, n in zip(base, candidate):
            audited = compare(b, n)
            qid = b['id']
            require(qid not in result and qid in questions, 'Duplicate or unexpected question')
            require(b['question'] == questions[qid]['question'] and set(b['docs']) <= known, 'Question or corpus differs')
            require(audited['old'] == before[qid]['answer'] and audited['new'] == retrained[qid]['answer'], 'Source ZIP replay differs')
            answer, trace = duo_agreement.predict(b, n)
            require(answer[:4] == before[qid]['answer'][:4] and len(answer) == len(set(answer)) == 5,
                    'Protected baseline prefix or submission format changed')
            require(len(set(answer)-set(before[qid]['answer'])) <= 1, 'More than one replacement')
            result[qid] = {'answer': answer}
            if trace['replaced']:
                changes.append({'id': qid, 'before': before[qid]['answer'], 'after': answer, 'trace': trace})
    require(set(result) == set(questions), 'Incomplete final predictions')
    write_submission(SimpleNamespace(out=args.out, questions=questions_path), result, 'agreement', known,
                     {'evidence': {'policy': duo_agreement.POLICY['name']}, 'source': marker})
    output = args.out/'submission-private-agreement.zip'
    require(predictions(output) == result, 'Final ZIP content mismatch')
    report = {'questions': len(result), 'changed_sets_vs_09510': len(changes),
              'protected_first_four_verified': True, 'maximum_replacements_per_question': 1,
              'changed_sets_vs_09500': sum(set(v['answer']) != set(retrained[q]['answer']) for q,v in result.items()),
              'zip_sha256': file_hash(output), 'private_score': None,
              'development_recall': .9526029526029526, 'validation_evaluated': False,
              'status': 'Experimental candidate requested by user; no private improvement established.'}
    write_json(args.out/'review.json', report)
    write_json(args.out/'changes.json', changes)
    (args.out/'submission-private-agreement.zip.sha256').write_text(file_hash(output)+'  '+output.name+'\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
