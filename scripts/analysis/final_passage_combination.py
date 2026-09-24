"""One final fixed CPU trial: extend admission to .95, retaining every .97 rescue."""
from fractions import Fraction
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import agreement_097, passage_combination as previous, passage_combination_final as final
from legalir2.h100 import digest, guard
from legalir2.labeled_gate import summarize
from legalir2.private_infer import write_submission
from legalir2.store import file_hash, read_json, write_json
from combine_passage_scores import verified_rows, zip_predictions, SOURCES

OUT = ROOT/'submissions/private/passage-combination-final-095-v1'
FROZEN = ROOT/'releases/handoff/20260923-best-09538-v1/recovery'
BEST_SHA = 'c179e33fddeab08f0a8d9cd11d8657845f3d3cbd4415df9e3ba4cc15e9d0f4bb'
SCORES_SHA = 'ed84ae72ab7839924d5d395c146851dea90d02aa09739e2a25249a74b14d6eff'


def compare(rows):
    anchor, _ = agreement_097.predict(*rows[:2])
    before, _ = previous.predict(*rows)
    after, trace = final.predict(*rows)
    if after[:4] != before[:4] or (before != anchor and after != before):
        raise ValueError('Final trial changed a protected prefix or an existing winning replacement')
    return before, after, trace, before != anchor


def main():
    if file_hash(FROZEN/'data/scores.json') != SCORES_SHA:
        raise ValueError('Frozen winning scores changed')
    manifest = read_json(FROZEN/'manifest.json')
    if manifest['best_zip_sha256'] != BEST_SHA: raise ValueError('Frozen winner differs')
    for name, expected in manifest['files'].items():
        path = (FROZEN/name).resolve()
        if not path.is_relative_to(FROZEN.resolve()) or file_hash(path) != expected:
            raise ValueError('Recovery source changed: '+name)
    for name in ('passage_combination.py', 'agreement_097.py', 'duo_agreement.py'):
        if file_hash(ROOT/'legalir2'/name) != manifest['files']['legalir2/'+name]:
            raise ValueError('Winning policy code changed')
    plan = {'policy': final.POLICY, 'winning_zip_sha256': BEST_SHA, 'scores_sha256': SCORES_SHA,
        'frozen_manifest_sha256': file_hash(FROZEN/'manifest.json'),
        'code': {name: file_hash(ROOT/'legalir2'/name) for name in
                 ('passage_combination_final.py', 'passage_combination.py', 'agreement_097.py', 'duo_agreement.py')},
        'script': file_hash(__file__), 'verified_loader': file_hash(Path(__file__).with_name('combine_passage_scores.py')),
        'development': SOURCES['development'], 'parameter_sweep': False,
        'private_labels_used': False, 'gpu_rerun_performed': False,
        'limitation': 'One fixed exploratory extension after private leaderboard feedback; reused development, no guarantee of gain.'}
    guard(OUT/'plan.json', plan)
    measured = []; dev_changes = []
    for item, ordered in verified_rows('development'):
        before, after, trace, _ = compare([item['original'], item['control'], *ordered])
        gold = set(item['original']['gold'])
        if not gold: raise ValueError('Missing development labels')
        a, b = (Fraction(len(gold & set(answer)), len(gold)) for answer in (before, after))
        measured.append((a, b, set(before) != set(after)))
        if before != after: dev_changes.append({'id': item['original']['id'], 'before': before,
            'after': after, 'delta': float(b-a), 'trace': trace})
    if len(measured) != 429: raise ValueError('Incomplete development evaluation')
    development = summarize(measured)
    write_json(OUT/'development.json', development); write_json(OUT/'development-changes.json', dev_changes)
    print('Development:', development, flush=True)
    if development['delta'] < 0:
        write_json(OUT/'review.json', {'status': 'Development regression; keep 0.9538.', 'development': development})
        return
    best_path = ROOT/'submissions/ready/submission-best-0.9538.zip'
    if file_hash(best_path) != BEST_SHA: raise ValueError('Ready winner changed')
    best = zip_predictions(best_path); questions = read_json(FROZEN/'data/questions.json')
    known = set(read_json(FROZEN/'data/known-docs.json')); answers = {}; changes = []; retained = 0
    for item in read_json(FROZEN/'data/scores.json'):
        rows = [item[k] for k in ('baseline', 'control', 'ordered_original', 'ordered_control')]
        qid = rows[0]['id']
        before, after, trace, rescued = compare(rows)
        if (qid in answers or before != best[qid]['answer'] or qid != item['id']
                or rows[0]['question'] != questions[qid]['question']):
            raise ValueError('Winning predictions or questions do not replay')
        if len(after) != 5 or len(set(after)) != 5 or not set(after) <= known:
            raise ValueError('Invalid private answer')
        answers[qid] = {'answer': after}; retained += rescued
        if before != after: changes.append({'id': qid, 'before': before, 'after': after, 'trace': trace})
    if len(answers) != 2080 or set(answers) != set(questions) or retained != 43:
        raise ValueError('Incomplete answers or missing winning replacements')
    duplicates = [p.name for p in (ROOT/'submissions/ready').glob('*.zip')
        if p.name != 'submission-final-passage-combination-095.zip' and zip_predictions(p) == answers]
    review = {'questions': 2080, 'changed_sets_vs_09538': len(changes), 'retained_winning_replacements': retained,
        'first_four_preserved': True, 'development': development, 'duplicate_of': duplicates,
        'private_score': None, 'promotion_gate_passed': False, 'plan': digest(plan), 'limitation': plan['limitation']}
    write_json(OUT/'changes.json', changes)
    if not changes or duplicates:
        review['status'] = 'Duplicate candidate; do not spend the final attempt.'
    else:
        write_submission(SimpleNamespace(out=OUT, questions=FROZEN/'data/questions.json'), answers,
                         'passage-combination-final-095', known, {'evidence': final.POLICY, 'source': digest(plan)})
        source = OUT/'submission-private-passage-combination-final-095.zip'
        if zip_predictions(source) != answers: raise ValueError('Packed answers differ')
        ready = ROOT/'submissions/ready/submission-final-passage-combination-095.zip'
        if ready.exists() and file_hash(ready) != file_hash(source): raise ValueError('Existing final trial differs')
        if not ready.exists(): ready.write_bytes(source.read_bytes())
        review.update(status='Final experimental candidate, not submitted or promoted.', zip_sha256=file_hash(source))
    write_json(OUT/'review.json', review); print(review, flush=True)


if __name__ == '__main__': main()
