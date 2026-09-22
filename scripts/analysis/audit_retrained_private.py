"""CPU-only integrity, ranking-churn and bounded evidence audit of the retrained duo."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from legalir2 import fulltext_prior as prior
from legalir2.h100 import digest
from legalir2.private_infer import validate_comparison_fit
from legalir2.store import read_json, write_json, file_hash

OUT = ROOT/'reports/retrain-private-audit'
REFERENCE = ROOT/'reports/retrain-cloud-restore-check/reference/inputs.json'


def predictions(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ['submission.json'] and archive.testzip() is None
        return json.loads(archive.read('submission.json'))


def order(row):
    score = .75*np.asarray(row['prior'])+.25*np.asarray(row['ability'])
    return sorted(row['docs'], key=lambda d: (-score[row['docs'].index(d)], d)), score


def validate(row):
    n = len(row['docs'])
    assert n >= 5 and len(set(row['docs'])) == n
    for key in ('prior', 'ability'):
        assert np.asarray(row[key]).shape == (n,) and np.isfinite(row[key]).all()
    p = np.asarray(row['probabilities'])
    assert p.shape == (n, n) and np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all() and np.allclose(p+p.T, 1., atol=1e-6)
    validate_comparison_fit(row)


def compare(base, new):
    validate(base)
    validate(new)
    assert new['base'] == digest(base)
    assert all(base[k] == new[k] for k in ('id', 'question', 'gold', 'docs', 'prior'))
    old_order, old_score = order(base)
    new_order, new_score = order(new)
    old, after = set(old_order[:5]), set(new_order[:5])
    added, removed = sorted(after-old), sorted(old-after)
    docs = base['docs']
    pairs = []
    for a in added:
        for r in removed:
            i, j = docs.index(a), docs.index(r)
            pairs.append({'added': a, 'removed': r, 'old_p_added_beats_removed': base['probabilities'][i][j],
                'new_p_added_beats_removed': new['probabilities'][i][j],
                'old_score_advantage_removed': float(old_score[j]-old_score[i]),
                'new_score_advantage_added': float(new_score[i]-new_score[j])})
    return {'id': base['id'], 'question': base['question'], 'old': old_order[:5], 'new': new_order[:5],
        'added': added, 'removed': removed, 'replacements': len(added), 'pairs': pairs,
        'best_removed_rank': min((old_order.index(d)+1 for d in removed), default=0),
        'deepest_added_old_rank': max((old_order.index(d)+1 for d in added), default=0),
        'old_boundary_margin': float(old_score[docs.index(old_order[4])]-old_score[docs.index(old_order[5])]) if len(docs)>5 else None,
        'new_boundary_margin': float(new_score[docs.index(new_order[4])]-new_score[docs.index(new_order[5])]) if len(docs)>5 else None}


def aggregate(changes, base_rows, new_rows):
    changed = [r for r in changes if r['replacements']]
    pairs = [p for r in changed for p in r['pairs']]
    stats = {'questions': len(changes), 'changed_sets': len(changed),
        'order_only_changes': sum(r['old'] != r['new'] and not r['replacements'] for r in changes),
        'replaced_slots': sum(r['replacements'] for r in changed),
        'replacement_histogram': dict(Counter(r['replacements'] for r in changed)),
        'best_removed_rank_histogram': dict(Counter(r['best_removed_rank'] for r in changed)),
        'added_previous_rank_histogram': dict(Counter(r['deepest_added_old_rank'] for r in changed)),
        'changed_old_margin_quantiles': np.quantile([r['old_boundary_margin'] for r in changed], [0,.25,.5,.75,1]).tolist(),
        'changed_new_margin_quantiles': np.quantile([r['new_boundary_margin'] for r in changed], [0,.25,.5,.75,1]).tolist(),
        'exchange_pairs': len(pairs),
        'new_added_loses_head_to_head': sum(p['new_p_added_beats_removed'] < .5 for p in pairs),
        'old_removed_wins_90pct': sum(p['old_p_added_beats_removed'] < .1 for p in pairs)}
    for label, rows in [('baseline', base_rows), ('candidate', new_rows)]:
        probabilities = np.concatenate([np.asarray(r['probabilities'])[np.triu_indices(len(r['docs']),1)] for r in rows])
        p = np.clip(probabilities, 1e-15, 1-1e-15)
        stats[label+'_pair_confidence'] = {'pairs': len(p), 'fraction_outside_01_09': float(np.mean((p<.1)|(p>.9))),
            'mean_binary_entropy_bits': float(np.mean(-(p*np.log2(p)+(1-p)*np.log2(1-p))))}
    return stats


def main():
    reference = read_json(REFERENCE)
    baseline_zip = OUT/'baseline/submission-private-duo.zip'
    candidate_zip = OUT/'candidate/submission-private-duo.zip'
    assert file_hash(baseline_zip) == prior.SCORED_ZIP
    choice = read_json(OUT/'candidate/submission-choice.json')
    assert file_hash(candidate_zip) == choice['zip_sha256']
    inputs = read_json(OUT/'candidate/inputs.json')
    resume = read_json(OUT/'candidate/cache/control/private/resume.json')
    request = read_json(OUT/'candidate/request.json')
    assert resume == {'inputs': digest(inputs), 'model': request['model'], 'split': 'private'}
    before, after = predictions(baseline_zip), predictions(candidate_zip)
    questions = read_json(ROOT/'data/private-official.json')
    known = set(read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')['known_docs'])
    assert set(before) == set(after) == set(questions) and len(before) == 2080
    changes, base_rows, new_rows, evidence = [], [], [], {}
    source_hashes = {}
    for name, expected in sorted(reference['source_files'].items()):
        if not name.startswith('cache/duo/part-'):
            continue
        path = OUT/'baseline'/name
        assert file_hash(path) == expected, name
        old = read_json(path)
        target = OUT/'candidate/cache/control/private'/path.name
        new = read_json(target)
        source_hashes[path.name] = {'baseline': expected, 'candidate': file_hash(target)}
        assert len(old) == len(new)
        for b, n in zip(old, new):
            c = compare(b, n)
            qid = c['id']
            assert b['question'] == questions[qid]['question']
            assert c['old'] == before[qid]['answer'] and c['new'] == after[qid]['answer']
            assert set(b['docs']) <= known
            changes.append(c)
            base_rows.append(b)
            new_rows.append(n)
            evidence[qid] = b['evidence']
    assert len(changes) == len({r['id'] for r in changes}) == 2080
    summary = {'private': aggregate(changes, base_rows, new_rows),
        'scores_user_reported': {'baseline': .9510, 'candidate': .9500},
        'submission_hashes': {'baseline': file_hash(baseline_zip), 'candidate': file_hash(candidate_zip)},
        'integrity': 'All 2080 cached predictions replay exactly; all referenced baseline shards match frozen hashes; candidate fits and source digests verified.'}
    changed = [r for r in changes if r['replacements']]
    selected, used = [], set()
    categories = [('largest_change', sorted(changed, key=lambda r: (-r['replacements'], r['id']))),
        ('strong_baseline_reversal', sorted(changed, key=lambda r: (-max(p['old_score_advantage_removed'] for p in r['pairs']), r['id']))),
        ('small_new_margin', sorted(changed, key=lambda r: (r['new_boundary_margin'], r['id']))),
        ('hash_control', sorted(changed, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest()))]
    for label, rows in categories:
        picked = 0
        for row in rows:
            if row['id'] in used:
                continue
            used.add(row['id'])
            selected.append(dict(row, selection=label, evidence={d: evidence[row['id']]['documents'][d] for d in row['added']+row['removed']}))
            picked += 1
            if picked == 2:
                break
    write_json(OUT/'changes.json', changes)
    write_json(OUT/'sample.json', selected)
    write_json(OUT/'source-hashes.json', source_hashes)
    # Replay only the already-computed development split. No validation scoring.
    base_dir = ROOT/'reports/fulltext-retrain-release-check/prepare-check/run/reference/cache/base/development'
    dev_b, dev_n, dev_o, dev_changes = [], [], [], []
    labeled_details = []
    for path in sorted(base_dir.glob('part-*.json')):
        b = read_json(path)
        n = read_json(OUT/'labeled/cache/control/development'/path.name)
        o = read_json(OUT/'labeled/cache/old-evidence-control/development'/path.name)
        for old, new, control in zip(b, n, o):
            c = compare(old, new)
            compare(old, control)
            gold = set(old['gold'])
            c.update(gold=sorted(gold), old_hits=sorted(gold & set(c['old'])), new_hits=sorted(gold & set(c['new'])),
                     missing_from_pool=sorted(gold-set(old['docs'])), missed_in_pool=sorted((gold & set(old['docs']))-set(c['new'])))
            c['delta'] = float(Fraction(len(c['new_hits'])-len(c['old_hits']), len(gold)))
            for doc in c['missed_in_pool']:
                prior_order = sorted(new['docs'], key=lambda d: (-new['prior'][new['docs'].index(d)], d))
                ability_order = sorted(new['docs'], key=lambda d: (-new['ability'][new['docs'].index(d)], d))
                labeled_details.append({'id': c['id'], 'question': c['question'], 'gold_doc': doc,
                    'prior_rank': prior_order.index(doc)+1, 'ability_rank': ability_order.index(doc)+1,
                    'final_rank': order(new)[0].index(doc)+1,
                    'p_beats_fifth': new['probabilities'][new['docs'].index(doc)][new['docs'].index(c['new'][4])],
                    'evidence': old['evidence']['documents'][doc]})
            dev_changes.append(c)
        dev_b.extend(b); dev_n.extend(n); dev_o.extend(o)
    screen = read_json(OUT/'labeled/screen.json')
    assert len(dev_changes) == 429 and digest([dev_b, dev_o, dev_n]) == screen['scores']
    summary['development'] = aggregate(dev_changes, dev_b, dev_n)
    summary['development'].update(wins=sum(r['delta']>0 for r in dev_changes), losses=sum(r['delta']<0 for r in dev_changes),
        neutral_changed=sum(r['delta']==0 and r['replacements']>0 for r in dev_changes),
        perfect_recall_questions=sum(set(r['gold'])<=set(r['new']) for r in dev_changes),
        misses_outside_pool=sum(len(r['missing_from_pool']) for r in dev_changes),
        misses_inside_pool=sum(len(r['missed_in_pool']) for r in dev_changes),
        questions_with_outside_misses=sum(bool(r['missing_from_pool']) for r in dev_changes),
        questions_with_inside_misses=sum(bool(r['missed_in_pool']) for r in dev_changes),
        labeled_removed=sum(len(set(r['removed'])&set(r['gold'])) for r in dev_changes),
        labeled_added=sum(len(set(r['added'])&set(r['gold'])) for r in dev_changes))
    summary['development']['candidate_coverage_macro'] = float(sum(Fraction(len(r['gold'])-len(r['missing_from_pool']), len(r['gold'])) for r in dev_changes)/429)
    summary['development']['baseline_recall'] = float(sum(Fraction(len(r['old_hits']), len(r['gold'])) for r in dev_changes)/429)
    summary['development']['candidate_recall'] = float(sum(Fraction(len(r['new_hits']), len(r['gold'])) for r in dev_changes)/429)
    v9_rows = read_json(ROOT/'reports/private/fulltext-prior-release-check/dataset/v9-development.json')
    v9 = {row['id']: row for row in v9_rows}
    admission = []
    for row in dev_changes:
        for doc in row['missing_from_pool']:
            docs = v9[row['id']]['docs']
            admission.append({'id': row['id'], 'gold_doc': doc,
                'cached_v9_position': docs.index(doc)+1 if doc in docs else None})
    summary['development']['outside_pool_present_in_v9'] = sum(r['cached_v9_position'] is not None for r in admission)
    summary['development']['outside_pool_absent_from_saved_v9'] = sum(r['cached_v9_position'] is None for r in admission)
    write_json(OUT/'labeled-admission-misses.json', admission)
    write_json(OUT/'labeled-in-pool-misses.json', labeled_details)
    write_json(OUT/'development-changes.json', dev_changes)
    write_json(OUT/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    text = []
    for row in selected:
        text += [f"QUESTION {row['id']} [{row['selection']}] {row['question']}",
                 f"old={row['old']} new={row['new']}", json.dumps(row['pairs'])]
        for doc, trace in row['evidence'].items():
            text += [f"{'ADDED' if doc in row['added'] else 'REMOVED'} {doc}: {trace['label']}"]
            text += [p['text'] for p in trace['packed']]
        text += ['']
    (OUT/'sample-passages.txt').write_text('\n'.join(text), encoding='utf-8')


if __name__ == '__main__':
    main()
