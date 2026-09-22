"""Audit the completed full-text experiment using its signed development cache."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from legalir2 import fulltext_prior as prior
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json, write_json

SOURCE = ROOT/'reports/private/20260920-prior-development/remote'
DATASET = ROOT/'reports/private/fulltext-prior-release-check/dataset'


def load():
    marker = read_json(SOURCE/'inputs.json')
    choice = read_json(SOURCE/'calibration.json')
    if choice['inputs'] != digest(marker):
        raise ValueError('Calibration identity changed')
    for name, expected in marker['datasets'].items():
        if file_hash(DATASET/name) != expected:
            raise ValueError('Dataset differs: '+name)
    records = read_json(DATASET/'development.json')
    known = set(read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')['known_docs'])
    rows = []
    for kind in ('base', 'relevance'):
        directory = SOURCE/'cache'/kind/'development'
        if read_json(directory/'resume.json') != {'inputs': digest(marker), 'split': 'development'}:
            raise ValueError('Cache marker differs')
    for start, items in prior.private.shard_items(records, 10):
        name = f'part-{start//10:05d}.json'
        base = read_json(SOURCE/'cache/base/development'/name)
        relevance = read_json(SOURCE/'cache/relevance/development'/name)
        prior.validate_base(base, items, known)
        prior.validate_pointwise(relevance, base)
        rows.extend(dict(b, relevance=r['relevance']) for b, r in zip(base, relevance))
    score_hash = digest([{k: v for k, v in r.items() if k != 'evidence'} for r in rows])
    if score_hash != choice['scores']:
        raise ValueError('Calibrated scores differ')
    return rows


def main():
    rows = load()
    v9 = {r['id']: r for r in read_json(DATASET/'v9-development.json')}
    counts, misses, changed = Counter(), [], []
    recall, oracle, old_pool = [], [], []
    for row in rows:
        docs, gold = row['docs'], set(row['gold'])
        before = prior.predict(row, prior.BASELINE)
        after = prior.predict(row, {'alpha': .25, 'strength': .25})
        relevance = np.array(row['relevance'])
        order = sorted(range(len(docs)), key=lambda i: (-relevance[i], docs[i]))
        rank = {docs[i]: r+1 for r, i in enumerate(order)}
        added = {a['doc'] for a in row['evidence']['additions']}
        recall.append(len(gold & set(before))/len(gold))
        oracle.append(len(gold & set(docs))/len(gold))
        old_pool.append(len(gold & set(v9[row['id']]['docs']))/len(gold))
        counts['added_gold_in_pool'] += len(added & gold)
        counts['added_gold_selected'] += len(added & gold & set(before))
        counts['probability_clipped'] += int((np.abs(relevance) >= 13.8).sum())
        counts['candidates'] += len(docs)
        for doc in gold-set(before):
            counts['missing_outside_pool' if doc not in docs else 'missing_ranked_out'] += 1
        def describe(doc):
            if doc not in docs:
                return {'doc': doc, 'outside_pool': True, 'in_v9_top50': doc in v9[row['id']]['docs']}
            i = docs.index(doc)
            trace = row['evidence']['documents'][doc]
            return {'doc': doc, 'gold': doc in gold, 'added': doc in added,
                'baseline_rank': sorted(docs, key=lambda d: (-(.75*row['prior'][docs.index(d)]+.25*row['ability'][docs.index(d)]), d)).index(doc)+1,
                'relevance_rank': rank[doc], 'relevance_logit': row['relevance'][i],
                'prior': row['prior'][i], 'ability': row['ability'][i], 'title': trace['label'],
                'passages': [p['text'] for p in trace['packed']]}
        if set(before) != set(after):
            gained = (set(after)-set(before)) & gold
            lost = (set(before)-set(after)) & gold
            counts['changed_questions'] += 1
            counts['better_questions'] += bool(len(gained) > len(lost))
            counts['worse_questions'] += bool(len(gained) < len(lost))
            changed.append({'id': row['id'], 'question': row['question'], 'gain': list(gained), 'loss': list(lost),
                'removed': [describe(d) for d in before if d not in after],
                'added': [describe(d) for d in after if d not in before]})
        if gold-set(before):
            misses.append({'id': row['id'], 'question': row['question'],
                'missing': [describe(d) for d in sorted(gold-set(before))],
                'tail': [describe(d) for d in before[-2:]]})
    report = {'counts': dict(counts), 'baseline_recall': float(np.mean(recall)),
        'candidate_pool_oracle': float(np.mean(oracle)), 'v9_top50_oracle': float(np.mean(old_pool)),
        'misses': misses, 'changes': changed,
        'limitation': 'Development diagnostics only; not new validation or private accuracy.'}
    write_json(SOURCE.parent/'audit.json', report)
    print({k: v for k, v in report.items() if k not in ('misses', 'changes')})
    for change in changed:
        if change['gain'] or change['loss']:
            print(json.dumps(change, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
