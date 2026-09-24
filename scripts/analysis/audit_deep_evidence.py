"""Export five diagnosed development misses with verified saved model inputs.

This is an explanatory audit selected after evaluation, not a policy search or
an independent test. It must not be used to hard-code submission corrections.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.h100 import digest
from legalir2.store import read_json, write_json, file_hash


def main():
    out = ROOT / 'reports/append-only-deep-v1'
    targets = {r['id']: r['docs'] for r in read_json(out / 'evaluation.json')['expanded_gold']}
    records = []
    for path in sorted((out / 'raw').glob('part-*.json')):
        comparisons = {r['id']: r for r in read_json(
            out / 'source/cache/comparisons/development' / path.name)}
        for row in read_json(path):
            if row['id'] not in targets:
                continue
            comparison = comparisons[row['id']]
            if comparison['raw'] != digest(row):
                raise ValueError('Saved evidence does not match comparison inputs')
            if row['question'] != comparison['question'] or row['gold'] != comparison['gold']:
                raise ValueError('Question identity changed')
            for doc in targets[row['id']]:
                evidence = row['evidence']['documents'][doc]
                import hashlib
                if hashlib.sha256(json.dumps(evidence['input_ids']).encode()).hexdigest() != evidence['sha256']:
                    raise ValueError('Saved token trace changed')
                if len(evidence['input_ids']) > evidence['budget']:
                    raise ValueError('Saved evidence exceeds token budget')
                scores = row['tables'][0][doc]
                if len(scores) != len(evidence['packed']):
                    raise ValueError('Passage scores do not align with packed evidence')
                records.append({'id': row['id'], 'question': row['question'], 'gold_doc': doc,
                    'raw_sha256': file_hash(path), 'raw_digest': digest(row),
                    'evidence': evidence, 'first_reranker_passage_scores': scores,
                    'second_reranker_bundle_score': row['tables'][1][doc],
                    'comparator_preferences': dict(zip(comparison['docs'],
                        comparison['probabilities'][comparison['docs'].index(doc)]))})
    if len(records) != sum(map(len, targets.values())):
        raise ValueError('Missing or duplicated audit cases')
    write_json(out / 'evidence-audit.json', {'selection': 'post-evaluation error analysis',
        'not_an_independent_evaluation': True, 'cases': records})
    print(f'Exported {len(records)} hash-verified development evidence cases.')


if __name__ == '__main__':
    main()
