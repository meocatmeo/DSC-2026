"""Independently verify rebuilt pair archives and export a fixed before/after evidence sample."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.cache import normalise_question
from legalir2 import duo
from legalir2.fulltext_training import partition, quarantine_empty
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'reports/duo-fulltext-training/20260921-v3')
    args = parser.parse_args()
    plan = read_json(args.out/'inputs.json')
    ready = read_json(args.out/'ready.json')
    audit = read_json(args.out/'audit.json')
    marker = digest(plan)
    if ready['inputs'] != marker or audit['inputs'] != marker or ready['outputs'] != audit['outputs']:
        raise ValueError('Completion markers differ')
    source_path = ROOT/'reports/duo-fulltext-training/source/training.json.gz'
    if file_hash(source_path) != plan['old_training']['sha256']:
        raise ValueError('Original pair archive changed')
    old = {g['id']: g for g in duo.read_compressed(source_path)}
    datasets = ROOT/'reports/private/fulltext-prior-release-check/dataset'
    heldout = {s: read_json(datasets/(s+'.json')) for s in ('development', 'validation')}
    if {s: digest(r) for s, r in heldout.items()} != plan['heldout']:
        raise ValueError('External held-out identities differ')
    selected, roles, exclusions = partition(list(old.values()), dict(heldout['development'], **heldout['validation']))
    selected, empty_exclusions = quarantine_empty(selected, set(plan['supported_documents']))
    exclusions += empty_exclusions
    roles = {g['id']: roles[g['id']] for g in selected}
    if roles != plan['roles'] or [g['id'] for g in selected] != plan['selected_ids'] or exclusions != plan['exclusions']:
        raise ValueError('Split cannot be reproduced')
    from transformers import AutoTokenizer
    tokenizer_path = ROOT/'artifacts2/duo-tokenizer'
    if any(file_hash(tokenizer_path/n) != sha for n, sha in plan['tokenizer'].items()):
        raise ValueError('Tokenizer changed')
    codec = duo.PairTokenizer(AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True), 2048)
    ids = set(plan['selected_ids'])
    heldout_text = {normalise_question(r['question']) for records in heldout.values() for r in records.values()}
    reports, rebuilt = {}, {}
    for variant in ('control', 'hard-negatives'):
        path = args.out/(variant+'.json.gz')
        if file_hash(path) != ready['outputs'][variant]['sha256']:
            raise ValueError('Prepared archive hash differs: '+variant)
        groups = duo.read_compressed(path)
        if len(groups) != len(ids) or {g['id'] for g in groups} != ids:
            raise ValueError('Prepared archive membership differs')
        count, max_tokens = Counter(), 0
        for group in groups:
            original = old[group['id']]
            trace = read_json(args.out/'groups'/(group['id']+'.json'))
            if trace['inputs'] != marker or trace['source'] != digest(original):
                raise ValueError('Prepared question identity differs')
            expected = trace['control' if variant == 'control' else 'augmented']
            if group != expected or any(group[k] != original[k] for k in ('id', 'question', 'gold')):
                raise ValueError('Prepared labels or evidence differ')
            if group['role'] != roles[group['id']] or group['development'] != (group['role'] == 'monitor'):
                raise ValueError('Prepared fit/monitor role differs')
            if normalise_question(group['question']) in heldout_text:
                raise ValueError('External held-out text leaked')
            pairs = group['pairs']
            if variant == 'control' and pairs != original['pairs']:
                raise ValueError('Control altered training pairs')
            if variant == 'hard-negatives' and pairs[:len(original['pairs'])] != original['pairs']:
                raise ValueError('Augmentation altered original pairs')
            if len(pairs) != len({tuple(p) for p in pairs}) or any(p not in group['gold'] or n in group['gold'] for p, n in pairs):
                raise ValueError('Invalid pair labels')
            if set(group['evidence']) != {d for pair in pairs for d in pair}:
                raise ValueError('Pair/evidence membership differs')
            for doc, pieces in group['evidence'].items():
                item = trace['trace']['documents'][doc]
                if pieces != [item['input_ids']] or len(pieces[0]) > item['budget'] or not item['packed']:
                    raise ValueError('Evidence differs from inference-format token trace')
                overhead = len(codec.prefix+codec.encode(group['question'])[:256]+codec.middle_a+codec.middle_b+codec.suffix)
                if item['budget'] != (codec.max_length-overhead)//2:
                    raise ValueError('Document budget differs from actual pair tokenizer')
                max_tokens = max(max_tokens, overhead+2*len(pieces[0]))
            count[group['role']+'_pairs_both_orders'] += 2*len(pairs)
            count[group['role']+'_groups'] += 1
        if max_tokens > 2048 or count['fit_pairs_both_orders'] != ready['outputs'][variant]['fit_pairs_both_orders'] or count['monitor_pairs_both_orders'] != ready['outputs'][variant]['monitor_pairs_both_orders']:
            raise ValueError('Pair count or token budget mismatch')
        reports[variant] = {**count, 'max_pair_token_upper_bound': max_tokens, 'sha256': file_hash(path)}
        rebuilt[variant] = {g['id']: g for g in groups}
        print(variant, reports[variant], flush=True)
    sample_ids = sorted(ids, key=lambda q: hashlib.sha256(('fulltext-training-review-v1:'+q).encode()).digest())[:16]
    aligned_old = [dict(old[qid], role=roles[qid], development=roles[qid] == 'monitor') for qid in plan['selected_ids']]
    old_control_path = args.out/'old-evidence-control.json.gz'
    duo.write_compressed(old_control_path, aligned_old)
    samples = []
    for qid in sample_ids:
        original = old[qid]
        trace = read_json(args.out/'groups'/(qid+'.json'))
        docs = original['gold']+[original['pairs'][0][1]]
        documents = []
        for doc in docs:
            item = trace['trace']['documents'][doc]
            before = codec.document(original['evidence'][doc], item['budget'])
            documents.append({'doc': doc, 'gold': doc in original['gold'],
                'old_visible_text': codec.tokenizer.decode(before),
                'new_visible_text': codec.tokenizer.decode(item['input_ids']),
                'new_windows': item['packed'], 'label': item['label']})
        samples.append({'id': qid, 'question': original['question'], 'role': roles[qid], 'documents': documents,
                        'extra_negatives': trace['extra']})
    write_json(args.out/'evidence-review.json', {'selection': 'First16 salted-ID SHA256 order, independent of text/labels/scores', 'samples': samples})
    result = {'inputs': marker, 'verification_code': file_hash(Path(__file__)), 'archives': reports,
              'aligned_old_control': {'sha256': file_hash(old_control_path), 'groups': len(aligned_old),
                  'fit_pairs_both_orders': sum(2*len(g['pairs']) for g in aligned_old if g['role'] == 'fit'),
                  'monitor_pairs_both_orders': sum(2*len(g['pairs']) for g in aligned_old if g['role'] == 'monitor')},
              'external_heldout_overlap': 0, 'sample_questions': sample_ids, 'model_trained': False,
              'limitation': 'Structural and provenance verification does not establish better evidence relevance or final recall.'}
    write_json(args.out/'verified.json', result)
    print('Verified complete rebuild and wrote fixed evidence review.', flush=True)


if __name__ == '__main__':
    main()
