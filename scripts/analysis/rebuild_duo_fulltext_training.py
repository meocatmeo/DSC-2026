"""Audit and rebuild full-text duo training examples on CPU; never train a model."""
import argparse
from collections import Counter
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import duo, fulltext_training as ft
from legalir2.duo_evidence import Evidence, index_metadata, open_index
from legalir2.h100 import digest, guard
from legalir2.impact import load_view
from legalir2.store import file_hash, read_json, write_json

_codec = _evidence = _weights = None


def initialize(tokenizer, index, evidence_index, weights, supported):
    global _codec, _evidence, _weights
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    _codec = duo.PairTokenizer(AutoTokenizer.from_pretrained(tokenizer, local_files_only=True), 2048)
    _evidence = Evidence(Path(index), _codec, evidence_index)
    _evidence.supported = set(supported)
    _weights = weights


def compute(task):
    group, row, role, path, marker = task
    result = ft.rebuild(group, row, role, _codec, _evidence, _weights)
    result['inputs'] = marker
    write_json(path, result)
    return result['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'reports/duo-fulltext-training/20260921-v3')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-prepared', type=Path, help='Import compatible already-built nonempty question shards with provenance')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Use 1 to 8 CPU workers')
    source = ROOT/'analysis/impact/duo-20260918T110105Z'
    old_path = ROOT/'reports/duo-fulltext-training/source/training.json.gz'
    signature, ready = read_json(source/'inputs.json'), read_json(source/'training.ready.json')
    if file_hash(old_path) != ready['sha256']:
        raise ValueError('Original training archive hash differs')
    tokenizer, index = ROOT/'artifacts2/duo-tokenizer', ROOT/'artifacts2/frozen-20260918'
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        if file_hash(tokenizer/name) != signature['source_inputs']['files']['models2/partner_qwen/'+name]:
            raise ValueError('Frozen tokenizer differs')
    expected = signature['source_inputs']['files']['artifacts-ranking-h100/store.sqlite']
    metadata = index_metadata(index/'duo-fulltext-v2.sqlite')
    if file_hash(index/'store.sqlite') != expected or metadata['store_sha256'] != expected:
        raise ValueError('Frozen evidence store differs')
    dataset = ROOT/'reports/private/fulltext-prior-release-check/dataset'
    heldout = {s: read_json(dataset/(s+'.json')) for s in ('development', 'validation')}
    if len(heldout['development']) != 429 or len(heldout['validation']) != 1050 or set(heldout['development']) & set(heldout['validation']):
        raise ValueError('External held-out membership differs')
    if set(heldout['development']) != set(signature['development']):
        raise ValueError('Original development split differs')
    original = duo.read_compressed(old_path)
    if len(original) != ready['queries']:
        raise ValueError('Original training count differs')
    selected, roles, exclusions = ft.partition(original, dict(heldout['development'], **heldout['validation']))
    with closing(open_index(index/'duo-fulltext-v2.sqlite')) as db:
        supported = {r[0] for r in db.execute("SELECT DISTINCT doc FROM passages WHERE length(trim(body))>0")}
    selected, empty_exclusions = ft.quarantine_empty(selected, supported)
    exclusions += empty_exclusions
    roles = {g['id']: roles[g['id']] for g in selected}
    train = read_json(ROOT/'data/train.json')
    for group in original:
        if group['question'] != train[group['id']]['question'] or group['gold'] != train[group['id']]['answer']:
            raise ValueError('Original training labels differ')
    source_run = ROOT/'analysis/ranking-h100/results/20260918T013034Z'
    if read_json(source_run/'inputs.json') != signature['source_inputs']:
        raise ValueError('Frozen candidate scoring source differs')
    records = {qid: train[qid] for qid in signature['source_inputs']['train_ids']}
    rows = {r['id']: r for r in load_view(source_run, 'retrieval', 'train', records)}
    plan = {'schema': ft.SCHEMA, 'old_training': ready, 'signature': digest(signature),
        'tokenizer': {n: file_hash(tokenizer/n) for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json')},
        'index': metadata, 'index_sha256': file_hash(index/'duo-fulltext-v2.sqlite'),
        'heldout': {s: digest(r) for s, r in heldout.items()}, 'roles': roles,
        'selected_ids': [g['id'] for g in selected], 'exclusions': exclusions,
        'pair_policy': 'control: exact old pairs; augmented: up to 2 fulltext phrase then V9 top16 negatives',
        'supported_documents': sorted(supported),
        'empty_source_policy': 'Quarantine any original pair group with a document lacking corpus body; exclude unsupported new negatives',
        'monitor_policy': 'normalized-text SHA256 salt fulltext-monitor-v1 modulo10=0; no external dev/val examples',
        'code': {n: file_hash(p) for n, p in [('builder', Path(__file__)), ('policy', Path(ft.__file__)),
            ('duo', Path(duo.__file__)), ('evidence', ROOT/'legalir2/duo_evidence.py')]},
        'limitations': 'Original V9 training scores may be in-sample. Unlabeled relevant negatives may exist. Neither/both labels are not invented. No new training or accuracy claim.'}
    if args.reuse_prepared:
        previous = read_json(args.reuse_prepared/'inputs.json')
        for key in ('old_training', 'signature', 'tokenizer', 'index', 'index_sha256', 'heldout', 'pair_policy', 'monitor_policy'):
            if previous[key] != plan[key]:
                raise ValueError('Cannot import incompatible prepared evidence: '+key)
        for key in ('duo', 'evidence'):
            if previous['code'][key] != plan['code'][key]:
                raise ValueError('Prepared evidence implementation differs')
        plan['import'] = {'directory': str(args.reuse_prepared.resolve()), 'inputs': digest(previous)}
    guard(args.out/'inputs.json', plan)
    marker = digest(plan)
    jobs = []
    for group in selected:
        path = args.out/'groups'/(group['id']+'.json')
        prior_path = args.reuse_prepared/'groups'/(group['id']+'.json') if args.reuse_prepared else None
        if not path.exists() and prior_path and prior_path.exists():
            result = read_json(prior_path)
            if result['inputs'] != plan['import']['inputs'] or result['source'] != digest(group) or result['row'] != digest(rows[group['id']]) or result['role'] != roles[group['id']]:
                raise ValueError('Imported prepared shard differs')
            if all(d in supported and t['packed'] for d, t in result['trace']['documents'].items()):
                result['imported_from'] = {'inputs': result['inputs'], 'sha256': file_hash(prior_path)}
                result['inputs'] = marker
                write_json(path, result)
        if path.exists():
            result = read_json(path)
            if (result['inputs'] != marker or result['source'] != digest(group) or result['row'] != digest(rows[group['id']])):
                raise ValueError('Prepared shard identity changed')
        else:
            jobs.append((group, rows[group['id']], roles[group['id']], path, marker))
    print(f'Eligible={len(selected)} roles={dict(Counter(roles.values()))}; excluded={len(exclusions)}; pending={len(jobs)}', flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
            initargs=(tokenizer, index, index/'duo-fulltext-v2.sqlite', signature['weights'], supported)) as workers:
        for number, qid in enumerate(workers.map(compute, jobs, chunksize=1), 1):
            if number % 10 == 0 or number == len(jobs):
                elapsed = time.monotonic()-started
                print(f'fulltext-training: {number}/{len(jobs)} new; {elapsed/max(1,number):.2f}s/group; eta={(len(jobs)-number)*elapsed/max(1,number)/60:.1f}min', flush=True)
    control, augmented, totals, maxima = [], [], Counter(), Counter()
    for group in selected:
        result = read_json(args.out/'groups'/(group['id']+'.json'))
        control.append(result['control'])
        augmented.append(result['augmented'])
        totals.update(result['stats'])
        for k in ('old_max_pair_tokens', 'new_max_pair_tokens'):
            maxima[k] = max(maxima[k], result[k])
    outputs = {}
    for name, groups in [('control', control), ('hard-negatives', augmented)]:
        path = args.out/(name+'.json.gz')
        duo.write_compressed(path, groups)
        outputs[name] = {'sha256': file_hash(path), 'groups': len(groups),
            'fit_pairs_both_orders': sum(2*len(g['pairs']) for g in groups if g['role'] == 'fit'),
            'monitor_pairs_both_orders': sum(2*len(g['pairs']) for g in groups if g['role'] == 'monitor')}
    report = {'inputs': marker, 'original': ready, 'groups': len(selected), 'roles': dict(Counter(roles.values())),
        'excluded': dict(Counter(r['reason'] for r in exclusions)), 'statistics': dict(totals), 'maxima': dict(maxima),
        'outputs': outputs, 'no_external_holdout_examples': True, 'model_trained': False}
    write_json(args.out/'audit.json', report)
    write_json(args.out/'ready.json', {'inputs': marker, 'outputs': outputs})
    print(report, flush=True)


if __name__ == '__main__':
    main()
