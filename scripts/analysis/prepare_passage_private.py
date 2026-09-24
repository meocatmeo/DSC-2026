"""Prepare private passage order from existing immutable scores; no GPU or labels."""
from pathlib import Path
import sys
import zipfile
import json
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2 import passage_private as run, passage_order, passage_order_run as dev, duo, deadline_agreement
from legalir2.h100 import digest, guard
from legalir2.store import read_json, file_hash, write_json
from audit_retrained_private import compare


def main():
    from transformers import AutoTokenizer
    out = ROOT/'reports/passage-private-v1'; source = ROOT/'reports/passage-private-source'
    audit = ROOT/'reports/retrain-private-audit'; hashes = read_json(audit/'source-hashes.json')
    dev_plan = read_json(ROOT/'reports/passage-order-v2/plan.json')
    choice = read_json(source/'submission-choice.json')
    expanded = [r for p in sorted((source/'cache/comparisons/private').glob('part-*.json')) for r in read_json(p)]
    if len(expanded) != 2080 or digest(expanded) != choice['private_scores']:
        raise ValueError('Complete saved private deep comparisons required')
    by_id = {r['id']: r for r in expanded}
    if len(by_id) != 2080:
        raise ValueError('Duplicate private comparison')
    models = ROOT/'releases/handoff/20260922-best-09523-v1/model-vault'
    for arm in run.ARMS:
        dev.verify_checkpoint(models/arm, dev_plan['models'][arm])
    codec = duo.PairTokenizer(AutoTokenizer.from_pretrained(models/'original-duo', local_files_only=True, padding_side='left'), 2048)
    questions = read_json(ROOT/'data/private-official.json')
    known = read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')['known_docs']
    best_path = ROOT/'submissions/ready/submission-best-0.9525.zip'
    if file_hash(best_path) != run.BEST_SHA:
        raise ValueError('Best ZIP differs')
    with zipfile.ZipFile(best_path) as archive:
        best = json.loads(archive.read('submission.json'))
    guard(out/'data/questions.json', questions); guard(out/'data/known-docs.json', known)
    target = out/'data/best-09525.zip'
    if target.exists() and file_hash(target) != run.BEST_SHA:
        raise ValueError('Different best ZIP already present')
    if not target.exists():
        target.write_bytes(best_path.read_bytes())
    plan = {'schema': run.SCHEMA, 'policy': passage_order.POLICY,
        'agreement': deadline_agreement.POLICIES['fifth-strict'], 'code': run.code_identity(),
        'models': dev_plan['models'], 'packages': dev.PACKAGES, 'batch': 4,
        'attention': 'flash_attention_2', 'dtype': 'bfloat16', 'replay_tolerance': 1e-4,
        'questions': 2080, 'private_labels_used': False, 'validation_evaluated': False,
        'user_requested_submission': True, 'promotion_gate_passed': False,
        'baseline_source_hashes': digest(hashes), 'raw_scores_linked_by': choice['private_scores'],
        'data_files': {name: file_hash(out/name) for name in ('data/questions.json', 'data/known-docs.json', 'data/best-09525.zip')}}
    guard(out/'plan.json', plan)
    qid_to_shard = {}
    for name, expected in sorted(hashes.items()):
        path = audit/'candidate/cache/control/private'/name
        if file_hash(path) != expected['candidate']:
            raise ValueError('Candidate source changed')
        for row in read_json(path):
            if row['id'] in qid_to_shard:
                raise ValueError('Duplicate candidate source')
            qid_to_shard[row['id']] = name
    cached_name = None; cached = {}; files = {}; seen = set(); changed_docs = total_docs = edge_count = all_edges = 0
    for path in sorted((source/'cache/deep/private').glob('part-*.json')):
        for raw in read_json(path):
            qid = raw['id']
            if qid in seen or qid not in questions or digest(raw) != by_id[qid]['raw']:
                raise ValueError('Raw private score identity differs')
            name = qid_to_shard[qid]
            if name != cached_name:
                first = audit/'baseline/cache/duo'/name; second = audit/'candidate/cache/control/private'/name
                if file_hash(first) != hashes[name]['baseline'] or file_hash(second) != hashes[name]['candidate']:
                    raise ValueError('Comparator shard changed')
                base, new = read_json(first), read_json(second)
                if len(base) != len(new):
                    raise ValueError('Unequal comparator shard lengths')
                cached = {b['id']: (b, n) for b, n in zip(base, new)}; cached_name = name
            b, n = cached[qid]; compare(b, n)
            before_answer, _ = deadline_agreement.predict(b, n, 'fifth-strict')
            if (before_answer != best[qid]['answer'] or b['question'] != questions[qid]['question']
                    or any(raw[k] != b[k] for k in ('id', 'question'))
                    or by_id[qid]['base'] != digest(b) or raw['retained'] != b['docs']):
                raise ValueError('Private baseline replay or original pool differs')
            before, after, ordering = {}, {}, {}
            for doc in b['docs']:
                trace = b['evidence']['documents'][doc]
                if trace != raw['evidence']['documents'][doc]:
                    raise ValueError('Passage scorer used different original evidence')
                reordered = passage_order.reorder(trace, raw['tables'][0][doc], codec)
                before[doc], after[doc] = trace['input_ids'], reordered['input_ids']; ordering[doc] = reordered['ordering']
            edges = passage_order.changed_edges(b['docs'], before, after)
            item = {'original': dev.compact(b), 'control': dev.compact(n), 'before': before, 'after': after,
                'ordering': ordering, 'raw': digest(raw), 'base': digest(b), 'edges': [list(pair) for pair in edges]}
            filename = f'questions/{len(files):05d}.json'
            guard(out/filename, item); files[filename] = file_hash(out/filename); seen.add(qid)
            changed_docs += sum(before[d] != after[d] for d in b['docs']); total_docs += len(b['docs'])
            edge_count += len(edges); all_edges += len(b['docs'])*(len(b['docs'])-1)//2
        print(f'Prepared {len(files)}/2080 private questions.', flush=True)
    if seen != set(questions) or len(files) != 2080:
        raise ValueError('Incomplete private preparation')
    report = {'questions': 2080, 'changed_documents': changed_docs, 'documents': total_docs,
              'pairs_to_rescore_per_model': edge_count, 'total_pairs_per_model': all_edges,
              'best_replayed': True, 'private_score': None}
    guard(out/'prepared.json', {'plan': digest(plan), 'files': files, 'report': report})
    print(report, flush=True)


if __name__ == '__main__':
    main()
