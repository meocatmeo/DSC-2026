"""Resumable BF16 ViRanker scores on the union of frozen current-component pools.

No retraining, baseline reranking, private inference or submission is performed.
Run ``python -m legalir2.fourth_voter --help`` for the labeled scoring command.
"""
import argparse
from contextlib import closing
from pathlib import Path
import time

import numpy as np

from .current_fusion import COMPONENTS, parameter_audit, score_order
from .download import verify
from .duo_evidence import Evidence, index_metadata
from .h100 import digest, guard
from .models import checkpoint_identity
from .store import file_hash, read_json, write_json

REPOSITORY = 'namdp-ptit/ViRanker'
SCHEMA = 'current-fourth-voter-v1'


def pool(row):
    rankings = row['rankings']
    if set(rankings) != set(COMPONENTS):
        raise ValueError('Expected all three current component orderings')
    for values in rankings.values():
        if len(values) < 5 or len(values) != len(set(values)):
            raise ValueError('Invalid source ordering')
    return list(dict.fromkeys(d for name in COMPONENTS for d in rankings[name]))


def passages(evidence, query, doc):
    """Two document-local full-text hits; deterministic corpus prefix fallback."""
    hits = evidence.search(query, doc=doc, limit=2)
    if hits:
        return [label+'\n'+body for _, _, _, label, body, _ in hits]
    fallback = evidence.db.execute('SELECT label,body FROM passages WHERE doc=? ORDER BY start,rowid LIMIT 2', (doc,)).fetchall()
    if not fallback:
        raise ValueError('No full-text corpus windows for '+doc)
    return [label+'\n'+body for label, body in fallback]


def validate_result(result, row, marker):
    docs = pool(row)
    if result['input'] != digest(row) or result['run'] != marker or result['id'] != row['id'] or result['docs'] != docs:
        raise ValueError('Fourth-voter checkpoint belongs to different inputs')
    if score_order(docs, result['scores']) != result['ordering']:
        raise ValueError('Fourth-voter ranking does not replay')
    if len(result['passage_sha256']) != len(docs):
        raise ValueError('Incomplete fourth-voter evidence identity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True, help='Complete labeled split JSON; private null labels are rejected')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--base-audit', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True, help='Directory containing frozen store.sqlite')
    parser.add_argument('--evidence-index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    if args.batch < 1:
        raise ValueError('Batch size must be positive')
    bundle, records = read_json(args.rankings), read_json(args.questions)
    rows = bundle['rows']
    if len(rows) != len(records) or {r['id'] for r in rows} != set(records):
        raise ValueError('Incomplete labeled ranking export')
    for row in rows:
        record = records[row['id']]
        if not record['answer'] or row['gold'] != record['answer'] or row['question'] != record['question']:
            raise ValueError('Only matching nonempty labeled questions are accepted')
        pool(row)
    provenance = verify(args.model)
    if provenance.get('repository') != REPOSITORY or not provenance.get('files') or not provenance.get('commit'):
        raise ValueError('Require an immutable verified off-the-shelf ViRanker checkpoint')
    metadata = index_metadata(args.evidence_index)
    if metadata['store_sha256'] != file_hash(args.index/'store.sqlite'):
        raise ValueError('Full-text evidence and frozen corpus differ')
    base = read_json(args.base_audit)
    value = {'schema': SCHEMA, 'rankings': file_hash(args.rankings), 'questions': file_hash(args.questions),
        'model': checkpoint_identity(provenance), 'base_audit': base, 'index': metadata,
        'evidence_sha256': file_hash(args.evidence_index),
        'code': {n: file_hash(Path(__file__).with_name(n)) for n in
                 ('fourth_voter.py', 'current_fusion.py', 'rerank.py', 'models.py', 'duo_evidence.py')},
        'precision': 'bfloat16', 'quantization': None, 'passages': 2, 'max_length': 512,
        'aggregation': 'highest + 0.2 * second-highest logit', 'batch': args.batch}
    guard(args.out/'inputs.json', value)
    marker = digest(value)
    from .rerank import Reranker
    model = None
    started, completed = time.monotonic(), 0
    results = []
    try:
        with closing(Evidence(args.index, None, args.evidence_index)) as evidence:
            known = set(evidence.store.doc_position)
            for number, row in enumerate(rows, 1):
                docs = pool(row)
                if not set(docs) <= known:
                    raise ValueError('Unknown fourth-voter candidate')
                path = args.out/'scores'/f'part-{number-1:05d}.json'
                if path.exists():
                    result = read_json(path)
                    validate_result(result, row, marker)
                else:
                    texts = [passages(evidence, row['question'], d) for d in docs]
                    if model is None:
                        model = Reranker(args.model, device='cuda', max_length=512)
                        torch = model.local.torch
                        if (getattr(model.local.model, 'is_quantized', False) or
                                any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in model.local.model.parameters())):
                            raise ValueError('This experiment requires unquantized BF16 weights')
                        guard(args.out/'audit.json', parameter_audit(base, model.parameters))
                    logits = model.score(row['question'], [text for group in texts for text in group], args.batch, 512)
                    if len(logits) != sum(map(len, texts)) or not np.isfinite(logits).all():
                        raise ValueError('Incomplete or nonfinite fourth-voter scores')
                    scores, offset = [], 0
                    for group in texts:
                        ranked = sorted(logits[offset:offset+len(group)], reverse=True)
                        scores.append(ranked[0]+(.2*ranked[1] if len(ranked) > 1 else 0.))
                        offset += len(group)
                    result = {'id': row['id'], 'input': digest(row), 'run': marker, 'docs': docs,
                        'scores': scores, 'ordering': score_order(docs, scores),
                        'passage_sha256': [digest(group) for group in texts]}
                    validate_result(result, row, marker)
                    write_json(path, result)
                    completed += 1
                results.append(result)
                if number % 5 == 0 or number == len(rows):
                    seconds = (time.monotonic()-started)/max(1, completed)
                    print(f'fourth-voter: {number}/{len(rows)}; new={completed}; {seconds:.2f}s/new question', flush=True)
        saved_audit = read_json(args.out/'audit.json')
        if saved_audit != parameter_audit(base, saved_audit['components']['fourth_reranker']):
            raise ValueError('Saved model budget audit differs')
        write_json(args.out/'rankings.json', {'inputs': marker, 'source_rankings': value['rankings'], 'rows': results})
        print('Labeled fourth-voter rankings ready; no submission generated.', flush=True)
    finally:
        if model is not None:
            model.local.close()


if __name__ == '__main__':
    main()
