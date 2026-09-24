"""Fixed, label-free evidence-order ablation; preserve every saved passage token."""
from copy import deepcopy
import hashlib
import json
import math

POLICY = {'name': 'pointwise-passage-order-v1', 'scorer': 'existing-first-reranker',
          'sort': 'descending-score-stable-ties', 'content': 'all-original-packed-tokens',
          'candidate_pool': 'unchanged', 'prior': 'unchanged', 'retrain': False}


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def reorder(trace, scores, codec):
    """Reconstruct before changing anything; reject tokenizer or trace drift."""
    packed = trace['packed']
    if len(scores) != len(packed) or any(not math.isfinite(s) for s in scores):
        raise ValueError('Passage scores must be finite and aligned')
    if not packed or len({p['rowid'] for p in packed}) != len(packed):
        raise ValueError('Expected distinct nonempty packed passages')
    if token_hash(trace['input_ids']) != trace['sha256']:
        raise ValueError('Saved token hash differs')
    metadata = codec.encode(trace['label']+'\n')[:min(96, trace['budget']//5)]
    pieces = [codec.encode(p['text']) for p in packed]
    if any(len(ids) != p['tokens'] or not ids for ids, p in zip(pieces, packed)):
        raise ValueError('Passage text does not reproduce saved token length')

    def assemble(order):
        ids = list(metadata)
        for position, i in enumerate(order):
            ids += (codec.separator if position else codec.encode('\n')) + pieces[i]
        return ids

    original = list(range(len(packed)))
    if assemble(original) != trace['input_ids']:
        raise ValueError('Tokenizer/text does not reproduce the exact saved evidence')
    order = sorted(original, key=lambda i: (-scores[i], i))
    result = deepcopy(trace)
    result['input_ids'] = assemble(order)
    if len(result['input_ids']) != len(trace['input_ids']) or len(result['input_ids']) > trace['budget']:
        raise ValueError('Evidence ordering changed the token budget')
    result['packed'] = [deepcopy(packed[i]) for i in order]
    result['sha256'] = token_hash(result['input_ids'])
    result['ordering'] = {'policy': POLICY['name'], 'original_sha256': trace['sha256'],
                          'indices': order, 'scores': list(scores)}
    return result


def changed_edges(docs, before, after):
    """Only edges incident to a changed input require new model predictions."""
    if len(docs) != len(set(docs)) or set(docs) != set(before) or set(docs) != set(after):
        raise ValueError('Evidence membership must exactly match the frozen pool')
    changed = {i for i, doc in enumerate(docs) if before[doc] != after[doc]}
    return [(i, j) for i in range(len(docs)) for j in range(i+1, len(docs))
            if i in changed or j in changed]


def rescore(local, codec, row, encoded, edges, batch):
    """Reuse unchanged edges, measure changed edges in both A/B orientations."""
    import numpy as np
    from .duo_agreement import scores as validate_scores
    validate_scores(row)
    if batch < 1 or len(set(edges)) != len(edges) or any(
            not 0 <= i < j < len(row['docs']) for i, j in edges):
        raise ValueError('Invalid batch or edge set')
    probability = np.array(row['probabilities'], dtype=float, copy=True)
    examples = [(i, j) for a, b in edges for i, j in ((a, b), (b, a))]
    values = []
    for start in range(0, len(examples), batch):
        ids = [codec.pair(row['question'], [encoded[row['docs'][i]]], [encoded[row['docs'][j]]])
               for i, j in examples[start:start+batch]]
        inputs = codec.tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt')
        with local.torch.inference_mode():
            logits = local.model(**{k: v.to(local.device) for k, v in inputs.items()},
                                 logits_to_keep=1).logits[:, -1, codec.labels].float()
            values.extend(logits.softmax(dim=-1)[:, 0].cpu().tolist())
    if len(values) != len(examples) or not np.isfinite(values).all():
        raise ValueError('Incomplete or non-finite model scores')
    for k, (i, j) in enumerate(edges):
        p = (values[2*k]+1-values[2*k+1])/2
        probability[i, j], probability[j, i] = p, 1-p
    return probability
