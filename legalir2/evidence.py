"""V11 deterministic evidence packing. Opt-in; no change to deployed v9 scoring.

Consumes retrieval scores and corpus text only. The final token IDs include the
checkpoint's scoring template and are ready for causal yes/no scoring unchanged.
"""
import hashlib
import math

from .models import INSTRUCTION, rerank_prompt

SCHEMA = 'v11-evidence-2048-v1'


def bundle_from_store(store, document_id, query, chunk_prior, tokenizer, template):
    """Use a frozen candidate's existing chunk retrieval scores, without labels."""
    rows = []
    for chunk_id in store.document_chunk_ids(document_id):
        owner, _, _, heading, body = store.chunk(chunk_id)
        if owner != document_id:
            raise ValueError('Chunk owner mismatch')
        rows.append({'id': chunk_id, 'text': body, 'heading': heading,
                     'score': chunk_prior.get(chunk_id, 0.0)})
    return build_bundle(query, store.label(document_id), rows, tokenizer, template)


def build_bundle(query, title, chunks, tokenizer, template, *, max_tokens=2048,
                 max_passages=6, overlap_threshold=0.8):
    """Chunks: {id, text, heading, score}; ties use numeric chunk ID.

    Suppress exact/near duplicates without filling spare slots with duplicates.
    Pack in retrieval-score order. At the budget boundary truncate one passage
    at a word boundary and omit the rest. Record every omission and exact input.
    """
    if not 1 <= max_passages <= 6 or max_tokens < 1:
        raise ValueError('Invalid passage count or token budget')
    if not 0 <= overlap_threshold <= 1:
        raise ValueError('Invalid overlap threshold')
    rows = list(chunks)
    if len({int(c['id']) for c in rows}) != len(rows):
        raise ValueError('Duplicate chunk IDs')
    if any(not math.isfinite(float(c['score'])) for c in rows):
        raise ValueError('Non-finite retrieval score')
    rows.sort(key=lambda c: (-float(c['score']), int(c['id'])))
    selected, dropped, bags = [], [], []
    for c in rows:
        words = set(c['text'].casefold().split())
        reason = None
        if not words:
            reason = 'empty'
        elif any(len(words & b) / max(1, len(words | b)) > overlap_threshold for b in bags):
            reason = 'overlap'
        elif len(selected) >= max_passages:
            reason = 'passage_limit'
        if reason:
            dropped.append({'id': int(c['id']), 'reason': reason})
        else:
            selected.append(c)
            bags.append(words)

    prefix, suffix = rerank_prompt(template)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    instruction = template.get('instruction', INSTRUCTION)
    stem = f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: '
    encode = lambda document: prefix_ids + tokenizer.encode(
        stem + document, add_special_tokens=False) + suffix_ids
    document = title.strip()
    reserved = len(encode(document))
    if reserved >= max_tokens:
        raise ValueError('Query, title and scoring template leave no evidence budget')
    packed = []
    for c in selected:
        heading = c.get('heading', '').strip()
        separator = '\n\n---\n' + (heading + '\n' if heading else '')
        full_text = c['text'].strip()
        proposed = document + separator + full_text
        used = full_text
        if len(encode(proposed)) > max_tokens:
            # Word-prefix search; every accepted prefix is counted using the actual
            # tokenizer, including its interaction with query/title/separators.
            words = full_text.split()
            lo, hi, used = 0, len(words), ''
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = ' '.join(words[:mid])
                if len(encode(document + separator + candidate)) <= max_tokens:
                    used, lo = candidate, mid + 1
                else:
                    hi = mid - 1
            if not used:
                dropped.append({'id': int(c['id']), 'reason': 'token_budget'})
                continue
            proposed = document + separator + used
        document = proposed
        packed.append({'id': int(c['id']), 'retrieval_score': float(c['score']),
                       'heading': heading, 'text': used,
                       'source_tokens': len(tokenizer.encode(full_text, add_special_tokens=False)),
                       'included_tokens': len(tokenizer.encode(used, add_special_tokens=False)),
                       'source_sha256': hashlib.sha256(full_text.encode()).hexdigest(),
                       'truncated': used != full_text})
    ids = encode(document)
    assert len(ids) <= max_tokens
    if not packed and selected:
        raise ValueError('No nonempty evidence fits the budget')
    # Empty-passage documents are still retrieval candidates (their titles are
    # indexed). Score the real title, and preserve the empty provenance, rather
    # than crashing the entire collection when one reaches the top 50.
    return {'schema': SCHEMA, 'query': query, 'document': document,
            'input_ids': ids, 'input_sha256': hashlib.sha256(
                ','.join(map(str, ids)).encode()).hexdigest(),
            'total_tokens': len(ids), 'reserved_tokens': reserved,
            'max_tokens': max_tokens, 'max_passages': max_passages,
            'overlap_threshold': overlap_threshold,
            'selected_chunk_ids': [int(c['id']) for c in selected],
            'packed': packed, 'dropped': dropped,
            'truncated': any(c['truncated'] for c in packed) or
                         any(c['reason'] == 'token_budget' for c in dropped)}
