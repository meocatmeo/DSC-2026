"""CPU rebuild of duo training evidence, isolated from frozen inference artifacts."""
from collections import Counter, defaultdict
import hashlib

from .cache import normalise_question
from .h100 import digest

SCHEMA = 'duo-fulltext-training-v1'


def partition(groups, heldout):
    """Exclude held-out IDs/text, quarantine conflicting duplicate labels, group monitor split."""
    blocked = {normalise_question(r['question']) for r in heldout.values()}
    eligible, exclusions = defaultdict(list), []
    if len({g['id'] for g in groups}) != len(groups):
        raise ValueError('Duplicate original group IDs')
    for group in groups:
        text = normalise_question(group['question'])
        if group['development'] or group['id'] in heldout or text in blocked:
            exclusions.append({'id': group['id'], 'reason': 'external-heldout-id-or-text'})
        else:
            eligible[text].append(group)
    selected, roles = [], {}
    for text, matches in sorted(eligible.items()):
        if len({tuple(sorted(g['gold'])) for g in matches}) > 1:
            exclusions.extend({'id': g['id'], 'reason': 'conflicting-duplicate-labels'} for g in matches)
            continue
        matches.sort(key=lambda g: g['id'])
        group = matches[0]
        exclusions.extend({'id': g['id'], 'reason': 'duplicate-training-text'} for g in matches[1:])
        # This monitor is newly withheld from this rebuild, not a fresh model-level
        # holdout: the previous checkpoint may have trained on these questions.
        monitor = int(hashlib.sha256(('fulltext-monitor-v1:'+text).encode()).hexdigest(), 16) % 10 == 0
        roles[group['id']] = 'monitor' if monitor else 'fit'
        selected.append(group)
    return sorted(selected, key=lambda g: g['id']), roles, exclusions


def challengers(group, v9_order, phrase_docs, limit=2):
    """Training-only negatives: phrase additions first, then unrepresented high V9 ranks."""
    gold = set(group['gold'])
    original = {n for p, n in group['pairs']}
    selected = []
    for doc, origin in [(d, 'fulltext-phrase') for d in phrase_docs]+[(d, 'v9-rank') for d in v9_order[:16]]:
        if doc in gold or doc in original or any(d['doc'] == doc for d in selected):
            continue
        selected.append({'doc': doc, 'source': origin})
        if len(selected) == limit:
            break
    return selected


def quarantine_empty(groups, supported):
    clean, excluded = [], []
    for group in groups:
        missing = sorted({d for pair in group['pairs'] for d in pair}-set(supported))
        if missing:
            excluded.append({'id': group['id'], 'reason': 'empty-corpus-evidence', 'documents': missing,
                             'gold_documents': sorted(set(missing) & set(group['gold']))})
        else:
            clean.append(group)
    return clean, excluded


def validate_group(group, codec, known, allow_empty=False):
    gold = set(group['gold'])
    if not gold or not gold <= known or len(gold) != len(group['gold']):
        raise ValueError('Invalid training positives')
    pairs = [tuple(p) for p in group['pairs']]
    if not pairs or len(pairs) != len(set(pairs)) or any(p not in gold or n in gold or n not in known for p, n in pairs):
        raise ValueError('Invalid or contradictory A/B training pairs')
    docs = set(d for pair in pairs for d in pair)
    if docs != set(group['evidence']) or not docs <= known:
        raise ValueError('Pair/evidence document mismatch')
    for pieces in group['evidence'].values():
        if (not allow_empty and (not pieces or any(not piece for piece in pieces))) or any(any(type(t) is not int or t < 0 for t in piece) for piece in pieces):
            raise ValueError('Invalid tokenized evidence')
    lengths = []
    for p, n in pairs:
        lengths.extend((len(codec.pair(group['question'], group['evidence'][p], group['evidence'][n])),
                        len(codec.pair(group['question'], group['evidence'][n], group['evidence'][p]))))
    return max(lengths)


def rebuild(group, row, role, codec, evidence, weights):
    """Keep the control pairs exact; build a separate hard-negative augmentation."""
    import numpy as np
    if any(group[k] != row[k] for k in ('id', 'question', 'gold')):
        raise ValueError('Original training and current ranking cache differ')
    if not hasattr(evidence, 'supported'):
        raise ValueError('Evidence index must declare supported corpus documents')
    known = set(evidence.store.doc_position)
    old_max = validate_group(group, codec, known, allow_empty=True)
    scores = np.asarray(row['features1'])@np.asarray(weights)
    order = [d for _, d in sorted(zip(-scores, row['docs']))]
    proposed, _ = evidence.expand(group['question'], order[:16], sorted(scores, reverse=True)[:16], known)
    supported = evidence.supported
    extra = challengers(group, [d for d in order[:16] if d in supported], [d for d in proposed[16:] if d in supported])
    old_docs = list(dict.fromkeys(d for pair in group['pairs'] for d in pair))
    docs = old_docs+[r['doc'] for r in extra]
    encoded = evidence.encode(group['question'], docs)
    if any(not evidence.trace['documents'][d]['packed'] for d in docs):
        raise ValueError('No usable corpus body for a rebuilt training document')
    common = {k: group[k] for k in ('id', 'question', 'gold')}
    common.update(development=role == 'monitor', role=role)
    control = dict(common, pairs=group['pairs'], evidence={d: encoded[d] for d in old_docs})
    added_pairs = [[p, r['doc']] for p in group['gold'] for r in extra]
    augmented = dict(common, pairs=group['pairs']+added_pairs, evidence=encoded)
    maximum = max(validate_group(control, codec, known), validate_group(augmented, codec, known))
    stats = Counter(documents=len(old_docs), gold_documents=len(group['gold']), old_pairs=len(group['pairs']),
                    extra_pairs=len(added_pairs), extra_negatives=len(extra))
    changes = []
    for doc in old_docs:
        trace = evidence.trace['documents'][doc]
        before = codec.document(group['evidence'][doc], trace['budget'])
        after = encoded[doc][0]
        changed = before != after
        stats['changed_documents'] += changed
        stats['changed_gold_documents'] += changed and doc in group['gold']
        stats['old_visible_tokens'] += len(before)
        stats['new_visible_tokens'] += len(after)
        stats['old_unpacked_tokens'] += sum(map(len, group['evidence'][doc]))
        stats['old_empty_documents'] += not before
        stats['old_empty_gold_documents'] += not before and doc in group['gold']
        stats['new_documents_without_body'] += not trace['packed']
        stats['gold_with_window_after_word320'] += doc in group['gold'] and any(p['start_word'] >= 320 for p in trace['packed'])
        changes.append({'doc': doc, 'gold': doc in group['gold'], 'changed': changed,
                        'old_visible_tokens': len(before), 'new_visible_tokens': len(after),
                        'old_visible_sha256': digest(before), 'new_visible_sha256': digest(after)})
    return {'id': group['id'], 'source': digest(group), 'row': digest(row), 'role': role,
            'control': control, 'augmented': augmented, 'extra': extra, 'stats': dict(stats),
            'old_max_pair_tokens': old_max, 'new_max_pair_tokens': maximum,
            'changes': changes, 'trace': evidence.trace}
