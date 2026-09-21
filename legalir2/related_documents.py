"""Retrieve supporting authorities through corpus citations and training co-labels.

Candidate proposals only: a related document is not necessarily an answer.
Held-out question labels must never enter the training records passed here.
"""
from collections import Counter, defaultdict
import math
import re
import unicodedata

from .store import file_hash, read_json, source_records, write_json

SCHEMA = 'related-documents-v1'
NUMBER = re.compile(r'(?<!\w)(\d{1,4})\s*/\s*(\d{4})\s*/\s*([\w]+(?:\s*[-–]\s*[\w]+)*)', re.UNICODE)


def numbers(text):
    text = unicodedata.normalize('NFC', text).upper()
    return set('/'.join((a.lstrip('0') or '0', year, re.sub(r'\s+', '', suffix).replace('–', '-')))
               for a, year, suffix in NUMBER.findall(text))


def build_graph(corpus, store):
    import json
    owners, bodies = defaultdict(set), {}
    for name, raw in source_records(corpus):
        record = json.loads(raw.decode('utf-8'))
        doc = str(record['id'])
        if doc in bodies or doc not in store.doc_position:
            raise ValueError('Duplicate or unknown citation source')
        body = record.get('passage', '')
        bodies[doc] = body
        # Ownership comes from the stored document identity, never a reference
        # appearing later in its body. Ambiguous numbers are excluded.
        for number in numbers(store.label(doc)):
            owners[number].add(doc)
    if set(bodies) != set(store.doc_ids):
        raise ValueError('Incomplete citation corpus')
    unique = {number: next(iter(docs)) for number, docs in owners.items() if len(docs) == 1}
    outgoing = {}
    incoming = defaultdict(list)
    for doc, body in bodies.items():
        links = sorted({unique[n] for n in numbers(body) if n in unique and unique[n] != doc})
        outgoing[doc] = links
        for target in links:
            incoming[target].append(doc)
    return {'schema': SCHEMA, 'numbers': unique, 'outgoing': outgoing,
            'incoming': dict(incoming), 'documents': sorted(bodies)}


class Relations:
    def __init__(self, graph, training):
        if graph['schema'] != SCHEMA:
            raise ValueError('Unsupported citation graph')
        self.graph = graph
        self.known = set(graph['documents'])
        self.frequency = Counter()
        self.pairs = defaultdict(Counter)
        for record in training.values():
            answers = set(record['answer'])
            if not answers or not answers <= self.known:
                raise ValueError('Invalid training answers')
            self.frequency.update(answers)
            for first in answers:
                self.pairs[first].update(answers-{first})

    def propose(self, anchor, existing, traces=None, limit=8, mode='combined'):
        if mode not in ('citations', 'training', 'combined'):
            raise ValueError('Unknown relationship channel')
        if len(set(anchor)) != len(anchor) or not set(anchor) <= self.known:
            raise ValueError('Invalid relationship anchor')
        scores, reasons = defaultdict(float), defaultdict(list)
        blocked = set(existing)
        for rank, source in enumerate(anchor, 1):
            weight = 1/math.sqrt(rank)
            if mode in ('training', 'combined'):
                for target, count in self.pairs[source].items():
                    if count < 2:
                        continue
                    # Shrink rare pairs, discount globally frequent authorities.
                    score = weight*count/(self.frequency[source]+5)/math.sqrt(1+self.frequency[target])
                    scores[target] += score
                    reasons[target].append({'source': source, 'channel': 'training', 'support': count})
            if mode in ('citations', 'combined'):
                local = set()
                trace = (traces or {}).get(source, {})
                text = ' '.join(p['text'] for p in trace.get('packed', []))
                for number in numbers(text):
                    target = self.graph['numbers'].get(number)
                    if target:
                        local.add(target)
                for target in self.graph['outgoing'].get(source, []):
                    degree = len(self.graph['incoming'].get(target, []))
                    score = weight*(.3 if target in local else .03)/math.sqrt(1+degree)
                    scores[target] += score
                    reasons[target].append({'source': source, 'channel': 'local_citation' if target in local else 'citation'})
        ranked = sorted((d for d in scores if d in self.known and d not in blocked), key=lambda d: (-scores[d], d))[:limit]
        return [{'doc': doc, 'score': scores[doc], 'reasons': reasons[doc]} for doc in ranked]
