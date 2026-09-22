"""Propose supporting authorities explicitly mentioned in selected answer passages.

Identity comes from corpus metadata. Ambiguous aliases are excluded. This module
does not see labels, rescore documents or modify the existing answer ranking.
"""
from collections import defaultdict
import math
import re
import unicodedata

from .duo_evidence import overlap

SCHEMA = 'local-authorities-v2'
POLICY = {'schema': SCHEMA, 'anchors': 4, 'maximum_additions': 4,
          'context_characters_each_side': 240, 'minimum_law_name_words': 3}
NUMBER = re.compile(r'(?<!\w)(\d{1,5})\s*/\s*(?:(\d{4})\s*/\s*)?([a-zđ][a-zđ0-9]*(?:\s*[-–]\s*[a-zđ0-9]+)*)(?!\w)', re.I)


def normalized(text):
    text = unicodedata.normalize('NFD', text.casefold().replace('đ','d'))
    return ' '.join(re.sub(r'[^a-z0-9]+',' ', ''.join(c for c in text if not unicodedata.combining(c))).split())


def reference_numbers(text):
    text = unicodedata.normalize('NFC', text).casefold()
    for match in NUMBER.finditer(text):
        num, year, suffix = match.groups()
        suffix = re.sub(r'\s+', '', suffix).replace('–', '-')
        yield '/'.join([str(int(num))]+([year] if year else [])+[suffix]), match.span()


class Authorities:
    def __init__(self, documents):
        self.known = set(documents)
        owners, aliases = defaultdict(set), defaultdict(set)
        versions = defaultdict(set)
        for doc, meta in documents.items():
            # Ownership must come from the dedicated number field, never body citations.
            for number, _ in reference_numbers(meta.get('number', '')):
                owners[number].add(doc)
            title = normalized(meta.get('title',''))
            if not re.match(r'^(bo luat|luat)\b',title):
                continue
            title = re.sub(r'\s+\d{5,}$','',title)
            # Amendments must not claim the identity of the amended law.
            if 'sua doi' in title or 'bo sung' in title:
                continue
            year = re.search(r'\b(?:19|20)\d{2}\b',title)
            if year:
                title = title[:year.end()]
            plain = re.sub(r'\s+(?:nam\s+)?(?:19|20)\d{2}$','',title)
            if len(plain.split()) < POLICY['minimum_law_name_words']:
                continue
            aliases[plain].add(doc)
            if year:
                versions[plain, year[0]].add(doc)
        self.numbers = {n:next(iter(ids)) for n,ids in owners.items() if len(ids)==1}
        self.aliases = {n:next(iter(ids)) for n,ids in aliases.items() if len(ids)==1}
        self.versions = {key:next(iter(ids)) for key,ids in versions.items() if len(ids)==1}
        # Keep ambiguous names in the matcher so a shorter unique alias cannot
        # steal a longer name. Explicit unknown years never fall back to a name.
        names = '|'.join(re.escape(n) for n in sorted(aliases,key=lambda n:(-len(n),n)))
        self.pattern = re.compile(r'(?<!\w)(?P<name>'+names+r')(?!\w)(?:\s+(?:nam\s+)?(?P<year>(?:19|20)\d{2})(?!\w))?') if names else None

    def references(self, text):
        found = []
        for number,(start,end) in reference_numbers(text):
            if number in self.numbers:
                found.append((self.numbers[number], 'number', number, text[max(0,start-240):end+240]))
        normal = normalized(text)
        if self.pattern:
            for match in self.pattern.finditer(normal):
                if not match['year'] and re.match(r'\s+(?:sua doi|bo sung)\b', normal[match.end():]):
                    continue
                target = self.versions.get((match['name'],match['year'])) if match['year'] else self.aliases.get(match['name'])
                if target:
                    found.append((target, 'law_name', match[0], normal[max(0,match.start()-240):match.end()+240]))
        return found

    def propose(self, query, anchor, existing, traces):
        return self.candidates(query,anchor,existing,traces)[:POLICY['maximum_additions']]

    def candidates(self, query, anchor, existing, traces):
        """Unbounded links for coverage diagnostics; inference uses propose."""
        if len(set(anchor)) != len(anchor) or not set(anchor)<=self.known or not set(existing)<=self.known:
            raise ValueError('Invalid corpus identity in authority proposal')
        support = defaultdict(dict)
        for rank,source in enumerate(anchor[:POLICY['anchors']],1):
            for passage in traces.get(source,{}).get('packed',[]):
                for target,channel,reference,context in self.references(passage['text']):
                    if target==source or target in existing:
                        continue
                    longest,coverage = overlap(normalized(query),normalized(context))
                    score = (1.+coverage+min(longest,8)/8)/math.sqrt(rank)
                    reason = {'source':source,'channel':channel,'reference':reference,
                              'context':context,'query_term_coverage':coverage,'phrase_words':longest,'score':score}
                    previous = support[target].get(source)
                    if previous is None or score>previous['score']:
                        support[target][source] = reason
        ranked = sorted(support,key=lambda d:(-sum(v['score'] for v in support[d].values()),d))
        return [{'doc':doc,'score':sum(v['score'] for v in support[doc].values()),
                 'reasons':list(support[doc].values())} for doc in ranked]
