"""Label-free FTS candidate proposals for an offline coverage threshold sweep."""
from itertools import product
import math

from .duo_evidence import open_index, overlap, tokens

SCHEMA = 'fts-threshold-coverage-v1'
POLICIES = [dict(phrase_words=p, term_coverage=c, extra=e, search_limit=n)
            for p, c, e, n in product((3, 4, 5, 6), (.25, .4, .55), (2, 4, 8), (96, 256))]


def phrase_query(query, size):
    if size not in (3, 4, 5, 6):
        raise ValueError('Unsupported phrase length')
    stream = tokens(query)
    phrases = dict.fromkeys(' '.join(stream[i:i+size]) for i in range(len(stream)-size+1))
    return ' OR '.join('"'+p+'"' for p in phrases)


def additions(hits, existing, known, policy):
    if policy not in POLICIES:
        raise ValueError('Unregistered FTS policy')
    excluded = set(existing)
    ranked = []
    for hit in hits[:policy['search_limit']]:
        if hit['doc'] not in known or hit['doc'] in excluded:
            continue
        if not math.isfinite(hit['bm25']) or not 0 <= hit['coverage'] <= 1:
            raise ValueError('Invalid FTS score')
        if hit['longest'] >= policy['phrase_words'] and hit['coverage'] >= policy['term_coverage']:
            ranked.append(hit)
    ranked.sort(key=lambda r: (-r['longest'], -r['coverage'], r['bm25'], r['doc'], r['rowid'], r['start']))
    result = []
    for hit in ranked:
        if hit['doc'] in excluded:
            continue
        result.append(hit['doc'])
        excluded.add(hit['doc'])
        if len(result) == policy['extra']:
            break
    return result


class Search:
    def __init__(self, path):
        self.db = open_index(path)

    def hits(self, query, phrase_words):
        expression = phrase_query(query, phrase_words)
        if not expression:
            return []
        rows = self.db.execute('SELECT rowid,doc,start,body,bm25(passages,0,0,0.2,1) AS score '
            'FROM passages WHERE passages MATCH ? ORDER BY score,rowid LIMIT 256',
            ('body : ('+expression+')',)).fetchall()
        result = []
        for rowid, doc, start, body, score in rows:
            longest, coverage = overlap(query, body)
            result.append({'rowid': rowid, 'doc': doc, 'start': start, 'bm25': score,
                           'longest': longest, 'coverage': coverage})
        return result

    def close(self):
        self.db.close()
