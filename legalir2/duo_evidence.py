"""Full-corpus evidence for the saved duo comparator, without rebuilding V9.

Sliding windows cover every source word, including text before recovered legal
headings. A separate SQLite FTS index proposes exact-phrase candidates and finds
document-local evidence. No question IDs, answer labels or learned weights.
"""
import argparse
from contextlib import closing
from functools import lru_cache
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from .store import Store, file_hash, source_records

SCHEMA = 'duo-fulltext-v2'
WINDOW = 220
OVERLAP = 64
EXTRA_CANDIDATES = 2
TOKEN = re.compile(r'\w+', re.UNICODE)


@lru_cache(maxsize=512)
def tokens(text):
    return tuple(TOKEN.findall(unicodedata.normalize('NFC', text).casefold()))


def windows(text, size=WINDOW, overlap=OVERLAP):
    words = unicodedata.normalize('NFC', text).split()
    for start in range(0, len(words), size - overlap):
        yield start, ' '.join(words[start:start + size])
        if start + size >= len(words):
            break


def match_query(query):
    # Quotes are constructed from tokens, never from caller SQL or FTS syntax.
    return ' OR '.join('"' + t + '"' for t in dict.fromkeys(tokens(query)))


def phrase_query(query):
    stream = tokens(query)
    phrases = dict.fromkeys(' '.join(stream[i:i + 6]) for i in range(len(stream) - 5))
    return ' OR '.join('"' + p + '"' for p in phrases)


@lru_cache(maxsize=1024)
def overlap(query, text):
    """Contiguous phrase agreement and distinct query-term coverage."""
    q, body = tokens(query), tokens(text)
    if not q or not body:
        return 0, 0.
    body_set = set(body)
    coverage = len(set(q) & body_set) / len(set(q))
    # Longest common contiguous run, capped at the original 24-token limit.
    # Track matching query positions only, instead of materializing sets of
    # every 2..24-token source n-gram for every candidate passage.
    positions = {}
    for i, token in enumerate(q):
        positions.setdefault(token, []).append(i)
    previous, longest = {}, 0
    for token in body:
        current = {i: previous.get(i - 1, 0) + 1 for i in positions.get(token, ())}
        if current:
            longest = max(longest, max(current.values()))
            if longest >= 24:
                return 24, coverage
        previous = current
    return longest if longest >= 2 else 0, coverage


def build(corpus, store_path, target):
    """Build an independent index atomically; verify complete corpus membership."""
    target = Path(target)
    if target.resolve() == Path(store_path).resolve():
        raise ValueError('Evidence index cannot overwrite the frozen store')
    if target.exists():
        raise ValueError('Evidence index exists; use a new filename for a rebuild')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.building')
    if temporary.exists():
        raise ValueError('Incomplete evidence build exists; choose a new filename')
    store = Store(store_path)
    db = sqlite3.connect(temporary)
    count, seen, digest = 0, set(), hashlib.sha256()
    try:
        db.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE passages USING fts5(doc, start UNINDEXED,
                label, body, tokenize='unicode61 remove_diacritics 0');
        ''')
        for name, raw in source_records(corpus):
            record = json.loads(raw.decode('utf-8'))
            doc = str(record['id'])
            if doc in seen or doc not in store.doc_position:
                raise ValueError(f'Duplicate or unknown corpus document: {doc}')
            seen.add(doc)
            digest.update(name.encode() + b'\0' + raw + b'\0')
            for start, body in windows(record.get('passage', '')):
                db.execute('INSERT INTO passages(doc,start,label,body) VALUES (?,?,?,?)',
                           (doc, start, store.label(doc), body))
                count += 1
            if len(seen) % 1000 == 0:
                print(f'Full-text evidence: {len(seen)} documents, {count} windows', flush=True)
        if seen != set(store.doc_ids):
            raise ValueError('Evidence corpus is incomplete')
        # The source store records this exact source-byte digest. Refuse a newer
        # corpus with matching IDs but different bodies against a frozen model.
        if digest.hexdigest() != store.metadata['corpus_sha256']:
            raise ValueError('Evidence corpus bytes differ from the frozen store source')
        metadata = {'schema': SCHEMA, 'store_sha256': file_hash(store_path),
                    'corpus_sha256': digest.hexdigest(), 'documents': len(seen),
                    'windows': count, 'window_words': WINDOW, 'overlap_words': OVERLAP,
                    'search_format': 2}
        db.execute('INSERT INTO metadata VALUES (?,?)', ('index', json.dumps(metadata)))
        db.commit()
    finally:
        db.close()
        store.close()
    temporary.replace(target)
    return metadata


def open_index(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Build the full-text evidence index first: {path}')
    return sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)


def index_metadata(path):
    with closing(open_index(path)) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='index'").fetchone()
        if row is None:
            raise ValueError('Incomplete full-text evidence index')
        metadata = json.loads(row[0])
        if metadata['schema'] != SCHEMA or metadata.get('search_format') != 2:
            raise ValueError('Unsupported evidence schema')
        return metadata


class Evidence:
    def __init__(self, index, codec, evidence_index):
        self.store = Store(Path(index) / 'store.sqlite')
        try:
            self.db = open_index(evidence_index)
        except Exception:
            self.store.close()
            raise
        self.codec = codec
        self.trace = {}

    def search(self, query, doc=None, phrase=False, limit=48):
        expression = phrase_query(query) if phrase else match_query(query)
        if not expression:
            return []
        expression = 'body : (' + expression + ')'
        if doc is not None:
            expression += ' AND doc : "' + doc.replace('"', '""') + '"'
        sql = ('SELECT rowid,doc,start,label,body,bm25(passages,0,0,0.2,1) AS score '
               'FROM passages WHERE passages MATCH ?')
        params = [expression]
        if doc is not None:
            sql += ' AND doc=?'
            params.append(doc)
        sql += ' ORDER BY score,rowid LIMIT ?'
        params.append(limit)
        return self.db.execute(sql, params).fetchall()

    def expand(self, query, docs, base, known):
        """At most two phrase-supported additions; never force a final answer."""
        docs, base = list(docs), list(base)
        additions, ranked = [], []
        for rowid, doc, start, label, body, bm25 in self.search(query, phrase=True, limit=96):
            if doc in docs or doc not in known:
                continue
            longest, coverage = overlap(query, body)
            if longest >= 6 and coverage >= .55:
                ranked.append((-longest, -coverage, bm25, doc, rowid, start))
        for length, coverage, _, doc, rowid, start in sorted(ranked):
            if doc in docs:
                continue
            docs.append(doc)
            # Match the existing question-memory fallback: new candidates must
            # win comparisons; phrase agreement is not an automatic promotion.
            base.append(min(base) - 1.)
            additions.append({'doc': doc, 'rowid': rowid, 'start_word': start,
                              'phrase_words': -length, 'coverage': -coverage})
            if len(additions) == EXTRA_CANDIDATES:
                break
        self.trace = {'schema': SCHEMA, 'additions': additions, 'documents': {}}
        return docs, base

    def focus(self, query, body, budget):
        """Select a query-centered span that fits actual tokens, not a prefix."""
        if budget <= 0:
            return [], ''
        encode = self.codec.encode
        full = encode(body)
        if len(full) <= budget:
            return full, body
        words = body.split()
        # Short overlapping windows preserve neighboring conditions around the
        # match; rank before tokenization so an early header cannot use the budget.
        spans = [(start, ' '.join(words[start:start + 80]))
                 for start in range(0, len(words), 24)] or [(0, '')]
        _, best = max(spans, key=lambda item: (*overlap(query, item[1]), -item[0]))
        ids = encode(best)
        if len(ids) <= budget:
            return ids, best
        # Keep the strongest match when even an 80-word window is too long.
        q = set(tokens(query))
        candidates = best.split()
        scores = [len(q & set(tokens(w))) for w in candidates]
        width = max(1, int(len(candidates) * budget / len(ids)))
        while width:
            start = max(range(len(candidates) - width + 1),
                        key=lambda i: (sum(scores[i:i + width]), -i))
            text = ' '.join(candidates[start:start + width])
            ids = encode(text)
            if len(ids) <= budget:
                return ids, text
            width -= 1
        return [], ''

    def encode(self, query, docs):
        c = self.codec
        overhead = len(c.prefix + c.encode(query)[:256] + c.middle_a + c.middle_b + c.suffix)
        budget = (c.max_length - overhead) // 2
        if budget < 128:
            raise ValueError('Pair context too short for full-text evidence')
        result = {}
        for doc in docs:
            rows = self.search(query, doc, limit=12)
            if not rows:
                rows = self.db.execute('SELECT rowid,doc,start,label,body,0 FROM passages '
                                       'WHERE doc=? ORDER BY rowid LIMIT 1', (doc,)).fetchall()
            rows.sort(key=lambda r: (*(-v for v in overlap(query, r[4])), r[5], r[0]))
            label = self.store.label(doc)
            metadata = c.encode(label + '\n')[:min(96, budget // 5)]
            selected, bags = [], []
            for row in rows:
                bag = set(tokens(row[4]))
                if any(len(bag & b) / max(1, len(bag | b)) > .8 for b in bags):
                    continue
                selected.append(row)
                bags.append(bag)
                if len(selected) == 3:
                    break
            ids, packed = list(metadata), []
            for i, row in enumerate(selected):
                separator = c.separator if packed else c.encode('\n')
                remaining = budget - len(ids) - len(separator)
                # Put most of the budget into the best clause while retaining
                # room for a second provision from a different part of the law.
                quota = remaining if i == len(selected) - 1 else int(remaining * .65)
                piece, text = self.focus(query, row[4], quota)
                if piece:
                    ids += separator + piece
                    packed.append({'rowid': row[0], 'start_word': row[2], 'text': text,
                                   'tokens': len(piece)})
            if len(ids) > budget:
                raise ValueError('Full-text evidence exceeded its token budget')
            # One pre-budgeted piece prevents the original 60/20/20 prefix split
            # from truncating our selected clause a second time.
            result[doc] = [ids]
            self.trace.setdefault('documents', {})[doc] = {'label': label, 'packed': packed,
                'input_ids': ids, 'budget': budget,
                'sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        return result

    def close(self):
        self.db.close()
        self.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.corpus, args.store, args.out)))


if __name__ == '__main__':
    main()
