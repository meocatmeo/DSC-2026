"""Complement frozen retrieval with full-text bigrams and distinctive evidence.

No answer labels or question IDs enter the search policy. This is an opt-in
experiment; the scored fulltext-v2 evidence module remains unchanged.
"""
from functools import lru_cache
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from .duo_evidence import Evidence, match_query, open_index, tokens
from .store import Store, file_hash, source_records

SCHEMA = 'focused-fulltext-v2'
# Function words should not outrank a legal concept because they happen to be
# uncommon within one statute. Keep negation tokens: they change legal meaning.
FUNCTION_WORDS = frozenset('là thì mà và của có được cho với về từ đến vào ra tại theo những các một như thế nào bao nhiêu'.split())


def build_documents(corpus, store_path, target):
    """A document-level index avoids letting long laws monopolize window hits."""
    if Path(target).resolve() == Path(store_path).resolve():
        raise ValueError('Document index must not overwrite the frozen store')
    store = Store(store_path)
    target = Path(target)
    expected = {'schema': 'fulltext-documents-v1', 'store_sha256': file_hash(store_path),
                'corpus_sha256': store.metadata['corpus_sha256'], 'documents': len(store.doc_ids)}
    if target.exists():
        with open_index(target) as db:
            actual = json.loads(db.execute("SELECT value FROM metadata WHERE key='index'").fetchone()[0])
        db.close()
        store.close()
        if actual != expected:
            raise ValueError('Existing document index differs')
        return expected
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.sqlite.tmp')
    if temporary.exists():
        raise ValueError('Unfinished document index exists; choose a new output path')
    db = sqlite3.connect(temporary)
    try:
        db.executescript("""
            PRAGMA journal_mode=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE VIRTUAL TABLE passages USING fts5(doc,start UNINDEXED,label,body,
                tokenize='unicode61 remove_diacritics 0');
        """)
        checksum, seen = hashlib.sha256(), set()
        for name, raw in source_records(corpus):
            record = json.loads(raw.decode('utf-8'))
            doc = str(record['id'])
            if doc in seen or doc not in store.doc_position:
                raise ValueError('Duplicate or unknown corpus document')
            seen.add(doc)
            checksum.update(name.encode()+b'\0'+raw+b'\0')
            db.execute('INSERT INTO passages VALUES (?,?,?,?)', (doc, 0, store.label(doc), record.get('passage', '')))
        if seen != set(store.doc_ids) or checksum.hexdigest() != expected['corpus_sha256']:
            raise ValueError('Document corpus differs from the frozen source')
        db.execute('INSERT INTO metadata VALUES (?,?)', ('index', json.dumps(expected)))
        db.commit()
    finally:
        db.close()
        store.close()
    temporary.replace(target)
    return expected


def bigram_query(query):
    stream = tokens(query)
    phrases = dict.fromkeys(' '.join(pair) for pair in zip(stream, stream[1:]))
    return ' OR '.join('"'+phrase+'"' for phrase in phrases)


class FocusedSearch:
    def __init__(self, path, document_index=None):
        self.db = open_index(path)
        self.documents_db = open_index(document_index) if document_index else self.db
        if document_index:
            try:
                passage = json.loads(self.db.execute("SELECT value FROM metadata WHERE key='index'").fetchone()[0])
                document = json.loads(self.documents_db.execute("SELECT value FROM metadata WHERE key='index'").fetchone()[0])
                if document['schema'] != 'fulltext-documents-v1' or any(
                        document[k] != passage[k] for k in ('store_sha256', 'corpus_sha256', 'documents')):
                    raise ValueError('Document and passage indexes differ')
            except Exception:
                self.close()
                raise
        self.db.execute('CREATE VIRTUAL TABLE temp.query_vocab USING fts5vocab(main, passages, row)')
        self.size = self.db.execute('SELECT count(*) FROM passages').fetchone()[0]

    def close(self):
        if self.documents_db is not self.db:
            self.documents_db.close()
        self.db.close()

    def channel(self, expression, doc=None, limit=128, *, documents=False):
        if not expression:
            return []
        match = 'body : ('+expression+')'
        if doc is not None:
            match += ' AND doc : "'+str(doc).replace('"', '""')+'"'
        sql = ('SELECT rowid,doc,start,label,body,bm25(passages,0,0,0,1) '
               'FROM passages WHERE passages MATCH ?')
        params = [match]
        if doc is not None:
            sql += ' AND doc=?'
            params.append(str(doc))
        db = self.documents_db if documents else self.db
        return db.execute(sql+' ORDER BY 6,rowid LIMIT ?', (*params, limit)).fetchall()

    def search(self, query, doc=None, limit=128):
        channels = [self.channel(expression, doc, limit) for expression in
                    (match_query(query), bigram_query(query))]
        rows, score = {}, {}
        for channel in channels:
            for rank, row in enumerate(channel, 1):
                key = row[0]
                rows[key] = row
                score[key] = score.get(key, 0.)+1/(60+rank)
        return [(*rows[key][:5], -score[key]) for key in sorted(rows, key=lambda k: (-score[k], k))][:limit]

    def documents(self, query, limit=32):
        # Fuse document ranks after deduplication so a long law cannot occupy
        # every retrieval slot with similar windows.
        scores, best = {}, {}
        for expression in (match_query(query), bigram_query(query)):
            seen = set()
            for row in self.channel(expression, limit=64, documents=True):
                doc = row[1]
                if doc in seen:
                    continue
                seen.add(doc)
                scores[doc] = scores.get(doc, 0.)+1/(60+len(seen))
                best.setdefault(doc, row)
        return [(doc, scores[doc], best[doc][0]) for doc in
                sorted(scores, key=lambda d: (-scores[d], d))[:limit]]

    @lru_cache(maxsize=512)
    def query_weights(self, query):
        stream = tuple(dict.fromkeys(tokens(query)))
        if not stream:
            return {}
        marks = ','.join('?' for _ in stream)
        frequency = dict(self.db.execute(f'SELECT term,doc FROM query_vocab WHERE term IN ({marks})', stream))
        return {term: math.log((self.size+1)/(frequency.get(term, 0)+1)) for term in stream}

    def focus_score(self, query, text):
        return self.coverage_score(query, [text])

    def coverage_score(self, query, texts):
        weights = self.query_weights(query)
        streams = [tokens(text) for text in texts]
        words = {word for body in streams for word in body}
        unigram = sum(weights.get(word, 0.) for word in words)
        query_tokens = tokens(query)
        query_pairs = set(zip(query_tokens, query_tokens[1:]))
        body_pairs = {pair for body in streams for pair in zip(body, body[1:])}
        phrase = sum(min(weights[a], weights[b]) for a, b in query_pairs & body_pairs)
        return unigram+2*phrase


class FocusedEvidence(Evidence):
    """Keep exact pair token budgets, but choose evidence by weighted retrieval."""
    def __init__(self, index, codec, evidence_index):
        super().__init__(index, codec, evidence_index)
        self.focused = FocusedSearch(evidence_index)

    def search(self, query, doc=None, phrase=False, limit=48):
        if phrase:
            return super().search(query, doc, phrase, limit)
        rows = self.focused.search(query, doc, max(limit, 32))
        return rows[:limit]

    def focus(self, query, body, budget):
        if budget <= 0:
            return [], ''
        if len(self.codec.encode(body)) <= budget:
            return self.codec.encode(body), body
        words = body.split()
        choices = [(i, ' '.join(words[i:i+80])) for i in range(0, len(words), 24)]
        _, text = max(choices, key=lambda item: (self.focused.focus_score(query, item[1]), -item[0]))
        words = text.split()
        while words:
            ids = self.codec.encode(' '.join(words))
            if len(ids) <= budget:
                return ids, ' '.join(words)
            # Drop the less informative edge; never exceed the actual token budget.
            left, right = ' '.join(words[1:]), ' '.join(words[:-1])
            words = words[1:] if self.focused.focus_score(query, left) > self.focused.focus_score(query, right) else words[:-1]
        return [], ''

    def encode(self, query, docs):
        codec = self.codec
        overhead = len(codec.prefix+codec.encode(query)[:256]+codec.middle_a+codec.middle_b+codec.suffix)
        budget = (codec.max_length-overhead)//2
        if budget < 128:
            raise ValueError('Pair context too short for focused evidence')
        result = {}
        self.trace = {'schema': SCHEMA, 'documents': {}}
        for doc in docs:
            rows = self.search(query, doc, limit=96)
            if not rows:
                rows = self.db.execute('SELECT rowid,doc,start,label,body,0 FROM passages '
                                       'WHERE doc=? ORDER BY rowid LIMIT 1', (doc,)).fetchall()
            # Within a long statute, generic query phrases recur everywhere.
            # Downweight that repetition to retain rare clauses such as housing
            # allowances rather than another passage about employees in general.
            query_terms = set(tokens(query))-FUNCTION_WORDS
            query_pairs = {pair for pair in zip(tokens(query), tokens(query)[1:])
                           if not set(pair) & FUNCTION_WORDS}
            occurrences = Counter()
            features = {}
            for row in rows:
                stream = tokens(row[4])
                features[row[0]] = ({('word', t) for t in set(stream) & query_terms} |
                    {('pair', a, b) for a, b in set(zip(stream, stream[1:])) & query_pairs})
                occurrences.update(features[row[0]])
            global_weights = self.focused.query_weights(query)
            def feature_weight(feature):
                weight = min(global_weights[t] for t in feature[1:])
                return weight*(4 if feature[0] == 'pair' else 1)*math.log((len(rows)+1)/(occurrences[feature]+1))
            weights = {f: feature_weight(f) for f in occurrences}
            selected, bags, covered = [], [], set()
            while rows:
                row = max(rows, key=lambda r: (sum(weights[f]*(.15 if f in covered else 1.)
                    for f in features[r[0]]), -r[5], -r[0]))
                rows.remove(row)
                bag = set(tokens(row[4]))
                if any(len(bag & old)/max(1, len(bag | old)) > .8 for old in bags):
                    continue
                selected.append(row)
                bags.append(bag)
                covered.update(features[row[0]])
                if len(selected) == 3:
                    break
            label = self.store.label(doc)
            ids = codec.encode(label+'\n')[:min(96, budget//5)]
            packed = []
            for i, row in enumerate(selected):
                separator = codec.separator if packed else codec.encode('\n')
                remaining = budget-len(ids)-len(separator)
                quota = remaining if i == len(selected)-1 else int(remaining*.65)
                piece, text = self.focus(query, row[4], quota)
                if piece:
                    ids += separator+piece
                    packed.append({'rowid': row[0], 'start_word': row[2], 'text': text, 'tokens': len(piece)})
            if len(ids) > budget:
                raise ValueError('Focused evidence exceeded token budget')
            result[doc] = [ids]
            self.trace['documents'][doc] = {'label': label, 'packed': packed,
                'input_ids': ids, 'budget': budget, 'sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        return result

    def close(self):
        self.focused.close()
        super().close()


def rescue_candidates(existing, fulltext, memory_evidence, memory_ranking, known, limit=8):
    """At most eight additional candidates; never discard a baseline candidate."""
    if limit < 0 or len(set(existing)) != len(existing):
        raise ValueError('Invalid rescue pool')
    result, seen = [], set(existing)
    # Two complementary channels receive equal rank-fusion weight. Memory uses
    # the frozen training-only records and a fixed minimum similarity.
    channels = [[doc for doc, _, _ in fulltext],
                [doc for doc in memory_ranking[:10] if memory_evidence[doc][0] >= .45]]
    score = {}
    for channel in channels:
        for rank, doc in enumerate(dict.fromkeys(channel), 1):
            if doc in known and doc not in seen:
                score[doc] = score.get(doc, 0.)+1/(60+rank)
    for doc in sorted(score, key=lambda d: (-score[d], d))[:limit]:
        result.append(doc)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a frozen-corpus document search index')
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(build_documents(arguments.corpus, arguments.store, arguments.out)))
