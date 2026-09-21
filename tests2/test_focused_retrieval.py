"""Query safety, phrase retrieval, and preservation of the baseline pool."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

from legalir2.focused_retrieval import FocusedSearch, FocusedEvidence, bigram_query, rescue_candidates, build_documents
from legalir2.duo import PairTokenizer
from legalir2.duo_evidence import build as build_passages
from legalir2.store import build as build_store, write_json, file_hash
from tests2.test_duo import CharacterTokenizer


class TestFocusedRetrieval(unittest.TestCase):
    def test_frozen_index_identity_and_real_packing_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root/'corpus'
            query = 'Where is the rare housing allowance clause?'
            for doc, body in [('a', 'ordinary general rules '*100+' rare housing allowance clause is here'),
                              ('b', 'ordinary general rules '*100)]:
                write_json(corpus/f'context_{doc}.json', {'id': doc, 'name': doc, 'passage': body})
            store, passages, documents = root/'store.sqlite', root/'passages.sqlite', root/'documents.sqlite'
            build_store(corpus, store)
            original = file_hash(store)
            build_passages(corpus, store, passages)
            expected = build_documents(corpus, store, documents)
            self.assertEqual(build_documents(corpus, store, documents), expected)
            self.assertEqual(file_hash(store), original)
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                build_documents(corpus, store, store)
            search = FocusedSearch(passages, documents)
            try:
                self.assertEqual(search.documents(query)[0][0], 'a')
            finally:
                search.close()
            codec = PairTokenizer(CharacterTokenizer(), 2048)
            evidence = FocusedEvidence(root, codec, passages)
            try:
                encoded = evidence.encode(query, ['a', 'b'])
                pair = codec.pair(query, encoded['a'], encoded['b'])
                self.assertLessEqual(len(pair), 2048)
                self.assertIn('rare housing allowance', ''.join(map(chr, pair)))
                self.assertEqual(encoded, evidence.encode(query, ['a', 'b']))
            finally:
                evidence.close()

    def test_bigrams_are_tokenized_not_sql_fragments(self):
        self.assertEqual(bigram_query('Nhà ở'), '"nhà ở"')
        self.assertEqual(bigram_query('" OR *'), '')
        self.assertEqual(bigram_query(''), '')

    def test_search_deduplicates_laws_and_weights_distinctive_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'search.sqlite'
            with sqlite3.connect(path) as db:
                db.execute("CREATE VIRTUAL TABLE passages USING fts5(doc,start UNINDEXED,label,body,tokenize='unicode61 remove_diacritics 0')")
                for i in range(30):
                    db.execute('INSERT INTO passages VALUES (?,?,?,?)',
                        ('common', i*100, '', 'người lao động có được hưởng trợ cấp theo quy định'))
                db.execute('INSERT INTO passages VALUES (?,?,?,?)', ('target', 0, '', 'phụ cấp nhà ở'))
            db.close()
            search = FocusedSearch(path)
            try:
                query = 'người lao động có được hưởng phụ cấp nhà ở'
                docs = [r[0] for r in search.documents(query)]
                self.assertEqual(len(docs), len(set(docs)))
                self.assertIn('target', docs)
                self.assertGreater(search.focus_score(query, 'phụ cấp nhà ở'),
                                   search.focus_score(query, 'người lao động có được hưởng'))
                self.assertTrue(all(r[1] == 'target' for r in search.search(query, doc='target')))
                self.assertEqual(search.search(''), [])
            finally:
                search.close()

    def test_expansion_keeps_baseline_and_rejects_unknown_candidates(self):
        existing = ['a', 'b', 'c', 'd', 'e']
        memory = {'f': [.7], 'g': [.2]}
        result = rescue_candidates(existing, [('a', 1, 1), ('f', .5, 2), ('unknown', .4, 3)],
            memory, ['f', 'g'], set('abcdefg'), limit=8)
        self.assertEqual(existing, list('abcde'))
        self.assertEqual(result, ['f'])
        self.assertEqual(rescue_candidates(existing, [], {}, [], set(existing)), [])
        with self.assertRaises(ValueError):
            rescue_candidates(['a', 'a'], [], {}, [], {'a'})


if __name__ == '__main__':
    unittest.main()
