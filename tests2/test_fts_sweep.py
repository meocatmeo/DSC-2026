"""FTS threshold proposals preserve the source pool and frozen search semantics."""
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from legalir2 import fts_sweep as fts
from legalir2.duo_evidence import Evidence, phrase_query


class TestFTSSweep(unittest.TestCase):
    def test_phrase_syntax_and_frozen_six_word_query(self):
        query = 'one two "three" four five six seven; OR *'
        self.assertEqual(fts.phrase_query(query, 6), phrase_query(query))
        self.assertEqual(fts.phrase_query('one "two" three', 3), '"one two three"')
        self.assertEqual(fts.phrase_query('two words', 3), '')
        with self.assertRaises(ValueError):
            fts.phrase_query(query, 2)

    def test_readonly_search_and_baseline_equivalence(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'index.sqlite'
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE VIRTUAL TABLE passages USING fts5(doc, start UNINDEXED, label, body)')
                db.executemany('INSERT INTO passages VALUES (?,?,?,?)', [
                    ('a', 0, '', 'one two three four five six seven'),
                    ('a', 1, '', 'one two three four five six'),
                    ('b', 0, '', 'one two three four five six'),
                    ('c', 0, '', 'one two three other'),
                    ('unknown', 0, '', 'one two three four five six seven')])
                db.commit()
            with closing(fts.Search(path)) as search:
                query = 'one two three four five six seven'
                policy = dict(phrase_words=6, term_coverage=.55, extra=2, search_limit=96)
                baseline = Evidence.__new__(Evidence)
                baseline.db = search.db
                before = ['c']
                expected, _ = baseline.expand(query, before, [1.], {'a', 'b', 'c'})
                actual = fts.additions(search.hits(query, 6), before, {'a', 'b', 'c'}, policy)
                self.assertEqual(before+actual, expected)
                self.assertEqual(before, ['c'])
                self.assertEqual(set(actual), {'a', 'b'})
                with self.assertRaises(sqlite3.OperationalError):
                    search.db.execute('DELETE FROM passages')

    def test_limits_thresholds_and_no_unknown_or_duplicate_candidates(self):
        def hit(doc, longest=4, coverage=.4):
            return dict(doc=doc, longest=longest, coverage=coverage, bm25=-1., rowid=1, start=0)
        policy = dict(phrase_words=4, term_coverage=.4, extra=2, search_limit=96)
        hits = [hit('old')]*96+[hit('new')]
        self.assertEqual(fts.additions(hits, ['old'], {'old', 'new'}, policy), [])
        self.assertEqual(fts.additions(hits, ['old'], {'old', 'new'}, dict(policy, search_limit=256)), ['new'])
        hits = [hit('x'), hit('b'), hit('a'), hit('a'), hit('c', coverage=.39), hit('d', longest=3)]
        self.assertEqual(fts.additions(hits, [], {'a', 'b', 'c', 'd'}, policy), ['a', 'b'])
        with self.assertRaises(ValueError):
            fts.additions([hit('a', coverage=float('nan'))], [], {'a'}, policy)


if __name__ == '__main__':
    unittest.main()
