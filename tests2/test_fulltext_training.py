"""Training preparation never admits held-out aliases or contradictory pairs."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from legalir2.fulltext_training import challengers, partition, quarantine_empty, rebuild, validate_group
from legalir2.duo import PairTokenizer
from legalir2.duo_evidence import Evidence, build
from legalir2.store import build as build_store, write_json
from tests2.test_duo import CharacterTokenizer


class TestFulltextTraining(unittest.TestCase):
    def test_empty_source_quarantine_preserves_pairs_in_remaining_groups(self):
        good = dict(id='good', gold=['a'], pairs=[['a', 'b']])
        bad = dict(id='bad', gold=['empty'], pairs=[['empty', 'b']])
        kept, excluded = quarantine_empty([good, bad], {'a', 'b'})
        self.assertEqual(kept, [good])
        self.assertEqual(excluded[0]['gold_documents'], ['empty'])

    def test_external_alias_and_conflicting_duplicates_are_excluded(self):
        def group(qid, question, gold=('a',)):
            return dict(id=qid, question=question, gold=list(gold), development=False)
        groups = [group('alias', 'HELD OUT?'), group('a', 'same'), group('b', 'same', ('b',)),
                  group('c', 'unique'), group('d', 'unique!')]
        selected, roles, exclusions = partition(groups, {'external': {'question': 'held out'}})
        self.assertEqual([g['id'] for g in selected], ['c'])
        self.assertEqual(len(exclusions), 4)
        self.assertEqual(partition(groups[::-1], {'external': {'question': 'held out'}})[1], roles)

    def test_hard_negatives_preserve_original_and_exclude_all_gold(self):
        group = dict(gold=['a', 'b'], pairs=[['a', 'c'], ['b', 'c']])
        self.assertEqual(challengers(group, list('abcdef'), ['a', 'd', 'd']),
                         [{'doc': 'd', 'source': 'fulltext-phrase'}, {'doc': 'e', 'source': 'v9-rank'}])
        self.assertEqual(group['pairs'], [['a', 'c'], ['b', 'c']])

    def test_pair_budget_and_positive_negative_integrity(self):
        codec = PairTokenizer(CharacterTokenizer(), 2048)
        group = dict(question='question', gold=['a'], pairs=[['a', 'b']], evidence={'a': [[97]*1200], 'b': [[98]*1200]})
        self.assertLessEqual(validate_group(group, codec, {'a', 'b'}), 2048)
        with self.assertRaises(ValueError):
            validate_group(dict(group, gold=['a', 'b']), codec, {'a', 'b'})
        empty = dict(group, evidence={'a': [[]], 'b': [[98]]})
        validate_group(empty, codec, {'a', 'b'}, allow_empty=True)
        with self.assertRaises(ValueError):
            validate_group(empty, codec, {'a', 'b'})

    def test_rebuild_changes_only_evidence_in_control_and_keeps_all_gold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root/'corpus'
            corpus.mkdir()
            for doc in 'abcde':
                write_json(corpus/('context_'+doc+'.json'), {'id': doc, 'name': doc,
                    'passage': 'one two three four five six legal provision '+doc})
            build_store(corpus, root/'store.sqlite')
            build(corpus, root/'store.sqlite', root/'fulltext.sqlite')
            codec = PairTokenizer(CharacterTokenizer(), 2048)
            evidence = Evidence(root, codec, root/'fulltext.sqlite')
            evidence.supported = {r[0] for r in evidence.db.execute(
                "SELECT DISTINCT doc FROM passages WHERE length(trim(body))>0")}
            self.addCleanup(evidence.close)
            group = dict(id='q', question='one two three four five six', gold=['a', 'b'], development=False,
                pairs=[['a', 'c'], ['b', 'c']], evidence={d: [[ord(c) for c in 'old snippet']] for d in 'abc'})
            row = {k: group[k] for k in ('id', 'question', 'gold')}
            row.update(docs=list('abcde'), features1=[[5], [4], [3], [2], [1]])
            result = rebuild(group, row, 'monitor', codec, evidence, [1])
            evidence.close()
            self.assertEqual(result['control']['pairs'], group['pairs'])
            self.assertEqual(result['control']['gold'], ['a', 'b'])
            self.assertEqual(len(result['augmented']['pairs']), 6)
            self.assertTrue(result['control']['development'])
            self.assertEqual(result['stats']['changed_documents'], 3)
            self.assertLessEqual(result['new_max_pair_tokens'], 2048)
            self.assertEqual(set(result['trace']['documents']), set('abcde'))
            del evidence.supported
            with self.assertRaisesRegex(ValueError, 'must declare supported'):
                rebuild(group, row, 'monitor', codec, evidence, [1])


if __name__ == '__main__':
    unittest.main()
