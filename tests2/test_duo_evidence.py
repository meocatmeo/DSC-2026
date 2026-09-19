"""Lossless corpus coverage, clause visibility and frozen-cache isolation."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from legalir2.duo import PairTokenizer
from legalir2.duo_evidence import Evidence, build, index_metadata, overlap, windows
from legalir2.store import build as build_store, read_json, write_json
from legalir2 import private_infer as private
from tests2.test_duo import CharacterTokenizer


class TestFullTextEvidence(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / 'corpus'
        self.corpus.mkdir()
        self.query = 'Thông báo miễn giảm thuế theo hiệp định quốc tế dùng mẫu nào?'
        self.clause = 'Thông báo miễn giảm thuế theo hiệp định quốc tế dùng mẫu 77/TEST.'
        records = {
            'a': 'BỘ TÀI CHÍNH\nSố: 77/2021/TT-BTC\n' + self.clause + '\nI. PHỤ LỤC\nkhác\nII. KHÁC\nkhác',
            'b': 'thuế thông báo quốc tế giảm hiệp theo định miễn ' * 20,
            'c': 'Các quy định khác về tổ chức nhân sự.',
            'd': 'BỘ Y TẾ\nSố: 88/2021/TT-BYT\n' + 'không liên quan ' * 100 + self.clause,
        }
        for doc, text in records.items():
            write_json(self.corpus / f'context_{doc}.json', {'id': doc, 'name': doc, 'passage': text})
        self.store = self.root / 'store.sqlite'
        build_store(self.corpus, self.store, header_words=4)
        self.index = self.root / 'evidence.sqlite'
        build(self.corpus, self.store, self.index)
        self.codec = PairTokenizer(CharacterTokenizer(), 2048)
        self.evidence = Evidence(self.root, self.codec, self.index)
        self.addCleanup(self.evidence.close)

    def test_every_source_word_is_covered_including_prefix_before_roman_headings(self):
        words = [f'word{i}' for i in range(913)]
        recovered = {}
        for start, text in windows(' '.join(words)):
            recovered.update({start + i: w for i, w in enumerate(text.split())})
        self.assertEqual([recovered[i] for i in range(len(words))], words)
        # Original parser drops the body preceding the first Roman appendix.
        self.assertFalse(any('77/TEST' in self.evidence.store.chunk(c)[4]
                             for c in self.evidence.store.document_chunk_ids('a')))
        rows = self.evidence.search(self.query, 'a')
        self.assertTrue(any('77/TEST' in row[4] for row in rows))
        self.assertEqual(index_metadata(self.index)['documents'], 4)

    def test_phrase_candidates_require_contiguous_evidence_and_known_ids(self):
        docs, scores = self.evidence.expand(self.query, ['c'], [2.], {'a', 'b', 'c'})
        self.assertEqual(docs, ['c', 'a'])
        self.assertEqual(scores, [2., 1.])
        self.assertEqual(self.evidence.trace['additions'][0]['doc'], 'a')
        self.assertEqual(overlap('one two three four five six', 'six five four three two one')[0], 0)
        self.assertEqual(self.evidence.expand('???', ['c'], [2.], {'c'})[0], ['c'])

    def test_late_clause_survives_token_budget_and_preserves_document_identity(self):
        self.evidence.expand(self.query, ['d', 'c'], [2., 1.], {'c', 'd'})
        encoded = self.evidence.encode(self.query, ['d', 'c'])
        text = ''.join(map(chr, encoded['d'][0]))
        self.assertIn('77/TEST', text)
        self.assertIn('88/2021/TT-BYT', text)
        pair = self.codec.pair(self.query, encoded['d'], encoded['c'])
        self.assertLessEqual(len(pair), self.codec.max_length)
        self.assertIn('77/TEST', ''.join(map(chr, pair)))
        self.assertTrue(self.evidence.trace['documents']['d']['packed'])
        self.assertEqual(encoded, self.evidence.encode(self.query, ['d', 'c']))

    def test_fts_syntax_in_question_is_data_and_output_is_deterministic(self):
        query = '" OR NEAR(*) : thông báo'
        self.assertEqual(self.evidence.search(query), self.evidence.search(query))
        self.assertEqual(self.evidence.search(''), [])

    def test_conditions_are_not_discarded_when_the_whole_window_fits(self):
        body = self.clause + ' context' * 150 + ' except the stated exemption'
        ids, text = self.evidence.focus(self.query, body, len(body) + 10)
        self.assertEqual(text, body)
        self.assertEqual(''.join(map(chr, ids)), body)

    def test_rebuild_never_overwrites_frozen_store_or_completed_sidecar(self):
        for target in (self.store, self.index):
            with self.assertRaises(ValueError):
                build(self.corpus, self.store, target)


class TestBaselineImport(unittest.TestCase):
    def test_changed_policy_reuses_only_verified_baseline_and_rejects_wrong_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'output'
            records = {'q': {'question': 'question', 'answer': []}}
            signature = {'known_docs': list('abcde')}
            marker = {k: k for k in ('schema', 'questions_sha256', 'questions', 'duo_inputs',
                                     'runtime', 'assets', 'packages')}
            marker.update(question_ids=['q'], shard_size=1, code='old', evidence={'policy': 'original'})
            write_json(source / 'inputs.json', marker)
            write_json(source / 'cache/v9/resume.json', {'inputs': private.h100.digest(marker), 'stage': 'v9'})
            features = [[float(i)] + [0.] * 19 for i in range(5)]
            rows = [{'id': 'q', 'question': 'question', 'gold': [], 'docs': list('abcde'),
                     'features1': features, 'features2': features}]
            write_json(source / 'cache/v9/part-00000.json', rows)
            updated = dict(marker, code='new', evidence={'policy': 'fulltext-v2'})
            args = SimpleNamespace(out=output, shard_size=10)
            private.import_v9(args, records, signature, updated, source)
            self.assertEqual(read_json(output / 'cache/v9/part-00000.json'), rows)
            self.assertFalse((output / 'cache/duo').exists())
            with self.assertRaisesRegex(ValueError, 'questions'):
                private.import_v9(args, records, signature, dict(updated, questions='changed'), source)


if __name__ == '__main__':
    unittest.main()
