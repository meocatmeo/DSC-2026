import unittest
from types import SimpleNamespace

from legalir2.evidence import build_bundle
from legalir2.models import TEMPLATES, rerank_prompt


class Tokenizer:
    # A strict character tokenizer makes token budget failures obvious without GPU.
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.template = TEMPLATES['qwen3-reranker']

    def bundle(self, chunks, **kwargs):
        return build_bundle('question', 'title', chunks, self.tokenizer, self.template, **kwargs)

    def test_deterministic_ties_and_no_duplicate_backfill(self):
        rows = [{'id': 9, 'text': 'same passage', 'score': 1},
                {'id': 3, 'text': 'same passage', 'score': 1},
                {'id': 5, 'text': 'different evidence', 'score': 0}]
        a = self.bundle(rows)
        self.assertEqual(a, self.bundle(list(reversed(rows))))
        self.assertEqual(a['selected_chunk_ids'], [3, 5])
        self.assertIn({'id': 9, 'reason': 'overlap'}, a['dropped'])

    def test_budget_preserves_template_and_query(self):
        result = self.bundle([{'id': 1, 'text': 'long evidence ' * 500,
                               'heading': 'Article 1', 'score': 1}], max_tokens=600)
        self.assertLessEqual(result['total_tokens'], 600)
        self.assertTrue(result['packed'][0]['truncated'])
        prefix, suffix = rerank_prompt(self.template)
        actual = ''.join(map(chr, result['input_ids']))
        self.assertTrue(actual.startswith(prefix))
        self.assertTrue(actual.endswith(suffix))
        self.assertIn('<Query>: question', actual)
        self.assertIn('Article 1', result['document'])

    def test_no_silent_query_truncation(self):
        with self.assertRaisesRegex(ValueError, 'leave no evidence'):
            self.bundle([{'id': 1, 'text': 'body', 'score': 1}], max_tokens=20)

    def test_empty_passage_candidate_scores_its_actual_title(self):
        result = self.bundle([{'id': 1, 'text': '', 'score': 1}])
        self.assertEqual(result['document'], 'title')
        self.assertEqual(result['packed'], [])
        self.assertEqual(result['dropped'], [{'id': 1, 'reason': 'empty'}])

    def test_limit_and_invalid_scores(self):
        rows = [{'id': i, 'text': f'unique{i}', 'score': i} for i in range(8)]
        result = self.bundle(rows)
        self.assertEqual(result['selected_chunk_ids'], [7, 6, 5, 4, 3, 2])
        with self.assertRaises(ValueError):
            self.bundle([{'id': 1, 'text': 'x', 'score': float('nan')}])
        with self.assertRaises(ValueError):
            self.bundle([rows[0], rows[0]])

    def test_scoring_rejects_mismatched_tokenizer_before_model_call(self):
        from legalir2.rerank import Reranker
        model = object.__new__(Reranker)
        model.head, model.context_limit = 'causal', 32768
        model.template = self.template
        prefix, suffix = rerank_prompt(self.template)
        model.prefix, model.suffix = [self.tokenizer.encode(x) for x in (prefix, suffix)]
        model.local = SimpleNamespace(tokenizer=self.tokenizer)
        bundle = self.bundle([{'id': 1, 'text': 'evidence', 'score': 1}])
        bundle['query'] = 'different question'
        with self.assertRaisesRegex(ValueError, 'tokenizer/prompt'):
            model.score_bundles([bundle])

    def test_store_adapter_uses_only_frozen_retrieval_scores(self):
        from legalir2.evidence import bundle_from_store
        store = SimpleNamespace(document_chunk_ids=lambda d: [1, 2],
                                chunk=lambda c: ('doc', c, 'article', f'Article {c}', f'body{c}'),
                                label=lambda d: 'title')
        bundle = bundle_from_store(store, 'doc', 'question', {2: 2.0},
                                   self.tokenizer, self.template)
        self.assertEqual(bundle['selected_chunk_ids'], [2, 1])


if __name__ == '__main__':
    unittest.main()
