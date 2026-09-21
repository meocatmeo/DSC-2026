"""Regression checks for the teammate tournament integration."""
import unittest

import numpy as np

from legalir2 import consensus, fulltext_prior
from tests2.test_fulltext_prior import fixture


class TestConsensus(unittest.TestCase):
    def test_gating_preserves_incumbent_without_head_to_head_support(self):
        docs = list('abcdef')
        matrix = np.full((6, 6), .5)
        scores = np.array([5., 4., 3., 2., 1., 20.])
        for profile in ('borda', 'graph'):
            self.assertEqual(set(consensus.rerank(docs, scores, matrix, docs[:5], profile)), set(docs[:5]))
            matrix[5, 4], matrix[4, 5] = .9, .1
            self.assertIn('f', consensus.rerank(docs, scores, matrix, docs[:5], profile))
            matrix[:] = .5

    def test_document_permutation_does_not_reassign_scores(self):
        docs = list('abcdef')
        rng = np.random.default_rng(8)
        matrix = rng.uniform(.05, .95, (6, 6))
        matrix = .5+(matrix-matrix.T)/2
        score = rng.normal(size=6)
        permutation = [4, 2, 5, 0, 3, 1]
        for profile in ('borda', 'graph'):
            expected = consensus.rerank(docs, score, matrix, docs[:5], profile)
            actual = consensus.rerank([docs[i] for i in permutation], score[permutation],
                matrix[np.ix_(permutation, permutation)], docs[:5], profile)
            self.assertEqual(actual, expected)

    def test_copeland_counts_wins_not_duplicated_probability_mass(self):
        matrix = np.full((6, 6), .5)
        matrix[0, 1:], matrix[1:, 0] = [.99, .51, .51, .1, .1], [.01, .49, .49, .9, .9]
        borda, _, copeland = consensus.signals(matrix)
        self.assertFalse(np.allclose(borda, copeland))

    def test_invalid_graph_and_duplicate_anchor_rejected(self):
        matrix = np.full((6, 6), .5)
        with self.assertRaises(ValueError):
            consensus.rerank(list('abcdef'), np.zeros(6), matrix, list('aabcd'), 'graph')
        matrix[0, 1] = float('nan')
        with self.assertRaises(ValueError):
            consensus.signals(matrix)

    def test_integrated_policy_uses_no_gold_at_prediction_time(self):
        row = fixture()
        policy = {'alpha': .5, 'strength': .15, 'consensus': 'graph'}
        expected = fulltext_prior.predict(row, policy)
        row['gold'] = ['a']
        self.assertEqual(expected, fulltext_prior.predict(row, policy))
        del row['gold']
        self.assertEqual(expected, fulltext_prior.predict(row, policy))


if __name__ == '__main__':
    unittest.main()
