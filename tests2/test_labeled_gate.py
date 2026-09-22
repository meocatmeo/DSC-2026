"""Complete paired evaluation, weighted pooling and exact deadband boundaries."""
from copy import deepcopy
from fractions import Fraction
import unittest

from legalir2.labeled_gate import DEADBAND, decision, evaluate


class TestLabeledGate(unittest.TestCase):
    def setUp(self):
        self.known = set('abcdefg')
        self.datasets = {'development': {'q1': {'answer': ['a', 'f']}},
                         'validation': {'q2': {'answer': ['a']}, 'q3': {'answer': ['g']}}}
        self.predictions = {s: {q: {'baseline': list('abcde'), 'proposed': list('abcde')} for q in rows}
                            for s, rows in self.datasets.items()}

    def test_weighted_pool_uses_question_macro_recall(self):
        self.predictions['development']['q1']['proposed'] = list('abcdf')
        report = evaluate(self.datasets, self.predictions, self.known)
        self.assertAlmostEqual(report['pooled']['delta'], 1/6)
        self.assertEqual(report['pooled']['wins'], 1)
        self.assertEqual(report['pooled']['changed_sets'], 1)
        self.assertEqual(report['pooled']['decision'], 'inconclusive')
        self.assertFalse(report['pooled']['gate_eligible'])
        self.assertNotIn('decision', report['development'])
        self.assertNotIn('decision', report['validation'])
        self.assertEqual(len(report['pooled']['paired_bootstrap_ci95']), 2)

    def test_uncertain_above_deadband_is_not_promoted(self):
        self.assertEqual(decision(Fraction(53, 10000), [-.001, .011]), 'inconclusive')
        self.assertEqual(decision(Fraction(53, 10000), [.001, .011]), 'promote')
        self.assertEqual(decision(-DEADBAND, [-.011, .001]), 'inconclusive')

    def test_exact_gate_boundaries_and_neutral_order_changes(self):
        for delta, expected in [(DEADBAND, 'promote'), (-DEADBAND, 'reject'),
                                (DEADBAND-Fraction(1, 1000000), 'inconclusive'),
                                (-DEADBAND+Fraction(1, 1000000), 'inconclusive')]:
            self.assertEqual(decision(delta), expected)
        self.predictions['validation']['q2']['proposed'].reverse()
        report = evaluate(self.datasets, self.predictions, self.known)
        self.assertEqual(report['pooled']['decision'], 'inconclusive')
        self.assertEqual(report['pooled']['changed_sets'], 0)

    def test_incomplete_unlabeled_and_invalid_predictions_fail(self):
        missing = deepcopy(self.predictions)
        del missing['validation']['q2']
        with self.assertRaises(ValueError):
            evaluate(self.datasets, missing, self.known)
        self.predictions['validation']['q2']['proposed'] = list('aaaaa')
        with self.assertRaises(ValueError):
            evaluate(self.datasets, self.predictions, self.known)
        self.datasets['validation']['q2']['answer'] = None
        with self.assertRaises(ValueError):
            evaluate(self.datasets, self.predictions, self.known)


if __name__ == '__main__':
    unittest.main()
