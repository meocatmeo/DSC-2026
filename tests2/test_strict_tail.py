from copy import deepcopy
import unittest
from legalir2.strict_tail import predict
from tests2.test_duo_agreement import fixture


class TestStrictTail(unittest.TestCase):
    def pair(self, confidence):
        row = fixture()
        row['probabilities'][5][4], row['probabilities'][4][5] = confidence, 1-confidence
        return row, deepcopy(row)

    def test_reverts_only_the_borderline_fifth(self):
        base, other = self.pair(.985)
        answer, trace = predict(base, other)
        self.assertEqual(answer, list('abcde'))
        self.assertTrue(trace['reverted'])

    def test_retains_exact_threshold_and_above(self):
        for confidence in (.99, .999):
            base, other = self.pair(confidence)
            self.assertEqual(predict(base, other)[0], list('abcdf'))

    def test_uses_weaker_model_and_never_reads_gold(self):
        base, other = self.pair(.999)
        other['probabilities'][5][4], other['probabilities'][4][5] = .985, .015
        base['gold'], other['gold'] = ['f'], ['e']
        self.assertEqual(predict(base, other)[0], list('abcde'))

    def test_no_rescue_stays_unchanged(self):
        base, other = self.pair(.7)
        answer, trace = predict(base, other)
        self.assertEqual(answer, list('abcde'))
        self.assertFalse(trace['reverted'])


if __name__ == '__main__':
    unittest.main()
