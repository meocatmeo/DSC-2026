from copy import deepcopy
import unittest
from legalir2.agreement_097 import predict
from tests2.test_duo_agreement import fixture


class TestAgreement097(unittest.TestCase):
    def test_threshold_and_first_four_preserved(self):
        for value, fifth in ((.96, 'e'), (.97, 'f'), (.975, 'f'), (.99, 'f')):
            base = fixture(); base['probabilities'][5][4], base['probabilities'][4][5] = value, 1-value
            self.assertEqual(predict(base, deepcopy(base))[0], list('abcd')+[fifth])

    def test_requires_both_models(self):
        base = fixture(); other = deepcopy(base)
        base['probabilities'][5][4], base['probabilities'][4][5] = .999, .001
        self.assertEqual(predict(base, other)[0], list('abcde'))

    def test_ability_support_still_required(self):
        base = fixture(); base['probabilities'][5][4], base['probabilities'][4][5] = .99, .01
        other = deepcopy(base); other['ability'][5] = 1.
        self.assertEqual(predict(base, other)[0], list('abcde'))


if __name__ == '__main__':
    unittest.main()
