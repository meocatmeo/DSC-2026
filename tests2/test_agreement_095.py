from copy import deepcopy
import unittest
from legalir2.agreement_095 import predict
from tests2.test_duo_agreement import fixture


class TestAgreement095(unittest.TestCase):
    def test_boundary_and_protected_prefix(self):
        for confidence, fifth in ((.949999, 'e'), (.95, 'f'), (.96, 'f'), (.97, 'f')):
            base = fixture()
            base['probabilities'][5][4], base['probabilities'][4][5] = confidence, 1-confidence
            self.assertEqual(predict(base, deepcopy(base))[0], list('abcd')+[fifth])

    def test_requires_both_models_and_positive_ability_margin(self):
        base = fixture(); other = deepcopy(base)
        other['probabilities'][5][4], other['probabilities'][4][5] = .949, .051
        self.assertEqual(predict(base, other)[0], list('abcde'))
        other = deepcopy(base); other['ability'][5] = other['ability'][4]
        self.assertEqual(predict(base, other)[0], list('abcde'))


if __name__ == '__main__':
    unittest.main()
