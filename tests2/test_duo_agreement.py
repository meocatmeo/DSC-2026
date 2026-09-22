"""Agreement rescue must preserve the baseline and require both models' support."""
from copy import deepcopy
import unittest
import numpy as np
from legalir2.duo_agreement import predict


def fixture():
    row = dict(id='q', question='query', docs=list('abcdef'), prior=[6.,5.,4.,3.,2.,-3.],
               ability=[6.,5.,4.,3.,2.,4.], probabilities=np.full((6,6), .5).tolist())
    row['probabilities'][5][4], row['probabilities'][4][5] = .95, .05
    return row


class TestAgreement(unittest.TestCase):
    def test_only_fifth_replaced_and_labels_not_used(self):
        first = fixture()
        second = deepcopy(first)
        answer, trace = predict(first, second)
        self.assertEqual(answer, list('abcdf'))
        self.assertTrue(trace['replaced'])
        first['gold'], second['gold'] = ['a'], ['f']
        self.assertEqual(predict(first, second)[0], answer)

    def test_disagreement_or_ability_opposition_keeps_baseline(self):
        first, second = fixture(), fixture()
        second['probabilities'][5][4], second['probabilities'][4][5] = .8, .2
        self.assertEqual(predict(first, second)[0], list('abcde'))
        second = fixture()
        second['ability'][5] = 1.
        self.assertEqual(predict(first, second)[0], list('abcde'))

    def test_invalid_alignment_and_matrix_fail(self):
        first, second = fixture(), fixture()
        second['docs'] = list('abcdeg')
        with self.assertRaises(ValueError):
            predict(first, second)
        second = fixture()
        second['probabilities'][5][4] = float('nan')
        with self.assertRaises(ValueError):
            predict(first, second)

    def test_candidate_order_does_not_change_result(self):
        first = fixture()
        expected = predict(first, first)[0]
        p = [5,2,0,4,1,3]
        permuted = deepcopy(first)
        for k in ('docs', 'prior', 'ability'):
            permuted[k] = [first[k][i] for i in p]
        permuted['probabilities'] = np.asarray(first['probabilities'])[np.ix_(p,p)].tolist()
        self.assertEqual(predict(permuted, permuted)[0], expected)


if __name__ == '__main__':
    unittest.main()
