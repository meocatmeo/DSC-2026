import copy
import unittest

import numpy as np

from legalir2.duo_boundary import predict


class TestBoundary(unittest.TestCase):
    def row(self):
        p = np.full((6, 6), .5)
        p[5, 4], p[4, 5] = .8, .2
        return {'docs': list('abcdef'), 'baseline': list('abcde'),
                'prior': [6., 5., 4., 3., 2., 1.],
                'ability': [6., 5., 4., 3., 0., 9.], 'probabilities': p.tolist()}

    def test_only_supported_proposed_replacement(self):
        row = self.row()
        original = copy.deepcopy(row)
        self.assertEqual(set(predict(row, .75)), set('abcdf'))
        self.assertEqual(set(predict(row, .9)), set('abcde'))
        self.assertEqual(predict(row, strength=0), list('abcde'))
        self.assertEqual(row, original)

    def test_labels_cannot_change_predictions(self):
        row = self.row()
        expected = predict(row)
        row.update(gold=['e'], question='ignored', id='ignored')
        self.assertEqual(predict(row), expected)
        row['gold'] = ['f']
        self.assertEqual(predict(row), expected)

    def test_invalid_inputs_fail(self):
        for threshold in (.4, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                predict(self.row(), threshold)
        row = self.row()
        row['probabilities'][0][1] = float('nan')
        with self.assertRaises(ValueError):
            predict(row)


if __name__ == '__main__':
    unittest.main()
