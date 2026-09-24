from copy import deepcopy
import unittest
from legalir2 import fulltext_prior
from legalir2.h100 import digest
from legalir2.passage_order_run import ARMS, validate_result
from tests2.test_duo_agreement import fixture


def case():
    row = fixture(); row['gold'] = ['a']
    row['ability'] = fulltext_prior.fit(row['prior'], row['probabilities']).tolist()
    item = {'original': row, 'control': deepcopy(row), 'edges': [[0, 1]]}
    result = deepcopy(row); result['source'] = digest(item)
    return item, result


class TestPassageOrderRun(unittest.TestCase):
    def test_correct_result_and_changed_edge_fit(self):
        item, result = case()
        validate_result(result, item, ARMS[0])
        result['probabilities'][0][1], result['probabilities'][1][0] = .8, .2
        result['ability'] = fulltext_prior.fit(result['prior'], result['probabilities']).tolist()
        validate_result(result, item, ARMS[1])

    def test_changed_prior_or_cache_identity_rejected(self):
        for key, value in (('source', 'wrong'), ('prior', [0.]*6)):
            item, result = case(); result[key] = value
            with self.assertRaises(ValueError):
                validate_result(result, item, ARMS[0])

    def test_altering_unmeasured_edge_rejected_even_with_valid_fit(self):
        item, result = case()
        result['probabilities'][2][3], result['probabilities'][3][2] = .8, .2
        result['ability'] = fulltext_prior.fit(result['prior'], result['probabilities']).tolist()
        with self.assertRaises(ValueError):
            validate_result(result, item, ARMS[0])


if __name__ == '__main__':
    unittest.main()
