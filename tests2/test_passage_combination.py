from copy import deepcopy
import unittest
from legalir2.passage_combination import predict
from legalir2.agreement_097 import predict as anchor
from tests2.test_duo_agreement import fixture


def pair(row, first, second, value):
    row['probabilities'][first][second] = value
    row['probabilities'][second][first] = 1-value


def four_rows():
    old = fixture()
    new = deepcopy(old); pair(new, 5, 4, .97)
    return [old, deepcopy(old), new, deepcopy(new)]


class TestPassageCombination(unittest.TestCase):
    def test_new_support_changes_only_fifth_without_reading_labels(self):
        rows = four_rows(); before, _ = anchor(*rows[:2])
        answer, trace = predict(*rows)
        self.assertEqual(before, list('abcde'))
        self.assertEqual(answer, list('abcdf'))
        self.assertTrue(trace['replaced'])
        for row in rows: row['gold'] = ['unrelated']
        self.assertEqual(predict(*rows)[0], answer)

    def test_each_required_preference_and_margin(self):
        for arm in range(4):
            rows = four_rows()
            pair(rows[arm], 5, 4, .5 if arm < 2 else .969999)
            self.assertEqual(predict(*rows)[0], anchor(*rows[:2])[0])
            rows = four_rows(); rows[arm]['ability'][5] = rows[arm]['ability'][4]
            self.assertEqual(predict(*rows)[0], anchor(*rows[:2])[0])

    def test_protects_old_prefix_even_if_new_ranking_changes(self):
        rows = four_rows()
        for row in rows[2:]: row['ability'][0] = -100.
        self.assertEqual(predict(*rows)[0][:4], list('abcd'))

    def test_challenges_actual_anchor_fifth_not_original_victim(self):
        rows = four_rows()
        for row in rows[:2]: pair(row, 5, 4, .99)
        # Original victim e strongly beats rescued f only under the new order.
        for row in rows[2:]: pair(row, 4, 5, .999); row['ability'][4] = 5.
        self.assertEqual(anchor(*rows[:2])[0], list('abcdf'))
        self.assertEqual(predict(*rows)[0], list('abcdf'))

    def test_misaligned_or_invalid_scores_fail(self):
        rows = four_rows(); rows[2]['prior'][0] += 1
        with self.assertRaises(ValueError): predict(*rows)
        rows = four_rows(); rows[3]['probabilities'][5][4] = float('nan')
        with self.assertRaises(ValueError): predict(*rows)

    def test_candidate_order_does_not_change_prediction(self):
        rows = four_rows(); expected = predict(*rows)[0]; order = [5, 2, 0, 4, 1, 3]
        permuted = []
        for row in rows:
            new = deepcopy(row)
            for key in ('docs', 'prior', 'ability'): new[key] = [row[key][i] for i in order]
            new['probabilities'] = [[row['probabilities'][i][j] for j in order] for i in order]
            permuted.append(new)
        self.assertEqual(predict(*permuted)[0], expected)


if __name__ == '__main__':
    unittest.main()
