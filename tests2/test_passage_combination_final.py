from copy import deepcopy
import unittest
from legalir2 import agreement_097, passage_combination as previous, passage_combination_final as final
from tests2.test_passage_combination import four_rows, pair


class TestFinalCombination(unittest.TestCase):
    def test_fixed_boundary(self):
        for preference, fifth in ((.949999, 'e'), (.95, 'f'), (.96, 'f'), (.97, 'f')):
            rows = four_rows()
            for row in rows[2:]: pair(row, 5, 4, preference)
            self.assertEqual(final.predict(*rows)[0], list('abcd')+[fifth])

    def test_keeps_existing_replacement(self):
        rows = four_rows()
        self.assertEqual(final.predict(*rows)[0], previous.predict(*rows)[0])
        self.assertEqual(final.predict(*rows)[0], list('abcdf'))

    def test_old_support_and_each_ability_still_required(self):
        for arm in range(4):
            rows = four_rows(); rows[arm]['ability'][5] = rows[arm]['ability'][4]
            self.assertEqual(final.predict(*rows)[0], agreement_097.predict(*rows[:2])[0])
        rows = four_rows(); pair(rows[0], 5, 4, .5)
        self.assertEqual(final.predict(*rows)[0], list('abcde'))


if __name__ == '__main__': unittest.main()
