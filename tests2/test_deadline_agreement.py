from copy import deepcopy
import unittest
import numpy as np
from legalir2 import deadline_agreement as deadline, duo_agreement
from tests2.test_duo_agreement import fixture


class TestDeadline(unittest.TestCase):
    def test_wide_admits_additional_and_strict_reverts(self):
        row=fixture()
        self.assertEqual(deadline.predict(row,row,'fifth-wide')[0],list('abcdf'))
        self.assertEqual(deadline.predict(row,row,'fifth-strict')[0],list('abcde'))
        row['probabilities'][5][4],row['probabilities'][4][5]=.85,.15
        self.assertEqual(duo_agreement.predict(row,row)[0],list('abcde'))
        self.assertEqual(deadline.predict(row,row,'fifth-wide')[0],list('abcdf'))

    def test_fourth_requires_separate_outsider_and_keeps_fifth(self):
        row=fixture()
        row['probabilities'][5][3],row['probabilities'][3][5]=.99,.01
        self.assertEqual(deadline.predict(row,row,'fourth-rescue')[0],list('abcdf'))
        # f no longer qualifies for fifth, but can replace fourth.
        row['probabilities'][5][4],row['probabilities'][4][5]=.5,.5
        self.assertEqual(deadline.predict(row,row,'fourth-rescue')[0],list('abcfe'))

    def test_no_label_access_and_requires_both_models(self):
        row=fixture(); other=deepcopy(row)
        other['probabilities'][5][4],other['probabilities'][4][5]=.7,.3
        for policy in ('fifth-wide','fifth-strict'):
            self.assertEqual(deadline.predict(row,other,policy)[0],list('abcde'))
        for policy in deadline.POLICIES:
            expected=deadline.predict(row,other,policy)[0]
            row['gold'],other['gold']=['a'],['f']
            self.assertEqual(deadline.predict(row,other,policy)[0],expected)

    def test_permutation_and_invalid_policy(self):
        row=fixture(); indices=[5,2,0,4,1,3]; other=deepcopy(row)
        for key in ('docs','prior','ability'):
            other[key]=[row[key][i] for i in indices]
        other['probabilities']=np.asarray(row['probabilities'])[np.ix_(indices,indices)].tolist()
        for policy in deadline.POLICIES:
            self.assertEqual(deadline.predict(row,row,policy)[0],deadline.predict(other,other,policy)[0])
        with self.assertRaises(ValueError):
            deadline.predict(row,row,'unknown')


if __name__=='__main__':
    unittest.main()
