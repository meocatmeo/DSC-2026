from copy import deepcopy
import unittest
import numpy as np
from legalir2 import append_only_deep as model
from legalir2.h100 import digest
from tests2.test_duo_agreement import fixture


def rows():
    base=fixture();base['probabilities'][5][4]=.5;base['probabilities'][4][5]=.5
    p=np.full((7,7),.5);p[:6,:6]=base['probabilities']
    p[6,4],p[4,6]=.99,.01
    p[6,:4]=.6;p[:4,6]=.4
    expanded={'id':base['id'],'question':base['question'],'base':digest(base),
              'docs':base['docs']+['g'],'probabilities':p.tolist()}
    return base,deepcopy(base),expanded


class TestAppendOnly(unittest.TestCase):
    def test_only_new_candidate_can_replace_fifth(self):
        b,n,e=rows();answer,trace=model.predict(b,n,e)
        self.assertEqual(answer,list('abcdg'));self.assertTrue(trace['replaced'])
        e['prior']=[float('nan')]*7;e['ability']=[999]*7
        self.assertEqual(model.predict(b,n,e)[0],answer)

    def test_requires_direct_and_consistent_comparison_support(self):
        b,n,e=rows();e['probabilities'][6][4],e['probabilities'][4][6]=.8,.2
        self.assertEqual(model.predict(b,n,e)[0],list('abcde'))
        b,n,e=rows()
        for i in range(4):
            e['probabilities'][6][i],e['probabilities'][i][6]=.4,.6
        self.assertEqual(model.predict(b,n,e)[0],list('abcde'))

    def test_changed_old_edges_or_source_fail(self):
        b,n,e=rows();e['probabilities'][0][1]=.9
        with self.assertRaises(ValueError):
            model.predict(b,n,e)
        b,n,e=rows();e['base']='different'
        with self.assertRaises(ValueError):
            model.predict(b,n,e)

    def test_keeps_current_agreement_prefix_and_evaluates_its_fifth(self):
        b,n,e=rows()
        for row in (b,n):
            row['probabilities'][5][4],row['probabilities'][4][5]=.99,.01
        e['base']=digest(b)
        e['probabilities'][5][4],e['probabilities'][4][5]=.99,.01
        # g beats the old fifth e but not the agreement fifth f: no replacement.
        self.assertEqual(model.predict(b,n,e)[0],list('abcdf'))


if __name__=='__main__':
    unittest.main()
