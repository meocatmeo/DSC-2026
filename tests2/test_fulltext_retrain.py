"""Matched-arm integrity and fail-closed promotion without GPU execution."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from legalir2 import fulltext_retrain as run


class TestFulltextRetrain(unittest.TestCase):
    def setUp(self):
        self.old = [dict(id='q', question='unique query', gold=['a'], role='fit', development=False,
                         pairs=[['a', 'b']], evidence={'a': [[1]], 'b': [[2]]})]
        self.new = deepcopy(self.old)
        self.new[0]['evidence']['a'] = [[3]]
        self.extra = deepcopy(self.new)
        self.extra[0]['pairs'].append(['a', 'c'])
        self.extra[0]['evidence']['c'] = [[4]]

    def test_matching_pairs_with_distinct_evidence(self):
        run.check_groups(self.old, self.new, self.extra, {}, {'q': 'fit'})
        self.new[0]['pairs'] = [['a', 'c']]
        with self.assertRaisesRegex(ValueError, 'altered original pairs'):
            run.check_groups(self.old, self.new, self.extra, {}, {'q': 'fit'})

    def test_heldout_alias_and_split_change_fail(self):
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            run.check_groups(self.old, self.new, self.extra, {'other': {'question': 'UNIQUE QUERY!'}}, {'q': 'fit'})
        self.extra[0]['development'] = True
        with self.assertRaisesRegex(ValueError, 'split changed'):
            run.check_groups(self.old, self.new, self.extra, {}, {'q': 'fit'})

    def test_failed_screen_never_reads_validation_or_promotes(self):
        with patch.object(run, 'screen', return_value={'continue_validation': False}), \
                patch.object(run, 'manifest', return_value={}), patch.object(run, 'collect') as collect:
            self.assertFalse(run.gate(SimpleNamespace())['deploy'])
            collect.assert_not_called()

    def test_cached_scores_cannot_adopt_another_evidence_row(self):
        base = dict(id='q', question='query', gold=['a'], docs=list('abcde'), prior=[0.]*5,
                    evidence={'token_trace': [1, 2, 3]})
        row = {k: base[k] for k in ('id', 'question', 'gold', 'docs', 'prior')}
        row.update(base=run.digest(base), ability=[0.]*5, probabilities=[[.5]*5 for _ in range(5)])
        run.validate_rows([row], [base])
        changed = deepcopy(base)
        changed['evidence']['token_trace'].append(4)
        with self.assertRaisesRegex(ValueError, 'pool, evidence or prior changed'):
            run.validate_rows([row], [changed])
        row['ability'][0] = 1.
        with self.assertRaisesRegex(ValueError, 'do not match'):
            run.validate_rows([row], [base])


if __name__ == '__main__':
    unittest.main()
