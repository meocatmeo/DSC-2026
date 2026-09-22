"""Direct validation rejects corrupt caches without an unnecessary second solve."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts/gpu'))
import retrain_private_resume as repair


class TestPrivateResume(unittest.TestCase):
    def setUp(self):
        self.base = dict(id='q', question='query', gold=[], docs=list('abcdef'),
                         prior=[3., 2., 1., 0., -1., -2.], evidence={'saved': True})
        probability = np.full((6, 6), .5)
        self.row = {k: deepcopy(self.base[k]) for k in ('id', 'question', 'gold', 'docs', 'prior')}
        self.row.update(base=repair.runner.digest(self.base), probabilities=probability.tolist(),
                        ability=repair.retrain.prior.fit(self.base['prior'], probability).tolist())

    def test_second_solve_failure_does_not_invalidate_stationary_saved_fit(self):
        before = deepcopy(self.row)
        with patch.object(repair.retrain.duo, 'tournament', side_effect=ValueError('Comparison tournament did not converge')) as solve:
            with self.assertRaisesRegex(ValueError, 'did not converge'):
                repair.retrain.validate_rows([self.row], [self.base])
            repair.validate_rows([self.row], [self.base])
            self.assertEqual(solve.call_count, 1)
        self.assertEqual(self.row, before)
        self.assertEqual(repair.retrain.prior.predict(self.row, repair.retrain.prior.BASELINE),
                         repair.retrain.prior.predict(before, repair.retrain.prior.BASELINE))

    def test_rejects_invalid_matrices_stale_scores_and_wrong_source(self):
        mutations = [lambda r: r['probabilities'].pop(),
                     lambda r: r['probabilities'][0].__setitem__(1, float('nan')),
                     lambda r: r['probabilities'][0].__setitem__(1, 1.1),
                     lambda r: r['probabilities'][0].__setitem__(1, .9),
                     lambda r: r['ability'].__setitem__(0, r['ability'][0]+.01),
                     lambda r: r.__setitem__('base', 'foreign'),
                     lambda r: r['prior'].__setitem__(0, 999.),
                     lambda r: r['ability'].pop()]
        for mutate in mutations:
            row = deepcopy(self.row)
            mutate(row)
            with self.assertRaises(ValueError):
                repair.validate_rows([row], [self.base])
        with self.assertRaises(ValueError):
            repair.validate_rows([], [self.base])

    def test_actual_saved_shard_passes_without_mutating_predictions(self):
        path = ROOT/'reports/retrain-convergence-audit/cache/control/private/part-00114.json'
        if not path.exists():
            self.skipTest('Cloud audit shard not downloaded')
        rows = repair.runner.read_json(path)
        for row in rows:
            # Source digest check is tested above; retrieve no private texts in test output.
            base = {k: deepcopy(row[k]) for k in ('id', 'question', 'gold', 'docs', 'prior')}
            checked = deepcopy(row)
            checked['base'] = repair.runner.digest(base)
            before = repair.retrain.prior.predict(checked, repair.retrain.prior.BASELINE)
            repair.validate_rows([checked], [base])
            self.assertEqual(before, repair.retrain.prior.predict(checked, repair.retrain.prior.BASELINE))


if __name__ == '__main__':
    unittest.main()
