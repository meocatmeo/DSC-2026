"""Full-rank fusion retains deeper candidates and accounts for independent model weights."""
import unittest

from legalir2.current_fusion import FIXED_POLICY, cached_orderings, fuse, parameter_audit


class TestCurrentFusion(unittest.TestCase):
    def test_deeper_consensus_can_beat_top_five_only_fusion(self):
        rankings = {'v9': list('abcdex'), 'duo': list('fghijx'), 'fulltext_duo': list('klmnox')}
        self.assertEqual(fuse(rankings)[0], 'x')
        self.assertNotIn('x', fuse({k: r[:5] for k, r in rankings.items()}))

    def test_alignment_full_order_and_baseline_replay(self):
        base = dict(id='q', question='question', gold=['a'], docs=list('abcdef'))
        v9 = dict(base, features1=[[v] for v in (6, 5, 4, 3, 2, 1)])
        duo = dict(base, prior=[6, 5, 4, 3, 2, 1], ability=[6, 5, 4, 3, 2, 1], baseline=list('abcde'))
        self.assertEqual(cached_orderings(v9, duo, duo, [1])['v9'], list('abcdef'))
        with self.assertRaises(ValueError):
            cached_orderings(v9, dict(duo, id='wrong'), duo, [1])
        with self.assertRaises(ValueError):
            cached_orderings(v9, dict(duo, baseline=list('bcdef')), duo, [1])

    def test_missing_voter_duplicate_and_nonfinite_weight_fail(self):
        rankings = {name: list('abcde') for name in FIXED_POLICY['weights']}
        with self.assertRaises(ValueError):
            fuse(dict(rankings, fourth=list('abcde')))
        with self.assertRaises(ValueError):
            fuse(dict(rankings, duo=list('aaaaa')))
        policy = dict(FIXED_POLICY, weights=dict(FIXED_POLICY['weights'], duo=float('nan')))
        with self.assertRaises(ValueError):
            fuse(rankings, policy)

    def test_fourth_voter_and_budget(self):
        rankings = {name: list('abcde') for name in FIXED_POLICY['weights']}
        rankings['fourth'] = list('fghij')
        policy = dict(FIXED_POLICY, weights=dict(FIXED_POLICY['weights'], fourth=1.))
        self.assertEqual(set(fuse(rankings, policy)), set('abcdefghij'))
        base = {'components': {'existing': 2327063553}, 'total': 2327063553}
        self.assertEqual(parameter_audit(base, 567755777)['total'], 2894819330)
        with self.assertRaises(ValueError):
            parameter_audit(base, 2_000_000_000)


if __name__ == '__main__':
    unittest.main()
