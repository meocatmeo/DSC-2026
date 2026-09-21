"""Exercise anchor comparisons, deployment gates, cache guards and submission."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from legalir2 import duo, retrieval_rescue as rescue
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json
from tests2.test_fulltext_prior import fixture


def relations():
    return SimpleNamespace(known=set('abcdefg'),
        propose=Mock(return_value=[{'doc': 'g', 'score': .1, 'reasons': []}]))


def scored(gold='g'):
    base, related = fixture(gold), relations()
    proposed = rescue.plan(base, related)
    wins = np.full((7, 5), .5)
    wins[5] = .2
    wins[6] = .95
    trace = deepcopy(base['evidence']['documents']['a'])
    row = dict(proposed, id='q', question=base['question'], gold=base['gold'],
        base=digest(base), wins=wins.tolist(),
        evidence={'documents': {d: deepcopy(trace) for d in proposed['docs']}})
    return row, base, related


class TestRescue(unittest.TestCase):
    def test_plan_replays_baseline_and_excludes_existing_citation_candidates(self):
        row, base, related = scored()
        self.assertEqual(row['baseline'], rescue.prior.predict(base, rescue.prior.BASELINE))
        self.assertEqual(row['docs'], list('abcdefg'))
        self.assertEqual(related.propose.call_args.args[1], base['docs'])
        self.assertEqual(related.propose.call_args.kwargs['mode'], 'citations')

    def test_measured_edges_equal_full_comparator_in_both_orders(self):
        import torch
        class Tokenizer:
            def pad(self, values, **kwargs):
                return {'input_ids': torch.tensor(values['input_ids'])}
        class Model:
            calls = 0
            def __call__(self, input_ids, **kwargs):
                self.calls += len(input_ids)
                # Deliberately include A/B position bias; both orders remove it.
                diff = input_ids[:, 0]-input_ids[:, 1]+.7
                return SimpleNamespace(logits=torch.stack((diff, torch.zeros_like(diff)), dim=1)[:, None, :])
        docs = [str(i) for i in range(16)]
        evidence = {d: int(d) for d in docs}
        codec = SimpleNamespace(tokenizer=Tokenizer(), labels=[0, 1], pair=lambda q, a, b: [a, b])
        local = SimpleNamespace(torch=torch, model=Model(), device='cpu')
        anchors = rescue.compare_anchors(local, codec, 'q', evidence, docs, 4)
        self.assertEqual(local.model.calls, 130)
        full = duo.compare_documents(local, codec, 'q', evidence, docs, 4)
        np.testing.assert_allclose(anchors, full[:, :5], atol=1e-7)
        self.assertTrue(np.isfinite(anchors).all())

    def test_policy_protects_top_three_and_requires_confidence(self):
        row, _, _ = scored()
        self.assertEqual(rescue.predict(row, rescue.BASELINE), list('abcde'))
        self.assertEqual(rescue.predict(row, {'scope': 'focused', 'threshold': .75}), list('abcde'))
        changed = rescue.predict(row, {'scope': 'citations', 'threshold': .9})
        self.assertEqual(changed[:3], list('abc'))
        self.assertEqual(len(set(changed)-set(row['baseline'])), 1)
        self.assertIn('g', changed)
        row['wins'][6] = [.89]*5
        self.assertEqual(rescue.predict(row, {'scope': 'citations', 'threshold': .9}), row['baseline'])
        row['wins'][6] = [0., 0., 0., .95, .95]
        self.assertEqual(rescue.predict(row, {'scope': 'citations', 'threshold': .9}), row['baseline'])

    def test_corrupt_cache_and_changed_source_fail_closed(self):
        row, base, related = scored()
        rescue.validate_scores([row], [base], related)
        for mutate in (
            lambda r: r.update(base='changed'),
            lambda r: r['wins'][6].__setitem__(0, float('nan')),
            lambda r: r['wins'][0].__setitem__(1, .8),
            lambda r: r['evidence']['documents']['g'].update(input_ids=[3]),
            lambda r: r['docs'].__setitem__(6, 'unknown'),
        ):
            changed = deepcopy(row)
            mutate(changed)
            with self.assertRaises(ValueError):
                rescue.validate_scores([changed], [base], related)

    def test_resume_reuses_valid_shard_without_loading_gpu(self):
        row, base, related = scored()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory))
            target = args.out/'cache/rescue/development/part-00000.json'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps([row]))
            (target.parent/'resume.json').write_text(json.dumps({
                'inputs': digest({'id': 'fixture'}), 'split': 'development'}))
            with patch.object(rescue, 'manifest', return_value={'id': 'fixture'}), \
                    patch.object(rescue, 'graph', return_value=related), \
                    patch.object(rescue.prior, 'records', return_value={'q': {}}), \
                    patch.object(rescue, 'base_parts', return_value=[(target.name, [base], [])]), \
                    patch.object(rescue, 'Local') as model:
                rescue.collect(args, 'development')
                model.assert_not_called()

    def test_validation_blocks_regression_and_no_gain_skips_work(self):
        row, _, _ = scored()
        policy = {'scope': 'citations', 'threshold': .9}
        self.assertTrue(rescue.decision([row], policy)['deploy'])
        row['gold'] = list(set(row['baseline'])-set(rescue.predict(row, policy)))
        self.assertFalse(rescue.decision([row], policy)['deploy'])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory))
            with patch.object(rescue, 'choice', return_value={'selected': rescue.BASELINE}), \
                    patch.object(rescue, 'manifest', return_value={}), \
                    patch.object(rescue, 'scores') as scores:
                rescue.validate(args)
                report = rescue.deployment(args)
                self.assertFalse(report['deploy'])
                self.assertFalse(report['evaluated'])
                scores.assert_not_called()

    def test_calibration_ties_keep_baseline_and_private_labels_unused(self):
        row, _, _ = scored()
        row['gold'] = ['a']
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory))
            with patch.object(rescue, 'scores', return_value=[row]) as scores, \
                    patch.object(rescue, 'manifest', return_value={}):
                rescue.calibrate(args)
                self.assertEqual(read_json(args.out/'calibration.json')['selected'], rescue.BASELINE)
                scores.assert_called_once_with(args, 'development')

    def test_submit_exact_fallback_and_candidate_keep_v9_unchanged(self):
        row, _, related = scored()
        for deploy in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                old = {'q': {'answer': list('abcde')}}
                for name in ('duo', 'v9'):
                    with zipfile.ZipFile(root/f'submission-private-{name}.zip', 'w') as archive:
                        archive.writestr('submission.json', json.dumps(old))
                marker = {'source_files': {'submission-private-v9.zip': file_hash(root/'submission-private-v9.zip')}, 'assets': {}}
                args = SimpleNamespace(out=root/'out', source=root)
                policy = {'scope': 'citations', 'threshold': .9}
                with patch.object(rescue, 'deployment', return_value={'deploy': deploy, 'policy': policy}), \
                        patch.object(rescue.prior, 'manifest', return_value=marker), \
                        patch.object(rescue.prior, 'SCORED_ZIP', file_hash(root/'submission-private-duo.zip')), \
                        patch.object(rescue, 'scores', return_value=[row]), \
                        patch.object(rescue, 'graph', return_value=related), \
                        patch.object(rescue.private, 'write_submission') as write:
                    rescue.submit(args)
                    self.assertEqual(write.call_args_list[0].args[1], old)
                    expected = {'q': {'answer': rescue.predict(row, policy)}} if deploy else old
                    self.assertEqual(write.call_args_list[1].args[1], expected)


if __name__ == '__main__':
    unittest.main()
