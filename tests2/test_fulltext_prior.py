"""Calibration isolation, cache provenance and full-text scoring without a GPU."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from legalir2 import duo, fulltext_prior as prior
from legalir2.models import TEMPLATES, rerank_prompt
from legalir2.store import file_hash, read_json, write_json
from tests2.test_duo import CharacterTokenizer


def fixture(gold='f'):
    probability = np.full((6, 6), .5)
    base, ability = duo.tournament(np.arange(6., 0., -1), probability)
    trace = {'label': 'Actual document title', 'packed': [
        {'rowid': 1, 'text': 'Actual relevant full-text clause.', 'tokens': 2}],
        'input_ids': [1, 2], 'budget': 100,
        'sha256': hashlib.sha256(json.dumps([1, 2]).encode()).hexdigest()}
    return {'id': 'q', 'question': 'Which document?', 'gold': [gold], 'docs': list('abcdef'),
            'prior': base.tolist(), 'ability': ability.tolist(), 'probabilities': probability.tolist(),
            'relevance': [-10., 0., 0., 0., 0., 10.],
            'evidence': {'documents': {d: deepcopy(trace) for d in 'abcdef'}}}


class TestFulltextPrior(unittest.TestCase):
    def test_memory_rejects_heldout_ids_and_normalized_aliases(self):
        memory = {'train': {'question': 'Training question', 'answer': ['a']}}
        prior.validate_memory(memory, {'val': {'question': 'Different question'}})
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            prior.validate_memory(memory, {'train': {'question': 'Different question'}})
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            prior.validate_memory(memory, {'alias': {'question': '  TRAINING   question '}})

    def test_zero_alpha_reproduces_frozen_duo_exactly(self):
        row = fixture()
        expected = duo.diagnostics([row], .25)['q']['predicted']
        self.assertEqual(prior.predict(row, prior.BASELINE), expected)
        self.assertNotIn('f', expected)

    def test_new_fulltext_signal_can_admit_low_prior_candidate(self):
        row = fixture()
        before = deepcopy(row)
        self.assertIn('f', prior.predict(row, {'alpha': 1., 'strength': .25}))
        self.assertEqual(row, before)
        fitted = prior.fit(np.array(row['prior']), np.array(row['probabilities']))
        np.testing.assert_allclose(fitted, row['ability'], atol=1e-6)

    def test_select_uses_labels_and_keeps_baseline_on_tie(self):
        row = fixture()
        self.assertNotEqual(prior.select([row])['selected'], prior.BASELINE)
        row['gold'] = list('abcdef')
        self.assertEqual(prior.select([row])['selected'], prior.BASELINE)
        row['gold'] = []
        with self.assertRaisesRegex(ValueError, 'labeled'):
            prior.select([row])

    def test_validation_regression_blocks_private_deployment(self):
        chosen = prior.select([fixture()])['selected']
        self.assertTrue(prior.decision([fixture()], chosen)['deploy'])
        lost = set(prior.predict(fixture(), prior.BASELINE))-set(prior.predict(fixture(), chosen))
        result = prior.decision([fixture(sorted(lost)[0])], chosen)
        self.assertFalse(result['deploy'])
        self.assertLess(result['delta'], 0)
        with patch.object(prior, 'deployment', return_value=result), \
                patch.object(prior, 'collect_pointwise') as score:
            prior.private_scores(SimpleNamespace())
        score.assert_not_called()

    def test_invalid_probabilities_and_relevance_fail_closed(self):
        row = fixture()
        row['probabilities'][0][1] = 2
        with self.assertRaises(ValueError):
            prior.fit(np.array(row['prior']), np.array(row['probabilities']))
        row = fixture()
        row['relevance'][0] = float('nan')
        with self.assertRaises(ValueError):
            prior.predict(row, {'alpha': .5, 'strength': .25})
        with self.assertRaises(ValueError):
            prior.predict(fixture(), {'alpha': .123, 'strength': .25})

    def test_no_development_gain_skips_validation_without_fake_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory))
            chosen = {'selected': prior.BASELINE}
            with patch.object(prior, 'choice', return_value=chosen), \
                    patch.object(prior, 'manifest', return_value={'id': 1}), \
                    patch.object(prior, 'scored') as score:
                prior.validate(args)
                report = prior.deployment(args)
                self.assertFalse(report['deploy'])
                self.assertFalse(report['evaluated'])
                self.assertIsNone(report['delta'])
                score.assert_not_called()

    def test_pointwise_reads_actual_passages_with_exact_prompt_budget(self):
        model = SimpleNamespace(local=SimpleNamespace(tokenizer=CharacterTokenizer()),
                                template=TEMPLATES['qwen3-reranker'])
        packed = prior.bundles(fixture(), model)
        self.assertEqual(len(packed), 6)
        prefix, suffix = rerank_prompt(model.template)
        for bundle in packed:
            text = ''.join(map(chr, bundle['input_ids']))
            self.assertIn('Actual relevant full-text clause.', text)
            self.assertIn('Actual document title', text)
            self.assertTrue(text.startswith(prefix) and text.endswith(suffix))
            self.assertLessEqual(len(bundle['input_ids']), 2048)

    def test_evidence_corruption_is_detected(self):
        row = fixture()
        prior.validate_base([row], [('q', {'question': row['question'], 'answer': row['gold']})], set(row['docs']))
        row['evidence']['documents']['a']['input_ids'] = [3]
        with self.assertRaisesRegex(ValueError, 'token trace'):
            prior.validate_base([row], [('q', {'question': row['question'], 'answer': row['gold']})], set(row['docs']))

    def test_pointwise_resume_skips_model_and_rejects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory), models=Path(directory), batch=4)
            row = fixture()
            model = Mock()
            model.local.tokenizer = CharacterTokenizer()
            model.template = TEMPLATES['qwen3-reranker']
            model.score_bundles.return_value = [.1, .2, .3, .4, .5, .9]
            with patch.object(prior, 'manifest', return_value={'id': 1}), \
                    patch.object(prior, 'records', return_value={'q': {}}), \
                    patch.object(prior, 'bases', side_effect=lambda *a: iter([(Path('part-00000.json'), [row])])), \
                    patch.object(prior, 'Reranker', return_value=model) as load:
                prior.collect_pointwise(args, 'development')
                prior.collect_pointwise(args, 'development')
                self.assertEqual(load.call_count, 1)
                model.close.assert_called_once()
                row['evidence']['documents']['a']['label'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'another comparison row'):
                    prior.collect_pointwise(args, 'development')

    def test_submit_packages_candidate_or_preserves_baseline_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root/'out', source=root/'source', duo_run=root/'model',
                                   questions=root/'questions.json')
            args.source.mkdir()
            write_json(args.questions, {'q': {'question': 'Which document?', 'answer': None}})
            write_json(args.duo_run/'inputs.json', {'known_docs': list('abcdef')})
            baseline = {'q': {'answer': list('abcde')}}
            for name in ('v9', 'duo'):
                with zipfile.ZipFile(args.source/f'submission-private-{name}.zip', 'w') as archive:
                    archive.writestr('submission.json', json.dumps(baseline))
            marker = {'assets': {'strength': .25, 'questions_sha256': file_hash(args.questions)},
                      'source_files': {'submission-private-v9.zip': file_hash(args.source/'submission-private-v9.zip')}}
            for deploy in (False, True):
                report = {'deploy': deploy, 'policy': {'alpha': 1., 'strength': .25}}
                with patch.object(prior, 'deployment', return_value=report), \
                        patch.object(prior, 'manifest', return_value=marker), \
                        patch.object(prior, 'scored', return_value=[fixture()]) as scoring, \
                        patch.object(prior, 'SCORED_ZIP', file_hash(args.source/'submission-private-duo.zip')):
                    prior.submit(args)
                with zipfile.ZipFile(args.out/'submission-private-duo.zip') as archive:
                    self.assertEqual(archive.namelist(), ['submission.json'])
                    result = json.loads(archive.read('submission.json'))
                self.assertEqual('f' in result['q']['answer'], deploy)
                if not deploy:
                    self.assertEqual(result, baseline)
                    scoring.assert_not_called()


if __name__ == '__main__':
    unittest.main()
