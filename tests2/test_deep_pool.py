"""No-GPU coverage of deep collection, trained fusion, cache and deployment."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from legalir2 import deep_pool as deep, deep_fusion as fusion, duo
from legalir2.h100 import digest
from legalir2.models import TEMPLATES
from legalir2.store import file_hash, read_json, write_json
from tests2.test_duo import CharacterTokenizer
from tests2.test_fulltext_prior import fixture


def raw_fixture():
    original = fixture()
    docs = list('abcdefgh')
    old = {'id': 'q', 'question': original['question'], 'gold': original['gold'],
        'docs': list('abcdef'), 'features1': np.arange(120).reshape(6, 20).tolist()}
    weights = np.ones(20)
    fused = [(d, float(8-i)) for i, d in enumerate(docs)]
    channels = {k: list(fused) for k in ('fused', 'dense', 'lexical')}
    tables = [{d: [float(i), float(i)-.2] for i, d in enumerate(docs)},
              {d: [float(i)/10] for i, d in enumerate(docs)}]
    v9 = dict(zip(old['docs'], (np.asarray(old['features1'])@weights).tolist()))
    traces = deepcopy(original['evidence'])
    traces['documents'].update({d: deepcopy(traces['documents']['a']) for d in 'gh'})
    raw = {'id': 'q', 'question': old['question'], 'gold': old['gold'], 'docs': docs,
        'v9_input': digest(old), 'v9': v9, 'retained': original['docs'], 'retrieved': fused,
        'channels': channels, 'tables': tables, 'features': fusion.features(docs, tables, channels, v9).tolist(),
        'evidence': traces}
    model = {'features': fusion.FEATURES, 'weights': [1.]*len(fusion.FEATURES)}
    return raw, original, old, weights, model


def compared_fixture():
    raw, old, _, _, model = raw_fixture()
    docs, initial = fusion.shortlist(raw, old, model)
    p = np.full((len(docs), len(docs)), .5)
    p[:6, :6] = old['probabilities']
    row = {'id': raw['id'], 'question': raw['question'], 'gold': raw['gold'], 'docs': docs,
        'prior': initial.tolist(), 'ability': deep.prior.fit(initial, p).tolist(), 'probabilities': p.tolist(),
        'raw': digest(raw), 'base': digest(old), 'baseline': deep.prior.predict(old, deep.prior.BASELINE)}
    return row, raw, old, model


class TestDeepPool(unittest.TestCase):
    def test_pool_keeps_original_candidates_beyond_cutoff(self):
        fused = [(str(i), 200-i) for i in range(150)]
        docs = fusion.candidate_pool(fused, ['130', '10'], {str(i) for i in range(150)})
        self.assertEqual(docs[:100], [str(i) for i in range(100)])
        self.assertEqual(docs[-1], '130')
        self.assertEqual(len(docs), 101)
        with self.assertRaises(ValueError):
            fusion.candidate_pool(fused, ['unknown'], {str(i) for i in range(150)})

    def test_raw_scores_replay_features_and_detect_tampering(self):
        raw, original, old, weights, _ = raw_fixture()
        args = ([('q', {'question': old['question'], 'answer': old['gold']})], {'q': old}, set('abcdefgh'), weights, {'q': original})
        deep.validate_raw([raw], *args)
        for mutate in (
            lambda r: r['features'][0].__setitem__(0, 99.),
            lambda r: r['tables'][0]['a'].__setitem__(0, float('nan')),
            lambda r: r['evidence']['documents']['a'].update(label='changed title'),
            lambda r: r.update(retained=list('abcde')),
        ):
            changed = deepcopy(raw)
            mutate(changed)
            with self.assertRaises(ValueError):
                deep.validate_raw([changed], *args)
        self.assertEqual(raw['features'][-1][-2:], [0., 0.])

    def test_labels_never_enter_features_or_shortlist(self):
        raw, original, _, _, model = raw_fixture()
        result = fusion.shortlist(raw, original, model)
        changed = deepcopy(raw)
        changed['gold'] = ['another-answer']
        changed_old = deepcopy(original)
        changed_old['gold'] = ['another-answer']
        new = fusion.shortlist(changed, changed_old, model)
        self.assertEqual(result[0], new[0])
        np.testing.assert_array_equal(result[1], new[1])
        self.assertEqual(result[0][:len(original['docs'])], original['docs'])
        self.assertLessEqual(len(result[0])-len(original['docs']), fusion.EXTRA)

    def test_training_split_rejects_ids_and_normalized_aliases(self):
        train = {'t': {'question': 'Training query', 'answer': ['a']}}
        deep.validate_split(train, {'d': {'question': 'Development query'}}, {'v': {'question': 'Validation query'}})
        with self.assertRaises(ValueError):
            deep.validate_split(train, {'t': {'question': 'Other'}}, {})
        with self.assertRaises(ValueError):
            deep.validate_split(train, {}, {'v': {'question': ' TRAINING   query? '}})

    def test_collection_scores_both_models_and_preserves_old_passages(self):
        raw, original, old, weights, _ = raw_fixture()
        first = SimpleNamespace(max_length=768, score=Mock(side_effect=lambda q, texts, **kw: [1.]*len(texts)))
        second = SimpleNamespace(local=SimpleNamespace(tokenizer=CharacterTokenizer()),
            template=TEMPLATES['qwen3-reranker'], score_bundles=Mock(side_effect=lambda bundles, **kw: [.6]*len(bundles)))
        def candidates(query, channels):
            channels.update(deepcopy(raw['channels']))
            return raw['retrieved'], {}
        model = SimpleNamespace(candidates=candidates, rerankers=[first, second])
        class Evidence:
            trace = {'documents': {'stale-query-document': {}}}
            def encode(self, query, docs):
                self.seen = docs
                self.trace.setdefault('documents', {}).update({d: deepcopy(raw['evidence']['documents'][d]) for d in docs})
        evidence = Evidence()
        result = deep.score_raw(SimpleNamespace(batch=4), model, evidence, None, 'q',
            {'question': old['question'], 'answer': old['gold']}, old, original, weights, set(raw['docs']))
        self.assertEqual(evidence.seen, list('gh'))
        self.assertNotIn('stale-query-document', result['evidence']['documents'])
        self.assertEqual(len(second.score_bundles.call_args.args[0]), len(raw['docs']))
        deep.validate_raw([result], [('q', {'question': old['question'], 'answer': old['gold']})],
            {'q': old}, set(raw['docs']), weights, {'q': original})

    def test_only_new_edges_are_scored_and_match_full_tournament(self):
        import torch
        class Tokenizer:
            def pad(self, values, **kwargs):
                return {'input_ids': torch.tensor(values['input_ids'])}
        class Model:
            calls = 0
            def __call__(self, input_ids, **kwargs):
                self.calls += len(input_ids)
                diff = input_ids[:, 0]-input_ids[:, 1]+.7
                return SimpleNamespace(logits=torch.stack((diff, torch.zeros_like(diff)), dim=1)[:, None, :])
        local = SimpleNamespace(torch=torch, model=Model(), device='cpu')
        codec = SimpleNamespace(tokenizer=Tokenizer(), labels=[0, 1], pair=lambda q, a, b: [a, b])
        docs = [str(i) for i in range(8)]
        evidence = {d: int(d) for d in docs}
        old = {'docs': docs[:6], 'probabilities': duo.compare_documents(local, codec, 'q', evidence, docs[:6], 4).tolist()}
        local.model.calls = 0
        expanded = deep.expand_probabilities(local, codec, 'q', evidence, old, docs, 4)
        self.assertEqual(local.model.calls, 26)
        expected = duo.compare_documents(local, codec, 'q', evidence, docs, 4)
        np.testing.assert_allclose(expanded, expected, atol=1e-7)
        np.testing.assert_array_equal(expanded[:6, :6], old['probabilities'])

    def test_comparison_cache_retains_baseline_and_fitted_identity(self):
        row, raw, original, model = compared_fixture()
        deep.validate_comparisons([row], [raw], {'q': original}, model)
        changed = deepcopy(row)
        changed['probabilities'][0][1] = .8
        with self.assertRaisesRegex(ValueError, 'Original'):
            deep.validate_comparisons([changed], [raw], {'q': original}, model)
        changed = deepcopy(row)
        changed['ability'][0] += .01
        with self.assertRaises(ValueError):
            deep.validate_comparisons([changed], [raw], {'q': original}, model)

    def test_fit_can_admit_candidates_missing_from_v9(self):
        samples = []
        for i in range(8):
            matrix = np.zeros((6, len(fusion.FEATURES)))
            matrix[5, 0] = 1.
            matrix[:5, -1] = 1.
            samples.append({'id': str(i), 'docs': list('abcdef'), 'gold': ['f'], 'features': matrix.tolist()})
        with patch('builtins.print'):
            model = fusion.train(samples)
        self.assertEqual(int(np.argmax(fusion.scores(samples[0], model))), 5)

    def test_no_development_gain_skips_gpu_validation_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory))
            with patch.object(deep, 'choice', return_value={'selected': 'baseline'}), \
                    patch.object(deep, 'manifest', return_value={}), patch.object(deep, 'compared') as scores:
                deep.validate(args)
                report = deep.deployment(args)
                self.assertFalse(report['deploy'])
                self.assertFalse(report['evaluated'])
                scores.assert_not_called()
            with patch('sys.argv', ['deep', 'validation', '--out', directory]), \
                    patch.object(deep, 'choice', return_value={'selected': 'baseline'}), \
                    patch.object(deep, 'run_split') as run, patch.object(deep.prior, 'collect_base') as collect:
                deep.main()
                run.assert_not_called()
                collect.assert_not_called()
            with patch('sys.argv', ['deep', 'private', '--out', directory]), \
                    patch.object(deep, 'deployment', return_value={'deploy': False}), patch.object(deep, 'run_split') as run:
                deep.main()
                run.assert_not_called()

    def test_validation_regression_rejected_and_ties_keep_baseline(self):
        row, _, _, _ = compared_fixture()
        old, new = set(row['baseline']), set(fusion.predict(row, 'deep'))
        self.assertNotEqual(old, new)
        row['gold'] = [next(iter(old-new))]
        self.assertFalse(deep.decision([row])['deploy'])
        row['gold'] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory))
            with patch.object(deep, 'manifest', return_value={}), \
                    patch.object(deep, 'fitted', return_value={'training': digest([])}), \
                    patch.object(deep, 'training_rows', return_value=[]), patch.object(deep, 'compared', return_value=[row]), \
                    patch.object(deep, 'coverage', return_value={}):
                deep.calibrate(args)
                self.assertEqual(deep.choice(args)['selected'], 'baseline')
                record = read_json(args.out/'calibration.json')
                record['trials'][1]['recall'] = 2.
                record['selected'] = 'deep'
                write_json(args.out/'calibration.json', record)
                with self.assertRaises(ValueError):
                    deep.choice(args)

    def test_coverage_reports_where_gold_is_dropped(self):
        row, raw, original, _ = compared_fixture()
        row['gold'] = raw['gold'] = original['gold'] = ['g']
        with patch.object(deep, 'raw_parts', return_value=[(Path('part.json'), [raw])]), \
                patch.object(deep, 'base_rows', return_value=[original]):
            report = deep.coverage(SimpleNamespace(), 'development', [row])
        self.assertEqual(report['original_duo_pool'], 0.)
        self.assertEqual(report['expanded_duo_pool'], 1.)
        self.assertEqual(report['scored_pool'], 1.)

    def test_submit_packages_fallback_and_valid_candidate(self):
        row, _, _, model = compared_fixture()
        for deploy in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = SimpleNamespace(out=root/'out', source=root, questions=root/'questions.json', duo_run=root/'model')
                write_json(args.questions, {'q': {'question': row['question'], 'answer': None}})
                write_json(args.duo_run/'inputs.json', {'known_docs': list('abcdefgh')})
                write_json(args.out/'audit.json', {'components': {}, 'total': 23})
                baseline = {'q': {'answer': row['baseline']}}
                for name in ('v9', 'duo'):
                    with zipfile.ZipFile(root/f'submission-private-{name}.zip', 'w') as archive:
                        archive.writestr('submission.json', json.dumps(baseline))
                marker = {'assets': {'strength': .25}, 'source_files': {p.name: file_hash(p) for p in root.glob('*.zip')}}
                with patch.object(deep, 'deployment', return_value={'deploy': deploy}), \
                        patch.object(deep.prior, 'manifest', return_value=marker), \
                        patch.object(deep.prior, 'SCORED_ZIP', file_hash(root/'submission-private-duo.zip')), \
                        patch.object(deep, 'compared', return_value=[row]) as scoring, patch.object(deep, 'fitted', return_value=model):
                    deep.submit(args)
                    if not deploy:
                        scoring.assert_not_called()
                for name in ('v9', 'duo'):
                    with zipfile.ZipFile(args.out/f'submission-private-{name}.zip') as archive:
                        self.assertEqual(archive.namelist(), ['submission.json'])
                        expected = {'q': {'answer': fusion.predict(row, 'deep')}} if name == 'duo' and deploy else baseline
                        self.assertEqual(json.loads(archive.read('submission.json')), expected)


if __name__ == '__main__':
    unittest.main()
