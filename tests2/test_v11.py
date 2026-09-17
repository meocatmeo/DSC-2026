import unittest
import tempfile
from pathlib import Path
import gzip
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from legalir2.v11 import matrix, normalized, checkpoint_row, prepare_resume, load_cache


class V11Tests(unittest.TestCase):
    def test_public_submission_uses_treatment_and_correct_zip_layout(self):
        from legalir2.v11 import pack_public
        from legalir2.stack import feature_names
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = dict(schema='schema', settings={}, lexical={}, store={}, model_identities=[])
            fitted = {'training_manifest': metadata, 'treatment': {
                'features': list(feature_names(2)) + ['qwen_bundle_score_v1'],
                'weights': [0]*20 + [1]}}
            (root / 'fusion.json').write_text(json.dumps(fitted))
            (root / 'public.manifest.json').write_text(json.dumps(metadata))
            public = root / 'questions.json'
            public.write_text(json.dumps({'q': {'question': 'public question'}}))
            row = {'id': 'q', 'question': 'public question', 'docs': ['a','b','c','d','e'],
                   'features': [[0]*20]*5, 'bundle_scores': [0,1,2,3,4]}
            with gzip.open(root / 'public.jsonl.gz', 'wt') as stream:
                stream.write(json.dumps(row) + '\n')
            store = SimpleNamespace(doc_ids=row['docs'], close=lambda: None)
            args = SimpleNamespace(output=root, public=public, store=root / 'unused')
            with patch('legalir2.store.Store', return_value=store):
                pack_public(args)
            with zipfile.ZipFile(root / 'submission-v11.zip') as archive:
                self.assertEqual(archive.namelist(), ['submission.json'])
                self.assertEqual(json.loads(archive.read('submission.json'))['q']['answer'], ['e','d','c','b','a'])
            public.write_text(json.dumps({'q': {'question': 'public question'}, 'missing': {'question': 'other'}}))
            with self.assertRaisesRegex(ValueError, 'cover all'):
                pack_public(args)

    def test_resume_keeps_committed_row_and_ignores_killed_write(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            items = [('1', {'question': 'q1', 'answer': ['a']}), ('2', {'question': 'q2', 'answer': ['b']})]
            signature = {'model': 'frozen', 'limit': 2}
            self.assertEqual(prepare_resume(directory, signature, items), {})
            checkpoint_row(directory, {'id': '1', 'question': 'q1', 'gold': ['a'], 'seconds': 3})
            (directory / 'killed.json.gz.partial').write_bytes(b'broken')
            self.assertEqual(set(prepare_resume(directory, signature, items)), {'1'})
            checkpoint_row(directory, {'id': '2', 'question': 'q2', 'gold': ['b'], 'seconds': 4})
            done = prepare_resume(directory, signature, items)
            joined = directory / 'complete.gz'
            joined.write_bytes(b''.join(done[k][0].read_bytes() for k,_ in items))
            self.assertEqual([r['id'] for r in load_cache(joined)], ['1', '2'])
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                prepare_resume(directory, {'model': 'changed'}, items)

    def test_resume_rejects_wrong_question(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            items = [('1', {'question': 'q', 'answer': ['a']})]
            prepare_resume(directory, {}, items)
            checkpoint_row(directory, {'id': '1', 'question': 'different', 'gold': ['a'], 'seconds': 1})
            with self.assertRaisesRegex(ValueError, 'Invalid checkpoint'):
                prepare_resume(directory, {}, items)

    def test_bundle_feature_does_not_replace_v9_features(self):
        old = np.arange(60).reshape(3, 20).tolist()
        row = {'features': old, 'bundle_scores': [0.1, 0.3, 0.2]}
        treatment = matrix(row, True)
        np.testing.assert_array_equal(matrix(row, False), old)
        np.testing.assert_array_equal(treatment[:, :20], old)
        np.testing.assert_allclose(treatment[:, 20], [0, 1, 0.5])

    def test_constant_score_and_duplicate_question(self):
        result = matrix({'features': [[1]*20, [2]*20], 'bundle_scores': [0.5, 0.5]}, True)
        np.testing.assert_array_equal(result[:, -1], [0, 0])
        self.assertEqual(normalized(' Câu  hỏi? '), normalized('câu hỏi'))
