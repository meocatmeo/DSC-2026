"""Private submission uses the saved control model without evaluating labeled splits."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('retrain_private_submit', ROOT/'scripts/gpu/retrain_private_submit.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class TestPrivateSubmit(unittest.TestCase):
    def test_preparation_preserves_source_and_copies_only_saved_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root/'source', root/'candidate'
            marker, model = {'schema': 'example'}, {'model': 'verified'}
            arm = source/'arms/control'
            files = {'inputs.json': marker, 'reference/inputs.json': {},
                     'arms/control/inputs.json': {}, 'arms/control/model/duo.json': model,
                     'gate.json': {'inputs': tool.digest(marker), 'deploy': False}}
            for name, value in files.items():
                tool.write_json(source/name, value)
            before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            with patch.object(tool.retrain, 'manifest', return_value=marker), \
                    patch.object(tool.retrain, 'arm_inputs', return_value=(arm, {})), \
                    patch.object(tool.retrain.duo, 'verify_model', return_value=model), \
                    patch.object(tool.retrain, 'screen', side_effect=AssertionError('No screen')), \
                    patch.object(tool.retrain, 'gate', side_effect=AssertionError('No gate')):
                tool.prepare(source, out)
                tool.prepare(source, out)
            self.assertTrue((out/'arms/control/model/duo.json').is_file())
            self.assertFalse((out/'arms/old-evidence-control').exists())
            self.assertFalse(tool.read_json(out/'request.json')['validation_requested'])
            for name, data in before.items():
                self.assertEqual((source/name).read_bytes(), data)
            with self.assertRaisesRegex(ValueError, 'separate output'):
                tool.prepare(source, source)

    def test_submit_uses_private_cache_and_labels_override_honestly(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            args = tool.options(out)
            args.duo_run = out/'duo'
            marker, model = {'schema': 'example'}, {'checkpoint': 'verified'}
            request = {'inputs': tool.digest(marker), 'model': tool.digest(model),
                       'tool_sha256': tool.file_hash(Path(tool.__file__)), 'user_requested_submission': True}
            tool.write_json(out/'request.json', request)
            tool.write_json(args.duo_run/'inputs.json', {'known_docs': list('abcdef')})
            def pack(options, predictions, *rest):
                self.assertEqual(predictions, {'q': {'answer': list('abcde')}})
                (options.out/'submission-private-duo.zip').write_bytes(b'packed by tested submission writer')
            with patch.object(tool.retrain, 'manifest', return_value=marker), \
                    patch.object(tool.retrain, 'arm_inputs', return_value=(out/'model', {})), \
                    patch.object(tool.retrain.duo, 'verify_model', return_value=model), \
                    patch.object(tool.retrain, 'collect', return_value=[{'id': 'q'}]) as collect, \
                    patch.object(tool.retrain.prior, 'predict', return_value=list('abcde')), \
                    patch.object(tool.retrain.private, 'write_submission', side_effect=pack), \
                    patch.object(tool.retrain, 'gate', side_effect=AssertionError('No gate')), \
                    patch.object(tool.retrain, 'screen', side_effect=AssertionError('No screen')):
                tool.submit(args)
            collect.assert_called_once_with(args, 'private', 'control', compute=False)
            choice = tool.read_json(out/'submission-choice.json')
            self.assertTrue(choice['candidate_deployed'])
            self.assertFalse(choice['validation_evaluated'])
            self.assertFalse(choice['promotion_gate_passed'])
            self.assertEqual(choice['gate']['decision'], 'user-requested-unvalidated')


if __name__ == '__main__':
    unittest.main()
