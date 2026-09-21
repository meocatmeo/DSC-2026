"""Verify nested sampling and reuse at the partial 256-question shard boundary."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from legalir2 import deep_pilot512 as extension
from legalir2.deep_pilot import training_subset
from legalir2.store import write_json, read_json


class TestExtension(unittest.TestCase):
    def test_nested_sampling(self):
        records = {str(i): {'question': f'Question {i}', 'answer': ['a']} for i in range(800)}
        self.assertEqual(list(training_subset(records, 512))[:256], list(training_subset(records, 256)))

    def test_exactly_256_new_forwards_and_zero_old_forwards_including_partial_shard(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            args = SimpleNamespace(out=root/'new', pilot_out=root/'pilot', duo_run=root/'model',
                deep_assets=root/'assets', batch=4, index=root, evidence_index=root/'evidence')
            questions = {str(i): {'question': f'Question {i}', 'answer': ['a']} for i in range(512)}
            pilot_questions = dict(list(questions.items())[:256])
            cached = [{'id': q} for q in pilot_questions]
            parts = [(Path(f'part-{i//5:05d}.json'), cached[i:i+5]) for i in range(0, 256, 5)]
            write_json(args.duo_run/'inputs.json', {'weights': [], 'known_docs': [], 'memory': {}})
            stack.enter_context(patch.object(extension.deep, 'records', side_effect=lambda a, s: questions if a.out == args.out else pilot_questions))
            stack.enter_context(patch.object(extension.deep, 'manifest', return_value={}))
            stack.enter_context(patch.object(extension.deep, 'v9_rows', return_value=[{'id': q} for q in questions]))
            stack.enter_context(patch.object(extension.deep, 'raw_parts', return_value=parts))
            stack.enter_context(patch.object(extension.deep, 'validate_raw'))
            stack.enter_context(patch.object(extension.deep, 'require_bf16'))
            stack.enter_context(patch.object(extension.deep, 'Evidence'))
            stack.enter_context(patch.object(extension.deep, 'QuestionMemory'))
            stack.enter_context(patch.object(extension.deep.duo, 'PairTokenizer'))
            stack.enter_context(patch.object(extension.deep.private, 'progress'))
            model = Mock()
            model.rerankers = [Mock(), Mock()]
            loader = stack.enter_context(patch.object(extension.deep, 'pipeline', return_value=model))
            forward = stack.enter_context(patch.object(extension.deep, 'score_raw', side_effect=lambda a, m, e, mem, q, *rest: {'id': q}))
            extension.training(args)
            self.assertEqual(forward.call_count, 256)
            self.assertEqual([c.args[4] for c in forward.call_args_list], list(questions)[256:])
            mixed = read_json(args.out/'cache/deep/train/part-00051.json')
            self.assertEqual([r['id'] for r in mixed], ['255', '256', '257', '258', '259'])
            rows = [r for p in sorted((args.out/'cache/deep/train').glob('part-*.json')) for r in read_json(p)]
            self.assertEqual([r['id'] for r in rows], list(questions))
            forward.reset_mock()
            loader.reset_mock()
            extension.training(args)
            forward.assert_not_called()
            loader.assert_not_called()
            write_json(args.out/'cache/deep/train/part-00000.json', [{'id': 'tampered'}])
            with self.assertRaises(ValueError):
                extension.training(args)

    def test_development_raw_import_needs_no_gpu(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            args = SimpleNamespace(out=root/'new', pilot_out=root/'old', duo_run=root/'model')
            questions = {str(i): {'question': str(i), 'answer': ['a']} for i in range(429)}
            rows = [{'id': q} for q in questions]
            parts = [(Path(f'part-{i//5:05d}.json'), rows[i:i+5]) for i in range(0, 429, 5)]
            write_json(args.duo_run/'inputs.json', {'weights': [], 'known_docs': []})
            for name, value in [('records', questions), ('manifest', {}), ('v9_rows', rows), ('base_rows', rows), ('raw_parts', parts)]:
                stack.enter_context(patch.object(extension.deep, name, return_value=value))
            validate = stack.enter_context(patch.object(extension.deep, 'validate_raw'))
            gpu = stack.enter_context(patch.object(extension.deep, 'pipeline'))
            self.assertEqual(extension.import_raw(args, 'development'), {})
            self.assertEqual(validate.call_count, 86)
            gpu.assert_not_called()
            self.assertEqual(sum(len(read_json(p)) for p in (args.out/'cache/deep/development').glob('part-*.json')), 429)


if __name__ == '__main__':
    unittest.main()
