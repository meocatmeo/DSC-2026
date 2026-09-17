"""Run the real collect/resume/finish flow with deterministic CPU model doubles."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from legalir2.v11 import collect, load_cache


class CollectTests(unittest.TestCase):
    def test_interruption_resume_completed_skip_and_changed_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(baseline=root / 'baseline', output=root / 'out',
                                   store=root / 'store', dense=root / 'dense', lexical=root / 'lexical',
                                   embedding=root / 'embedding', first=root / 'first', qwen=root / 'qwen',
                                   questions=root / 'questions.json', validation=root / 'validation.json',
                                   stage='collect', split='train', limit=None, device='cpu', bundle_batch=1)
            settings = {'candidates': 5, 'stage2_candidates': 5, 'reranker_weights': [1, 1]}
            for directory in [args.baseline, args.dense, args.embedding, args.first, args.qwen]:
                directory.mkdir()
                (directory / 'config.json').write_text('{}')
            (args.dense / 'manifest.json').write_text('{}')
            args.store.write_text('fixture')
            (args.baseline / 'v9-ensemble.run.json').write_text(json.dumps({'settings': settings, 'lexical': {}, 'store': {}}))
            (args.baseline / 'v9-stacker-ens.json').write_text('{}')
            args.questions.write_text(json.dumps({f'q{i}': {'question': f'question{i}', 'answer': ['a']} for i in range(2)}))
            args.validation.write_text('{}')
            docs = list('abcde')
            calls, closes = [], []
            fail = [True]

            def candidates(query, channels):
                calls.append(query)
                if query == 'question1' and fail[0]:
                    raise RuntimeError('simulated GPU failure')
                return [(d, 1.0) for d in docs], {}

            local = SimpleNamespace(torch=None, tokenizer=None, identity={})
            model = SimpleNamespace(local=local, head='causal', context_limit=4096,
                                    template={}, score_bundles=lambda b, n: [0.1] * len(b))
            pipeline = SimpleNamespace(rerankers=[model, model], candidates=candidates,
                                       _rerank=lambda *a: {d: [0.1] for d in docs}, store=None,
                                       stacker=None, close=lambda: closes.append(True),
                                       manifest=lambda: {'settings': settings, 'lexical': {}, 'store': {}})
            with patch('legalir2.pipeline.Pipeline', return_value=pipeline) as constructor, \
                 patch('legalir2.pipeline.final_order', return_value=docs), \
                 patch('legalir2.v11.bundle_from_store', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'simulated GPU failure'):
                    collect(args)
                fail[0] = False
                collect(args)
                self.assertEqual(calls, ['question0', 'question1', 'question1'])
                self.assertEqual([r['id'] for r in load_cache(args.output / 'train.jsonl.gz')], ['q0', 'q1'])
                self.assertEqual(len(closes), 2)
                self.assertFalse((args.output / 'train.rows').exists())
                count = constructor.call_count
                collect(args)
                self.assertEqual(constructor.call_count, count)  # no GPU load for finished split
                args.limit = 1
                with self.assertRaisesRegex(ValueError, 'Completed cache inputs'):
                    collect(args)


if __name__ == '__main__':
    unittest.main()
