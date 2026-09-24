"""Passage-order watcher requires successful complete results and cloud backup."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location('vast_stop_passage',
    Path(__file__).resolve().parents[1]/'scripts/gpu/vast_stop_passage.py')
helper = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(helper)


class TestPassageStop(unittest.TestCase):
    def fixture(self, root):
        out = root/'outputs/passage-order/run'; out.mkdir(parents=True)
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding='utf8')
        plan = {'questions': 429, 'models': {'original-duo': {}, 'retrained-control': {}},
                'replay_tolerance': .0001}
        identity = helper.digest(plan); files = {}; scores = {}
        put(out/'plan.json', plan)
        for i in range(429):
            name = f'questions/{i:05d}.json'; question = {'id': str(i)}
            put(out/name, question); files[name] = helper.file_hash(out/name)
            for arm in plan['models']:
                score_name = f'scores/{arm}/{i:05d}.json'
                put(out/score_name, {'source': helper.digest(question)})
                scores[f'{arm}/{i:05d}.json'] = helper.file_hash(out/score_name)
        for arm in plan['models']:
            marker = {'plan': identity, 'model': helper.digest({})}
            put(out/'scores'/arm/'resume.json', marker)
            put(out/'scores'/arm/'replay.json', dict(marker,
                maximum_probability_error=0., baseline_predictions_identical=True))
        put(out/'prepared.json', {'plan': identity, 'files': files})
        put(out/'evaluation.json', {'plan': identity, 'development': {'questions': 429},
                                   'scores': helper.digest(scores)})
        (out/'.stage').write_text('exited status=0; final backup')
        return out, identity

    def test_complete_evaluation_required_and_failure_never_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, identity = self.fixture(root)
            with patch.object(helper, 'EXPECTED_PLAN', identity):
                self.assertEqual(helper.ready(root, 'run'), out)
                for stage in ('score', 'completed', 'exited status=1; final backup',
                              'exited status=1; final upload failed'):
                    (out/'.stage').write_text(stage)
                    with self.assertRaises(RuntimeError):
                        helper.ready(root, 'run')

    def test_changed_or_missing_results_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out, identity = self.fixture(root)
            with patch.object(helper, 'EXPECTED_PLAN', identity):
                target = out/'scores/original-duo/00428.json'
                target.write_text('{"source":"wrong"}')
                with self.assertRaises(RuntimeError):
                    helper.ready(root, 'run')
                target.unlink()
                with self.assertRaises(FileNotFoundError):
                    helper.ready(root, 'run')

    def test_backup_before_stop_and_backup_failure_prevents_api(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); out = root/'outputs/passage-order/run'; out.mkdir(parents=True)
                (out/'.stage').write_text('exited status=0; final backup')
                config = root/'settings.json'
                config.write_text(json.dumps({'root': str(root), 'run': 'run',
                    'instance': '52083026', 'key': 'fake'}))
                events = []
                def backup(*args):
                    events.append('backup')
                    if fail:
                        raise RuntimeError('backup failed')
                with patch.dict('sys.modules', {'fcntl': MagicMock()}), \
                        patch.object(helper, 'ready', return_value=out), \
                        patch.object(helper, 'verify_backup', side_effect=backup), \
                        patch.object(helper, 'api', side_effect=lambda *a, **k: events.append('stop')) as api:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'backup failed'):
                            helper.watch(config)
                        api.assert_not_called()
                    else:
                        helper.watch(config)
                        api.assert_called_once_with('52083026', 'fake', stop=True)
                    self.assertEqual(events, ['backup'] + ([] if fail else ['stop']))

    def test_backup_uses_only_passage_order_remote_and_hash_check(self):
        with patch.object(helper.subprocess, 'run') as run:
            helper.verify_backup(Path('/workspace/out'), 'run')
        command = run.call_args.args[0]
        self.assertIn('gdrive:DSC-2026-results/runs/passage-order/run/artifacts', command)
        self.assertIn('check', command)
        self.assertNotIn('--size-only', command)
        self.assertTrue(run.call_args.kwargs['check'])

    def test_explicit_instance_conflict_fails_before_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root/'scripts/gpu/gpu-passage-order-4090.sh'
            script.parent.mkdir(parents=True); script.write_text('fixture')
            with patch.object(helper.sys, 'platform', 'linux'), \
                    patch.object(helper.sys, 'argv', ['watcher', '--root', str(root), '--instance', '52083026']), \
                    patch.dict(helper.os.environ, {'CONTAINER_ID': '123', 'VAST_CONTAINERLABEL': ''}), \
                    patch.object(helper, 'api') as api:
                with self.assertRaisesRegex(RuntimeError, 'differs from this GPU terminal'):
                    helper.main()
                api.assert_not_called()


if __name__ == '__main__':
    unittest.main()
