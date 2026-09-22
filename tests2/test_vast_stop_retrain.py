"""Retraining auto-stop cannot fire on a failed run, corrupt ZIP or missing backup."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

SPEC = importlib.util.spec_from_file_location('vast_stop_retrain', Path(__file__).resolve().parents[1]/'scripts/gpu/vast_stop_retrain.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class TestRetrainStop(unittest.TestCase):
    def fixture(self, root, deploy=False):
        out = root/'outputs/retrain/run'
        out.mkdir(parents=True)
        def put(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding='utf-8')
        put(root/'data/private-official.json', {'q': {'answer': None}})
        put(root/'outputs/duo/duo-20260918T110105Z/inputs.json', {'known_docs': list('abcdef')})
        inputs = {'schema': 'duo-fulltext-retrain-v1'}
        gate = {'inputs': helper.digest(inputs), 'deploy': deploy}
        put(out/'inputs.json', inputs)
        put(out/'gate.json', gate)
        with zipfile.ZipFile(out/'submission-private-duo.zip', 'w') as archive:
            archive.writestr('submission.json', json.dumps({'q': {'answer': list('abcde')}}))
        put(out/'submission-choice.json', {'gate': gate, 'candidate_deployed': deploy,
            'zip_sha256': helper.file_hash(out/'submission-private-duo.zip')})
        (out/'.stage').write_text('exited status=0; final backup')
        return out

    def test_single_duo_zip_and_both_deployment_outcomes(self):
        for deploy in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                out = self.fixture(root, deploy)
                self.assertEqual(helper.ready(root, 'run'), out)
                for stage in ('startup', 'completed', 'exited status=1; final backup', 'exited status=1; final upload failed'):
                    (out/'.stage').write_text(stage)
                    with self.assertRaises(RuntimeError):
                        helper.ready(root, 'run')

    def test_changed_zip_and_foreign_gate_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = self.fixture(root)
            (out/'gate.json').write_text(json.dumps({'inputs': 'wrong', 'deploy': False}))
            with self.assertRaisesRegex(RuntimeError, 'decision'):
                helper.ready(root, 'run')

    def test_explicit_unvalidated_candidate_can_stop_after_valid_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = self.fixture(root, deploy=True)
            gate = helper.read_json(out/'gate.json')
            gate.update(decision='user-requested-unvalidated', promotion_gate_passed=False, validation_evaluated=False)
            (out/'gate.json').write_text(json.dumps(gate))
            choice = helper.read_json(out/'submission-choice.json')
            choice.update(gate=gate, promotion_gate_passed=False, validation_evaluated=False)
            (out/'submission-choice.json').write_text(json.dumps(choice))
            self.assertEqual(helper.ready(root, 'run'), out)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = self.fixture(root)
            with (out/'submission-private-duo.zip').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'ZIP differs'):
                helper.ready(root, 'run')

    def test_backup_checks_only_retrain_namespace(self):
        with patch.object(helper.subprocess, 'run') as run:
            helper.verify_backup(Path('/work/outputs/retrain/run'), 'run')
        command = run.call_args.args[0]
        self.assertIn('gdrive:DSC-2026-results/runs/retrain/run/artifacts', command)
        self.assertIn('check', command)
        self.assertNotIn('--size-only', command)
        self.assertTrue(run.call_args.kwargs['check'])

    def test_waits_for_target_exit_and_lock_then_backup_before_stop(self):
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                out = self.fixture(root)
                (out/'.stage').write_text('train')
                config = root/'config.json'
                config.write_text(json.dumps({'root': str(root), 'run': 'run', 'instance': '123', 'key': 'fake'}))
                events = []
                locks = MagicMock()
                locks.flock.side_effect = [BlockingIOError(), None]
                def sleep(seconds):
                    events.append('wait')
                    (out/'.stage').write_text('exited status=0; final backup')
                def backup(*args):
                    events.append('backup')
                    if fail:
                        raise RuntimeError('backup failed')
                with patch.dict('sys.modules', {'fcntl': locks}), patch.object(helper.time, 'sleep', side_effect=sleep), \
                        patch.object(helper, 'verify_backup', side_effect=backup), \
                        patch.object(helper, 'api', side_effect=lambda *a, **kw: events.append('stop')) as api:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'backup failed'):
                            helper.watch(config)
                        api.assert_not_called()
                    else:
                        helper.watch(config)
                        api.assert_called_once_with('123', 'fake', stop=True)
                    self.assertEqual(events, ['wait', 'wait', 'backup'] + ([] if fail else ['stop']))

    def test_api_only_requests_stop_with_explicit_success(self):
        with patch.object(helper.urllib.request, 'urlopen') as open_url:
            open_url.return_value.__enter__.return_value.read.return_value = b'{"success":false}'
            with self.assertRaises(RuntimeError):
                helper.api('123', 'fake', stop=True)
            request = open_url.call_args.args[0]
            self.assertEqual(request.method, 'PUT')
            self.assertEqual(json.loads(request.data), {'state': 'stopped'})
            open_url.return_value.__enter__.return_value.read.return_value = b'{"instances":{"id":456}}'
            with self.assertRaises(RuntimeError):
                helper.api('123', 'fake')

    def test_old_success_marker_cannot_stop_a_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = self.fixture(root)
            cutoff = (out/'.stage').stat().st_mtime_ns+1000000000
            config = root/'config.json'
            config.write_text(json.dumps({'root': str(root), 'run': 'run', 'instance': '123',
                                          'key': 'fake', 'after_ns': cutoff}))
            with patch.dict('sys.modules', {'fcntl': MagicMock()}), \
                    patch.object(helper.time, 'sleep', side_effect=RuntimeError('still waiting')), \
                    patch.object(helper, 'api') as api, patch.object(helper, 'verify_backup') as backup:
                with self.assertRaisesRegex(RuntimeError, 'still waiting'):
                    helper.watch(config)
                api.assert_not_called()
                backup.assert_not_called()


if __name__ == '__main__':
    unittest.main()
