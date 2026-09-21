import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

SPEC = importlib.util.spec_from_file_location('vast_stop_after', Path(__file__).resolve().parents[1]/'scripts/gpu/vast_stop_after.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class TestStopAfter(unittest.TestCase):
    def test_requires_success_and_both_complete_zips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'outputs/private/run'
            out.mkdir(parents=True)
            (root/'data').mkdir()
            (root/'data/private-official.json').write_text(json.dumps({'q': {}}))
            for model in ('v9', 'duo'):
                with zipfile.ZipFile(out/f'submission-private-{model}.zip', 'w') as archive:
                    archive.writestr('submission.json', json.dumps({'q': {'answer': list('abcde')}}))
            for stage in ('duo', 'completed', 'exited status=1; final backup', 'exited status=1; final upload failed'):
                (out/'.stage').write_text(stage)
                with self.assertRaises(RuntimeError):
                    helper.ready(root, 'run')
            (out/'.stage').write_text('exited status=0; final backup')
            self.assertEqual(helper.ready(root, 'run'), out)
            with zipfile.ZipFile(out/'submission-private-duo.zip', 'w') as archive:
                archive.writestr('submission.json', '{}')
            with self.assertRaises(RuntimeError):
                helper.ready(root, 'run')

    def test_backup_is_read_only_and_scoped_to_dsc(self):
        with patch.object(helper.subprocess, 'run') as run:
            helper.verify_backup(Path('/work/outputs/private/run'), 'run')
        command = run.call_args.args[0]
        self.assertIn('check', command)
        self.assertIn('gdrive:DSC-2026-results/runs/private/run/artifacts', command)
        self.assertNotIn('delete', command)
        self.assertTrue(run.call_args.kwargs['check'])

    def test_api_stop_requires_explicit_success_and_never_deletes(self):
        with patch.object(helper.urllib.request, 'urlopen') as open_url:
            open_url.return_value.__enter__.return_value.read.return_value = b'{"success":false}'
            with self.assertRaises(RuntimeError):
                helper.api('123', 'fake', stop=True)
            request = open_url.call_args.args[0]
            self.assertEqual(request.method, 'PUT')
            self.assertEqual(json.loads(request.data), {'state': 'stopped'})
            self.assertEqual(request.full_url, 'https://console.vast.ai/api/v0/instances/123/')
            open_url.return_value.__enter__.return_value.read.return_value = b'{"success":true}'
            self.assertTrue(helper.api('123', 'fake', stop=True)['success'])

    def test_wrong_instance_identity_is_rejected(self):
        with patch.object(helper.urllib.request, 'urlopen') as open_url:
            open_url.return_value.__enter__.return_value.read.return_value = b'{"instances":{"id":456}}'
            with self.assertRaises(RuntimeError):
                helper.api('123', 'fake')

    def test_waits_for_lock_then_backup_before_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root/'config.json'
            config.write_text(json.dumps({'root': str(root), 'run': 'run', 'instance': '123', 'key': 'fake'}))
            for fail in (False, True):
                events = []
                lock = MagicMock()
                lock.flock.side_effect = [BlockingIOError(), None]
                def backup(*args):
                    events.append('backup')
                    if fail:
                        raise RuntimeError('backup failed')
                with patch.dict('sys.modules', {'fcntl': lock}), \
                        patch.object(helper.time, 'sleep') as sleep, \
                        patch.object(helper, 'ready', return_value=root), \
                        patch.object(helper, 'verify_backup', side_effect=backup), \
                        patch.object(helper, 'api', side_effect=lambda *a, **kw: events.append('stop')) as api:
                    if fail:
                        with self.assertRaises(RuntimeError):
                            helper.watch(config)
                        api.assert_not_called()
                    else:
                        helper.watch(config)
                        api.assert_called_once_with('123', 'fake', stop=True)
                    sleep.assert_called_once_with(30)
                    self.assertEqual(events, ['backup'] if fail else ['backup', 'stop'])


if __name__ == '__main__':
    unittest.main()
