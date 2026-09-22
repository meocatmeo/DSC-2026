"""A stale tmux shell must not hide startup failure or prevent a valid resume."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

SPEC = importlib.util.spec_from_file_location('launch_retrain', Path(__file__).resolve().parents[1]/'scripts/gpu/launch_retrain.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class TestRetrainLauncher(unittest.TestCase):
    def test_live_process_detection_excludes_idle_shells_and_monitor(self):
        self.assertTrue(launcher.is_job(['bash', 'scripts/gpu/gpu-fulltext-retrain-4090.sh', 'all']))
        self.assertTrue(launcher.is_job(['python3', '/work/launch_retrain.py', '--job']))
        self.assertTrue(launcher.is_job(['python3', '-m', 'legalir2.fulltext_retrain', 'train']))
        self.assertFalse(launcher.is_job(['bash']))
        self.assertFalse(launcher.is_job(['python3', '/work/launch_retrain.py']))
        self.assertFalse(launcher.is_job(['python3', '/work/vast_stop_retrain.py', '--watch', '/secret/config.json']))

    def test_restore_fills_missing_files_but_never_overwrites_modified_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root/'fulltext-retrain.zip'
            data = b'print("frozen")\n'
            manifest = hashlib.sha256(data).hexdigest()+'  legalir2/example.py\n'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('legalir2/example.py', data)
                bundle.writestr('private-files.sha256', manifest)
            with patch.object(launcher, 'ZIP_SHA', launcher.sha(archive)), patch.object(launcher.subprocess, 'run') as run:
                launcher.restore(root)
                self.assertEqual((root/'legalir2/example.py').read_bytes(), data)
                (root/'legalir2/example.py').write_text('user change')
                with self.assertRaisesRegex(RuntimeError, 'refusing to overwrite'):
                    launcher.restore(root)
                self.assertEqual((root/'legalir2/example.py').read_text(), 'user change')
                run.assert_not_called()

    def test_idle_existing_tmux_gets_new_window_after_arming_and_visible_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = SimpleNamespace(root=root, run='test-run')
            events = []
            def arm(*args):
                self.assertGreater(args[2], 0)
                events.append('arm')
            def tmux(command, **kwargs):
                events.append('start')
                self.assertEqual(command[:2], ['tmux', 'new-window'])
                self.assertNotIn('switch-client', command)
                return '%77\n'
            with patch.dict('sys.modules', {'fcntl': MagicMock()}), \
                    patch.object(launcher, 'running', return_value=[]), patch.object(launcher, 'restore'), \
                    patch.object(launcher, 'mode', return_value='run'), patch.object(launcher, 'preflight'), \
                    patch.object(launcher, 'arm', side_effect=arm), \
                    patch.object(launcher.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                    patch.object(launcher.subprocess, 'check_output', side_effect=tmux), \
                    patch.object(launcher, 'follow', return_value=0) as follow:
                self.assertEqual(launcher.launch(args), 0)
                self.assertEqual(events, ['arm', 'start'])
                self.assertTrue(follow.call_args.args[0].is_file())

    def test_existing_job_is_followed_without_restore_or_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            log = root/'run.log'
            log.write_text('training')
            with patch.dict('sys.modules', {'fcntl': MagicMock()}), \
                    patch.object(launcher, 'running', return_value=[123]), patch.object(launcher, 'restore') as restore, \
                    patch.object(launcher, 'arm') as arm, patch.object(launcher, 'latest_log', return_value=log), \
                    patch.object(launcher, 'follow', return_value=0), patch.object(launcher.subprocess, 'check_output') as tmux:
                launcher.launch(SimpleNamespace(root=root, run='test-run'))
                restore.assert_not_called()
                tmux.assert_not_called()
                arm.assert_called_once_with(root, 'test-run', 0)

    def test_failed_preflight_never_arms_or_starts_gpu_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict('sys.modules', {'fcntl': MagicMock()}), \
                    patch.object(launcher, 'running', return_value=[]), patch.object(launcher, 'restore'), \
                    patch.object(launcher, 'mode', return_value='all'), \
                    patch.object(launcher, 'preflight', side_effect=RuntimeError('Insufficient disk')), \
                    patch.object(launcher, 'arm') as arm, patch.object(launcher.subprocess, 'check_output') as tmux:
                with self.assertRaisesRegex(RuntimeError, 'Insufficient disk'):
                    launcher.launch(SimpleNamespace(root=root, run='test-run'))
                arm.assert_not_called()
                tmux.assert_not_called()

    def test_runner_failure_is_persisted_and_log_backup_attempted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            attempt = root/'attempt'
            with patch.object(launcher.subprocess, 'run', side_effect=[SimpleNamespace(returncode=19), SimpleNamespace(returncode=0)]) as run:
                code = launcher.job(SimpleNamespace(root=root, attempt=attempt, run='test-run', mode='run'))
                self.assertEqual(code, 19)
                self.assertEqual(json.loads((attempt/'status.json').read_text())['exit_code'], 19)
                self.assertIn('JOB EXIT STATUS: 19', (attempt/'launcher.log').read_text())
                self.assertEqual(run.call_args_list[-1].args[0][:2], ['rclone', 'copyto'])

    def test_private_job_invokes_only_the_private_submission_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(launcher.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
                code = launcher.job(SimpleNamespace(root=root, attempt=root/'attempt',
                    run='duo-fulltext-retrain-private-control-4090-v1', mode='private-submit'))
            self.assertEqual(code, 0)
            command = run.call_args_list[0].args[0]
            self.assertEqual(Path(command[1]).name, 'gpu-retrain-private-submit.sh')
            self.assertEqual(len(command), 2)

    def test_fresh_private_job_invokes_restore_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(launcher.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
                code = launcher.job(SimpleNamespace(root=root, attempt=root/'attempt',
                    run='duo-fulltext-retrain-private-control-4090-v1', mode='fresh-private'))
            self.assertEqual(code, 0)
            self.assertEqual(Path(run.call_args_list[0].args[0][1]).name, 'gpu-retrain-private-fresh.sh')


if __name__ == '__main__':
    unittest.main()
