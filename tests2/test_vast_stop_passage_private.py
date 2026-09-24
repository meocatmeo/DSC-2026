import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

spec = importlib.util.spec_from_file_location('stop_private', Path(__file__).resolve().parents[1]/'scripts/gpu/vast_stop_passage_private.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)


class TestPrivateStop(unittest.TestCase):
    def test_failed_or_unfinished_run_never_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root/'outputs/passage-order/run'; out.mkdir(parents=True)
            for state in ('score', 'completed', 'exited status=1; final backup'):
                (out/'.stage').write_text(state)
                with self.assertRaises(RuntimeError): helper.ready(root, 'run')

    def test_backup_failure_prevents_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'outputs/passage-order/run';out.mkdir(parents=True)
            (out/'.stage').write_text('exited status=0; final backup')
            config=root/'config.json';config.write_text(json.dumps({'root':str(root),'run':'run','instance':'52083026','key':'fake'}))
            with patch.dict('sys.modules', {'fcntl': MagicMock()}), patch.object(helper,'ready',return_value=out), \
                    patch.object(helper,'verify_backup',side_effect=RuntimeError('backup failed')), patch.object(helper,'api') as api:
                with self.assertRaisesRegex(RuntimeError,'backup failed'): helper.watch(config)
                api.assert_not_called()

    def test_requires_private_submission_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'outputs/passage-order/run';out.mkdir(parents=True)
            (out/'.stage').write_text('exited status=0; final backup')
            plan={'questions':2080};identity=helper.digest(plan)
            (out/'plan.json').write_text(json.dumps(plan))
            (out/'prepared.json').write_text(json.dumps({'plan':identity,'files':{str(i):'x' for i in range(2080)}}))
            with patch.object(helper,'EXPECTED_PLAN',identity):
                with self.assertRaises(FileNotFoundError): helper.ready(root,'run')


if __name__ == '__main__': unittest.main()
