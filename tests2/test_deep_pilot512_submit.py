"""Private-only continuation, archive integrity, and launcher failure handling."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from legalir2 import deep_pilot512_submit as direct
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json, write_json
from tests2.test_deep_pool import compared_fixture


class TestDirect(unittest.TestCase):
    def test_import_partial_private_snapshot_and_preserve_valid_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root/'new', reuse_out=root/'reuse', duo_run=root/'duo')
            source = {'assets': {'training_questions': 256}, 'batch': 4}
            target = dict(source, assets={'training_questions': 512})
            questions = {str(i): {'question': str(i), 'answer': None} for i in range(10)}
            write_json(args.duo_run/'inputs.json', {'known_docs': ['a'], 'weights': [1.]})
            source_dir = args.reuse_out/'cache/deep/private'
            write_json(source_dir/'resume.json', {'inputs': digest(source), 'split': 'private'})
            rows = [{'id': str(i)} for i in range(5)]
            write_json(source_dir/'part-00000.json', rows)
            with patch.object(direct, 'REUSE_INPUTS', digest(source)), \
                    patch.object(direct.deep, 'manifest', side_effect=lambda a: source if a.out == args.reuse_out else target), \
                    patch.object(direct.deep, 'v9_rows', return_value=rows), \
                    patch.object(direct.deep, 'base_rows', return_value=rows), \
                    patch.object(direct.deep, 'records', return_value=questions), \
                    patch.object(direct.deep, 'validate_raw') as validate, \
                    patch.object(direct.deep, 'pipeline') as gpu:
                direct.import_private(args)
                self.assertEqual(read_json(args.out/'private-reuse.json')['newly_imported_questions'], 5)
                self.assertEqual(read_json(args.out/'cache/deep/private/part-00000.json'), rows)
                self.assertFalse((args.out/'cache/deep/private/part-00001.json').exists())
                validate.assert_called_once()
                gpu.assert_not_called()
                direct.import_private(args)
                self.assertEqual(read_json(args.out/'private-reuse.json')['newly_imported_questions'], 0)
                target['batch'] = 8
                with self.assertRaises(ValueError):
                    direct.import_private(args)
                target['batch'] = 4
                write_json(source_dir/'resume.json', {'inputs': 'tampered', 'split': 'private'})
                with self.assertRaises(ValueError):
                    direct.import_private(args)

    def test_identity_requires_exact_winning_pilot(self):
        selected = {'selected': 'deep', 'scores': direct.SCORES}
        with patch.object(direct.deep, 'choice', return_value=selected), \
                patch.object(direct.deep, 'fitted', return_value={}), \
                patch.object(direct.deep, 'manifest', return_value={}), \
                patch.object(direct, 'INPUTS', digest({})), patch.object(direct, 'FUSION', digest({})):
            self.assertFalse(direct.identity(None)['validation_evaluated'])
            selected['selected'] = 'baseline'
            with self.assertRaises(ValueError):
                direct.identity(None)
            selected.update(selected='deep', scores='changed')
            with self.assertRaises(ValueError):
                direct.identity(None)

    def test_submit_new_answers_without_validation_and_reject_missing_rows(self):
        row, _, _, _ = compared_fixture()
        row['gold'] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root/'out', source=root, questions=root/'questions.json', duo_run=root/'model')
            write_json(args.questions, {'q': {'question': row['question'], 'answer': None}})
            write_json(args.duo_run/'inputs.json', {'known_docs': list('abcdefgh')})
            write_json(args.out/'audit.json', {'total': 2327063576})
            baseline = {'q': {'answer': row['baseline']}}
            for name in ('v9', 'duo'):
                with zipfile.ZipFile(root/f'submission-private-{name}.zip', 'w') as archive:
                    archive.writestr('submission.json', json.dumps(baseline))
            marker = {'assets': {'strength': .25}, 'source_files': {p.name: file_hash(p) for p in root.glob('*.zip')}}
            with patch.object(direct, 'prepared', return_value={'validation_evaluated': False}), \
                    patch.object(direct.deep.prior, 'manifest', return_value=marker), \
                    patch.object(direct.deep.prior, 'SCORED_ZIP', file_hash(root/'submission-private-duo.zip')), \
                    patch.object(direct.deep, 'compared', return_value=[row]) as scores:
                direct.submit(args)
                for name in ('v9', 'duo'):
                    with zipfile.ZipFile(args.out/f'submission-private-{name}.zip') as archive:
                        expected = baseline if name == 'v9' else {'q': {'answer': direct.deep.fusion.predict(row, 'deep')}}
                        self.assertEqual(json.loads(archive.read('submission.json')), expected)
                report = read_json(args.out/'submission-choice.json')
                self.assertTrue(report['candidate_deployed'])
                self.assertEqual(report['changed_answer_sets'], 1)
                self.assertFalse(report['validation_evaluated'])
                scores.return_value = []
                with self.assertRaises(ValueError):
                    direct.submit(args)

    def test_launcher_runs_only_prepare_private_submit_and_backs_up_failure(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for fail in ('', 'private'):
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, text):
                    path = fixture/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(text.encode())
                    path.chmod(0o755)
                runner = 'scripts/gpu/gpu-deep-pilot512-submit-4090.sh'
                put(runner, (root/runner).read_text())
                put('checked', 'fixture')
                for manifest in ('private-files.sha256', 'pilot512-submit-files.sha256', 'pilot512-files.sha256'):
                    put(manifest, hashlib.sha256(b'fixture').hexdigest()+'  checked\n')
                put('artifacts-ranking-h100/duo-fulltext-v2.sqlite', 'fixture')
                put('bin/flock', '#!/usr/bin/env bash\nexit 0\n')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
echo "$*" >> "$FIXTURE/calls"
if [ -n "$FAIL" ] && [[ "$*" == *"legalir2.deep_pilot512_submit $FAIL "* ]]; then exit 19; fi
''')
                put('scripts/gpu/gpu-private-h100.sh', '#!/usr/bin/env bash\nexit 0\n')
                put('scripts/gpu/gpu-duo-runtime.sh', '''step() { :; }
stage_now() { echo "$1" > "$OUT/.stage"; }
ship_once() { echo backup >> "$FIXTURE/calls"; }
rclone_run() { echo "rclone $*" >> "$FIXTURE/calls"; }
start_worker() { :; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; ship_once; exit "$code"; }
''')
                env = dict(os.environ, FIXTURE=fixture.as_posix(), FAIL=fail, AUTO_STOP='1')
                result = subprocess.run([str(bash), '-c', 'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; '
                    'bash "$FIXTURE/scripts/gpu/gpu-deep-pilot512-submit-4090.sh" run'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if fail else 0, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text().splitlines()
                stages = [s.split('legalir2.deep_pilot512_submit ', 1)[1].split()[0]
                          for s in calls if 'legalir2.deep_pilot512_submit ' in s]
                self.assertEqual(stages, ['prepare', 'private'] if fail else ['prepare', 'private', 'submit'])
                self.assertTrue(any('private-deep-pilot512-4090-v1/artifacts' in s for s in calls))
                self.assertTrue(any('vast_stop_after.py' in s for s in calls))
                self.assertEqual(calls[-1], 'backup')


if __name__ == '__main__':
    unittest.main()
