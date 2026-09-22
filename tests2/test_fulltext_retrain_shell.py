"""Run the real Bash stage dispatcher with external GPU/cloud services stubbed."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TestRetrainShell(unittest.TestCase):
    def test_private_only_stage_order_and_failure_backup(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for failure in ('', 'private'):
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, text):
                    path = fixture/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(text.encode())
                    path.chmod(0o755)
                put('tools/gpu-retrain-private-submit.sh', (root/'scripts/gpu/gpu-retrain-private-submit.sh').read_text())
                put('checked.txt', 'fixture')
                put('private-files.sha256', hashlib.sha256(b'fixture').hexdigest()+'  checked.txt\n')
                put('bin/flock', '#!/usr/bin/env bash\nexit 0\n')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FIXTURE/calls"
if [ -n "$FAIL_STAGE" ] && [[ "$*" == *"retrain_private_submit.py $FAIL_STAGE "* ]]; then exit 19; fi
''')
                put('scripts/gpu/gpu-duo-runtime.sh', '''step() { :; }
stage_now() { printf '%s\\n' "$1" > "$OUT/.stage"; }
ship_once() { echo backup >> "$FIXTURE/calls"; }
start_worker() { :; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; ship_once; exit "$code"; }
''')
                env = dict(os.environ, FIXTURE=fixture.as_posix(), ROOT=fixture.as_posix(), FAIL_STAGE=failure,
                           RETRAIN_RUN_ID='duo-fulltext-retrain-private-control-4090-v1')
                result = subprocess.run([str(bash), '-c',
                    'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; bash "$FIXTURE/tools/gpu-retrain-private-submit.sh"'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if failure else 0, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text().splitlines()
                stages = [c.split('retrain_private_submit.py ', 1)[1].split(' --out', 1)[0]
                          for c in calls if 'retrain_private_submit.py ' in c]
                self.assertEqual(stages, ['prepare', 'private'] if failure else ['prepare', 'private', 'submit'])
                self.assertEqual(calls[-1], 'backup')

    def test_matched_stage_order_and_failures_preserve_backup(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for failure in ('', 'train', 'validation'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, text):
                    path = fixture/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(text.encode())
                    path.chmod(0o755)
                put('scripts/gpu/gpu-fulltext-retrain-4090.sh', (root/'scripts/gpu/gpu-fulltext-retrain-4090.sh').read_text())
                put('checked.txt', 'fixture')
                put('private-files.sha256', hashlib.sha256(b'fixture').hexdigest()+'  checked.txt\n')
                put('artifacts-ranking-h100/duo-fulltext-v2.sqlite', 'fixture')
                for tool in ('rclone', 'flock'):
                    put('bin/'+tool, '#!/usr/bin/env bash\nexit 0\n')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FIXTURE/calls"
if [ -n "$FAIL_STAGE" ] && [[ "$*" == *"legalir2.fulltext_retrain $FAIL_STAGE "* ]]; then exit 19; fi
''')
                put('scripts/gpu/gpu-private-h100.sh', '#!/usr/bin/env bash\necho restore >> "$FIXTURE/calls"\n')
                put('scripts/gpu/gpu-duo-runtime.sh', '''step() { :; }
stage_now() { printf '%s\n' "$1" > "$OUT/.stage"; }
ship_once() { echo backup >> "$FIXTURE/calls"; }
rclone_run() { :; }
start_worker() { :; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; ship_once; exit "$code"; }
''')
                env = dict(os.environ, FIXTURE=fixture.as_posix(), FAIL_STAGE=failure,
                           BASE_RESULTS='gdrive:DSC-2026-results', RETRAIN_RUN_ID='fixture')
                result = subprocess.run([str(bash), '-c',
                    'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; '
                    'bash "$FIXTURE/scripts/gpu/gpu-fulltext-retrain-4090.sh" run'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if failure else 0, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text().splitlines()
                self.assertEqual(calls[-1], 'backup')
                stages = [c.split('legalir2.fulltext_retrain ', 1)[1].split(' --out', 1)[0]
                          for c in calls if 'legalir2.fulltext_retrain ' in c]
                expected = ['prepare', 'train --arm old-evidence-control', 'train --arm control',
                            'development', 'screen', 'validation', 'gate', 'private', 'submit']
                if failure:
                    index = next(i for i, stage in enumerate(expected) if stage.split()[0] == failure)
                    expected = expected[:index+1]
                self.assertEqual(stages, expected)
                self.assertNotIn('train --arm hard-negatives', stages)


if __name__ == '__main__':
    unittest.main()
