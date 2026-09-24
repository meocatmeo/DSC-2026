"""Exercise the actual launcher's dispatcher with cloud and GPU calls stubbed."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TestPassageOrderShell(unittest.TestCase):
    def test_stage_order_failure_and_final_backup(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for failure in ('', 'score', 'evaluate'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)

                def put(name, text):
                    path = fixture/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(text.encode())
                    path.chmod(0o755)

                put('scripts/gpu/gpu-passage-order-4090.sh',
                    (root/'scripts/gpu/gpu-passage-order-4090.sh').read_text())
                put('checked.txt', 'fixture')
                put('private-files.sha256', hashlib.sha256(b'fixture').hexdigest()+'  checked.txt\n')
                for name in ('flock', 'rclone'):
                    put('bin/'+name, '#!/usr/bin/env bash\nexit 0\n')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
echo "$*" >> "$FIXTURE/calls"
if [ -n "$FAIL_STAGE" ] && [[ "$*" == *"legalir2.passage_order_run $FAIL_STAGE "* ]]; then exit 19; fi
''')
                put('scripts/gpu/gpu-duo-runtime.sh', '''step() { :; }
stage_now() { printf '%s\\n' "$1" > "$OUT/.stage"; }
ship_once() { echo backup >> "$FIXTURE/calls"; }
rclone_run() { :; }
start_worker() { :; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; ship_once; exit "$code"; }
''')
                env = dict(os.environ, FIXTURE=fixture.as_posix(), FAIL_STAGE=failure,
                    BASE_RESULTS='gdrive:DSC-2026-results', PASSAGE_RUN_ID='fixture')
                result = subprocess.run([str(bash), '-c',
                    'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; '
                    'bash "$FIXTURE/scripts/gpu/gpu-passage-order-4090.sh" run'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if failure else 0, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text().splitlines()
                self.assertEqual(calls[-1], 'backup')
                stages = [c.split('legalir2.passage_order_run ', 1)[1].split(' --out', 1)[0]
                          for c in calls if 'legalir2.passage_order_run ' in c]
                expected = ['score --max-questions 2', 'score', 'evaluate']
                if failure == 'score':
                    expected = expected[:1]
                self.assertEqual(stages, expected)
                self.assertFalse(any(' submit ' in c or ' private ' in c for c in calls))

    def test_scripts_parse(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for name in ('gpu-passage-order-4090.sh', 'start-passage-order-4090.sh'):
            result = subprocess.run([str(bash), '-n', str(root/'scripts/gpu'/name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
