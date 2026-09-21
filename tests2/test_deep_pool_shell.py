"""Exercise the real 4090 launcher with stubbed external services."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TestDeepPoolShell(unittest.TestCase):
    runner = 'gpu-deep-pool-4090.sh'
    module = 'legalir2.deep_pool'
    failure_stage = 'validation'
    success_stages = ['prepare', 'training', 'fit', 'development', 'calibrate', 'validation', 'validate', 'private', 'submit']

    def test_stage_order_submit_and_failure_backup(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for failure in ('', self.failure_stage):
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, text):
                    path = fixture/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(text.encode())
                    path.chmod(0o755)
                put('scripts/gpu/'+self.runner, (root/'scripts/gpu'/self.runner).read_text())
                put('checked.txt', 'fixture')
                put('private-files.sha256', hashlib.sha256(b'fixture').hexdigest()+'  checked.txt\n')
                put('artifacts-ranking-h100/duo-fulltext-v2.sqlite', 'fixture')
                for tool in ('rclone', 'flock', 'python3'):
                    put('bin/'+tool, '#!/usr/bin/env bash\nexit 0\n')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FIXTURE/calls"
if [ -n "$FAIL_STAGE" ] && [[ "$*" == *"legalir2.deep_pool $FAIL_STAGE "* ]]; then exit 19; fi
''')
                put('scripts/gpu/gpu-private-setup.sh', '#!/usr/bin/env bash\necho setup >> "$FIXTURE/calls"\n')
                put('scripts/gpu/gpu-private-h100.sh', '#!/usr/bin/env bash\necho restore >> "$FIXTURE/calls"\n')
                put('scripts/gpu/gpu-duo-runtime.sh', '''step() { :; }
stage_now() { printf '%s\n' "$1" > "$OUT/.stage"; }
ship_once() { echo backup >> "$FIXTURE/calls"; }
rclone_run() { :; }
start_worker() { :; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; ship_once; exit "$code"; }
''')
                env = dict(os.environ, FIXTURE=fixture.as_posix(), FAIL_STAGE=failure, AUTO_STOP='1', RUNNER=self.runner)
                result = subprocess.run([str(bash), '-c',
                    'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; '
                    'bash "$FIXTURE/scripts/gpu/$RUNNER" all'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if failure else 0, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text().splitlines()
                self.assertTrue(any('vast_stop_after.py' in line for line in calls))
                stages = [line.replace('legalir2.deep_pilot', 'legalir2.deep_pool').split('legalir2.deep_pool ', 1)[1].split()[0]
                          for line in calls if 'legalir2.deep_pool' in line or 'legalir2.deep_pilot' in line]
                expected = self.success_stages
                if failure:
                    expected = expected[:expected.index(failure)+1]
                self.assertEqual(stages, expected)
                if not failure:
                    self.assertTrue(any(self.module+' submit ' in line for line in calls))
                self.assertEqual(calls[-1], 'backup')


if __name__ == '__main__':
    unittest.main()
