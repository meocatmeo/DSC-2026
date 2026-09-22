"""Exercise fresh-host restore through the real Bash scripts with cloud/GPU stubs."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TestPrivateFresh(unittest.TestCase):
    def test_restore_then_private_only_and_failure_stops_before_inference(self):
        root = Path(__file__).resolve().parents[1]
        bash = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')
        if not bash.exists():
            self.skipTest('Bash unavailable')
        for fail in ('', 'restore', 'verify'):
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, text):
                    path = fixture/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(text.encode())
                    path.chmod(0o755)
                for name in ('gpu-retrain-private-fresh.sh', 'gpu-retrain-private-submit.sh'):
                    put('tools/'+name, (root/'scripts/gpu'/name).read_text())
                put('checked.txt', 'fixture')
                put('private-files.sha256', hashlib.sha256(b'fixture').hexdigest()+'  checked.txt\n')
                put('bin/flock', '#!/usr/bin/env bash\nexit 0\n')
                put('scripts/gpu/gpu-private-setup.sh', '#!/usr/bin/env bash\necho "setup $*" >> "$FIXTURE/calls"\n')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FIXTURE/calls"
if [ "$FAIL_STAGE" = verify ] && [[ "$*" == *verify_retrain_restore.py* ]]; then exit 19; fi
''')
                put('scripts/gpu/gpu-duo-runtime.sh', '''step() { :; }
stage_now() { printf '%s\\n' "$1" > "$OUT/.stage"; }
ship_once() { echo backup >> "$FIXTURE/calls"; }
start_worker() { :; }
stop_worker() { :; }
rclone_run() { echo "rclone $*" >> "$FIXTURE/calls"; if [ "$FAIL_STAGE" = restore ]; then return 19; fi; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; ship_once; exit "$code"; }
''')
                env = dict(os.environ, FIXTURE=fixture.as_posix(), ROOT=fixture.as_posix(), FAIL_STAGE=fail)
                result = subprocess.run([str(bash), '-c',
                    'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; bash "$FIXTURE/tools/gpu-retrain-private-fresh.sh"'],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if fail else 0, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text().splitlines()
                self.assertEqual(calls[0], 'setup env')
                stages = [c.split('retrain_private_submit.py ', 1)[1].split(' --out', 1)[0]
                          for c in calls if 'retrain_private_submit.py ' in c]
                self.assertEqual(stages, [] if fail else ['prepare', 'private', 'submit'])
                self.assertEqual(calls[-1], 'backup')
                transfers = '\n'.join(c for c in calls if c.startswith('rclone'))
                self.assertNotIn('assets/models', transfers)
                self.assertNotIn('/indexes', transfers)
                self.assertIn('/arms/control/model/**', transfers)
                if not fail:
                    verify = next(i for i, c in enumerate(calls) if 'verify_retrain_restore.py' in c)
                    infer = next(i for i, c in enumerate(calls) if 'retrain_private_submit.py private' in c)
                    self.assertLess(verify, infer)


if __name__ == '__main__':
    unittest.main()
