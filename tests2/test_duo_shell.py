"""The comparator launcher shares the tested H100 runtime and extends its timeout."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class TestDuoShell(unittest.TestCase):
    def test_help_requires_no_gpu_or_credentials(self):
        result = subprocess.run([BASH, str(ROOT / "scripts/gpu/gpu-duo-h100.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare -> train -> calibrate -> validate -> submit", result.stdout)

    def launch_fixture(self, fail="", stage="all"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value.encode())
                path.chmod(0o755)

            write("scripts/gpu/gpu-duo-h100.sh", (ROOT / "scripts/gpu/gpu-duo-h100.sh").read_text())
            write("artifacts-ranking-h100/store.sqlite", "fixture")
            for command in ("rclone", "flock", "nvidia-smi", "setsid"):
                write("bin/" + command, "#!/usr/bin/env bash\nexit 0\n")
            write("bin/timeout", '#!/usr/bin/env bash\nwhile [[ "$1" == --* ]]; do shift; done\nshift\nexec "$@"\n')
            write(".venv-h100/bin/python", '''#!/usr/bin/env bash
if [ "$1" = - ]; then cat >/dev/null; exit 0; fi
if [ "$1" = -u ]; then
  printf '%s\n' "$4" >> "$ROOT/calls"
  printf '%s\n' "$3" >> "$ROOT/modules"
  if [ "$4" = "${FAIL_STAGE:-}" ]; then exit 17; fi
fi
''')
            write("scripts/gpu/gpu-duo-runtime.sh", '''step() { echo "$*"; }
stage_now() { echo "stage=$1"; }
ship_once() { :; }
rclone_run() { :; }
start_worker() { :; }
run_cmd() { "$@"; }
finish_run() { local code=$?; trap - EXIT; exit "$code"; }
''')
            env = dict(os.environ, FIXTURE=root.as_posix(), SOURCE_RUN_ID="source-fixture",
                       RUN_ID="duo-fixture", FAIL_STAGE=fail, FIXTURE_STAGE=stage)
            result = subprocess.run([BASH, "-c", 'export PATH="$(cd "$FIXTURE/bin" && pwd):$PATH"; bash "$FIXTURE/scripts/gpu/gpu-duo-h100.sh" "$FIXTURE_STAGE"'],
                                    env=env, capture_output=True, text=True)
            calls = (root / "calls").read_text().splitlines() if (root / "calls").exists() else []
            if stage == "submit-approved" and result.returncode == 0:
                self.assertEqual((root / "modules").read_text().splitlines(), ["legalir2.duo", "legalir2.duo_submit"])
            return result, calls

    def test_all_runs_through_submit_in_order(self):
        result, calls = self.launch_fixture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ["prepare", "train", "calibrate", "validate", "submit"])

    def test_training_failure_stops_before_validation_and_submit(self):
        result, calls = self.launch_fixture("train")
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        self.assertEqual(calls, ["prepare", "train"])

    def test_explicit_approved_submit_skips_training_and_validation(self):
        result, calls = self.launch_fixture(stage="submit-approved")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ["prepare", "submit"])

    def test_launchers_parse(self):
        for name in ("scripts/gpu/gpu-duo-h100.sh", "scripts/gpu/gpu-duo-runtime.sh"):
            result = subprocess.run([BASH, "-n", str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_only_checkpoint_uploads_get_longer_deadline(self):
        env = dict(os.environ, ROOT=ROOT.as_posix(), OUT="/example/output")
        script = '''
set -euo pipefail
source "$ROOT/scripts/gpu/gpu-duo-runtime.sh"
timeout() { printf '%s\n' "$4"; }
TRANSFER_TIMEOUT=5m rclone_run copy "$OUT" cloud:artifacts
TRANSFER_TIMEOUT=2m rclone_run copy /snapshot cloud:logs
'''
        result = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["20m", "2m"])

    def test_checkpoint_backup_waits_for_long_upload_and_keeps_failure_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "run.log").write_text("training complete\n")
            env = dict(os.environ, ROOT=ROOT.as_posix(), OUT=out.as_posix(),
                       RUN_ID="fixture", RUN_REMOTE="cloud:fixture", ATTEMPT_ID="attempt")
            script = '''
set -euo pipefail
source "$ROOT/scripts/gpu/gpu-duo-runtime.sh"
flock() { test "$1" = -w && test "$2" = 1500 && test "$3" = 8; }
rclone_run() { printf 'transfer\n' >> "$OUT/calls"; }
ship_once
test "$(wc -l < "$OUT/calls")" -eq 3
rclone_run() { return 9; }
if ship_once; then exit 44; fi
flock() { return 1; }
if ship_once; then exit 45; fi
'''
            result = subprocess.run([BASH, "-c", script], env=env, capture_output=True, text=True, timeout=25)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Upload failed; inspect", result.stdout)
            self.assertIn("Upload lock timed out", result.stdout)
            self.assertFalse(list(out.glob(".ship.*")))


if __name__ == "__main__":
    unittest.main()
