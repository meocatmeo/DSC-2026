"""Exercise launcher failure handling with Bash and local rclone, without a GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = str(Path("C:/Program Files/Git/bin/bash.exe")) if os.name == "nt" else shutil.which("bash")
RCLONE = str(ROOT / "tools/bin/rclone.exe") if os.name == "nt" else shutil.which("rclone")


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash unavailable")
class TestH100Shell(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "out").mkdir()
        (self.root / "out/run.log").write_text("before\n")
        self.env = dict(os.environ, ROOT=ROOT.as_posix(), OUT=(self.root / "out").as_posix(),
                        RUN_REMOTE=(self.root / "cloud").as_posix(), RUN_ID="fixture",
                        ATTEMPT_ID="attempt", LOG_EVERY="0.1")
        if os.name == "nt":
            self.env["PATH"] = "C:/Program Files/Git/usr/bin;" + self.env["PATH"]

    def shell(self, body):
        # Git Bash does not implement Linux process groups/flock. Mock only these
        # OS primitives here; production uses real setsid/flock on the rental host.
        prelude = 'set -euo pipefail\nsource "$ROOT/scripts/gpu/gpu-ranking-h100-runtime.sh"\n'
        if os.name == "nt":
            prelude += 'flock() { :; }\nsetsid() { "$@"; }\n'
        return subprocess.run([BASH, "-c", prelude + body], env=self.env, cwd=ROOT,
                              capture_output=True, text=True, timeout=25)

    def test_scripts_have_valid_bash_syntax(self):
        for name in ("scripts/gpu/gpu-ranking-h100.sh", "scripts/gpu/gpu-ranking-h100-runtime.sh"):
            result = subprocess.run([BASH, "-n", str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_cmd_preserves_heredoc_input_and_failure_code(self):
        result = self.shell('''
run_cmd bash -s <<'SCRIPT'
printf 'stdin survived\n'
SCRIPT
run_cmd bash -c 'exit 37'
echo 'must not run'
''')
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertIn("stdin survived", result.stdout)
        self.assertNotIn("must not run", result.stdout)

    def test_final_upload_failure_never_masks_original_failure(self):
        for original, expected in ((0, 1), (42, 42)):
            result = self.shell(f'''
ship_once() {{ return 1; }}
trap finish_run EXIT
exit {original}
''')
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            self.assertIn("FINAL UPLOAD FAILED", result.stdout)

    def test_transient_upload_failure_retries_in_worker(self):
        result = self.shell(r'''
export -f step
ship_once() {
  local n=0
  [ ! -f "$OUT/calls" ] || n=$(cat "$OUT/calls")
  n=$((n+1)); printf '%s' "$n" > "$OUT/calls"
  [ "$n" -ne 1 ]
}
# Exercise the real worker loop with its uploader replaced by a deterministic
# failure. Source in the worker after its function definitions to inject the stub.
declare -f ship_once > "$OUT/stub.sh"
sed '/^  set -euo pipefail$/a\  source "$OUT/stub.sh"' "$ROOT/scripts/gpu/gpu-ranking-h100-runtime.sh" > "$OUT/worker.sh"
bash "$OUT/worker.sh" worker &
worker=$!
for attempt in {1..60}; do
  [ -f "$OUT/calls" ] && [ "$(cat "$OUT/calls")" -ge 3 ] && break
  sleep 0.1
done
kill -TERM "$worker"
wait "$worker" || true
test "$(cat "$OUT/calls")" -ge 3
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("retrying next heartbeat", result.stdout)

    @unittest.skipUnless(RCLONE and Path(RCLONE).is_file(), "rclone unavailable")
    def test_local_rclone_uses_frozen_logs_and_only_completed_artifacts(self):
        for name in ("document/train/part-00000.json", "document/train/resume.json",
                     "document/train/part-00001.json.tmp", "unfinished.partial", ".hidden"):
            path = self.root / "out" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        (self.root / "empty.conf").write_text("")
        self.env["RCLONE_BIN"] = Path(RCLONE).as_posix()
        self.env["RCLONE_CONFIG"] = (self.root / "empty.conf").as_posix()
        result = self.shell('''
rclone_run() {
  # A live append after snapshotting must not change the uploaded snapshot.
  printf 'during transfer\n' >> "$OUT/run.log"
  "$RCLONE_BIN" --config "$RCLONE_CONFIG" "$@"
}
ship_once
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.root / "cloud/logs/attempt/run.log").read_text(), "before\n")
        copied = {p.relative_to(self.root / "cloud/artifacts").as_posix()
                  for p in (self.root / "cloud/artifacts").rglob("*") if p.is_file()}
        self.assertEqual(copied, {"document/train/part-00000.json", "document/train/resume.json"})
        self.assertFalse(list((self.root / "out").glob(".ship.*")))


if __name__ == "__main__":
    unittest.main()
