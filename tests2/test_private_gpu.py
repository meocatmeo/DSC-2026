"""GPU profile checks and the complete fresh/resumed 4090 launch path."""
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from legalir2.private_gpu import check

ROOT = Path(__file__).resolve().parents[1]
BASH = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')


def device(name='NVIDIA GeForce RTX 4090', capability=(8, 9), memory=24, bf16=True):
    return SimpleNamespace(__version__='2.7.1+cu128', cuda=SimpleNamespace(
        is_available=lambda: True, is_bf16_supported=lambda: bf16,
        get_device_name=lambda i: name, get_device_capability=lambda i: capability,
        get_device_properties=lambda i: SimpleNamespace(total_memory=memory * 2**30)))


class TestPrivateGPU(unittest.TestCase):
    @unittest.skipUnless(BASH.exists(), 'Bash unavailable')
    def test_fast_wrapper_uses_separate_resume_identity(self):
        wrapper = ROOT/'scripts/gpu/gpu-duo-4090-fast.sh'
        if not wrapper.exists():
            self.skipTest('Fast release wrapper is not included in this package')
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            scripts = fixture/'scripts/gpu'
            scripts.mkdir(parents=True)
            (scripts/wrapper.name).write_bytes(wrapper.read_bytes())
            (scripts/'gpu-duo-4090.sh').write_bytes(b'#!/usr/bin/env bash\nprintf "%s %s" "$PRIVATE_RUN_ID" "$1"\n')
            for override in ('', 'custom-fast-b2'):
                env = dict(os.environ)
                env.pop('PRIVATE_RUN_ID', None)
                if override:
                    env['PRIVATE_RUN_ID'] = override
                result = subprocess.run([str(BASH), str(scripts/wrapper.name), 'run'],
                                        cwd=directory, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, (override or 'private-duo-fulltext-v2-4090-fast1') + ' run')

    def test_4090_and_h100_are_explicit_profiles(self):
        self.assertEqual(check(device(), '4090')['vram_gib'], 24)
        self.assertEqual(check(device('NVIDIA H100 80GB HBM3', (9, 0), 80), 'h100')['vram_gib'], 80)
        with self.assertRaisesRegex(ValueError, 'H100'):
            check(device(), 'h100')
        with self.assertRaisesRegex(ValueError, '4090'):
            check(device('NVIDIA H100 80GB HBM3', (9, 0), 80), '4090')

    def test_reject_wrong_device_memory_backend_or_packages(self):
        for model in (device('NVIDIA RTX 4080'), device(capability=(8, 6)),
                      device(memory=16), device(bf16=False)):
            with self.assertRaises(ValueError):
                check(model, '4090')
        model = device()
        model.__version__ = '2.8.0'
        with self.assertRaisesRegex(ValueError, 'saved torch'):
            check(model, '4090')
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            check(device(), 'unknown')

    @unittest.skipUnless(BASH.exists(), 'Bash unavailable')
    def test_fresh_resume_and_failures(self):
        cases = [('all', '', False), ('all', 'setup', False), ('all', 'gpu', False),
                 ('all', 'inference', False), ('env', '', False), ('run', '', False),
                 ('submit', '', False), ('duo', '', True), ('invalid', '', False)]
        for stage, fail, override in cases:
            with self.subTest(stage=stage, fail=fail, override=override), tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory)
                def put(name, body):
                    path = fixture/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(body.encode())
                    path.chmod(0o755)
                put('scripts/gpu/gpu-duo-4090.sh', (ROOT/'scripts/gpu/gpu-duo-4090.sh').read_text())
                put('scripts/gpu/gpu-private-setup.sh', '''#!/usr/bin/env bash
printf 'setup %s %s %s\n' "$1" "$GPU_PROFILE" "$MAX_JOBS" >> calls
[ "${FAIL_STAGE:-}" != setup ] || exit 19
''')
                put('.venv-h100/bin/python', '''#!/usr/bin/env bash
printf 'gpu %s\n' "$*" >> calls
[ "${FAIL_STAGE:-}" != gpu ] || exit 23
''')
                put('scripts/gpu/gpu-private-h100.sh', '''#!/usr/bin/env bash
printf 'inference %s %s %s %s %s\n' "$1" "$DUO_POLICY" "$INFER_BATCH" "$PRIVATE_RUN_ID" "$V9_SOURCE_RUN_ID" >> calls
[ "${FAIL_STAGE:-}" != inference ] || exit 29
''')
                env = dict(os.environ, FAIL_STAGE=fail)
                for name in ('PY', 'INFER_BATCH', 'MAX_JOBS', 'PRIVATE_RUN_ID', 'V9_SOURCE_RUN_ID'):
                    env.pop(name, None)
                if override:
                    env.update(INFER_BATCH='2', PRIVATE_RUN_ID='private-4090-b2', V9_SOURCE_RUN_ID='')
                result = subprocess.run([str(BASH), str(fixture/'scripts/gpu/gpu-duo-4090.sh'), stage],
                                        cwd=directory, env=env, capture_output=True, text=True)
                expected = {'setup': 19, 'gpu': 23, 'inference': 29}.get(fail, 2 if stage == 'invalid' else 0)
                self.assertEqual(result.returncode, expected, result.stdout+result.stderr)
                calls = (fixture/'calls').read_text().splitlines() if (fixture/'calls').exists() else []
                if stage == 'invalid':
                    self.assertEqual(calls, [])
                    continue
                if stage in ('all', 'env'):
                    self.assertEqual(calls[0], 'setup env 4090 2')
                if stage == 'env' or fail == 'setup':
                    self.assertEqual(len(calls), 1)
                    continue
                self.assertIn('gpu -m legalir2.private_gpu --profile 4090', calls)
                if fail == 'gpu':
                    self.assertFalse(any(c.startswith('inference') for c in calls))
                    continue
                next_stage = 'all' if stage in ('all', 'run') else stage
                suffix = '2 private-4090-b2 ' if override else '4 private-duo-fulltext-v2-4090 private-v9-duo-9da4e0cb-v1'
                self.assertEqual(calls[-1], f'inference {next_stage} fulltext-v2 {suffix}')


if __name__ == '__main__':
    unittest.main()
