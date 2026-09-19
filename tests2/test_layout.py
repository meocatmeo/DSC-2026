"""Verify organized entry points and a self-contained private release."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASH = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path('/bin/bash')


class TestLayout(unittest.TestCase):
    def test_no_loose_scripts_or_credentials_at_root(self):
        self.assertFalse([p.name for p in ROOT.iterdir() if p.is_file()
                          and p.suffix in ('.py', '.sh', '.conf', '.exe')])

    @unittest.skipUnless(BASH.exists(), 'Bash unavailable')
    def test_all_gpu_scripts_parse_and_root_sibling_paths_exist(self):
        for script in (ROOT/'scripts/gpu').glob('*.sh'):
            result = subprocess.run([str(BASH), '-n', str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, script.name+result.stderr)
            for name in re.findall(r'\$ROOT/(scripts/gpu/[\w-]+\.sh)', script.read_text()):
                self.assertTrue((ROOT/name).is_file(), f'{script.name}: missing {name}')

    @unittest.skipUnless(BASH.exists(), 'Bash unavailable')
    def test_extracted_release_help_from_another_working_directory(self):
        release = ROOT/'releases/private/20260919-layout-v2/private-inference-overlay.zip'
        if not release.exists():
            self.skipTest('Build private release before this integration check')
        with tempfile.TemporaryDirectory() as directory:
            extracted = Path(directory)/'checkout'
            with zipfile.ZipFile(release) as archive:
                names = archive.namelist()
                self.assertIn('data/private-official.json', names)
                self.assertIn('config/requirements-private-h100.txt', names)
                self.assertNotIn('gpu-private-h100.sh', names)
                archive.extractall(extracted)
            for name in ('gpu-private-h100.sh','gpu-private-setup.sh'):
                result = subprocess.run([str(BASH), str(extracted/'scripts/gpu'/name), '--help'],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertIn('scripts/gpu/', result.stdout)


if __name__ == '__main__':
    unittest.main()
