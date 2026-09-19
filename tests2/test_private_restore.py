"""Exact Qwen fallback restore, interrupted downloads and source mismatch protection."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from legalir2 import private_restore as restore


class TestPrivateRestore(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / 'partner_qwen'
        self.contents = {name: name.encode() for name in ('config.json', 'model.safetensors', 'tokenizer.json')}
        self.signature = {'source_inputs': {'files': {
            restore.PREFIX+name: hashlib.sha256(data).hexdigest() for name, data in self.contents.items()}}}

    def fetch(self, repo, commit, name, path):
        self.assertEqual((repo, commit), (restore.REPO, restore.COMMIT))
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.contents[name]
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest(), len(data)

    def test_complete_restore_and_reuse_without_network(self):
        with patch.object(restore, 'fetch', side_effect=self.fetch) as fetch:
            restore.restore(self.signature, self.directory)
            self.assertEqual(fetch.call_count, 3)
            fetch.reset_mock()
            restore.restore(self.signature, self.directory, check=True)
            fetch.assert_not_called()

    def test_mismatched_download_does_not_replace_local_file(self):
        self.directory.mkdir()
        path = self.directory / 'config.json'
        path.write_bytes(b'preserve until verified')
        self.contents['config.json'] = b'wrong upstream bytes'
        with patch.object(restore, 'fetch', side_effect=self.fetch):
            with self.assertRaisesRegex(ValueError, 'differs from saved run'):
                restore.restore(self.signature, self.directory)
        self.assertEqual(path.read_bytes(), b'preserve until verified')

    def test_interrupted_restore_reuses_verified_files(self):
        def interrupted(repo, commit, name, path):
            if name == 'model.safetensors':
                raise OSError('interrupted transfer')
            return self.fetch(repo, commit, name, path)
        with patch.object(restore, 'fetch', side_effect=interrupted):
            with self.assertRaises(OSError):
                restore.restore(self.signature, self.directory)
        with patch.object(restore, 'fetch', side_effect=self.fetch) as fetch:
            restore.restore(self.signature, self.directory)
            self.assertEqual(fetch.call_count, 2)


if __name__ == '__main__':
    unittest.main()
