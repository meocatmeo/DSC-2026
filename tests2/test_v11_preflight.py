import json
from pathlib import Path
import struct
import tempfile
import unittest

from legalir2.v11_preflight import model_ready, dense_ready


class PreflightTests(unittest.TestCase):
    def test_missing_shard_and_truncated_shard_are_not_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'a.safetensors', 'b': 'b.safetensors'}}))
            header = json.dumps({'a': {'data_offsets': [0, 4]}}).encode()
            weights = struct.pack('<Q', len(header)) + header + b'1234'
            (root / 'a.safetensors').write_bytes(weights)
            with self.assertRaises(FileNotFoundError):
                model_ready(root)
            (root / 'b.safetensors').write_bytes(weights[:-1])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                model_ready(root)
            (root / 'b.safetensors').write_bytes(weights)
            model_ready(root)

    def test_manifest_alone_does_not_make_dense_ready(self):
        from legalir2.store import file_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = root / 'store'
            store.write_bytes(b'fixture')
            (root / 'manifest.json').write_text(json.dumps({'store_sha256': file_hash(store), 'model_identity': {'path': str(root)}}))
            with self.assertRaises(FileNotFoundError):
                dense_ready(root, store, root)
