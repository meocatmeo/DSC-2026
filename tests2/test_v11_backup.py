from pathlib import Path
import tempfile
import unittest

from legalir2.v11_backup import snapshot_indexes


class SnapshotTests(unittest.TestCase):
    def test_snapshot_is_immutable_and_excludes_torn_utf8_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index = root / 'out/train.rows/index.jsonl'
            index.parent.mkdir(parents=True)
            complete = '{"id":"câu một"}\n'.encode()
            torn = '{"id":"câu hai"}'.encode()[:-3]
            index.write_bytes(complete + torn)
            snapshot_indexes(root / 'out', root / 'snapshot')
            frozen = root / 'snapshot/train.jsonl'
            self.assertEqual(frozen.read_bytes(), complete)
            index.write_bytes(complete + b'{"id":"next"}\n')
            self.assertEqual(frozen.read_bytes(), complete)

    def test_nested_benchmark_and_completed_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ['train', 'benchmark/train']:
                index = root / 'out' / (split + '.rows/index.jsonl')
                index.parent.mkdir(parents=True)
                index.write_bytes(b'{}\n')
            (root / 'out/train.jsonl.gz').write_bytes(b'complete')
            (root / 'out/train.manifest.json').write_text('{}')
            snapshot_indexes(root / 'out', root / 'snapshot')
            self.assertEqual([p.name for p in (root / 'snapshot').iterdir()], ['benchmark-train.jsonl'])
