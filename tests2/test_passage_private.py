from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from legalir2 import passage_private as run
from legalir2.store import write_json, file_hash


class TestPrivatePassage(unittest.TestCase):
    def test_changed_edge_inventory_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = {'original': {'id': 'q', 'docs': ['a', 'b']},
                'before': {'a': [1], 'b': [2]}, 'after': {'a': [3], 'b': [2]}, 'edges': [[0, 1]]}
            write_json(root/'questions/0.json', row)
            prepared = {'files': {'questions/0.json': file_hash(root/'questions/0.json')}}
            self.assertEqual(len(list(run.items(root, prepared))), 1)
            row['edges'] = []; write_json(root/'questions/0.json', row)
            with self.assertRaises(ValueError): list(run.items(root, prepared))
            prepared['files']['questions/0.json'] = file_hash(root/'questions/0.json')
            with self.assertRaisesRegex(ValueError, 'edge'): list(run.items(root, prepared))

    def test_partial_scores_cannot_create_submission(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'data').mkdir()
            with zipfile.ZipFile(root/'data/best-09525.zip', 'w') as z:
                z.writestr('submission.json', '{}')
            (root/'data/best-09533.zip').write_bytes((root/'data/best-09525.zip').read_bytes())
            write_json(root/'data/questions.json', {'q': {'question': 'Q'}})
            write_json(root/'data/known-docs.json', list('abcde'))
            with patch.object(run, 'load', return_value=({}, {})), \
                    patch.object(run, 'BEST_SHA', file_hash(root/'data/best-09525.zip')), \
                    patch.object(run, 'CURRENT_BEST_SHA', file_hash(root/'data/best-09533.zip')), \
                    patch.object(run, 'scored', return_value=iter([])):
                with self.assertRaisesRegex(ValueError, 'Complete 2080'):
                    run.submit(SimpleNamespace(out=root))
            self.assertFalse((root/'submission-private-passage-order.zip').exists())


if __name__ == '__main__': unittest.main()
