"""Recovery must reject changed payloads and unknown questions before inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[1]/'scripts/package/best_recovery_runtime.py'
spec=importlib.util.spec_from_file_location('best_recovery_runtime',MODULE)
runtime=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class TestRecovery(unittest.TestCase):
    def test_duplicate_json_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'invalid.json';path.write_text('{"q":1,"q":2}')
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                runtime.read(path)

    def test_tampered_release_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);payload=root/'payload.json';payload.write_text('{}')
            expected=runtime.sha(payload);payload.write_text('{"changed":true}')
            (root/'manifest.json').write_text(json.dumps({'files':{'payload.json':expected}}))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                runtime.recover(root,root/'output')
            self.assertFalse((root/'output').exists())

    def test_new_questions_are_not_silently_answered_from_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=root/'original.json';original.write_text('{}')
            questions=root/'questions.json';questions.write_text('{"new":"question"}')
            (root/'manifest.json').write_text(json.dumps({'files':{},'questions_sha256':runtime.sha(original)}))
            with self.assertRaisesRegex(ValueError,'only the frozen'):
                runtime.recover(root,root/'output',questions)
            self.assertFalse((root/'output').exists())


if __name__=='__main__':
    unittest.main()
