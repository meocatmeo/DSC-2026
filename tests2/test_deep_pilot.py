"""Check fixed pilot sampling and baseline-only completion on development gains."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from legalir2 import deep_pilot as pilot
from legalir2.store import file_hash, write_json, read_json
from tests2.test_deep_pool_shell import TestDeepPoolShell as ShellChecks


class TestPilotShell(ShellChecks):
    runner = 'gpu-deep-pilot-4090.sh'
    module = 'legalir2.deep_pilot'
    failure_stage = 'development'
    success_stages = ['prepare', 'training', 'fit', 'development', 'calibrate', 'submit']


class TestPilot(unittest.TestCase):
    def test_sampling_ignores_labels_scores_and_input_order(self):
        records = {str(i): {'question': f'Question {i}', 'answer': ['a']} for i in range(500)}
        before = pilot.training_subset(records)
        changed = deepcopy(dict(reversed(list(records.items()))))
        for row in changed.values():
            row.update(answer=['z'], score=-100)
        self.assertEqual(list(before), list(pilot.training_subset(changed)))
        self.assertEqual(len(before), 256)
        self.assertTrue(set(before) <= set(records))

    def test_development_gain_still_writes_only_original_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root/'out', source=root/'source', duo_run=root/'duo', questions=root/'questions.json')
            args.source.mkdir()
            known = ['a', 'b', 'c', 'd', 'e', 'f']
            answers = {'q': {'answer': known[:5]}}
            write_json(args.questions, {'q': {'question': 'Question', 'answer': None}})
            write_json(args.duo_run/'inputs.json', {'known_docs': known})
            write_json(args.out/'audit.json', {})
            hashes = {}
            for name in ('v9', 'duo'):
                path = args.source/f'submission-private-{name}.zip'
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr('submission.json', json.dumps(answers))
                hashes[path.name] = file_hash(path)
            marker = {'assets': {'strength': .25}, 'source_files': hashes}
            with patch.object(pilot.deep, 'choice', return_value={'selected': 'deep'}), \
                    patch.object(pilot.deep, 'fitted', return_value={'questions': 256}), \
                    patch.object(pilot.deep.prior, 'manifest', return_value=marker), \
                    patch.object(pilot.deep.prior, 'SCORED_ZIP', hashes['submission-private-duo.zip']):
                pilot.submit(args)
            report = read_json(args.out/'pilot-report.json')
            self.assertFalse(report['candidate_deployed'])
            self.assertFalse(report['validation_evaluated'])
            self.assertFalse(report['private_inference_performed'])
            for name in ('v9', 'duo'):
                with zipfile.ZipFile(args.out/f'submission-private-{name}.zip') as archive:
                    self.assertEqual(json.loads(archive.read('submission.json')), answers)


if __name__ == '__main__':
    unittest.main()
