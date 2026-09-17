"""Regressions for the V11 checkpoint/resume path and cache streaming.

Each test here corresponds to a defect that would have surfaced only after hours of
GPU time: a resume that always refuses, a fit stage that loads gigabytes it does not
read, and a checkpoint scan that reparses every committed row.
"""
import gzip
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from legalir2.v11 import (ANALYTIC_KEYS, canonical, checkpoint_row, load_cache,
                          prepare_resume, read_index, stream_cache)


def row_for(qid, question, gold, extra=None):
    payload = {'id': qid, 'question': question, 'gold': gold, 'docs': ['d1', 'd2'],
               'features': [[0.0] * 20, [1.0] * 20], 'bundle_scores': [0.2, 0.8],
               'frozen_prediction': ['d1'], 'seconds': 1.5}
    payload.update(extra or {})
    return payload


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.questions = {'q1': {'question': 'câu một', 'answer': ['d1']},
                          'q2': {'question': 'câu hai', 'answer': ['d2']}}
        self.items = list(self.questions.items())
        self.signature = {'schema': 'v11-evidence-2048-v1', 'items': self.items}
        self.directory = self.root / 'train.rows'

    def test_resume_accepts_its_own_guard(self):
        """`list(dict.items())` is tuples; JSON returns lists. Without canonicalising
        the guard, every restart raised and told you to discard the checkpoints."""
        prepare_resume(self.directory, self.signature, self.items)
        checkpoint_row(self.directory, row_for('q1', 'câu một', ['d1']))
        done = prepare_resume(self.directory, self.signature, self.items)
        self.assertEqual(set(done), {'q1'})
        self.assertEqual(done['q1'][1], 1.5)

    def test_resume_still_refuses_changed_inputs(self):
        prepare_resume(self.directory, self.signature, self.items)
        moved = dict(self.signature, items=[('q1', {'question': 'khác', 'answer': ['d9']})])
        with self.assertRaises(ValueError):
            prepare_resume(self.directory, moved, self.items)

    def test_only_audited_code_hashes_are_resume_compatible(self):
        import hashlib
        import legalir2.v11
        from legalir2.v11_compat import compatible_signature, EQUIVALENT
        for name, hashes in EQUIVALENT.items():
            source = Path(legalir2.v11.__file__).parent / name
            self.assertIn(hashlib.sha256(source.read_bytes()).hexdigest(), hashes)
            old = {'code': {name: hashes[0]}, 'store': 'same'}
            new = {'code': {name: hashes[-1]}, 'store': 'same'}
            self.assertEqual(compatible_signature(old), compatible_signature(new))
            self.assertNotEqual(compatible_signature(old), compatible_signature({'code': {name: 'unknown'}, 'store': 'same'}))
            self.assertEqual(old['code'][name], hashes[0])  # never mutates the guard

    def test_rows_without_a_guard_are_refused(self):
        self.directory.mkdir(parents=True)
        (self.directory / 'stray.json.gz').write_bytes(gzip.compress(b'{}'))
        with self.assertRaises(ValueError):
            prepare_resume(self.directory, self.signature, self.items)

    def test_crash_between_commit_and_index_rescores_one_question(self):
        """The row lands, the index line does not. That question must be rescored,
        never silently treated as committed and never fatal."""
        prepare_resume(self.directory, self.signature, self.items)
        checkpoint_row(self.directory, row_for('q1', 'câu một', ['d1']))
        index = self.directory / 'index.jsonl'
        index.write_text('', encoding='utf-8')
        self.assertEqual(read_index(self.directory, self.questions), {})

    def test_torn_final_index_line_is_survivable(self):
        prepare_resume(self.directory, self.signature, self.items)
        checkpoint_row(self.directory, row_for('q1', 'câu một', ['d1']))
        index = self.directory / 'index.jsonl'
        with index.open('a', encoding='utf-8') as stream:
            stream.write('{"id": "q2", "file": "trunc')
        self.assertEqual(set(read_index(self.directory, self.questions)), {'q1'})
        # Recovery must still work after writing the question that was interrupted.
        checkpoint_row(self.directory, row_for('q2', 'câu hai', ['d2']))
        self.assertEqual(set(read_index(self.directory, self.questions)), {'q1', 'q2'})

    def test_truncated_row_is_rescored_without_parsing_all_rows(self):
        prepare_resume(self.directory, self.signature, self.items)
        checkpoint_row(self.directory, row_for('q1', 'câu một', ['d1']))
        path = next(self.directory.glob('*.json.gz'))
        path.write_bytes(path.read_bytes()[:20])
        self.assertEqual(read_index(self.directory, self.questions), {})

    def test_index_ignores_rows_whose_file_vanished(self):
        prepare_resume(self.directory, self.signature, self.items)
        checkpoint_row(self.directory, row_for('q1', 'câu một', ['d1']))
        next(self.directory.glob('*.json.gz')).unlink()
        self.assertEqual(read_index(self.directory, self.questions), {})

    def test_rescoring_supersedes_the_earlier_index_line(self):
        prepare_resume(self.directory, self.signature, self.items)
        checkpoint_row(self.directory, row_for('q1', 'câu một', ['d1']))
        checkpoint_row(self.directory, row_for('q1', 'câu một', ['d1'], {'seconds': 9.0}))
        done = read_index(self.directory, self.questions)
        self.assertEqual(done['q1'][1], 9.0)
        self.assertEqual(len(done), 1)

    def test_committed_rows_concatenate_into_one_readable_stream(self):
        prepare_resume(self.directory, self.signature, self.items)
        for qid, record in self.items:
            checkpoint_row(self.directory, row_for(qid, record['question'], record['answer']))
        done = read_index(self.directory, self.questions)
        merged = self.root / 'train.jsonl.gz'
        with merged.open('wb') as stream:
            for qid, _ in self.items:
                with done[qid][0].open('rb') as member:
                    shutil.copyfileobj(member, stream)
        self.assertEqual([r['id'] for r in load_cache(merged)], ['q1', 'q2'])


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def write(self, rows):
        path = self.root / 'cache.jsonl.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        return path

    def test_provenance_is_dropped_but_every_consumed_key_survives(self):
        bulky = {'bundles': [{'input_ids': list(range(2048))}] * 50,
                 'baseline_passages': {'d1': ['x' * 4000]},
                 'passage_scores': [{'d1': [0.1, 0.2, 0.3]}], 'fused': [['d1', 0.9]],
                 'channels': {'dense': {}}}
        path = self.write([row_for('q1', 'câu', ['d1'], bulky)])
        [row] = load_cache(path)
        for key in ('id', 'question', 'gold', 'docs', 'features', 'bundle_scores',
                    'frozen_prediction'):
            self.assertIn(key, row)
        for key in bulky:
            self.assertNotIn(key, row)

    def test_streaming_holds_one_row_at_a_time(self):
        path = self.write([row_for(f'q{i}', 'câu', ['d1']) for i in range(5)])
        generator = stream_cache(path)
        self.assertEqual(next(generator)['id'], 'q0')
        self.assertEqual(next(generator)['id'], 'q1')
        generator.close()

    def test_blank_lines_are_skipped(self):
        path = self.root / 'cache.jsonl.gz'
        body = json.dumps(row_for('q1', 'câu', ['d1'])) + '\n\n'
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.write(body)
        self.assertEqual(len(load_cache(path)), 1)

    def test_analytic_keys_cover_what_the_stages_read(self):
        for key in ('id', 'question', 'gold', 'docs', 'features', 'bundle_scores',
                    'frozen_prediction'):
            self.assertIn(key, ANALYTIC_KEYS)

    def test_canonical_is_idempotent_through_json(self):
        value = {'items': [('q1', {'answer': ['d1']})]}
        once = canonical(value)
        self.assertEqual(once, json.loads(json.dumps(once)))
        self.assertNotEqual(once, value)


if __name__ == '__main__':
    unittest.main()
