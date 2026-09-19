"""An explicit user override changes promotion only, never checkpoint identity."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from legalir2 import duo, duo_submit
from legalir2.store import file_hash, read_json, write_json


class TestApprovedSubmit(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(out=self.root, index=self.root, public=self.root / "public.json",
                                    validation=self.root / "validation.json")
        write_json(self.root / "inputs.json", {})
        write_json(self.root / "model/duo.json", {})
        write_json(self.args.validation, {"q": {"question": "q", "answer": ["f"]}})
        choice = {"model": file_hash(self.root / "model/duo.json"), "inputs": duo.digest({}),
                  "chosen": {"strength": .25}}
        write_json(self.root / "calibration.json", choice)
        self.selection = {"promoted": False, "strength": .25,
                          "calibration": file_hash(self.root / "calibration.json"),
                          "comparison": {"questions": 1, "delta": .0022, "ci": [-.0019, .0068]}}
        write_json(self.root / "selection.json", self.selection)

    def test_inconclusive_submission_keeps_original_report_and_records_approval(self):
        before = (self.root / "selection.json").read_bytes()
        rows = [{"id": "q", "question": "q", "gold": [], "docs": list("abcdef"),
                 "prior": [0.] * 6, "ability": [0., 0., 0., 0., 0., 10.], "baseline": list("abcde")}]
        with patch.object(duo, "collect", return_value=rows) as collect, patch.object(duo_submit, "Store"), \
                patch.object(duo, "pack", side_effect=lambda source, store, public, target: target.write_bytes(b"zip fixture")):
            duo_submit.submit_approved(self.args)
        collect.assert_called_once_with(self.args, "public")
        self.assertEqual((self.root / "selection.json").read_bytes(), before)
        self.assertEqual(read_json(self.root / "submission-duo.json")["q"]["answer"][0], "f")
        self.assertFalse(read_json(self.root / "submission-approval.json")["automatic_promotion"])
        self.assertEqual(read_json(self.root / "submission-complete.json")["submission_sha256"],
                         file_hash(self.root / "submission-duo.zip"))

    def test_override_still_rejects_changed_model(self):
        write_json(self.root / "model/duo.json", {"changed": True})
        with patch.object(duo, "collect") as collect, self.assertRaisesRegex(ValueError, "checkpoint differ"):
            duo_submit.submit_approved(self.args)
        collect.assert_not_called()

    def test_override_cannot_change_calibrated_strength_or_skip_validation(self):
        for modification in ({"strength": .5}, {"calibration": "changed"},
                             {"comparison": {"questions": 1, "delta": -.01}},
                             {"comparison": {"questions": 0, "delta": .01}}):
            write_json(self.root / "selection.json", dict(self.selection, **modification))
            with patch.object(duo, "collect") as collect, self.assertRaises(ValueError):
                duo_submit.submit_approved(self.args)
            collect.assert_not_called()

    def test_entry_point_uses_frozen_guards_and_restores_handler_on_failure(self):
        original = duo.submit
        def guarded_main():
            self.assertIs(duo.submit, duo_submit.submit_approved)
            raise ValueError("frozen guard failed")
        with patch("sys.argv", ["duo_submit", "submit"]), patch.object(duo, "main", side_effect=guarded_main), \
                self.assertRaisesRegex(ValueError, "frozen guard failed"):
            duo_submit.main()
        self.assertIs(duo.submit, original)


class TestUnlabelledPublic(unittest.TestCase):
    def test_public_null_answers_score_resume_and_pack_without_changing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(out=root / "out", index=root / "index", baseline=root / "baseline",
                                   public=root / "public.json", train=root / "train.json",
                                   validation=root / "validation.json", infer_batch=8, device="cpu")
            docs = list("abcdef")
            records = {"null": {"question": "first", "answer": None},
                       "absent": {"question": "second"}, "empty": {"question": "third", "answer": []}}
            signature = {"attention": "sdpa", "weights": [1.] + [0.] * 19,
                         "memory": {}, "known_docs": docs, "runtime": {"attention": "sdpa", "batch": 8}}
            write_json(args.out / "inputs.json", signature)
            write_json(args.out / "model/duo.json", {})
            write_json(args.public, records)
            write_json(args.train, {})
            write_json(args.validation, {})
            write_json(args.baseline / "v9-ensemble.run.json", {"audit": {"components": {"baseline": 100}}})
            public_hash = file_hash(args.public)
            # Reproduce the resume marker already created before the failed GPU call.
            marker = {"inputs": duo.digest(signature), "model": file_hash(args.out / "model/duo.json"),
                      "questions": duo.digest(records), "batch": 8, "shard_size": 10}
            duo.guard(args.out / "scores/public/resume.json", marker)
            marker_hash = file_hash(args.out / "scores/public/resume.json")
            first_stage = Mock()
            first_stage.settings = {"candidates": 6, "aggregate": "max", "aggregate_weight": .5}
            first_stage.rerankers = ["first", "second"]
            first_stage.candidates.return_value = ([(d, 1.) for d in docs], {})
            first_stage._rerank.side_effect = lambda *a: {d: [float(i)] for i, d in enumerate(docs)}
            local = Mock(parameters=100)
            memory = Mock()
            memory.search.return_value = ({}, [])
            original = duo.score_row
            with patch.object(duo, "verify_model", return_value={"max_length": 2048}), \
                    patch.object(duo, "Local", return_value=local) as load, \
                    patch.object(duo, "PairTokenizer"), patch.object(duo, "Evidence"), \
                    patch.object(duo, "QuestionMemory", return_value=memory), \
                    patch.object(duo, "pipeline", return_value=first_stage) as pipeline, \
                    patch.object(duo, "compare_documents", return_value=np.full((6, 6), .5)):
                rows = duo_submit.collect_public(args)
                self.assertIs(duo.score_row, original)
                self.assertEqual(len(rows), 3)
                self.assertIsNone(rows[0]["gold"])
                load.reset_mock()
                pipeline.reset_mock()
                self.assertEqual(duo_submit.collect_public(args), rows)
                load.assert_not_called()
                pipeline.assert_not_called()
            self.assertEqual(file_hash(args.public), public_hash)
            self.assertEqual(file_hash(args.out / "scores/public/resume.json"), marker_hash)
            predictions = {q: {"answer": r["predicted"]} for q, r in duo.diagnostics(rows, .25).items()}
            path = args.out / "submission.json"
            write_json(path, predictions)
            target = args.out / "submission.zip"
            duo.pack(path, SimpleNamespace(doc_ids=docs), args.public, target)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.namelist(), ["submission.json"])

    def test_failed_public_scoring_restores_frozen_handler(self):
        original = duo.score_row
        with patch.object(duo, "collect", side_effect=RuntimeError("interrupted")), \
                self.assertRaisesRegex(RuntimeError, "interrupted"):
            duo_submit.collect_public(None)
        self.assertIs(duo.score_row, original)


if __name__ == "__main__":
    unittest.main()
