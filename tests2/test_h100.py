"""CPU checks for the rental runner's checkpoints, promotion and submission."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from legalir2 import h100
from legalir2.models import attention_for
from legalir2.stack import feature_names
from legalir2.store import file_hash, read_json, write_json


class TestH100(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.args = SimpleNamespace(out=root / "out", baseline=root / "baseline",
                                    index=root / "index", public=root / "public.json",
                                    train=root / "train.json", validation=root / "val.json",
                                    limit=10, shard_size=2)
        write_json(self.args.out / "inputs.json", {"code": "fixture"})
        self.runtime = {"attention": "sdpa", "batch": 8,
                        "input_digest": h100.digest(read_json(self.args.out / "inputs.json"))}
        self.records = {f"q{i}": {"question": f"Question {i}", "answer": ["b"]}
                        for i in range(5)}

    def row(self, model, qid, record, mode):
        matrix = np.zeros((2, 20))
        matrix[1, 0] = 1
        row = {"id": qid, "question": record["question"], "gold": record.get("answer", []),
               "docs": ["a", "b"], "candidate_ranks": {"b": 2},
               "features2": matrix.tolist()}
        if mode == "retrieval":
            row["features1"] = matrix.tolist()
        return row

    def test_resume_skips_valid_shards_and_repairs_corruption(self):
        model = Mock()
        with patch.object(h100, "pipeline", return_value=model) as load, \
                patch.object(h100, "score_row", side_effect=self.row) as score:
            original = h100.collect(self.args, self.records, "train", "retrieval", self.runtime)
            self.assertEqual(score.call_count, 5)
            model.close.assert_called_once()
            score.reset_mock()
            load.reset_mock()
            self.assertEqual(h100.collect(self.args, self.records, "train", "retrieval",
                                          self.runtime), original)
            load.assert_not_called()
            shard = self.args.out / "retrieval/train/part-00001.json"
            shard.write_text("{interrupted", encoding="utf-8")
            self.assertEqual(h100.collect(self.args, self.records, "train", "retrieval",
                                          self.runtime), original)
            self.assertEqual(score.call_count, 2)
        model.close.assert_called_once()

    def test_interruption_preserves_only_committed_shards(self):
        def interrupted(model, qid, record, mode):
            if qid == "q3":
                raise KeyboardInterrupt
            return self.row(model, qid, record, mode)
        model = Mock()
        with patch.object(h100, "pipeline", return_value=model), \
                patch.object(h100, "score_row", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                h100.collect(self.args, self.records, "train", "document", self.runtime)
        self.assertEqual(len(list((self.args.out / "document/train").glob("part-*.json"))), 1)
        model.close.assert_called_once()
        with patch.object(h100, "pipeline", return_value=Mock()), \
                patch.object(h100, "score_row", side_effect=self.row) as score:
            rows = h100.collect(self.args, self.records, "train", "document", self.runtime)
        self.assertEqual(len(rows), 5)
        self.assertEqual(score.call_count, 3)

    def test_resume_refuses_changed_runtime_or_unguarded_shards(self):
        with patch.object(h100, "pipeline", return_value=Mock()), \
                patch.object(h100, "score_row", side_effect=self.row):
            h100.collect(self.args, self.records, "train", "retrieval", self.runtime)
        with self.assertRaisesRegex(ValueError, "Inputs changed"):
            h100.collect(self.args, self.records, "train", "retrieval", dict(self.runtime, batch=32))
        (self.args.out / "retrieval/train/resume.json").unlink()
        with self.assertRaisesRegex(ValueError, "no input guard"):
            h100.collect(self.args, self.records, "train", "retrieval", self.runtime)

    def test_bad_feature_width_and_nonfinite_shards_are_rejected(self):
        for matrix in ([[0.0] * 19] * 2, [[float("nan")] * 20] * 2):
            row = self.row(None, "q0", self.records["q0"], "document")
            row["features2"] = matrix
            with self.assertRaisesRegex(ValueError, "features"):
                h100.validate_rows([row], [("q0", self.records["q0"])], "document")

    def test_calibration_excludes_heldout_ids_text_and_duplicate_text(self):
        write_json(self.args.train, {"held": {"question": "different", "answer": ["a"]},
                                    "dup": {"question": "Held out?", "answer": ["a"]},
                                    "yes": {"question": "unique", "answer": ["b"]},
                                    "again": {"question": "unique", "answer": ["b"]}})
        write_json(self.args.validation, {"held": {"question": "Held out?", "answer": ["a"]}})
        train, _ = h100.questions(self.args)
        self.assertEqual(list(train), ["yes"])

    def test_second_reranker_receives_first_reranker_order(self):
        model = Mock()
        model.settings = {"candidates": 2, "aggregate": "max", "aggregate_weight": .5}
        model.rerankers = ["first", "second"]
        model.candidates.return_value = ([("a", 1), ("b", .5)], {})
        model._rerank.side_effect = [{"a": [0.], "b": [2.]}, {"b": [3.], "a": [1.]}]
        h100.score_row(model, "q0", self.records["q0"], "retrieval")
        self.assertEqual(model._rerank.call_args_list[0].args[2], ["a", "b"])
        self.assertEqual(model._rerank.call_args_list[1].args[2], ["b", "a"])
        self.assertIs(model._rerank.call_args_list[0].args[4], model._rerank.call_args_list[1].args[4])

    def comparisons(self, lower=-.001):
        return {f"{arm}_vs_{base}": {"ci": [lower, .01]}
                for arm in ("ranks", "document") for base in ("control", "frozen")}

    def test_promotion_needs_both_controls_and_positive_lower_bound(self):
        scores = {"frozen": .95, "control": .95, "ranks": .96, "document": .97}
        checks = self.comparisons()
        self.assertEqual(h100.choose(checks, scores), "frozen")
        checks["ranks_vs_frozen"]["ci"][0] = .001
        self.assertEqual(h100.choose(checks, scores), "frozen")
        checks["ranks_vs_control"]["ci"][0] = 0
        self.assertEqual(h100.choose(checks, scores), "frozen")
        checks["ranks_vs_control"]["ci"][0] = .001
        self.assertEqual(h100.choose(checks, scores), "ranks")
        self.assertEqual(h100.choose(self.comparisons(.001), scores), "document")

    def selection(self, winner):
        write_json(self.args.out / "runtime.json", self.runtime)
        write_json(self.args.public, {"q0": {"question": "Question 0"}})
        write_json(self.args.baseline / "submission-v9.json", {"q0": {"answer": ["a"]}})
        for name in ("control", "ranks", "document"):
            write_json(self.args.out / f"stacker-{name}.json",
                       {"weights": [1.] + [0.] * 19, "features": feature_names(2),
                        "feature_version": 1 if name == "control" else 2})
        write_json(self.args.out / "comparison.json", {
            "recall": {"frozen": .95, "control": .95, "ranks": .96, "document": .97},
            "comparisons": self.comparisons(.001 if winner == "document" else -.001)})
        write_json(self.args.out / "selection.json", {
            "selected": winner, "input_digest": self.runtime["input_digest"],
            "runtime_digest": h100.digest(self.runtime),
            "comparison_sha256": file_hash(self.args.out / "comparison.json"),
            "stackers": {name: file_hash(self.args.out / f"stacker-{name}.json")
                         for name in ("control", "ranks", "document")}})

    def test_submit_fallback_keeps_v9_and_validates_real_zip_without_gpu(self):
        self.selection("frozen")
        with patch.object(h100, "Store", return_value=Mock(doc_ids=["a", "b"])), \
                patch.object(h100, "collect") as collect:
            h100.submit(self.args)
        collect.assert_not_called()
        self.assertEqual(read_json(self.args.out / "submission.json"),
                         read_json(self.args.baseline / "submission-v9.json"))
        with zipfile.ZipFile(self.args.out / "submission.zip") as archive:
            self.assertEqual(archive.namelist(), ["submission.json"])
            self.assertIsNone(archive.testzip())

    def test_submit_qualified_treatment_scores_public_and_checks_weights(self):
        self.selection("document")
        row = self.row(None, "q0", {"question": "Question 0"}, "document")
        with patch.object(h100, "Store", return_value=Mock(doc_ids=["a", "b"])), \
                patch.object(h100, "collect", return_value=[row]) as collect:
            h100.submit(self.args)
        self.assertEqual(collect.call_args.args[2:4], ("public", "document"))
        self.assertEqual(read_json(self.args.out / "submission.json")["q0"]["answer"], ["b", "a"])
        write_json(self.args.out / "stacker-document.json", {"weights": []})
        with self.assertRaisesRegex(ValueError, "weights changed"):
            h100.submit(self.args)

    def test_submit_rejects_changed_selection_evidence(self):
        self.selection("frozen")
        write_json(self.args.out / "comparison.json", {})
        with self.assertRaisesRegex(ValueError, "selection changed"):
            h100.submit(self.args)

    def test_baseline_drift_stops_before_training_pass(self):
        write_json(self.args.out / "runtime.json", self.runtime)
        write_json(self.args.train, self.records)
        validation = {"held": {"question": "held out", "answer": ["missing"]}}
        write_json(self.args.validation, validation)
        write_json(self.args.baseline / "v9-stacker-ens.json", {"weights": [0.] * 20})
        write_json(self.args.baseline / "v9-ensemble.diagnostics.json", {
            "held": {"relevant": ["missing"], "predicted": ["missing"],
                     "candidate_ranks": {"missing": 1}}})
        self.args.baseline_tolerance = .002
        row = self.row(None, "held", validation["held"], "retrieval")
        row["candidate_ranks"] = {"missing": None}
        with patch.object(h100, "collect", return_value=[row]) as collect, \
                patch.object(h100, "fit_arm") as fit:
            with self.assertRaisesRegex(ValueError, "Baseline drift"):
                h100.run(self.args)
        self.assertEqual(collect.call_count, 1)
        fit.assert_not_called()

    def test_flash_attention_is_only_routed_to_causal_cuda(self):
        with patch.dict(os.environ, {"LEGALIR_CAUSAL_ATTN": "flash_attention_2"}):
            self.assertEqual(attention_for("causal", "cuda"), "flash_attention_2")
            for kind, device in (("encoder", "cuda"), ("sequence", "cuda"), ("causal", "cpu")):
                self.assertEqual(attention_for(kind, device), "sdpa")
        with patch.dict(os.environ, {"LEGALIR_CAUSAL_ATTN": "typo"}):
            with self.assertRaises(ValueError):
                attention_for("causal", "cuda")


if __name__ == "__main__":
    unittest.main()
