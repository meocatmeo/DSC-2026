"""Regression coverage for missing channel ranks and document-local evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from legalir2.lexical import DEFAULTS as LEXICAL_DEFAULTS, Lexical
from legalir2.pipeline import DEFAULTS, Pipeline
from legalir2.stack import FEATURES, FEATURE_VERSION, Stacker, build_features, train


class TestMissingRanks(unittest.TestCase):
    def setUp(self):
        self.docs = ["missing", "deep", "first"]
        ranking = [("first", 5.0)] + [(f"other{i}", 1.0) for i in range(98)]
        ranking.append(("deep", 0.5))
        self.channels = {name: ranking for name in ("lexical", "dense", "fused")}
        self.tables = [{d: [1.0] for d in self.docs}]

    def test_absence_never_outranks_a_deep_real_result(self):
        matrix = build_features(self.docs, self.tables, self.channels)
        for name in ("lex_rr", "den_rr", "fused_rr"):
            column = matrix[:, FEATURES.index(name)]
            self.assertEqual(column[0], 0.0)
            self.assertGreater(column[1], column[0])
            self.assertGreater(column[2], column[1])
        self.assertTrue(np.isfinite(matrix).all())

    def test_empty_channels_produce_finite_zero_rank_features(self):
        matrix = build_features(self.docs, self.tables, {})
        for name in ("lex_rr", "den_rr", "fused_rr", "lex_den_gap"):
            np.testing.assert_array_equal(matrix[:, FEATURES.index(name)], 0.0)
        self.assertTrue(np.isfinite(matrix).all())

    def test_loading_old_weights_preserves_the_baseline_order(self):
        weights = np.zeros(len(FEATURES))
        weights[FEATURES.index("lex_rr")] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stacker.json"
            record = {"weights": weights.tolist(), "features": list(FEATURES)}
            path.write_text(json.dumps(record), encoding="utf-8")
            legacy = Stacker.load(path)
            self.assertEqual(legacy.version, 1)
            self.assertEqual(legacy.order(self.docs, self.tables, self.channels),
                             ["first", "missing", "deep"])
            record["feature_version"] = FEATURE_VERSION
            path.write_text(json.dumps(record), encoding="utf-8")
            corrected = Stacker.load(path)
            self.assertEqual(corrected.order(self.docs, self.tables, self.channels),
                             ["first", "deep", "missing"])

    def test_training_retains_cache_version_and_rejects_mixed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, output = [Path(directory) / name for name in ("cache.jsonl", "stacker.json")]
            row = {"labels": [0, 1, 0]}
            for version in (1, FEATURE_VERSION):
                row["features"] = build_features(self.docs, self.tables, self.channels,
                                                 version=version).tolist()
                if version == FEATURE_VERSION:
                    row["feature_version"] = version
                cache.write_text(json.dumps(row) + "\n", encoding="utf-8")
                train(cache, output, epochs=2, verbose=False)
                self.assertEqual(Stacker.load(output).version, version)
            other = dict(row, feature_version=1)
            cache.write_text(json.dumps(row) + "\n" + json.dumps(other), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mixed or unsupported"):
                train(cache, output, epochs=2, verbose=False)

    def test_unknown_feature_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "feature version"):
            Stacker(np.zeros(len(FEATURES)), version=999)
        with self.assertRaisesRegex(ValueError, "feature version"):
            build_features(self.docs, self.tables, self.channels, version=999)


class TestDocumentEvidence(unittest.TestCase):
    def pipeline(self, mode="document", dense=True):
        # A's first chunk wins lexical retrieval. B's header wins dense retrieval.
        # B's answering article is below BOTH global cutoffs, despite being its
        # best local combination. Different index orders catch positional joins.
        chunks = {"a": [1, 2, 3], "b": [10, 11, 12]}
        texts = {1: "a header", 2: "a unrelated", 3: "a other",
                 10: "b header", 11: "b background", 12: "b answering article"}
        holder = SimpleNamespace(doc_ids=["a", "b"],
                                 document_chunk_ids=chunks.get,
                                 chunk_documents=lambda: {c: d for d, ids in chunks.items() for c in ids},
                                 chunk_text=lambda c, with_label=True: texts[c], close=lambda: None)
        lexical = Lexical.__new__(Lexical)
        lexical.settings = dict(LEXICAL_DEFAULTS)
        lexical.unigram = SimpleNamespace(chunk_ids=np.array([1, 2, 3, 10, 11, 12]),
                                          chunks=6, doc_index=np.array([0, 0, 0, 1, 1, 1]),
                                          doc_ids=["a", "b"])
        lexical.chunk_scores = Mock(return_value=np.array([100, 0, 0, 0, 5, 8], dtype=float))
        dense_index = SimpleNamespace(chunk_ids=np.array([11, 3, 10, 1, 12, 2]),
                                      scores=Mock(return_value=np.array([.4, .01, 1, .8, .9, .02])),
                                      parameters=1, close=lambda: None)
        with patch("legalir2.pipeline.Store", return_value=holder), \
             patch("legalir2.pipeline.Lexical", return_value=lexical), \
             patch("legalir2.dense.DenseIndex", return_value=dense_index):
            pipeline = Pipeline("unused", "unused", dense_directory="unused" if dense else None,
                                settings={"chunk_candidates": 1, "passages": 1,
                                          "passage_selection": mode})
        self.addCleanup(pipeline.close)
        return pipeline

    def test_local_selection_recovers_evidence_without_changing_candidates(self):
        old, new = self.pipeline("retrieval"), self.pipeline()
        old_channels, new_channels = {}, {}
        before, old_prior = old.candidates("question", old_channels)
        after, new_prior = new.candidates("question", new_channels)
        self.assertEqual(before, after)
        self.assertEqual(old_channels, new_channels)
        self.assertEqual(old._passages("b", old_prior), ["b header"])
        self.assertEqual(new._passages("b", new_prior), ["b answering article"])
        new.lexical.chunk_scores.assert_called_once_with("question")
        new.dense.scores.assert_called_once_with("question")

    def test_both_rerankers_receive_the_new_evidence_and_same_token_budget(self):
        pipeline = self.pipeline()
        models = [SimpleNamespace(score=Mock(side_effect=lambda q, texts, batch, length:
                                             [9.0 if "answering" in t else 0.0 for t in texts]),
                                  close=lambda: None) for _ in range(2)]
        pipeline.rerankers = models
        self.assertEqual(pipeline.search("question", top_k=1), ["b"])
        for model in models:
            args = model.score.call_args.args
            self.assertIn("b answering article", args[1])
            self.assertNotIn("b header", args[1])
            self.assertEqual(args[3], DEFAULTS["rerank_max_length"])

    def test_each_query_gets_fresh_scores(self):
        pipeline = self.pipeline(dense=False)
        _, first = pipeline.candidates("first")
        self.assertEqual(pipeline._passages("b", first), ["b answering article"])
        pipeline.lexical.chunk_scores.return_value = np.array([100, 0, 0, 0, 8, 5])
        _, second = pipeline.candidates("second")
        self.assertEqual(pipeline._passages("b", second), ["b background"])
        self.assertEqual(pipeline.lexical.chunk_scores.call_count, 2)

    def test_no_matching_evidence_has_deterministic_fallback(self):
        pipeline = self.pipeline(dense=False)
        pipeline.lexical.chunk_scores.return_value = np.zeros(6)
        candidates, prior = pipeline.candidates("")
        self.assertEqual([d for d, _ in candidates], ["a", "b"])
        self.assertEqual(pipeline._passages("b", prior), ["b header"])

    def test_duplicate_windows_do_not_hide_an_article_beyond_the_old_pool(self):
        pipeline = self.pipeline()
        pipeline.store.document_chunk_ids = lambda doc: list(range(1, 13))
        pipeline.store.chunk_text = lambda c, with_label=True: (
            "repeated article text" if c <= 10 else f"distinct provision {c}")
        prior = {c: 1.0 / c for c in range(1, 13)}
        self.assertEqual(pipeline._passages("b", prior, 3),
                         ["repeated article text", "distinct provision 11", "distinct provision 12"])
        pipeline.settings["passage_selection"] = "retrieval"
        self.assertEqual(pipeline._passages("b", prior, 3), ["repeated article text"] * 3)

    def test_disabled_channel_does_not_select_passages(self):
        pipeline = self.pipeline()
        pipeline.settings["lexical_weight"] = 0.0
        _, prior = pipeline.candidates("question")
        self.assertEqual(pipeline._passages("b", prior), ["b header"])

    def test_invalid_selection_mode_fails_before_loading_models(self):
        with self.assertRaisesRegex(ValueError, "passage_selection"):
            Pipeline("unused", "unused", settings={"passage_selection": "typo"})


if __name__ == "__main__":
    unittest.main()
