"""Correctness checks for query memory and metric-weighted nonlinear ranking."""
import importlib.util
import unittest

import numpy as np

from legalir2.impact import (RecallObjective, context, evaluate, features, filtered_memory,
                             names, predicted)
from legalir2.question_memory import QuestionMemory


class TestRecallObjective(unittest.TestCase):
    def row(self, gold=("f",)):
        return {"id": "q", "question": "fixture", "docs": list("abcdef"), "gold": list(gold),
                "base": np.asarray([6., 5., 4., 3., 2., 1.]), "candidate_ranks": {}}

    def test_metric_gradient_promotes_missed_gold_and_demotes_top_five(self):
        row = self.row()
        gradient, hessian = RecallObjective([row], auxiliary=0)(row["base"])
        self.assertTrue((gradient[:5] > 0).all())
        self.assertLess(gradient[5], 0)
        self.assertAlmostEqual(gradient.sum(), 0)
        self.assertTrue((hessian > 0).all())

    def test_inside_top_five_swaps_have_zero_recall_weight(self):
        row = self.row(("a",))
        gradient, _ = RecallObjective([row], auxiliary=0)(row["base"])
        np.testing.assert_equal(gradient[1:5], 0)
        self.assertLess(gradient[0], 0)
        self.assertGreater(gradient[5], 0)

    def test_unretrieved_gold_still_counts_in_macro_denominator(self):
        one, two = self.row(), self.row(("f", "missing"))
        a = RecallObjective([one], auxiliary=0)(one["base"])
        b = RecallObjective([two], auxiliary=0)(two["base"])
        np.testing.assert_allclose(b, np.asarray(a) / 2)

    def test_no_positive_produces_no_training_signal(self):
        row = self.row(("missing",))
        gradient, hessian = RecallObjective([row])(row["base"])
        np.testing.assert_equal(gradient, 0)
        np.testing.assert_equal(hessian, 0)

    def test_query_groups_do_not_form_cross_query_pairs(self):
        first, second = self.row(), self.row(("a",))
        scores = np.concatenate((first["base"], second["base"]))
        joined = RecallObjective([first, second])(scores)
        separate = [RecallObjective([r])(r["base"]) for r in (first, second)]
        for i in (0, 1):
            np.testing.assert_allclose(joined[i], np.concatenate([r[i] for r in separate]))

    def test_prediction_ties_are_doc_id_stable(self):
        row = self.row()
        row["docs"] = list("fedcba")
        self.assertEqual(predicted(row, np.zeros(6)), list("abcde"))
        self.assertEqual(list(evaluate([row])), ["q"])

    def test_context_encodes_decision_boundary(self):
        values = np.asarray([6., 5., 4., 3., 2., 1.])
        result = context(values, list("abcdef"))
        self.assertEqual(result[4, 1], 0)
        self.assertEqual(result[5, 2], 0)
        self.assertEqual(result[0, 3], 1)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is optional")
class TestQuestionMemory(unittest.TestCase):
    def records(self):
        return {"a": {"question": "Bảo hiểm y tế", "answer": ["health"]},
                "copy": {"question": "  BẢO HIỂM Y TẾ?", "answer": ["health"]},
                "b": {"question": "Quy định bảo hiểm y tế", "answer": ["health", "rules"]},
                "c": {"question": "Thuế thu nhập doanh nghiệp", "answer": ["tax"]}}

    def test_self_and_duplicate_text_are_excluded(self):
        memory = QuestionMemory(self.records())
        evidence, ranking = memory.search("Bảo hiểm y tế", "a")
        self.assertTrue(ranking)
        self.assertEqual(evidence["health"][3], 1 / 32)
        self.assertEqual(evidence["rules"][3], 1 / 32)
        self.assertEqual(evidence["health"][1], evidence["rules"][1])

    def test_heldout_ids_and_text_cannot_enter_memory(self):
        held = {"a": {"question": "Bảo hiểm y tế"}}
        records = filtered_memory(self.records(), held, {"health", "rules", "tax"})
        self.assertEqual(set(records), {"b", "c"})
        self.assertEqual(filtered_memory(self.records(), {}, {"unknown"}), {})

    def test_unknown_terms_return_finite_zero_features(self):
        matrix, _ = QuestionMemory(self.records()).features("zzzxxx", ["missing"])
        np.testing.assert_equal(matrix, 0)

    def test_views_align_and_new_candidates_have_missing_neural_features(self):
        memory = QuestionMemory(self.records())
        matrix = np.zeros((6, 20))
        matrix[:, 0] = np.arange(6, 0, -1)
        row = {"id": "new", "question": "Mức đóng bảo hiểm y tế", "gold": [],
               "docs": list("abcdef"), "features1": matrix.tolist(),
               "features2": matrix.tolist(), "candidate_ranks": {}}
        result = features(row, row, np.asarray([1.] + [0.] * 19), memory,
                          set("abcdef") | {"health", "rules", "tax"})
        self.assertEqual(result["matrix"].shape[1], len(names()))
        self.assertIn("health", result["docs"])
        index = result["docs"].index("health")
        self.assertTrue(np.isnan(result["matrix"][index, :60]).all())
        self.assertEqual(result["matrix"][index, names().index("memory_only")], 1)
        self.assertEqual(predicted(result, result["base"]), list("abcde"))
        with self.assertRaisesRegex(ValueError, "same question"):
            features(row, dict(row, docs=list("fedcba")), np.zeros(20), memory, set("abcdef"))


if __name__ == "__main__":
    unittest.main()
