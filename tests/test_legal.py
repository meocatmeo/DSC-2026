import unittest

from legalir.legal import context_window, diverse_chunks, folded_terms, phrase_evidence


class LegalEvidenceTests(unittest.TestCase):
    def test_folded_vietnamese_titles(self):
        self.assertEqual(folded_terms("ĐỐI NGOẠI quốc phòng"), folded_terms("Doi-ngoai-quoc-phong"))

    def test_contiguous_concept_beats_scattered_words(self):
        scores = phrase_evidence("đối ngoại quốc phòng", ["Luật quốc phòng: đối ngoại quốc phòng",
                                                          "đối tượng ngoại lệ quốc gia phòng ban"])
        self.assertGreater(scores[0], scores[1])

    def test_empty_query(self):
        self.assertEqual(phrase_evidence("", ["abc", ""]), [0, 0])

    def test_passage_diversity(self):
        texts = {1: "thuế thu nhập cá nhân", 2: "thuế thu nhập cá nhân", 3: "hồ sơ đăng ký xe"}
        self.assertEqual(diverse_chunks([1, 2, 3], texts.__getitem__, 2), [1, 3])

    def test_context_reconstructs_overlap_and_single_title(self):
        texts = {10: "LAW a b c d", 11: "LAW c d e f", 12: "LAW e f g h", 13: "OTHER secret"}
        self.assertEqual(context_window([10, 11, 12], 11, texts.__getitem__, "LAW", 2),
                         "LAW a b c d e f g h")
        self.assertEqual(context_window([10, 11, 12], 12, texts.__getitem__, "LAW", 2),
                         "LAW c d e f g h")
        self.assertEqual(context_window([10], 10, texts.__getitem__, "LAW", 2), "LAW a b c d")

    def test_context_does_not_remove_unmatched_words(self):
        texts = {1: "a b", 2: "c d"}
        self.assertEqual(context_window([1, 2], 1, texts.__getitem__, "", 1), "a b c d")
        with self.assertRaises(ValueError):
            context_window([1, 2], 3, texts.__getitem__, "", 1)


if __name__ == "__main__":
    unittest.main()