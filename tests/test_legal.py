import unittest

from legalir.legal import diverse_chunks, folded_terms, phrase_evidence


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


if __name__ == "__main__":
    unittest.main()