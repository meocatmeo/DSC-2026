import unittest

from legalir2 import structure


HEADER = (
    "BỘ\n\nCÔNG THƯƠNG\n\n-------\n\nCỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n\n"
    "Độc lập - Tự do - Hạnh phúc\n\n------\n\nSố: 40/2010/TT-BCT\n\n"
    "Hà Nội, ngày 13 tháng 12 năm 2010\n\nTHÔNG TƯ\n\nQUY ĐỊNH VỀ TRÌNH TỰ GIẢI QUYẾT\n\n"
    "Căn cứ Luật Điện lực ngày 03 tháng 12 năm 2004;\n\n"
)
BODY = (
    "Chương I\n\nQUY ĐỊNH CHUNG\n\nĐiều 1. Phạm vi điều chỉnh\n\n"
    "Thông tư này quy định trình tự giải quyết tranh chấp.\n\n"
    "Điều 2. Đối tượng áp dụng\n\nThông tư này áp dụng với các đơn vị điện lực "
    "theo quy định tại Điều 1 của Thông tư này.\n\n"
    "Chương II\n\nTRÌNH TỰ\n\nĐiều 3. Thụ lý hồ sơ\n\nHồ sơ được thụ lý trong 05 ngày.\n\n"
)


class TestReflow(unittest.TestCase):
    def test_joins_wrapped_lines_without_losing_words(self):
        text = "Căn cứ Luật Điện lực ngày 03\n\ntháng 12 năm 2004;\n\nĐiều 1. Phạm vi\n\n"
        flowed = structure.reflow(text)
        self.assertIn("ngày 03 tháng 12 năm 2004;", flowed)
        self.assertTrue(any(line.startswith("Điều 1.") for line in flowed.split("\n")))

    def test_reflow_preserves_every_word(self):
        original = (HEADER + BODY).split()
        self.assertEqual(structure.reflow(HEADER + BODY).split(), original)


class TestHeader(unittest.TestCase):
    def test_document_number(self):
        self.assertEqual(structure.document_number(structure.reflow(HEADER)), "40/2010/TT-BCT")

    def test_document_number_keeps_vietnamese_letters(self):
        text = structure.reflow("Số: 1468/QĐ-BTP\n\nHà Nội, ngày 01 tháng 01 năm 2020\n\n")
        self.assertEqual(structure.document_number(text), "1468/QĐ-BTP")

    def test_subject_is_the_capitalised_title_not_the_boilerplate(self):
        lines = structure.reflow(HEADER).split("\n")
        subject = structure.subject_line(lines)
        self.assertIn("THÔNG TƯ", subject)
        self.assertIn("TRÌNH TỰ", subject)
        self.assertNotIn("CỘNG HÒA", subject)


class TestUnits(unittest.TestCase):
    def test_articles_become_separate_units_with_breadcrumbs(self):
        units = structure.parse_units(structure.reflow(HEADER + BODY))
        bodies = [body for _, body in units]
        self.assertEqual(len(units), 3)
        self.assertTrue(bodies[0].startswith("Điều 1."))
        self.assertTrue(bodies[2].startswith("Điều 3."))
        self.assertIn("Chương I", units[0][0])
        self.assertIn("Chương Ii", units[2][0].title())

    def test_in_text_article_reference_does_not_start_a_unit(self):
        units = structure.parse_units(structure.reflow(HEADER + BODY))
        self.assertIn("theo quy định tại Điều 1", units[1][1])

    def test_unstructured_document_still_produces_a_unit(self):
        units = structure.parse_units("Một đoạn văn bản không có điều khoản nào cả.")
        self.assertEqual(len(units), 1)
        self.assertTrue(units[0][1])

    def test_roman_sections_used_when_there_are_no_articles(self):
        text = ("I. PHẠM VI ĐIỀU CHỈNH\n\nThông tư này hướng dẫn.\n\n"
                "II. ĐỐI TƯỢNG ÁP DỤNG\n\nÁp dụng với mọi tổ chức.\n\n")
        units = structure.parse_units(structure.reflow(text))
        self.assertEqual(len(units), 2)


class TestChunking(unittest.TestCase):
    def test_every_document_yields_a_header_chunk_first(self):
        chunks = structure.document_chunks("Thong-tu-40-2010-TT-BCT-abc-1", HEADER + BODY)
        self.assertEqual(chunks[0]["unit"], "header")
        self.assertTrue(all(c["unit"] == "article" for c in chunks[1:]))

    def test_empty_passage_still_indexes_the_title(self):
        chunks = structure.document_chunks("Quyet-dinh-1-QD-X-2", "")
        self.assertTrue(chunks)
        self.assertIn("Quyet dinh 1 QD X 2", structure.chunk_text(chunks[0]))

    def test_long_article_is_split_with_overlap(self):
        long_article = "Điều 9. Tiêu đề " + " ".join(f"từ{i}" for i in range(900))
        pieces = structure.split_long(long_article, size=320, overlap=64)
        self.assertGreater(len(pieces), 2)
        first, second = pieces[0].split(), pieces[1].split()
        self.assertEqual(first[-64:], second[:64])

    def test_chunk_text_carries_label_and_breadcrumb(self):
        chunks = structure.document_chunks("Thong-tu-40-2010-TT-BCT-abc-1", HEADER + BODY)
        text = structure.chunk_text(chunks[3])
        self.assertIn("40/2010/TT-BCT", text)
        self.assertIn("Điều 3.", text)


if __name__ == "__main__":
    unittest.main()
