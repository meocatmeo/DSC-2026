import json
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

from legalir.data import chunks, connect, load_questions, prepare, read_json, split_questions, write_json
from legalir.retrieval import Retriever, document_ranking, fuse
from legalir.submission import evaluate, pack, validate


class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        documents = [
            ("1", "Đăng ký xe", "Thời hạn cấp đăng ký xe máy cho người nước ngoài là hai ngày."),
            ("2", "Thuế", "Nộp thuế thu nhập cá nhân theo quy định của pháp luật."),
            ("3", "", ""),
            ("4", "Bảo hiểm", "Người lao động được đóng bảo hiểm xã hội."),
            ("5", "Đất", "Quy hoạch sử dụng đất nông nghiệp."),
            ("6", "Thanh tra", "Quyết định thanh tra chuyên ngành."),
        ]
        for key, name, passage in documents:
            write_json(self.corpus / f"context_{key}.json", {"id": int(key), "name": name, "passage": passage})
        self.database = self.root / "corpus.sqlite"
        self.report = prepare(self.corpus, self.database, size=12, overlap=3)
        self.questions = {"q1": {"question": "Thời hạn đăng ký xe máy", "answer": ["1"]},
                          "q2": {"question": "thuế thu nhập cá nhân", "answer": ["2"]}}

    def tearDown(self):
        self.temporary.cleanup()

    def test_chunk_coverage_and_overlap(self):
        self.assertEqual(list(chunks("a b c d e", 3, 1)), ["a b c", "c d e"])
        self.assertEqual(list(chunks("", 3, 1)), [])
        with self.assertRaises(ValueError):
            list(chunks("abc", 3, 3))

    def test_corpus_and_retrieval(self):
        self.assertEqual(self.report["documents"], 6)
        self.assertEqual(self.report["empty_passage_ids"], ["3"])
        retriever = Retriever(self.database)
        try:
            answers = retriever.search(self.questions["q1"]["question"])
            self.assertEqual(answers[0], "1")
            self.assertEqual(len(answers), 5)
            self.assertEqual(len(set(answers)), 5)
            self.assertEqual(retriever.search(""), retriever.search(""))
            self.assertEqual(len(retriever.search('" OR * NOT: []')), 5)
        finally:
            retriever.close()

    def test_fusion_deduplicates_documents_before_ranking(self):
        self.assertEqual(document_ranking([(1, 1), (2, .9), (3, .8)], {1: "a", 2: "a", 3: "b"}),
                         [("a", 1), ("b", .8)])
        ranked = fuse([[("a", 1), ("b", .5)], [("b", 1)]])
        self.assertEqual(ranked[0][0], "b")

    def test_metrics_and_overfive_penalty(self):
        truth = {"a": ["1", "2"], "b": ["3"], "c": ["4"]}
        pred = {"a": {"answer": ["1", "5"]}, "b": {"answer": []},
                "c": {"answer": ["1", "2", "3", "4", "5", "6"]}}
        scores = evaluate(pred, truth)
        self.assertAlmostEqual(scores["recall"], 1 / 6)
        self.assertAlmostEqual(scores["precision"], 1 / 6)
        with self.assertRaises(ValueError):
            evaluate({}, truth)

    def test_submission_rejections(self):
        ids = {"1", "2", "3", "4", "5", "6"}
        invalid = [{}, {"answer": "1"}, {"answer": [1]}, {"answer": ["1", "1"]},
                   {"answer": ["999"]}, {"answer": list(ids)}]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate({"q1": row, "q2": {"answer": ["2"]}}, self.questions, ids)
        with self.assertRaises(ValueError):
            validate({"q1": {"answer": ["1"]}}, self.questions, ids)
        with self.assertRaises(ValueError):
            validate([], self.questions, ids)

    def test_duplicate_json_keys(self):
        path = self.root / "bad.json"
        path.write_text('{"q1":{},"q1":{}}', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_json(path)

    def test_pack_end_to_end(self):
        engine = Retriever(self.database)
        try:
            predictions = {key: {"answer": engine.search(value["question"], 1)}
                           for key, value in self.questions.items()}
        finally:
            engine.close()
        self.assertEqual(evaluate(predictions, self.questions)["recall"], 1)
        source = self.root / "submission.json"
        destination = self.root / "submission.zip"
        write_json(source, predictions)
        pack(source, destination, self.questions, {str(i) for i in range(1, 7)})
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual(archive.namelist(), ["submission.json"])
            self.assertEqual(json.loads(archive.read("submission.json").decode("utf-8")), predictions)

    def test_split_is_reproducible_and_groups_duplicates(self):
        records = dict(self.questions)
        records["q3"] = {"question": "  THUẾ thu nhập cá nhân ", "answer": ["2"]}
        source = self.root / "train.json"
        write_json(source, records)
        split_questions(source, self.root / "a", fraction=.5)
        split_questions(source, self.root / "b", fraction=.5)
        a = read_json(self.root / "a/validation.json")
        self.assertEqual(a, read_json(self.root / "b/validation.json"))
        self.assertEqual("q2" in a, "q3" in a)
        train = read_json(self.root / "a/train.json")
        self.assertFalse(set(train) & set(a))
        self.assertEqual(set(train) | set(a), set(records))

    def test_public_labels_cannot_be_evaluated(self):
        with self.assertRaises(ValueError):
            evaluate({"q": {"answer": ["1"]}}, {"q": {"answer": None}})

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Official scorer requires NumPy")
    def test_official_scorer_parity(self):
        import scoring
        truth = {"a": ["1", "2"], "b": ["3"], "c": ["4"]}
        predicted = {"a": {"answer": ["1", "5"]}, "b": {"answer": []},
                     "c": {"answer": ["1", "2", "3", "4", "5", "6"]}}
        official = scoring.eval_retrieval(predicted, truth)
        actual = evaluate(predicted, truth)
        for metric in ("precision", "recall"):
            self.assertAlmostEqual(official[metric], actual[metric])


if __name__ == "__main__":
    unittest.main()