import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from legalir2 import fuse, lexical, predict, store
from legalir2.mine import jaccard, token_set, _quota
from legalir2.models import audit, pooling_for


DOCUMENTS = [
    {"id": 101, "name": "Nghi-dinh-110-2018-ND-CP-quan-ly-le-hoi-1",
     "passage": ("CHÍNH PHỦ\n\n-----\n\nSố: 110/2018/NĐ-CP\n\nHà Nội, ngày 29 tháng 8 năm 2018\n\n"
                 "NGHỊ ĐỊNH\n\nQUY ĐỊNH VỀ QUẢN LÝ VÀ TỔ CHỨC LỄ HỘI\n\n"
                 "Chương I\n\nQUY ĐỊNH CHUNG\n\n"
                 "Điều 1. Phạm vi điều chỉnh\n\nNghị định này quy định về quản lý và tổ chức lễ hội "
                 "trên lãnh thổ Việt Nam.\n\n"
                 "Điều 2. Nguyên tắc tổ chức lễ hội\n\nViệc tổ chức lễ hội phải bảo đảm an ninh "
                 "trật tự và an toàn cho người tham gia lễ hội.\n\n")},
    {"id": 102, "name": "Thong-tu-05-2020-TT-BYT-bao-hiem-y-te-2",
     "passage": ("BỘ Y TẾ\n\n-----\n\nSố: 05/2020/TT-BYT\n\nHà Nội, ngày 01 tháng 3 năm 2020\n\n"
                 "THÔNG TƯ\n\nHƯỚNG DẪN THANH TOÁN BẢO HIỂM Y TẾ\n\n"
                 "Điều 1. Đối tượng áp dụng\n\nThông tư này áp dụng đối với người tham gia bảo "
                 "hiểm y tế.\n\n"
                 "Điều 2. Mức thanh toán\n\nMức thanh toán bảo hiểm y tế được xác định theo giá "
                 "dịch vụ khám bệnh chữa bệnh.\n\n")},
    {"id": 103, "name": "Quyet-dinh-9-2019-QD-TTg-ho-tro-nong-nghiep-3", "passage": ""},
]


TOPICS = ["đất đai", "giao thông đường bộ", "thuế thu nhập cá nhân", "xây dựng",
          "giáo dục nghề nghiệp", "an toàn thực phẩm", "chứng khoán", "lao động"]


def filler(number):
    """A plausible extra document, so tests that need a real candidate pool have one."""
    topic = TOPICS[number % len(TOPICS)]
    return {"id": 200 + number,
            "name": f"Nghi-dinh-{number}-2021-ND-CP-quy-dinh-ve-{number}",
            "passage": (f"CHÍNH PHỦ\n\n-----\n\nSố: {number}/2021/NĐ-CP\n\n"
                        f"Hà Nội, ngày 01 tháng 6 năm 2021\n\nNGHỊ ĐỊNH\n\n"
                        f"QUY ĐỊNH CHI TIẾT VỀ {topic.upper()}\n\n"
                        f"Điều 1. Phạm vi điều chỉnh\n\nNghị định này quy định chi tiết "
                        f"về {topic} trên lãnh thổ Việt Nam.\n\n"
                        f"Điều 2. Đối tượng áp dụng\n\nNghị định này áp dụng đối với tổ "
                        f"chức, cá nhân có hoạt động liên quan đến {topic}.\n\n")}


def make_corpus(directory, extra=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = list(DOCUMENTS) + [filler(n) for n in range(extra)]
    for number, record in enumerate(records, 1):
        (directory / f"context_{number}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return directory


class TestStoreAndLexical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        corpus = make_corpus(root / "contexts")
        cls.report = store.build(corpus, root / "store.sqlite")
        lexical.build(root / "store.sqlite", root / "lexical")
        cls.store = store.Store(root / "store.sqlite")
        cls.lexical = lexical.Lexical(root / "lexical")

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.temporary.cleanup()

    def test_every_document_is_indexed_including_the_empty_one(self):
        self.assertEqual(self.report["documents"], 3)
        self.assertEqual(self.report["empty_passage_ids"], ["103"])
        self.assertEqual(len(self.store.doc_ids), 3)

    def test_document_number_is_extracted_into_the_label(self):
        self.assertIn("110/2018/NĐ-CP", self.store.label("101"))
        self.assertIn("QUẢN LÝ VÀ TỔ CHỨC LỄ HỘI", self.store.label("101"))

    def test_retrieval_finds_the_right_document(self):
        documents, chunks = self.lexical.search("Việc tổ chức lễ hội dựa trên nguyên tắc nào?")
        self.assertEqual(documents[0][0], "101")
        self.assertTrue(chunks)

    def test_second_topic_retrieves_the_other_document(self):
        documents, _ = self.lexical.search("Mức thanh toán bảo hiểm y tế được xác định thế nào?")
        self.assertEqual(documents[0][0], "102")

    def test_bigram_channel_changes_the_scores(self):
        query = "tổ chức lễ hội"
        with_phrase = self.lexical.chunk_scores(query)
        self.lexical.apply({"beta": 0.0})
        without = self.lexical.chunk_scores(query)
        self.lexical.apply({"beta": lexical.DEFAULTS["beta"]})
        self.assertFalse((with_phrase == without).all())

    def test_empty_query_returns_no_positive_scores(self):
        documents, chunks = self.lexical.search("")
        self.assertEqual(chunks, [])
        self.assertEqual(documents, [])

    def test_document_positions_are_keyed_by_whole_document_id(self):
        """Regression: building this map by iterating the wrong variable keyed it by the
        first character of each id, which made every membership check silently false and
        mined zero training groups instead of raising."""
        self.assertEqual(sorted(self.store.doc_position), ["101", "102", "103"])
        self.assertEqual(self.store.doc_position["102"], 1)
        for doc in self.store.doc_ids:
            self.assertIn(doc, self.store.doc_position)

    def test_chunk_ids_agree_between_store_and_index(self):
        mapping = self.store.chunk_documents()
        self.assertEqual(sorted(mapping), sorted(int(c) for c in self.lexical.unigram.chunk_ids))


class TestFusion(unittest.TestCase):
    def test_rrf_prefers_documents_ranked_well_by_both_channels(self):
        first = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
        second = [("b", 0.9), ("c", 0.8), ("a", 0.1)]
        self.assertEqual(fuse.rrf([first, second])[0][0], "b")

    def test_weights_let_one_channel_dominate(self):
        first = [("a", 1.0), ("b", 0.5)]
        second = [("b", 1.0), ("a", 0.5)]
        self.assertEqual(fuse.rrf([first, second], [5.0, 1.0])[0][0], "a")

    def test_normalised_fusion_is_scale_free(self):
        plain = fuse.normalised([[("a", 1.0), ("b", 0.0)]])
        scaled = fuse.normalised([[("a", 1000.0), ("b", 0.0)]])
        self.assertEqual(plain, scaled)

    def test_diverse_skips_near_duplicate_passages(self):
        texts = {1: "a b c d e f g h", 2: "a b c d e f g h", 3: "x y z w v u t s"}
        picked = fuse.diverse([1, 2, 3], texts.get, 2)
        self.assertEqual(picked, [1, 3])

    def test_context_radius_zero_returns_the_anchor_alone(self):
        self.assertEqual(fuse.with_neighbours([1, 2, 3], 2, {1: "p", 2: "q", 3: "r"}.get, 0), "q")

    def test_context_radius_joins_neighbours_in_document_order(self):
        joined = fuse.with_neighbours([1, 2, 3], 2, {1: "p", 2: "q", 3: "r"}.get, 1)
        self.assertEqual(joined, "p\nq\nr")


class TestMiningHelpers(unittest.TestCase):
    def test_jaccard_detects_near_duplicate_law_text(self):
        first = token_set("điều 1 phạm vi điều chỉnh nghị định này quy định")
        second = token_set("điều 1 phạm vi điều chỉnh nghị định này quy định")
        self.assertEqual(jaccard(first, second), 1.0)

    def test_quota_spreads_negatives_across_tiers(self):
        self.assertEqual(_quota(12, 3), [4, 4, 4])
        self.assertEqual(sum(_quota(13, 3)), 13)


class TestMetricsAndSubmission(unittest.TestCase):
    def test_metrics_match_the_organiser_definition(self):
        truth = {"1": ["a"], "2": ["b", "c"]}
        predictions = {"1": {"answer": ["a", "z"]}, "2": {"answer": ["b"]}}
        result = predict.metrics(predictions, truth)
        self.assertAlmostEqual(result["recall"], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(result["precision"], (0.5 + 1.0) / 2)

    def test_more_than_five_ids_scores_zero(self):
        truth = {"1": ["a"]}
        predictions = {"1": {"answer": ["a", "b", "c", "d", "e", "f"]}}
        self.assertEqual(predict.metrics(predictions, truth)["recall"], 0.0)

    def test_empty_answer_scores_zero(self):
        self.assertEqual(predict.metrics({"1": {"answer": []}}, {"1": ["a"]})["recall"], 0.0)

    def test_validate_rejects_missing_and_oversized_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "q.json"
            questions.write_text(json.dumps({"1": {"question": "x"}, "2": {"question": "y"}}),
                                 encoding="utf-8")
            path = root / "p.json"
            path.write_text(json.dumps({"1": {"answer": ["a"]}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                predict.validate(path, questions)
            path.write_text(json.dumps({"1": {"answer": ["a"] * 6}, "2": {"answer": ["b"]}}),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                predict.validate(path, questions)
            path.write_text(json.dumps({"1": {"answer": ["a", "a"]}, "2": {"answer": ["b"]}}),
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                predict.validate(path, questions)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dup.json"
            path.write_text('{"1": {"question": "a"}, "1": {"question": "b"}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                predict.load_questions(path)

    def test_pack_writes_one_utf8_file_and_reopens_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = make_corpus(root / "contexts")
            store.build(corpus, root / "store.sqlite")
            holder = store.Store(root / "store.sqlite")
            questions = root / "q.json"
            questions.write_text(json.dumps({"7": {"question": "lễ hội"}}), encoding="utf-8")
            predictions = root / "p.json"
            predictions.write_text(json.dumps({"7": {"answer": ["101", "102"]}}), encoding="utf-8")
            archive = root / "submission.zip"
            predict.pack(predictions, holder, questions, archive)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.namelist(), ["submission.json"])
                payload = json.loads(zipped.read("submission.json").decode("utf-8"))
            self.assertEqual(payload["7"]["answer"], ["101", "102"])
            holder.close()

    def test_pack_rejects_unknown_document_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store.build(make_corpus(root / "contexts"), root / "store.sqlite")
            holder = store.Store(root / "store.sqlite")
            questions = root / "q.json"
            questions.write_text(json.dumps({"7": {"question": "x"}}), encoding="utf-8")
            predictions = root / "p.json"
            predictions.write_text(json.dumps({"7": {"answer": ["999999"]}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                predict.pack(predictions, holder, questions, root / "s.zip")
            holder.close()


class TestAudit(unittest.TestCase):
    def test_budget_is_strict(self):
        with self.assertRaises(ValueError):
            audit({"embedding": 2_000_000_000, "reranker": 2_000_000_000})
        report = audit({"embedding": 568_000_000, "reranker": 568_000_000})
        self.assertEqual(report["total"], 1_136_000_000)
        self.assertGreater(report["headroom"], 0)

    def test_pooling_follows_the_architecture(self):
        self.assertEqual(pooling_for("xlm-roberta"), "cls")
        self.assertEqual(pooling_for("qwen3"), "last")


if __name__ == "__main__":
    unittest.main()


class StubReranker:
    """A reranker with a fixed opinion, so the ensemble wiring can be tested
    without any model weights."""

    def __init__(self, preference, parameters=100):
        self.preference = preference          # document id -> score
        self.parameters = parameters
        self.calls = []

    def score(self, query, passages, batch_size=8, max_length=768):
        self.calls.append(len(passages))
        return [self.preference.get(self.owner_of(p), 0.0) for p in passages]

    def owner_of(self, passage):
        for key in self.preference:
            if key in passage:
                return key
        return None

    def close(self):
        pass


class TestRerankerEnsemble(unittest.TestCase):
    """The stub passages carry their document id as text, so a stub reranker can
    express a preference and the combination logic is what is under test."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        store.build(make_corpus(root / "contexts"), root / "store.sqlite")
        lexical.build(root / "store.sqlite", root / "lexical")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def build(self, rerankers, settings=None):
        from legalir2.pipeline import Pipeline
        pipeline = Pipeline(self.root / "store.sqlite", self.root / "lexical",
                            settings=dict({"candidates": 3, "passages": 1}, **(settings or {})))
        pipeline.rerankers = rerankers
        pipeline._passages = lambda doc, prior: [f"passage of {doc}"]
        return pipeline

    def test_single_reranker_decides_the_order(self):
        pipeline = self.build([StubReranker({"102": 9.0, "101": 1.0, "103": 0.5})])
        self.assertEqual(pipeline.search("lễ hội", top_k=1), ["102"])
        pipeline.store.close()

    def test_a_heavier_second_reranker_overturns_the_first(self):
        wide = StubReranker({"101": 5.0, "102": 4.0, "103": 0.0})
        precise = StubReranker({"101": 0.0, "102": 9.0, "103": 0.0})
        pipeline = self.build([wide, precise], {"stage2_candidates": 2, "rerank_blend": 0.0,
                                                "reranker_weights": [1.0, 3.0]})
        self.assertEqual(pipeline.search("lễ hội", top_k=1), ["102"])
        pipeline.store.close()

    def test_equal_weights_make_opposed_rankings_a_deterministic_tie(self):
        """Each ranking is min-max normalised before combining, so two models that
        disagree exactly cancel. The tie then breaks on document id, not on order of
        evaluation, which keeps a run reproducible."""
        wide = StubReranker({"101": 5.0, "102": 4.0, "103": 0.0})
        precise = StubReranker({"101": 0.0, "102": 9.0, "103": 0.0})
        settings = {"stage2_candidates": 2, "rerank_blend": 0.0}
        first = self.build([wide, precise], settings).search("lễ hội", top_k=2)
        second = self.build([wide, precise], settings).search("lễ hội", top_k=2)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), ["101", "102"])

    def test_second_stage_only_sees_the_promoted_documents(self):
        wide = StubReranker({"101": 5.0, "102": 4.0, "103": 0.0})
        precise = StubReranker({"101": 1.0, "102": 2.0, "103": 9.0})
        pipeline = self.build([wide, precise], {"stage2_candidates": 2, "rerank_blend": 0.0})
        order = pipeline.search("lễ hội", top_k=3)
        # 103 was never promoted, so its high second-stage score cannot reach it.
        self.assertEqual(precise.calls, [2])
        self.assertEqual(order[2], "103")
        pipeline.store.close()

    def test_promoted_documents_always_outrank_the_rest(self):
        wide = StubReranker({"101": 5.0, "102": 4.0, "103": 3.0})
        precise = StubReranker({"101": -50.0, "102": -60.0, "103": 0.0})
        pipeline = self.build([wide, precise], {"stage2_candidates": 2, "rerank_blend": 0.0})
        self.assertEqual(pipeline.search("lễ hội", top_k=3)[2], "103")
        pipeline.store.close()

    def test_ensemble_without_a_second_stage_scores_every_candidate_twice(self):
        first = StubReranker({"101": 1.0, "102": 2.0, "103": 3.0})
        second = StubReranker({"101": 1.0, "102": 2.0, "103": 3.0})
        pipeline = self.build([first, second], {"stage2_candidates": 0})
        pipeline.search("lễ hội")
        self.assertEqual(first.calls, second.calls)
        pipeline.store.close()

    def test_audit_counts_every_deployed_reranker(self):
        from legalir2.models import audit
        report = audit({"embedding": 568_000_000, "reranker1": 568_000_000,
                        "reranker2": 2_000_000_000})
        self.assertEqual(report["total"], 3_136_000_000)
        with self.assertRaises(ValueError):
            audit({"embedding": 568_000_000, "reranker1": 568_000_000,
                   "reranker2": 3_000_000_000})
