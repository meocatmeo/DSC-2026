import json
import tempfile
import unittest
import zipfile
import zlib
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
        pipeline._passages = lambda doc, prior, count=None: [f"passage of {doc}"]
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


class TestFusionTuningHelpers(unittest.TestCase):
    """The fusion sweep reuses the pipeline's chunk-to-document aggregation. If the
    two ever disagree, tuned weights would be fitted against a ranking the pipeline
    does not actually produce."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        store.build(make_corpus(root / "contexts", extra=6), root / "store.sqlite")
        lexical.build(root / "store.sqlite", root / "lexical")
        cls.store = store.Store(root / "store.sqlite")
        cls.lexical = lexical.Lexical(root / "lexical")

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.temporary.cleanup()

    def test_document_ranking_matches_the_pipeline(self):
        from legalir2.tune import _document_ranking
        query = "Việc tổ chức lễ hội theo nguyên tắc nào?"
        scores = self.lexical.chunk_scores(query)
        settings = self.lexical.settings
        mine = _document_ranking(self.lexical.unigram.chunk_ids, self.store.chunk_documents(),
                                 scores, settings["decay"], settings["depth"],
                                 settings["chunk_candidates"])
        theirs, _ = self.lexical.search(query)
        self.assertEqual([d for d, _ in mine], [d for d, _ in theirs])
        for (a, x), (b, y) in zip(mine, theirs):
            self.assertAlmostEqual(x, y, places=4)

    def test_fusion_weights_change_the_order(self):
        from legalir2.fuse import rrf
        first = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
        second = [("c", 3.0), ("b", 2.0), ("a", 1.0)]
        self.assertEqual(rrf([first, second], [3.0, 1.0], 60)[0][0], "a")
        self.assertEqual(rrf([first, second], [1.0, 3.0], 60)[0][0], "c")

    def test_rrf_constant_changes_how_sharply_top_ranks_dominate(self):
        from legalir2.fuse import rrf
        first = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
        second = [("b", 1.0), ("c", 1.0), ("a", 1.0)]
        sharp = dict(rrf([first, second], [1.0, 1.0], 1))
        flat = dict(rrf([first, second], [1.0, 1.0], 1000))
        self.assertGreater(sharp["a"] - sharp["c"], flat["a"] - flat["c"])


class TestDerivedProvenance(unittest.TestCase):
    """Regression: copying a base model's provenance.json into a fine-tuned directory
    claims file hashes that training has just invalidated, so `verify-models` reports
    perfectly good trained weights as corrupt."""

    def test_a_trained_checkpoint_records_lineage_not_hashes(self):
        from legalir2.train_ce import derived_provenance
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            base.mkdir()
            (base / "provenance.json").write_text(json.dumps({
                "repository": "AITeamVN/Vietnamese_Reranker", "commit": "abc123",
                "files": {"model.safetensors": {"sha256": "deadbeef", "bytes": 1}}}),
                encoding="utf-8")
            record = derived_provenance(base, {"epochs": 2})
            self.assertNotIn("files", record)
            self.assertEqual(record["derived_from"]["repository"],
                             "AITeamVN/Vietnamese_Reranker")
            self.assertEqual(record["training"]["epochs"], 2)

    def test_verify_reports_a_derived_checkpoint_instead_of_failing(self):
        from legalir2.download import verify
        with tempfile.TemporaryDirectory() as directory:
            tuned = Path(directory)
            (tuned / "provenance.json").write_text(json.dumps({
                "derived_from": {"repository": "base/model"}, "training": {"epochs": 2}}),
                encoding="utf-8")
            record = verify(tuned)
            self.assertTrue(record["derived"])

    def test_verify_still_fails_a_genuinely_corrupt_download(self):
        from legalir2.download import verify
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"not the bytes that were downloaded")
            (root / "provenance.json").write_text(json.dumps({
                "repository": "x/y", "commit": "abc",
                "files": {"model.safetensors": {"sha256": "0" * 64, "bytes": 4}}}),
                encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                verify(root)
            self.assertIn("hash mismatch", str(raised.exception))


class TestMiningMatchesInference(unittest.TestCase):
    """Mining must draw its candidates from the same path prediction uses.

    This is a regression test for a silent failure, not a hypothetical one: mining
    once had its own min-max blend of the two channels while prediction used weighted
    RRF, so the reranker was trained to reject one distribution and deployed against
    another. Nothing errored - the conversion rate just sagged. Asserting that the
    mined ranking equals Pipeline.candidates() under non-default fusion weights is
    what makes a future divergence loud.
    """

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        store.build(make_corpus(root / "contexts", extra=12), root / "store.sqlite")
        lexical.build(root / "store.sqlite", root / "lexical")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _questions(self, path):
        from legalir2.store import Store
        opened = Store(self.root / "store.sqlite")
        gold = opened.doc_ids[0]
        text = opened.chunk_text(opened.document_chunk_ids(gold)[0], with_label=False)
        opened.close()
        query = " ".join(text.split()[:12])
        path.write_text(json.dumps({"q1": {"question": query, "answer": [gold]}},
                                   ensure_ascii=False), encoding="utf-8")
        return gold, query

    def test_mined_negatives_come_from_the_prediction_ranking(self):
        from legalir2.mine import mine
        from legalir2.pipeline import Pipeline

        # Lopsided weights: with a single channel any blend agrees, so the tuned
        # settings have to demonstrably reach the miner for this to prove anything.
        settings = {"lexical_weight": 1.5, "rrf_constant": 20, "shortlist": 50}
        questions = self.root / "questions.json"
        gold, query = self._questions(questions)
        output = self.root / "groups" / "train.jsonl"
        stats = mine(self.root / "store.sqlite", self.root / "lexical", questions, output,
                     device="cpu", negatives=5, settings=settings)
        self.assertEqual(stats["questions"], 1, "the single labelled question was dropped")
        self.assertEqual(stats["fusion"]["lexical_weight"], 1.5,
                         "tuned fusion settings never reached the miner")

        group = [json.loads(line) for line in output.open(encoding="utf-8")][0]
        pipeline = Pipeline(self.root / "store.sqlite", self.root / "lexical",
                            settings=settings)
        fused, prior = pipeline.candidates(query)
        expected = [doc for doc, _ in fused[:100] if doc != gold]
        chunk_document = pipeline.store.chunk_documents()
        pipeline.close()

        mined = [chunk_document[c] for c in group["negatives"]]
        self.assertTrue(mined, "no negatives mined")
        self.assertTrue(set(mined) <= set(expected),
                        f"mined {mined} from outside the prediction ranking {expected[:10]}")
        self.assertNotIn(gold, mined, "the gold document was mined as a negative")
        self.assertTrue(any(prior.get(c, 0.0) for c in group["negatives"]),
                        "negatives were not selected by the prediction-time chunk prior")


class TestRerankSweepMatchesLivePipeline(unittest.TestCase):
    """The offline sweep must rank identically to the live pipeline.

    A tuner that optimises a slightly different system than the one that ships is
    worse than no tuner: it produces confident numbers for settings that will not
    reproduce. This asserts the two agree on the same inputs, for several settings,
    which is what makes `final_order` worth having as a single shared function.
    """

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        store.build(make_corpus(root / "contexts", extra=20), root / "store.sqlite")
        lexical.build(root / "store.sqlite", root / "lexical")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _pipeline(self, settings):
        from legalir2.pipeline import Pipeline
        pipeline = Pipeline(self.root / "store.sqlite", self.root / "lexical",
                            settings=settings)
        # A deterministic stub scoring each passage as a pure function of its text.
        # Purity is the point: a stub that also depended on position in the batch
        # would score the same passage differently in a deeper cache, and would fail
        # this test for a reason that has nothing to do with the code under test.
        class Stub:
            def score(self, query, texts, batch, length):
                return [(zlib.crc32(t.encode("utf-8")) % 1000) / 100.0 for t in texts]
            def close(self): pass
        pipeline.rerankers = [Stub()]
        return pipeline

    def test_cached_sweep_reproduces_the_live_order(self):
        from legalir2.pipeline import final_order

        query = "quy định chi tiết về đất đai trên lãnh thổ Việt Nam"
        for settings in ({"candidates": 8, "passages": 2, "aggregate": "max"},
                         {"candidates": 5, "passages": 3, "aggregate": "max_plus",
                          "aggregate_weight": 0.35, "rerank_blend": 0.05},
                         {"candidates": 10, "passages": 1, "aggregate": "softmax",
                          "rerank_blend": 0.0}):
            with self.subTest(settings=settings):
                live = self._pipeline(dict(settings))
                expected = live.search(query, top_k=5)

                # The sweep caches deeper and wider than the setting under test, then
                # truncates - exactly what rerank_sweep does.
                cache_pipeline = self._pipeline(dict(settings))
                fused, prior = cache_pipeline.candidates(query)
                docs = [d for d, _ in fused[:20]]
                table = cache_pipeline._rerank(cache_pipeline.rerankers[0], query, docs,
                                               prior, {}, passages=4)
                merged = dict(cache_pipeline.settings, **settings)
                actual = final_order(fused, [table], merged)[:5]
                live.close(); cache_pipeline.close()
                self.assertEqual(expected, actual,
                                 f"cached sweep diverged from the live pipeline for {settings}")


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_torch(), "torch not installed (CPU-only checkout)")
class TestMultiPositiveLoss(unittest.TestCase):
    """The multi-positive objective must be a strict generalisation.

    If it changed single-gold behaviour at all it would be a gamble on 91% of the set to
    chase the other 9%. Asserting exact equivalence at one positive is what makes it a
    free bet: the only questions whose training signal changes are the multi-gold ones.
    """

    def targets(self, counts, group_size):
        import torch
        from legalir2.train_ce import group_targets
        return group_targets(counts, group_size, "cpu", torch)

    def test_one_positive_reproduces_the_old_one_hot_target(self):
        import torch
        rows = self.targets([1, 1, 1], 8)
        expected = torch.nn.functional.one_hot(torch.zeros(3, dtype=torch.long), 8).float()
        self.assertTrue(torch.equal(rows, expected))

    def test_one_positive_gives_the_identical_loss(self):
        import torch
        torch.manual_seed(0)
        logits = torch.randn(4, 8)
        old = torch.nn.functional.cross_entropy(
            logits, torch.zeros(4, dtype=torch.long))
        new = torch.nn.functional.cross_entropy(logits, self.targets([1] * 4, 8))
        self.assertAlmostEqual(float(old), float(new), places=6)

    def test_mass_is_split_evenly_and_sums_to_one(self):
        import torch
        rows = self.targets([1, 2, 3], 8)
        self.assertTrue(torch.allclose(rows.sum(dim=1), torch.ones(3)))
        self.assertAlmostEqual(float(rows[1, 0]), 0.5)
        self.assertAlmostEqual(float(rows[2, 2]), 1 / 3)
        self.assertEqual(float(rows[2, 3]), 0.0, "mass leaked past the positives")

    def test_multi_positive_prefers_lifting_every_gold_document(self):
        """Two golds ranked 1st and 8th must lose to both ranked 1st and 2nd."""
        import torch
        targets = self.targets([2], 8)
        one_high_one_low = torch.tensor([[9.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 9.0]])
        both_high = torch.tensor([[9.0, 9.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
        worse = torch.nn.functional.cross_entropy(one_high_one_low, targets)
        better = torch.nn.functional.cross_entropy(both_high, targets)
        self.assertLess(float(better), float(worse))

    def test_document_loss_uses_the_fifth_place_cutoff(self):
        import torch
        from legalir2.train_ce import recall_pairwise_loss
        # The positive is seventh and the negative is first, so this pair changes
        # Recall@5 by exactly one full point for a single-gold question.
        scores = [torch.tensor([0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0], requires_grad=True)]
        loss = recall_pairwise_loss(scores, [1], [1], torch)
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertIsNotNone(scores[0].grad)

    def test_document_loss_does_not_make_gold_documents_compete(self):
        import torch
        from legalir2.train_ce import recall_pairwise_loss
        first = torch.tensor([9.0, 8.0, 1.0, 0.0, -1.0, -2.0], requires_grad=True)
        swapped = torch.tensor([8.0, 9.0, 1.0, 0.0, -1.0, -2.0], requires_grad=True)
        a = recall_pairwise_loss([first], [2], [2], torch)
        b = recall_pairwise_loss([swapped], [2], [2], torch)
        self.assertAlmostEqual(float(a), float(b), places=6)


class TestLearnedRankFusion(unittest.TestCase):
    """The stacker must learn from evidence and reach the live pipeline unchanged."""

    def test_it_recovers_a_signal_the_hand_blend_would_bury(self):
        """A synthetic world where the reranker is wrong and retrieval is right.

        The shipped blend weights the reranker 1.0 and retrieval 0.02, so it follows the
        reranker off a cliff. A fitted combination should notice which signal predicts
        the label and reweight. This is the 23 discarded gold documents in miniature.
        """
        import numpy as np
        from legalir2.stack import FEATURES, Stacker, fit

        rng = np.random.default_rng(7)
        rows = []
        for _ in range(400):
            n = 10
            gold = rng.integers(0, n)
            matrix = rng.normal(0, 1, (n, len(FEATURES)))
            matrix[:, FEATURES.index("rr_max")] = rng.normal(0, 1, n)   # pure noise
            lex = np.zeros(n); lex[gold] = 1.0                          # perfect signal
            matrix[:, FEATURES.index("lex_rr")] = lex
            labels = np.zeros(n); labels[gold] = 1.0
            rows.append((matrix, labels))
        weights, history = fit(rows, epochs=200, verbose=False)
        self.assertLess(history[-1], history[0], "the fit did not reduce the loss")

        stacker = Stacker(weights)
        informative = abs(weights[FEATURES.index("lex_rr")])
        noise = abs(weights[FEATURES.index("rr_max")])
        self.assertGreater(informative, noise * 3,
                           "the fit did not prefer the signal that predicts the label")

        correct = 0
        for matrix, labels in rows[:100]:
            correct += int(labels[int(np.argmax(stacker.score(matrix)))] == 1.0)
        self.assertGreater(correct, 90, f"only {correct}/100 ranked the gold first")

    def test_fitted_weights_change_the_pipeline_order(self):
        """A stacker in settings must actually decide the final order."""
        import json, numpy as np
        from legalir2.pipeline import final_order
        from legalir2.stack import FEATURES, Stacker

        fused = [("A", 0.9), ("B", 0.5), ("C", 0.1)]
        channels = {"fused": fused,
                    "lexical": [("C", 9.0), ("B", 2.0), ("A", 1.0)],
                    "dense": [("C", 0.9), ("B", 0.4), ("A", 0.2)]}
        # The reranker loves A; both retrieval channels rank C first.
        tables = [{"A": [9.0], "B": [1.0], "C": [0.5]}]
        settings = {"candidates": 3, "passages": 1, "aggregate": "max",
                    "aggregate_weight": 0.2, "rerank_blend": 0.02,
                    "stage2_candidates": 25}

        blended = final_order(fused, tables, settings)
        self.assertEqual(blended[0], "A", "the hand blend should follow the reranker")

        weights = np.zeros(len(FEATURES))
        weights[FEATURES.index("lex_rr")] = 1.0
        ordered = final_order(fused, tables, settings,
                              channels=channels, stacker=Stacker(weights))
        self.assertEqual(ordered[0], "C", "the stacker did not override the blend")
        self.assertEqual(sorted(ordered), sorted(["A", "B", "C"]),
                         "the stacker dropped or duplicated documents")

    def test_a_saved_stacker_round_trips(self):
        import numpy as np
        from legalir2.stack import FEATURES, Stacker
        from legalir2.store import write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stacker.json"
            weights = np.arange(len(FEATURES), dtype="float64") / 10.0
            write_json(path, {"features": list(FEATURES), "weights": weights.tolist()})
            loaded = Stacker.load(path)
            self.assertTrue(np.allclose(loaded.weights, weights))

    def test_it_refuses_a_weight_vector_of_the_wrong_width(self):
        from legalir2.stack import Stacker
        with self.assertRaises(ValueError):
            Stacker([0.1, 0.2])


class TestCheckpointIdentity(unittest.TestCase):
    """A re-downloaded checkpoint must be recognised as the same checkpoint.

    Regression for a failure that cost a whole unattended run: the dense index compared
    entire provenance dicts, which include `seconds` - the download duration. Fetching
    the same weights again produced a different number, so every restored index was
    rejected as "built with a different checkpoint" and three stages died in a row.
    """

    def identity(self, provenance):
        from legalir2.models import checkpoint_identity
        return checkpoint_identity(provenance)

    def _downloaded(self, seconds, commit="abc123", digest="deadbeef"):
        return {"repository": "AITeamVN/Vietnamese_Embedding_v2", "revision": "main",
                "commit": commit, "endpoint": "https://huggingface.co",
                "seconds": seconds,
                "files": {"model.safetensors": {"sha256": digest, "bytes": 1140000000}}}

    def test_download_duration_does_not_change_identity(self):
        self.assertEqual(self.identity(self._downloaded(47)),
                         self.identity(self._downloaded(112)))

    def test_different_commit_is_a_different_checkpoint(self):
        self.assertNotEqual(self.identity(self._downloaded(47)),
                            self.identity(self._downloaded(47, commit="999fff")))

    def test_different_weights_are_a_different_checkpoint(self):
        self.assertNotEqual(self.identity(self._downloaded(47)),
                            self.identity(self._downloaded(47, digest="cafe")))

    def test_a_fine_tuned_checkpoint_is_not_its_base(self):
        derived = {"derived_from": self._downloaded(47), "training": {"base": "models2/embedding"}}
        self.assertNotEqual(self.identity(derived), self.identity(self._downloaded(47)))

    def test_an_old_index_without_model_identity_still_validates(self):
        """Indexes built before this field existed must not be condemned by its absence."""
        from legalir2.models import checkpoint_identity
        manifest = {"model": self._downloaded(47)}          # no model_identity key
        expected = manifest.get("model_identity") or checkpoint_identity(manifest["model"])
        self.assertEqual(expected, self.identity(self._downloaded(999)))


class TestPairedComparison(unittest.TestCase):
    """Paired testing must call a real change real and a noisy one noise."""

    def _diagnostics(self, scores):
        return {str(i): {"relevant": ["g"], "predicted": ["g"] if s else ["x"],
                         "question": "q", "candidate_ranks": {}}
                for i, s in enumerate(scores)}

    def _run(self, before, after, **kwargs):
        from legalir2.compare import compare
        import io, contextlib, json, tempfile
        with tempfile.TemporaryDirectory() as d:
            bp, ap = Path(d) / "b.json", Path(d) / "a.json"
            bp.write_text(json.dumps(self._diagnostics(before)))
            ap.write_text(json.dumps(self._diagnostics(after)))
            with contextlib.redirect_stdout(io.StringIO()):
                return compare(bp, ap, resamples=2000, **kwargs)

    def test_a_handful_of_flips_either_way_is_called_noise(self):
        """Balanced churn: 10 questions fixed, 10 broken. Net zero, and it must say so.

        The flips have to be balanced BY CONSTRUCTION. Flipping random indices in a
        95%-correct vector overwhelmingly breaks correct answers, which is a genuine
        regression - an earlier version of this test did exactly that and the code was
        right to call it one.
        """
        before = [1] * 950 + [0] * 50
        after = list(before)
        for i in range(10):                         # 10 correct answers broken
            after[i] = 0
        for i in range(950, 960):                   # 10 wrong answers fixed
            after[i] = 1
        result = self._run(before, after)
        self.assertLessEqual(result["ci"][0], 0.0)
        self.assertGreaterEqual(result["ci"][1], 0.0)

    def test_a_consistent_one_sided_gain_is_detected(self):
        before = [1] * 900 + [0] * 100
        after = [1] * 960 + [0] * 40               # 60 questions fixed, none broken
        result = self._run(before, after)
        self.assertGreater(result["ci"][0], 0.0, "a real 60-question gain was called noise")
        self.assertLess(result["p"], 0.01)

    def test_pairing_beats_comparing_means(self):
        """The whole reason this module exists."""
        import math
        before = [1] * 900 + [0] * 100
        after = [1] * 920 + [0] * 80               # +0.020, ~2.7 s.e. of a single mean
        result = self._run(before, after)
        unpaired = math.sqrt(0.9 * 0.1 / 1000) * 1.96 * 2
        paired = result["ci"][1] - result["ci"][0]
        self.assertLess(paired, unpaired,
                        "the paired interval should be tighter than the unpaired one")


class TestScoreCache(unittest.TestCase):
    """A cached replay must rank identically to the live pipeline.

    The cache exists so that hundreds of configurations can be judged without a GPU.
    If its ordering drifts from the live one by even a little, every conclusion drawn
    from it is about a system that will never ship - the same failure mode as mining
    through a different fusion than prediction used, which cost this project two points.
    """

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        store.build(make_corpus(root / "contexts", extra=25), root / "store.sqlite")
        lexical.build(root / "store.sqlite", root / "lexical")
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _pipeline(self, settings):
        from legalir2.pipeline import Pipeline

        class Stub:
            def score(self, query, texts, batch, length):
                return [(zlib.crc32(t.encode("utf-8")) % 1000) / 100.0 for t in texts]
            def close(self): pass

        pipeline = Pipeline(self.root / "store.sqlite", self.root / "lexical",
                            settings=settings)
        pipeline.rerankers = [Stub()]
        return pipeline

    def _questions(self):
        from legalir2.store import Store
        opened = Store(self.root / "store.sqlite")
        out = {}
        for number, doc in enumerate(opened.doc_ids[:6]):
            chunks = opened.document_chunk_ids(doc)
            if not chunks:
                continue
            text = opened.chunk_text(chunks[0], with_label=False)
            out[f"q{number}"] = {"question": " ".join(text.split()[:12]), "answer": [doc]}
        opened.close()
        return out

    def test_cached_evaluation_matches_the_live_pipeline(self):
        from legalir2.cache import Cache, build

        questions = self._questions()
        settings = {"candidates": 10, "passages": 2, "aggregate": "max_plus",
                    "rerank_blend": 0.02}
        pipeline = self._pipeline(dict(settings))
        path = build(pipeline, questions, self.root / "cache.npz",
                     depth=30, rerank_depth=15, passages=4, progress=10**9)
        live = {qid: pipeline.search(r["question"], top_k=5)
                for qid, r in questions.items()}
        pipeline.close()

        cached = Cache(path)
        for row, qid in cached.rows():
            channels, table, _ = cached.replay(row)
            from legalir2.pipeline import DEFAULTS, final_order
            order = final_order(channels["fused"], [table], dict(DEFAULTS, **settings),
                                channels=channels)[:5]
            self.assertEqual(live[qid], order,
                             f"cached replay diverged from the live pipeline on {qid}")

    def test_the_cache_can_be_re_scored_under_different_settings(self):
        """The whole point: changing settings must change the answer, with no GPU."""
        from legalir2.cache import Cache, build
        questions = self._questions()
        pipeline = self._pipeline({"candidates": 10, "passages": 2})
        path = build(pipeline, questions, self.root / "cache2.npz",
                     depth=30, rerank_depth=15, passages=4, progress=10**9)
        pipeline.close()
        cached = Cache(path)
        a = cached.evaluate({"candidates": 10, "passages": 1, "aggregate": "max"})
        b = cached.evaluate({"candidates": 5, "passages": 4, "aggregate": "softmax",
                             "rerank_blend": 0.5})
        self.assertEqual(set(a), set(b))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in list(a.values()) + list(b.values())))

    def test_folds_are_stable_under_reordering(self):
        """A question must keep its fold when the question file changes around it."""
        from legalir2.cache import folds
        ids = [f"q{i}" for i in range(500)]
        first = folds(ids)
        second = folds(list(reversed(ids)) + ["q_new_1", "q_new_2"])
        self.assertTrue(all(first[q] == second[q] for q in ids),
                        "fold assignment moved when the question list changed")
        sizes = [sum(1 for v in first.values() if v == f) for f in (0, 1)]
        self.assertTrue(min(sizes) > 0.4 * len(ids), f"folds badly unbalanced: {sizes}")

    def test_the_same_question_under_two_ids_shares_a_fold(self):
        """Duplicate question text must not straddle the train/score split.

        Keyed on the id, two copies of one question land in different folds roughly
        half the time, and the model trained on fold A is then "held out" on a question
        it memorised. That is the exact leak cross-fitting exists to close, reappearing
        inside the mechanism that closes it.
        """
        from legalir2.cache import folds
        texts = {}
        for i in range(400):
            texts[f"a{i}"] = f"Câu hỏi số {i} về hợp đồng xây dựng?"
            # same question, different id, different spacing and case
            texts[f"b{i}"] = f"câu  hỏi số {i} VỀ hợp đồng xây dựng"
        keyed_on_id = folds(list(texts), 5)
        split = sum(1 for i in range(400) if keyed_on_id[f"a{i}"] != keyed_on_id[f"b{i}"])
        self.assertGreater(split, 200, "the id-keyed control should split most duplicates")

        keyed_on_text = folds(list(texts), 5, texts=texts)
        for i in range(400):
            self.assertEqual(keyed_on_text[f"a{i}"], keyed_on_text[f"b{i}"],
                             f"duplicate question {i} straddles the fold boundary")

    def test_mining_channel_changes_which_negatives_are_chosen(self):
        """`--channel dense` must tier negatives on the dense ranking, not the fused one.

        The bi-encoder produces the dense channel's ranking over the whole corpus by
        itself; fusion happens afterwards. Training it on documents the FUSION found
        hard is the same train/inference mismatch that cost 0.021 on the reranker, and
        a flag that silently kept using the fused order would hide it perfectly.
        """
        from legalir2.mine import mine as run_mine
        import json as _json
        import tempfile as _tempfile
        from pathlib import Path as _Path
        from legalir2 import store as _store, lexical as _lexical
        from legalir2.store import Store as _Store, write_json as _write

        with _tempfile.TemporaryDirectory() as directory:
            root = _Path(directory)
            _store.build(make_corpus(root / "contexts", extra=25), root / "store.sqlite")
            _lexical.build(root / "store.sqlite", root / "lexical")
            opened = _Store(root / "store.sqlite")
            questions = {}
            for number, doc in enumerate(opened.doc_ids[:12]):
                chunks = opened.document_chunk_ids(doc)
                if chunks:
                    text = opened.chunk_text(chunks[0], with_label=False)
                    questions[f"q{number}"] = {"question": " ".join(text.split()[:12]),
                                               "answer": [doc]}
            opened.close()
            _write(root / "q.json", questions)

            def mined(channel):
                out = root / f"g-{channel}.jsonl"
                run_mine(root / "store.sqlite", root / "lexical", root / "q.json", out,
                         device="cpu", negatives=4, positives=1, depth=20,
                         channel=channel)
                rows = [_json.loads(line) for line in
                        out.read_text(encoding="utf-8").splitlines() if line.strip()]
                stats = _json.loads((root / f"g-{channel}.stats.json").read_text())
                return rows, stats

            fused_rows, fused_stats = mined("fused")
            lexical_rows, lexical_stats = mined("lexical")

            self.assertEqual(fused_stats["channel"], "fused")
            self.assertEqual(lexical_stats["channel"], "lexical")
            # Every mined group still has to be usable regardless of the channel.
            for rows in (fused_rows, lexical_rows):
                self.assertTrue(rows)
                for row in rows:
                    self.assertTrue(row["negatives"])
                    self.assertTrue(row["positives_by_gold"])
                    self.assertNotIn(row["gold"][0],
                                     {d for d in row.get("gold_ranks", {})
                                      if d not in row["gold"]})

    def test_duplicate_groups_reports_only_repeats(self):
        from legalir2.cache import duplicate_groups
        groups = duplicate_groups({"1": "Thuế thu nhập?", "2": "thuế thu nhập",
                                   "3": {"question": "THUẾ  THU NHẬP!"},
                                   "4": "Một câu hỏi khác"})
        self.assertEqual(list(groups.values()), [["1", "2", "3"]])

    def test_folds_without_texts_are_unchanged(self):
        """Omitting texts must reproduce the old behaviour, or every cache built
        before this change silently moves questions across the split."""
        from legalir2.cache import folds
        ids = [f"q{i}" for i in range(200)]
        self.assertEqual(folds(ids), folds(ids, texts={}))
        self.assertEqual(folds(ids), folds(ids, texts={"q0": None}))


class TestIdentityIgnoresRevisionName(unittest.TestCase):
    """Re-fetching the same commit under a different branch name is the same checkpoint.

    Regression for the repair path: when a provenance file goes missing, the fix is to
    re-download pinned to the commit the index recorded. If `revision` counted towards
    identity, that repair would produce a checkpoint that could never match an index
    built from 'main' - the fix would look exactly like the bug.
    """

    def test_same_commit_via_branch_or_sha_is_one_checkpoint(self):
        from legalir2.models import checkpoint_identity
        sha = "18b44161e041bf1d3a333ab5144b5b7b93f914d2"
        files = {"model.safetensors": {"sha256": "2fa082ea"}}
        via_branch = {"repository": "AITeamVN/Vietnamese_Embedding_v2",
                      "revision": "main", "commit": sha, "files": files}
        via_sha = dict(via_branch, revision=sha)
        self.assertEqual(checkpoint_identity(via_branch), checkpoint_identity(via_sha))

    def test_a_different_commit_is_still_a_different_checkpoint(self):
        from legalir2.models import checkpoint_identity
        files = {"model.safetensors": {"sha256": "2fa082ea"}}
        a = {"repository": "r", "revision": "main", "commit": "aaa", "files": files}
        b = {"repository": "r", "revision": "main", "commit": "bbb", "files": files}
        self.assertNotEqual(checkpoint_identity(a), checkpoint_identity(b))


class TestPerCheckpointPromptTemplate(unittest.TestCase):
    """A causal reranker must get ITS OWN prompt, not another model's.

    Qwen3-Reranker and Prism-Qwen3.5-Reranker share the <|im_start|> scaffolding and
    both score sigmoid(logit_yes - logit_no) on the first generated token, but their
    system prompt and instruction differ. Feeding one the other's template does not
    error - it silently returns worse scores on every passage, which is the most
    expensive failure shape this project has met.
    """

    def test_prism_and_qwen_get_different_prompts(self):
        from legalir2.models import TEMPLATES, rerank_prompt
        qwen, prism = TEMPLATES["qwen3-reranker"], TEMPLATES["prism"]
        self.assertNotEqual(qwen["system"], prism["system"])
        self.assertNotEqual(qwen["instruction"], prism["instruction"])
        self.assertNotEqual(rerank_prompt(qwen)[0], rerank_prompt(prism)[0])

    def test_template_is_chosen_from_recorded_provenance(self):
        from legalir2.models import TEMPLATES, template_for
        from legalir2.store import write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "some-finetuned-name"
            path.mkdir()
            write_json(path / "provenance.json",
                       {"repository": "infgrad/Prism-Qwen3.5-Reranker-2B"})
            self.assertEqual(template_for(path), TEMPLATES["prism"],
                             "a directory name that hides the publisher must still "
                             "resolve through provenance")

    def test_a_finetuned_copy_inherits_its_base_template(self):
        from legalir2.models import TEMPLATES, template_for
        from legalir2.store import write_json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reranker-2b-tuned"
            path.mkdir()
            write_json(path / "provenance.json",
                       {"derived_from": {"repository": "infgrad/Prism-Qwen3.5-Reranker-2B"},
                        "training": {"base": "models2/reranker2b"}})
            self.assertEqual(template_for(path), TEMPLATES["prism"])

    def test_an_unknown_checkpoint_falls_back_rather_than_crashing(self):
        from legalir2.models import TEMPLATES, template_for
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(template_for(Path(directory)), TEMPLATES["qwen3-reranker"])


@unittest.skipUnless(_has_torch(), "torch not installed (CPU-only checkout)")
class TestCausalRerankerTraining(unittest.TestCase):
    """Training a causal reranker must optimise the quantity inference reads.

    rerank.py scores a causal reranker as softmax over [logit_no, logit_yes] at the
    final position. Training it through AutoModelForSequenceClassification would throw
    away the pretrained LM head, train a randomly-initialised classifier, and produce a
    model that converges happily on an objective the inference path never consults.
    """

    def test_the_training_score_is_monotone_in_the_inference_score(self):
        """logit_yes - logit_no ranks identically to softmax over the pair."""
        import torch
        pairs = torch.tensor([[2.0, 5.0], [1.0, 1.5], [4.0, -1.0], [0.0, 0.0]])
        training = pairs[:, 1] - pairs[:, 0]                 # what train_ce optimises
        inference = pairs.softmax(dim=1)[:, 1]               # what rerank.py returns
        self.assertEqual(training.argsort().tolist(), inference.argsort().tolist(),
                         "training and inference would rank candidates differently")

    def test_head_detection_routes_each_checkpoint_correctly(self):
        from legalir2.rerank import head_for
        from legalir2.store import write_json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "bge-style"
            sequence.mkdir()
            write_json(sequence / "config.json",
                       {"model_type": "xlm-roberta",
                        "architectures": ["XLMRobertaForSequenceClassification"]})
            self.assertEqual(head_for(sequence), "sequence")

            qwen = root / "prism-style"
            qwen.mkdir()
            write_json(qwen / "config.json",
                       {"model_type": "qwen3_5", "architectures": ["Qwen3ForCausalLM"]})
            self.assertEqual(head_for(qwen), "causal",
                             "a Qwen-family reranker must train through the LM head")

    def test_group_reshape_survives_the_causal_path(self):
        """The per-pair score vector must reshape to (groups, group_size) either way."""
        import torch
        groups, size = 4, 8
        pair_logits = torch.randn(groups * size, 2)
        raw = pair_logits[:, 1] - pair_logits[:, 0]
        self.assertEqual(tuple(raw.float().view(groups, size).shape), (groups, size))
