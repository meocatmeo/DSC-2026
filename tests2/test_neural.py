"""Neural plumbing checks against tiny randomly initialised local fixtures.

These exercise pooling, the two reranker head shapes, dense index building and the
fused pipeline without downloading anything. Random fixtures say nothing about
retrieval quality - they are software tests, and they are not training data.
Skipped automatically where PyTorch is not installed (for example on a CPU only
prototyping box).
"""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    from transformers import AutoConfig  # noqa: F401
    HAVE_TORCH = True
except Exception:                                     # pragma: no cover
    HAVE_TORCH = False

from legalir2 import lexical, store
from tests2.test_system import make_corpus


def tiny_encoder(directory):
    from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
    directory = Path(directory)
    config = XLMRobertaConfig(vocab_size=1000, hidden_size=32, num_hidden_layers=2,
                              num_attention_heads=2, intermediate_size=64,
                              max_position_embeddings=512)
    model = XLMRobertaModel(config)
    model.save_pretrained(directory, safe_serialization=True)
    _save_tokenizer(directory)
    return directory


def tiny_cross_encoder(directory):
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
    directory = Path(directory)
    config = XLMRobertaConfig(vocab_size=1000, hidden_size=32, num_hidden_layers=2,
                              num_attention_heads=2, intermediate_size=64,
                              max_position_embeddings=512, num_labels=1)
    model = XLMRobertaForSequenceClassification(config)
    model.save_pretrained(directory, safe_serialization=True)
    _save_tokenizer(directory)
    return directory


def _save_tokenizer(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=1000, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.train_from_iterator(
        ["điều 1 phạm vi lễ hội bảo hiểm y tế thanh toán nghị định thông tư"], trainer)
    tokenizer.save(str(Path(directory) / "tokenizer.json"))
    (Path(directory) / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
        "pad_token": "<pad>", "cls_token": "<s>", "sep_token": "</s>",
        "mask_token": "<mask>", "model_max_length": 512}), encoding="utf-8")


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed in this environment")
class TestNeural(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        store.build(make_corpus(cls.root / "contexts"), cls.root / "store.sqlite")
        lexical.build(cls.root / "store.sqlite", cls.root / "lexical")
        cls.encoder = tiny_encoder(cls.root / "encoder")
        cls.cross = tiny_cross_encoder(cls.root / "cross")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_embedder_returns_unit_vectors(self):
        import numpy as np

        from legalir2.dense import Embedder
        embedder = Embedder(self.encoder, device="cpu")
        vectors = embedder.encode(["điều 1 phạm vi", "lễ hội"], batch_size=2, max_length=32)
        self.assertEqual(vectors.shape[0], 2)
        norms = np.linalg.norm(vectors.astype("float32"), axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-2))
        embedder.close()

    def test_dense_index_round_trips_and_is_bound_to_its_store(self):
        from legalir2.dense import DenseIndex, build
        manifest = build(self.root / "store.sqlite", self.encoder, self.root / "dense",
                         device="cpu", batch_size=2, max_length=32)
        self.assertEqual(manifest["chunks"], 7)
        index = DenseIndex(self.root / "dense", self.root / "store.sqlite", self.encoder,
                           device="cpu")
        scores = index.scores("lễ hội")
        self.assertEqual(len(scores), 7)
        index.close()

    def test_sequence_reranker_produces_one_score_per_passage(self):
        from legalir2.rerank import Reranker
        reranker = Reranker(self.cross, device="cpu", max_length=64)
        self.assertEqual(reranker.head, "sequence")
        scores = reranker.score("lễ hội", ["điều 1 phạm vi", "bảo hiểm y tế", "thông tư"],
                                batch_size=2)
        self.assertEqual(len(scores), 3)
        reranker.close()

    def test_pipeline_with_dense_and_reranker_returns_five_ids(self):
        from legalir2.dense import build
        from legalir2.pipeline import Pipeline
        build(self.root / "store.sqlite", self.encoder, self.root / "dense",
              device="cpu", batch_size=2, max_length=32)
        pipeline = Pipeline(self.root / "store.sqlite", self.root / "lexical",
                            self.root / "dense", self.encoder, self.cross, device="cpu",
                            settings={"candidates": 3, "passages": 2, "rerank_max_length": 64,
                                      "rerank_batch": 2})
        answer = pipeline.search("Việc tổ chức lễ hội theo nguyên tắc nào?")
        self.assertEqual(len(answer), 3)
        self.assertEqual(len(set(answer)), len(answer))
        self.assertTrue(set(answer) <= set(pipeline.store.doc_ids))
        self.assertLess(pipeline.audit["total"], 4_000_000_000)
        pipeline.close()

    def test_stale_dense_index_is_refused(self):
        from legalir2.dense import DenseIndex, build
        build(self.root / "store.sqlite", self.encoder, self.root / "dense2",
              device="cpu", batch_size=2, max_length=32)
        other = self.root / "contexts2"
        make_corpus(other)
        (other / "context_4.json").write_text(
            json.dumps({"id": 104, "name": "Luat-x-4", "passage": "Điều 1. Khác"}),
            encoding="utf-8")
        store.build(other, self.root / "store2.sqlite")
        with self.assertRaises(ValueError):
            DenseIndex(self.root / "dense2", self.root / "store2.sqlite", self.encoder,
                       device="cpu")

    def test_mining_produces_positives_from_gold_and_negatives_from_others(self):
        from legalir2.mine import mine
        questions = self.root / "train.json"
        questions.write_text(json.dumps({
            "1": {"question": "Việc tổ chức lễ hội theo nguyên tắc nào?", "answer": ["101"]},
            "2": {"question": "Mức thanh toán bảo hiểm y tế thế nào?", "answer": ["102"]},
        }, ensure_ascii=False), encoding="utf-8")
        output = self.root / "groups.jsonl"
        stats = mine(self.root / "store.sqlite", self.root / "lexical", questions, output,
                     negatives=2, positives=1, device="cpu")
        self.assertEqual(stats["questions"], 2)
        groups = [json.loads(line) for line in output.open(encoding="utf-8")]
        holder = store.Store(self.root / "store.sqlite")
        for group in groups:
            for chunk in group["positives"]:
                self.assertIn(holder.chunk(chunk)[0], group["gold"])
            for chunk in group["negatives"]:
                self.assertNotIn(holder.chunk(chunk)[0], group["gold"])
        holder.close()




@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed in this environment")
class TestProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        store.build(make_corpus(cls.root / "contexts"), cls.root / "store.sqlite")
        lexical.build(cls.root / "store.sqlite", cls.root / "lexical")
        cls.encoder = tiny_encoder(cls.root / "encoder")
        cls.questions = cls.root / "train.json"
        cls.questions.write_text(json.dumps({
            "1": {"question": "Việc tổ chức lễ hội theo nguyên tắc nào?", "answer": ["101"]},
            "2": {"question": "Mức thanh toán bảo hiểm y tế thế nào?", "answer": ["102"]},
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_probe_reports_ranking_metrics_without_building_an_index(self):
        from legalir2.probe import run
        result = run(self.root / "store.sqlite", self.root / "lexical", self.questions,
                     self.encoder, mode="embedding", device="cpu", limit=2, distractors=4,
                     batch_size=2, max_length=32)
        self.assertEqual(result["questions"], 2)
        self.assertGreaterEqual(result["mrr"], 0.0)
        self.assertLessEqual(result["top1"], 1.0)
        self.assertFalse((self.root / "dense").exists())


if __name__ == "__main__":
    unittest.main()
