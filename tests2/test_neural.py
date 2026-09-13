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
        store.build(make_corpus(cls.root / "contexts", extra=8), cls.root / "store.sqlite")
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
        expected = store.Store(self.root / "store.sqlite")
        self.assertEqual(manifest["chunks"], expected.metadata["chunks"])
        expected.close()
        index = DenseIndex(self.root / "dense", self.root / "store.sqlite", self.encoder,
                           device="cpu")
        scores = index.scores("lễ hội")
        self.assertEqual(len(scores), manifest["chunks"])
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
        # Five ids: documents beyond `candidates` still fill the tail from fusion order.
        self.assertEqual(len(answer), 5)
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
        store.build(make_corpus(cls.root / "contexts", extra=8), cls.root / "store.sqlite")
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
                     batch_size=2, max_length=32, min_distractors=1)
        self.assertEqual(result["questions"], 2)
        self.assertGreaterEqual(result["mrr"], 0.0)
        self.assertLessEqual(result["top1"], 1.0)
        self.assertFalse((self.root / "dense").exists())

    def test_questions_with_too_few_distractors_are_dropped_and_counted(self):
        """One of the two questions retrieves only its own document plus one other, so
        at a floor of two distractors it cannot be scored. Dropping it silently would
        make the reported metric quietly unrepresentative, so the count is returned."""
        from legalir2.probe import run
        result = run(self.root / "store.sqlite", self.root / "lexical", self.questions,
                     self.encoder, mode="embedding", device="cpu", limit=2, distractors=4,
                     batch_size=2, max_length=32, min_distractors=2)
        self.assertEqual(result["questions"], 1)
        self.assertEqual(result["dropped"], 1)

    def test_probe_explains_itself_when_nothing_can_be_built(self):
        from legalir2.probe import run
        with self.assertRaises(ValueError) as raised:
            run(self.root / "store.sqlite", self.root / "lexical", self.questions,
                self.encoder, mode="embedding", device="cpu", limit=2, distractors=4,
                batch_size=2, max_length=32, min_distractors=50)
        self.assertIn("distractors", str(raised.exception))


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed in this environment")
class TestContextLimit(unittest.TestCase):
    """Approved checkpoints range from 256 usable tokens (PhoBERT rerankers) to 8192
    (bge-m3 and its Vietnamese derivatives). Asking for more than the position table
    holds fails inside the embedding layer, which on a full corpus index happens an
    hour into the run."""

    def test_limit_accounts_for_the_roberta_position_offset(self):
        from legalir2.models import context_limit
        self.assertEqual(context_limit({"model_type": "xlm-roberta",
                                        "max_position_embeddings": 514}), 512)
        self.assertEqual(context_limit({"model_type": "xlm-roberta",
                                        "max_position_embeddings": 8194}), 8192)
        self.assertEqual(context_limit({"model_type": "qwen3",
                                        "max_position_embeddings": 32768}), 32768)

    def test_no_position_information_means_no_clamp(self):
        from legalir2.models import clamp_length, context_limit
        self.assertIsNone(context_limit({"model_type": "qwen3"}))
        self.assertEqual(clamp_length(4096, {"model_type": "qwen3"}), 4096)

    def test_request_beyond_the_table_is_clamped_not_raised(self):
        from legalir2.models import clamp_length
        short = {"model_type": "xlm-roberta", "max_position_embeddings": 514}
        self.assertEqual(clamp_length(768, short), 512)
        self.assertEqual(clamp_length(256, short), 256)

    def test_a_short_context_encoder_still_indexes(self):
        """The fixture's position table holds 512; the pipeline asks for 768."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store.build(make_corpus(root / "contexts", extra=2), root / "store.sqlite")
            from legalir2.dense import build
            manifest = build(root / "store.sqlite", tiny_encoder(root / "enc"),
                             root / "dense", device="cpu", batch_size=2, max_length=768)
            self.assertLessEqual(manifest["max_length"], 512)


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed in this environment")
class TestMixedPrecision(unittest.TestCase):
    def test_cpu_trains_in_float32_with_no_scaler(self):
        import torch

        from legalir2.models import amp_setup
        enabled, dtype, scaler = amp_setup(torch, "cpu")
        self.assertFalse(enabled)
        self.assertIs(dtype, torch.float32)
        self.assertIsNone(scaler)

    def test_a_card_without_bfloat16_gets_float16_and_a_scaler(self):
        """Turing and older cannot autocast to bfloat16. Without this fallback the
        training loop raises on a cheap rented card, after the data is already there."""
        import torch

        from legalir2 import models
        real = torch.cuda.is_bf16_supported
        torch.cuda.is_bf16_supported = lambda *a, **k: False
        try:
            enabled, dtype, scaler = models.amp_setup(torch, "cuda")
        finally:
            torch.cuda.is_bf16_supported = real
        self.assertTrue(enabled)
        self.assertIs(dtype, torch.float16)
        self.assertIsNotNone(scaler)

    def test_cross_encoder_training_runs_end_to_end_on_cpu(self):
        """A real optimiser step through the listwise loss, on fixtures."""
        from legalir2.train_ce import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store.build(make_corpus(root / "contexts", extra=8), root / "store.sqlite")
            lexical.build(root / "store.sqlite", root / "lexical")
            questions = root / "train.json"
            questions.write_text(json.dumps({
                str(i): {"question": "Việc tổ chức lễ hội theo nguyên tắc nào?",
                         "answer": ["101"]} for i in range(6)
            }, ensure_ascii=False), encoding="utf-8")
            from legalir2.mine import mine
            groups = root / "groups.jsonl"
            mine(root / "store.sqlite", root / "lexical", questions, groups,
                 negatives=4, positives=1, device="cpu")
            train(root / "store.sqlite", groups, tiny_cross_encoder(root / "ce"),
                  root / "ce-tuned", device="cpu", epochs=1, group_size=3,
                  groups_per_step=2, max_length=32, checkpointing=False)
            self.assertTrue((root / "ce-tuned" / "config.json").is_file())
            self.assertTrue((root / "ce-tuned" / "training.json").is_file())


if __name__ == "__main__":
    unittest.main()
