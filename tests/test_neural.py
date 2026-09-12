"""Offline neural integration using tiny random fixtures, not competition data."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from legalir.data import prepare, write_json

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "numpy"))


@unittest.skipUnless(AVAILABLE, "Optional neural dependencies not installed")
class NeuralTests(unittest.TestCase):
    def test_local_qwen_embedding_index_and_reranking(self):
        import numpy as np
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model, Qwen3ForCausalLM
        from legalir.neural import Embedder, Reranker
        from legalir.retrieval import Retriever, build_dense

        torch.manual_seed(2026)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocabulary = {word: i for i, word in enumerate([
                "[PAD]", "[UNK]", "[EOS]", "yes", "no", "legal", "question", "tax", "vehicle", "document"])}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                                unk_token="[UNK]", eos_token="[EOS]")
            config = Qwen3Config(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                 head_dim=16, max_position_embeddings=2048, pad_token_id=0,
                                 eos_token_id=2, attention_dropout=0.0)
            for role, cls in (("embedding", Qwen3Model), ("reranker", Qwen3ForCausalLM)):
                path = root / role
                cls(config).save_pretrained(path)
                tokenizer.save_pretrained(path)
                write_json(path / "provenance.json", {"repo": "test-fixture", "revision": role})
            embedder = Embedder(root / "embedding", device="cpu")
            try:
                vectors = embedder.encode(["tax", "vehicle document"], batch_size=2)
                self.assertEqual(vectors.shape, (2, 32))
                np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
                single = embedder.encode(["tax"])[0]
                np.testing.assert_allclose(vectors[0], single, atol=1e-5)
            finally:
                embedder.close()
            reranker = Reranker(root / "reranker", device="cpu")
            try:
                scores = reranker.score("legal question", ["tax", "vehicle document"], batch_size=2)
                self.assertEqual(len(scores), 2)
                self.assertTrue(all(0 <= x <= 1 for x in scores))
            finally:
                reranker.close()
            corpus = root / "corpus"
            for key, text in (("1", "tax legal document"), ("2", "vehicle document")):
                write_json(corpus / f"context_{key}.json", {"id": key, "name": "", "passage": text})
            database = root / "corpus.sqlite"
            prepare(corpus, database)
            dense = root / "dense"
            manifest = build_dense(database, root / "embedding", dense, device="cpu", batch_size=2)
            self.assertEqual(manifest["chunks"], 2)
            engine = Retriever(database, dense=dense, embedding=root / "embedding",
                               reranker=root / "reranker", device="cpu", candidates=2)
            try:
                answer = engine.search("tax", 2)
                self.assertEqual(set(answer), {"1", "2"})
                self.assertLess(engine.manifest["parameters"], 4_000_000_000)
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()