"""Joint-document prompt, tournament, leakage and checkpoint invariants."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from legalir2.duo import (PairTokenizer, diagnostics, pair_examples, pool, read_compressed,
                          submit, tournament, validate_scores, verify_model, verify_source_inputs, write_compressed)
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json, write_json


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class TestDuo(unittest.TestCase):
    def test_restored_receipt_requires_exact_source_model_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models2/partner_qwen"
            model.mkdir(parents=True)
            names = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
            for name in names:
                (model / name).write_text("original " + name)
            provenance = {"repository": "Qwen/Qwen3-Reranker-0.6B", "seconds": 10,
                          "files": {n: {"sha256": file_hash(model / n)} for n in names}}
            write_json(model / "provenance.json", provenance)
            original = {"files": {p.relative_to(root).as_posix(): file_hash(p) for p in model.iterdir()}}
            verify_source_inputs(original, root)
            provenance["seconds"] = 999
            write_json(model / "provenance.json", provenance)
            verify_source_inputs(original, root)
            (model / "model.safetensors").write_text("different weights")
            # Even a self-consistent new receipt cannot bless new weights.
            provenance["files"]["model.safetensors"]["sha256"] = file_hash(model / "model.safetensors")
            write_json(model / "provenance.json", provenance)
            with self.assertRaisesRegex(ValueError, "Source-run input changed.*model.safetensors"):
                verify_source_inputs(original, root)

    def test_provenance_exception_does_not_allow_different_repository_or_other_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models2/partner_qwen"
            model.mkdir(parents=True)
            names = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
            for name in names:
                (model / name).write_text("original")
            provenance = {"repository": "different/model", "files": {n: {"sha256": file_hash(model / n)} for n in names}}
            write_json(model / "provenance.json", provenance)
            original = {"files": {p.relative_to(root).as_posix(): file_hash(p) for p in model.iterdir()}}
            original["files"]["models2/partner_qwen/provenance.json"] = "old receipt"
            with self.assertRaisesRegex(ValueError, "Restored partner provenance"):
                verify_source_inputs(original, root)
            provenance["repository"] = "Qwen/Qwen3-Reranker-0.6B"
            provenance["files"]["tokenizer.json"]["sha256"] = "incorrect"
            write_json(model / "provenance.json", provenance)
            with self.assertRaisesRegex(ValueError, "Restored partner provenance"):
                verify_source_inputs(original, root)
            (root / "questions.json").write_text("changed")
            original["files"]["questions.json"] = "original hash"
            with self.assertRaisesRegex(ValueError, "Source-run input changed: questions.json"):
                verify_source_inputs(original, root)

    def test_balanced_prompt_retains_both_documents_and_answer_suffix(self):
        codec = PairTokenizer(CharacterTokenizer(), 2048)
        first, second = [[901] * 3000, [902] * 1000], [[801] * 3000, [802] * 1000]
        ids = codec.pair("question " * 100, first, second)
        self.assertLessEqual(len(ids), 2048)
        self.assertEqual(ids[-len(codec.suffix):], codec.suffix)
        self.assertIn(901, ids)
        self.assertIn(902, ids)
        self.assertIn(801, ids)
        self.assertIn(802, ids)
        self.assertEqual(ids.count(901), ids.count(801))
        self.assertEqual(ids.count(902), ids.count(802))

    def test_prompt_refuses_budget_that_cannot_fit_evidence(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            PairTokenizer(CharacterTokenizer(), 128).pair("q", [[1]], [[2]])

    def test_tournament_solves_a_strong_preference_and_is_permutation_equivariant(self):
        probability = np.full((6, 6), .5)
        probability[5, :5] = .99
        probability[:5, 5] = .01
        prior, result = tournament(np.zeros(6), probability)
        self.assertEqual(int(np.argmax(result)), 5)
        permutation = np.asarray([5, 3, 1, 0, 2, 4])
        _, shuffled = tournament(np.zeros(6), probability[permutation][:, permutation])
        np.testing.assert_allclose(shuffled, result[permutation])
        self.assertAlmostEqual(float(result.sum()), float(prior.sum()))

    def test_cyclic_preferences_remain_finite(self):
        probability = np.asarray([[.5, .9, .1], [.1, .5, .9], [.9, .1, .5]])
        _, values = tournament([0., 0., 0.], probability)
        np.testing.assert_allclose(values, 0, atol=1e-8)

    def test_confident_tournament_opposing_the_prior_reaches_stationary_solution(self):
        probability = np.triu(np.ones((16, 16)), 1) + np.eye(16) * .5
        prior, values = tournament(np.arange(16), probability, precision=.001)
        fitted = 1 / (1 + np.exp(-(values[:, None] - values[None, :])))
        gradient = (fitted - probability).sum(axis=1) + .001 * (values - prior)
        self.assertLess(float(np.max(np.abs(gradient))), 1e-7)
        self.assertTrue((np.diff(values) < 0).all())

    def test_invalid_tournaments_are_rejected(self):
        for matrix in (np.zeros((3, 3)), np.full((3, 3), float("nan")), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                tournament([1., 2., 3.], matrix)

    def test_training_pairs_are_symmetric_query_balanced_and_split(self):
        groups = [{"id": "a", "pairs": [["p", "n"], ["p", "m"]], "development": False},
                  {"id": "b", "pairs": [["p", "n"]], "development": False},
                  {"id": "held", "pairs": [["p", "n"]], "development": True}]
        examples = pair_examples(groups, False)
        self.assertEqual(len(examples), 6)
        for group in groups[:2]:
            self.assertEqual(sum(e[4] for e in examples if e[0]["id"] == group["id"]), 1.)
        for first, second in zip(examples[::2], examples[1::2]):
            self.assertEqual(first[1:3], second[1:3][::-1])
            self.assertEqual((first[3], second[3]), (0, 1))
        self.assertEqual({e[0]["id"] for e in pair_examples(groups, True)}, {"held"})

    def test_memory_expands_candidate_pool_without_inventing_high_base_score(self):
        matrix = np.zeros((20, 20))
        matrix[:, 0] = np.arange(20, 0, -1)
        row = {"id": "q", "question": "q", "docs": [str(i) for i in range(20)], "features1": matrix}
        memory = Mock()
        memory.search.return_value = ({"extra": [.8], "untrusted": [.99]}, ["extra", "untrusted"])
        docs, scores = pool(row, np.asarray([1.] + [0.] * 19), memory, set(row["docs"]) | {"extra"})
        self.assertEqual(len(docs), 17)
        self.assertEqual(docs[-1], "extra")
        self.assertLess(scores[-1], min(scores[:-1]))

    def test_zero_strength_exactly_retains_baseline_and_other_strength_changes_it(self):
        row = {"id": "q", "question": "q", "gold": ["f"], "docs": list("abcdef"),
               "prior": [6., 5., 4., 3., 2., 1.], "ability": [0., 0., 0., 0., 0., 10.],
               "baseline": list("abcde")}
        self.assertEqual(diagnostics([row], 0)["q"]["predicted"], list("abcde"))
        self.assertEqual(diagnostics([row], 1)["q"]["predicted"][0], "f")

    def test_score_checkpoint_rejects_changed_question_or_invalid_values(self):
        row = {"id": "q", "question": "question", "gold": ["a"], "docs": list("abcde"),
               "prior": [1.] * 5, "ability": [1.] * 5, "probabilities": np.full((5, 5), .5).tolist()}
        items = [("q", {"question": "question", "answer": ["a"]})]
        validate_scores([row], items)
        with self.assertRaises(ValueError):
            validate_scores([dict(row, question="changed")], items)
        with self.assertRaises(ValueError):
            validate_scores([dict(row, ability=[float("nan")] * 5)], items)

    def test_compressed_data_roundtrip_and_model_file_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_compressed(root / "data.json.gz", [{"question": "Bảo hiểm", "ids": [1, 2]}])
            self.assertEqual(read_compressed(root / "data.json.gz")[0]["question"], "Bảo hiểm")
            (root / "weights").write_text("fixture")
            from legalir2.duo import SCHEMA
            write_json(root / "duo.json", {"schema": SCHEMA, "inputs": digest({}),
                       "files": {"weights": file_hash(root / "weights")}})
            verify_model(root, {})
            (root / "weights").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify_model(root, {})

    def test_failed_experiment_never_repackages_v9_as_a_new_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory))
            write_json(args.out / "calibration.json", {})
            write_json(args.out / "selection.json", {"promoted": False})
            with patch("legalir2.duo.check_choice"), patch("legalir2.duo.collect") as collect:
                submit(args)
            collect.assert_not_called()
            self.assertFalse(list(args.out.glob("submission*")))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Neural libraries unavailable")
class TestDuoNeural(unittest.TestCase):
    def test_real_tiny_qwen_logits_train_and_score_both_document_orders(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers
        from legalir2.duo import batch_inputs, compare_documents

        vocabulary = {word: i for i, word in enumerate(["<unk>", "<pad>", "A", "B", "question", "first", "second"])}
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", padding_side="left")
        codec = PairTokenizer(fast, 512)
        torch.manual_seed(2026)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocabulary), hidden_size=32,
                    intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=8, max_position_embeddings=512))
        evidence = {"a": [codec.encode("first " * 20)], "b": [codec.encode("second " * 20)]}
        group = {"question": "question", "evidence": evidence}
        inputs = batch_inputs(codec, [(group, "a", "b", 0, 1.)], "cpu")
        loss = torch.nn.functional.cross_entropy(model(**inputs, logits_to_keep=1).logits[:, -1, codec.labels], torch.tensor([0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        model.eval()
        local = SimpleNamespace(torch=torch, model=model, device="cpu")
        probability = compare_documents(local, codec, "question", evidence, ["a", "b"], batch=2)
        flipped = compare_documents(local, codec, "question", evidence, ["b", "a"], batch=2)
        np.testing.assert_allclose(probability + probability.T, 1.)
        np.testing.assert_allclose(flipped, probability[::-1, ::-1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
