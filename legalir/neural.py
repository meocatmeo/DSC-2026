"""Qwen local inference. No hosted inference and no remote model code."""

import gc
from pathlib import Path

EMBEDDING = "Qwen/Qwen3-Embedding-0.6B"
RERANKER = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = "Given a Vietnamese legal question, retrieve legal documents containing information needed to answer the question."
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LocalModel:
    def __init__(self, path, causal=False, device="cuda"):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        if not Path(path).is_dir():
            raise ValueError(f"Local model directory not found: {path}. Run download first.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Install CUDA PyTorch/use NVIDIA hardware, or explicitly use --device cpu.")
        self.torch = torch
        self.device = device
        dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        loader = AutoModelForCausalLM if causal else AutoModel
        self.model = loader.from_pretrained(path, local_files_only=True, torch_dtype=dtype,
                                            attn_implementation="sdpa").to(device).eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters >= 4_000_000_000:
            raise ValueError("Model exceeds parameter budget")

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class Embedder(LocalModel):
    def encode(self, texts, batch_size=8, max_length=1024, query=False):
        import numpy as np

        result = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            if query:
                batch = [f"Instruct: {INSTRUCTION}\nQuery: {text}" for text in batch]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                                    return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                # Left padding guarantees that the final position is a real token.
                vectors = self.model(**inputs).last_hidden_state[:, -1]
                vectors = self.torch.nn.functional.normalize(vectors.float(), p=2, dim=1)
            result.append(vectors.cpu().numpy())
        return np.concatenate(result) if result else np.empty((0, self.model.config.hidden_size), dtype="float32")


class Reranker(LocalModel):
    def __init__(self, path, device="cuda"):
        super().__init__(path, causal=True, device=device)
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.labels = [self.tokenizer.convert_tokens_to_ids(x) for x in ("no", "yes")]

    def score(self, query, texts, batch_size=4, max_length=1536):
        if max_length <= len(self.prefix) + len(self.suffix) + 32:
            raise ValueError("Reranker max length too short")
        scores = []
        for start in range(0, len(texts), batch_size):
            pairs = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {text}"
                     for text in texts[start:start + batch_size]]
            ids = self.tokenizer(pairs, add_special_tokens=False, truncation=True,
                                 max_length=max_length - len(self.prefix) - len(self.suffix))["input_ids"]
            inputs = self.tokenizer.pad({"input_ids": [self.prefix + x + self.suffix for x in ids]},
                                        padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, self.labels].float()
                scores.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
        return scores