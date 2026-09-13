"""Local model loading, pooling and the combined parameter audit.

Every checkpoint is loaded from a local directory with `local_files_only=True`:
no hosted inference, no remote code, no API of any kind. The deployed system is
audited as a whole against the competition's parameter budget.
"""
import gc
import json
from pathlib import Path

BUDGET = 4_000_000_000

# Query side instruction, only for the checkpoints whose model card asks for one.
INSTRUCTION = ("Given a Vietnamese legal question, retrieve the legal document that "
               "contains the information needed to answer it.")

RERANK_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
                 'based on the Query and the Instruct provided. Note that the answer can '
                 'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def family(path):
    config = read_json(Path(path) / "config.json")
    return (config.get("model_type") or "").lower(), config


def pooling_for(model_type):
    """bge-m3 / XLM-R style encoders pool the CLS token; Qwen decoders the last one."""
    if model_type.startswith("xlm-roberta") or model_type in ("bert", "roberta", "camembert"):
        return "cls"
    return "last"


def needs_instruction(path):
    name = str(path).lower()
    return "qwen3-embedding" in name or "qwen3.5-embed" in name


def torch_dtype(torch, device):
    if device == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Local:
    """One locally stored checkpoint plus its parameter count."""

    def __init__(self, path, kind="encoder", device="cuda", dtype=None):
        import torch
        from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                                  AutoModelForSequenceClassification, AutoTokenizer)

        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Local model directory not found: {path}. Run `download` first.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA build of PyTorch, "
                               "or pass --device cpu explicitly.")
        self.torch = torch
        self.path = path
        self.device = device
        self.kind = kind
        self.model_type, self.config = family(path)
        self.pooling = pooling_for(self.model_type)
        loader = {"encoder": AutoModel, "sequence": AutoModelForSequenceClassification,
                  "causal": AutoModelForCausalLM}[kind]
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True,
            padding_side="left" if (kind == "causal" or self.pooling == "last") else "right")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = loader.from_pretrained(
            path, local_files_only=True, torch_dtype=dtype or torch_dtype(torch, device),
            attn_implementation="sdpa", trust_remote_code=False).to(device).eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.provenance = (read_json(path / "provenance.json")
                           if (path / "provenance.json").is_file() else {"path": str(path)})

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def audit(parameters, budget=BUDGET):
    """Fail loudly if the deployed system is not strictly under the budget."""
    total = sum(parameters.values())
    if total >= budget:
        detail = ", ".join(f"{k}={v:,}" for k, v in parameters.items())
        raise ValueError(f"Deployed system has {total:,} parameters ({detail}), "
                         f"which is not strictly below {budget:,}")
    return {"components": parameters, "total": total, "budget": budget,
            "headroom": budget - total}
