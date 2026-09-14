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


def checkpoint_identity(provenance):
    """The part of a provenance record that says WHICH weights these are.

    Deliberately excludes `seconds` - how long the download took - and `endpoint`.
    Comparing whole provenance dicts to decide whether two checkpoints match means
    comparing a stopwatch reading, so re-downloading the same weights produces a
    "different" checkpoint and anything built against it (a dense index, say) is
    rejected as stale. Identity is the repository, the resolved commit, and the file
    hashes; everything else is a note about the download, not about the model.
    """
    if not provenance:
        return {}
    if "derived_from" in provenance:
        return {"derived_from": checkpoint_identity(provenance["derived_from"]),
                "training": provenance.get("training", {}).get("base")}
    keep = {k: provenance[k] for k in ("repository", "revision", "commit") if k in provenance}
    files = provenance.get("files")
    if files:
        keep["files"] = {name: record.get("sha256") for name, record in sorted(files.items())}
    elif "path" in provenance:
        keep["path"] = provenance["path"]
    return keep


def context_limit(config, tokenizer=None):
    """The longest sequence this checkpoint can actually accept.

    Approved checkpoints differ by more than an order of magnitude: bge-m3 and its
    Vietnamese derivatives extend the position table to 8192, plain XLM-R stops at
    512, and PhoBERT based rerankers stop at 256. Asking for more than the position
    table holds raises an index error deep inside the embedding layer - which, on a
    full corpus index, happens an hour into the run. Clamping up front turns that
    into a printed note.
    """
    positions = config.get("max_position_embeddings")
    limits = []
    if positions:
        # Learned absolute positions reserve slots for padding and the two specials.
        offset = 2 if (config.get("model_type") or "").startswith(("xlm-roberta", "roberta")) else 0
        limits.append(int(positions) - offset)
    declared = getattr(tokenizer, "model_max_length", None) if tokenizer else None
    if isinstance(declared, int) and 0 < declared < 1_000_000:
        limits.append(declared)
    return min(limits) if limits else None


def clamp_length(requested, config, tokenizer=None, label=""):
    limit = context_limit(config, tokenizer)
    if limit and requested > limit:
        print(f"  note: {label or 'model'} accepts at most {limit} tokens; "
              f"using {limit} instead of {requested}", flush=True)
        return limit
    return requested


def needs_instruction(path):
    name = str(path).lower()
    return "qwen3-embedding" in name or "qwen3.5-embed" in name


def torch_dtype(torch, device):
    if device == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def amp_setup(torch, device):
    """Pick the mixed precision mode this card actually supports.

    bfloat16 needs Ampere or newer. On older cards (Turing and before) bfloat16
    autocast is unavailable, so float16 is used with a gradient scaler instead -
    otherwise training either raises or silently runs at fp32 speed. Returns
    (enabled, dtype, scaler).
    """
    if not device.startswith("cuda"):
        return False, torch.float32, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, None
    print("  note: this GPU has no bfloat16 support; training in float16 with a "
          "gradient scaler", flush=True)
    return True, torch.float16, torch.amp.GradScaler("cuda")


def dtype_kwarg(value):
    """`torch_dtype` was renamed to `dtype` in transformers 5.

    The GPU environment pins 4.51.3 while a freshly created CPU environment gets 5.x,
    so the keyword is chosen at run time rather than pinned to one of them.
    """
    import transformers
    major = int(str(transformers.__version__).split(".")[0])
    return {"dtype": value} if major >= 5 else {"torch_dtype": value}


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
            path, local_files_only=True, attn_implementation="sdpa", trust_remote_code=False,
            **dtype_kwarg(dtype or torch_dtype(torch, device))).to(device).eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.context_limit = context_limit(self.config, self.tokenizer)
        self.provenance = (read_json(path / "provenance.json")
                           if (path / "provenance.json").is_file() else {"path": str(path)})
        self.identity = checkpoint_identity(self.provenance)

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
