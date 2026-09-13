"""Cross encoder reranking, for both head shapes present on the approved list.

* `sequence` - a classification head with one logit (bge-reranker-v2-m3 and the
  Vietnamese rerankers finetuned from it). The relevance score is that logit.
* `causal`   - the Qwen reranker prompt, where relevance is the probability the
  model assigns to "yes" over "no" at the final position.

Both run entirely from local weights. Scores are order preserving only within a
single question, so they are always used as a ranking, never compared across
questions.
"""
from .models import INSTRUCTION, RERANK_PREFIX, RERANK_SUFFIX, Local


def head_for(path):
    from .models import family
    model_type, config = family(path)
    if config.get("architectures") and any("SequenceClassification" in a
                                           for a in config["architectures"]):
        return "sequence"
    return "causal" if model_type.startswith("qwen") else "sequence"


class Reranker:
    def __init__(self, path, device="cuda", head=None, max_length=1024):
        self.head = head or head_for(path)
        self.local = Local(path, kind="sequence" if self.head == "sequence" else "causal",
                           device=device)
        self.parameters = self.local.parameters
        self.max_length = max_length
        if self.head == "causal":
            tokenizer = self.local.tokenizer
            self.prefix = tokenizer.encode(RERANK_PREFIX, add_special_tokens=False)
            self.suffix = tokenizer.encode(RERANK_SUFFIX, add_special_tokens=False)
            self.labels = [tokenizer.convert_tokens_to_ids(x) for x in ("no", "yes")]

    # ------------------------------------------------------------------ scoring

    def score(self, query, passages, batch_size=8, max_length=None):
        max_length = max_length or self.max_length
        if not passages:
            return []
        return (self._score_causal(query, passages, batch_size, max_length)
                if self.head == "causal"
                else self._score_sequence(query, passages, batch_size, max_length))

    def _score_sequence(self, query, passages, batch_size, max_length):
        torch = self.local.torch
        out = []
        for start in range(0, len(passages), batch_size):
            batch = passages[start:start + batch_size]
            inputs = self.local.tokenizer([query] * len(batch), batch, padding=True,
                                          truncation="only_second", max_length=max_length,
                                          return_tensors="pt")
            inputs = {k: v.to(self.local.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.local.model(**inputs).logits.float().view(-1)
            out.extend(logits.cpu().tolist())
        return out

    def _score_causal(self, query, passages, batch_size, max_length):
        torch = self.local.torch
        budget = max_length - len(self.prefix) - len(self.suffix)
        if budget < 64:
            raise ValueError("Reranker max length is too short for the prompt")
        out = []
        for start in range(0, len(passages), batch_size):
            batch = [f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {text}"
                     for text in passages[start:start + batch_size]]
            ids = self.local.tokenizer(batch, add_special_tokens=False, truncation=True,
                                       max_length=budget)["input_ids"]
            inputs = self.local.tokenizer.pad(
                {"input_ids": [self.prefix + x + self.suffix for x in ids]},
                padding=True, return_tensors="pt")
            inputs = {k: v.to(self.local.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.local.model(**inputs, logits_to_keep=1).logits[:, -1, self.labels]
                probability = logits.float().softmax(dim=1)[:, 1]
            out.extend(probability.cpu().tolist())
        return out

    def close(self):
        self.local.close()
