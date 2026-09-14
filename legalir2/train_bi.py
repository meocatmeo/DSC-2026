"""Fine tune the bi-encoder on the organiser's labels with mined hard negatives.

Objective: InfoNCE over one positive passage against that question's mined hard
negatives plus every other passage in the batch. Training data comes only from
`train.json` labels and corpus passages; the held out validation split is never
used for weight updates, only for choosing between finished checkpoints.
"""
import json
import math
import random
import time
from pathlib import Path

from .mine import load_groups
from .store import Store, write_json


def cosine_schedule(optimizer, warmup, total):
    def factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    import torch
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(store_path, groups_path, model_path, output, device="cuda", epochs=2,
          batch_size=8, negatives=7, learning_rate=1e-5, temperature=0.05,
          query_length=96, passage_length=512, warmup_ratio=0.05, seed=2026,
          accumulate=1, checkpointing=True, max_groups=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from .models import amp_setup, dtype_kwarg, family, pooling_for

    store = Store(store_path)
    groups = load_groups(groups_path)
    if max_groups:
        groups = groups[:max_groups]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model_type, _ = family(model_path)
    pooling = pooling_for(model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                             padding_side="left" if pooling == "last" else "right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                      attn_implementation="sdpa",
                                      **dtype_kwarg(torch.float32)).to(device)
    if checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    def pool(hidden, mask):
        if pooling == "cls":
            return hidden[:, 0]
        if pooling == "mean":
            weight = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weight).sum(1) / weight.sum(1).clamp(min=1e-6)
        return hidden[:, -1]

    def encode(texts, max_length):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                           return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=amp_dtype, enabled=use_amp):
            hidden = model(**inputs).last_hidden_state
        vectors = pool(hidden, inputs["attention_mask"]).float()
        return torch.nn.functional.normalize(vectors, p=2, dim=1)

    use_amp, amp_dtype, scaler = amp_setup(torch, device)
    # cosine_schedule counts *optimizer* steps, and with gradient accumulation the
    # optimizer advances once per `accumulate` forwards. Sizing the schedule in
    # forwards silently stretches it: warmup runs `accumulate` times too long and the
    # cosine only reaches 1/accumulate of its arc, so the learning rate finishes near
    # its peak instead of near zero and the final weights are a mid-training snapshot.
    # Nothing in the loss curve shows this - it just trains to a worse checkpoint.
    forwards = (len(groups) // batch_size) * epochs
    steps = max(1, forwards // max(1, accumulate))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = cosine_schedule(optimizer, int(warmup_ratio * steps), max(1, steps))
    started, step, history = time.time(), 0, []

    for epoch in range(epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = [groups[i] for i in order[start:start + batch_size]]
            queries, passages = [], []
            for group in batch:
                queries.append(group["question"])
                positive = rng.choice(group["positives"])
                pool_negatives = list(group["negatives"])
                rng.shuffle(pool_negatives)
                picked = pool_negatives[:negatives]
                while len(picked) < negatives and pool_negatives:
                    picked.append(rng.choice(pool_negatives))
                passages.append(store.chunk_text(positive))
                passages.extend(store.chunk_text(c) for c in picked)
            query_vectors = encode(queries, query_length)
            passage_vectors = encode(passages, passage_length)
            logits = query_vectors @ passage_vectors.T / temperature
            targets = torch.arange(len(batch), device=device) * (1 + negatives)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            scaled = loss / accumulate
            scaler.scale(scaled).backward() if scaler else scaled.backward()
            if (step + 1) % accumulate == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                with torch.no_grad():
                    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
                history.append({"step": step, "loss": round(loss.item(), 4),
                                "in_batch_accuracy": round(accuracy, 4)})
                print(f"  epoch {epoch} step {step}/{forwards} loss {loss.item():.4f} "
                      f"acc {accuracy:.3f} {time.time() - started:.0f}s", flush=True)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.to(torch.float32 if device == "cpu" else amp_dtype)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    record = {"base": str(model_path), "groups": len(groups), "epochs": epochs,
                "batch_size": batch_size, "negatives": negatives, "temperature": temperature,
                "learning_rate": learning_rate, "pooling": pooling, "steps": step, "optimizer_steps": steps,
                "seconds": round(time.time() - started), "history": history[-50:]}
    write_json(output / "training.json", record)
    from .train_ce import derived_provenance
    write_json(output / "provenance.json", derived_provenance(model_path, record))
    store.close()
    print(f"Saved fine tuned bi-encoder to {output}")
