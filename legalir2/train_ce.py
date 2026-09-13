"""Fine tune the cross encoder reranker, optionally distilling a larger teacher.

Objective: listwise softmax cross entropy over one positive passage and that
question's mined hard negatives. Listwise training matches how the reranker is
used - it only has to order one question's candidates - and in practice moves
Recall@5 far more than pointwise binary training does.

When a teacher score cache is supplied the loss adds a KL term against the
teacher's distribution over the same group. The teacher runs offline only; the
saved student is the sole model that ships, so the deployed parameter budget is
the student's.
"""
import json
import random
import time
from pathlib import Path

from .mine import load_groups
from .models import amp_setup, dtype_kwarg
from .store import Store, write_json
from .train_bi import cosine_schedule


def load_teacher(path):
    if not path:
        return {}
    cache = {}
    for line in Path(path).open(encoding="utf-8"):
        record = json.loads(line)
        cache[record["id"]] = {int(k): float(v) for k, v in record["scores"].items()}
    return cache


def train(store_path, groups_path, model_path, output, device="cuda", epochs=2,
          group_size=8, groups_per_step=4, learning_rate=1e-5, max_length=512,
          warmup_ratio=0.05, seed=2026, accumulate=1, checkpointing=True,
          teacher_path=None, distill_weight=0.4, distill_temperature=1.0,
          max_groups=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    store = Store(store_path)
    groups = load_groups(groups_path)
    if max_groups:
        groups = groups[:max_groups]
    teacher = load_teacher(teacher_path)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, attn_implementation="sdpa", num_labels=1,
        **dtype_kwarg(torch.float32)).to(device)
    if checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    negatives = group_size - 1
    steps = (len(groups) // groups_per_step) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = cosine_schedule(optimizer, int(warmup_ratio * steps), max(1, steps))
    started, step, history = time.time(), 0, []
    use_amp, amp_dtype, scaler = amp_setup(torch, device)

    for epoch in range(epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for start in range(0, len(order) - groups_per_step + 1, groups_per_step):
            batch = [groups[i] for i in order[start:start + groups_per_step]]
            pairs, teacher_rows, keep = [], [], []
            for group in batch:
                positive = rng.choice(group["positives"])
                pool = [c for c in group["negatives"] if c != positive]
                rng.shuffle(pool)
                picked = pool[:negatives]
                if len(picked) < negatives:
                    if not picked:
                        continue
                    while len(picked) < negatives:
                        picked.append(rng.choice(pool))
                chunks = [positive] + picked
                keep.append(group)
                pairs.extend((group["question"], store.chunk_text(c)) for c in chunks)
                cache = teacher.get(group["id"], {})
                teacher_rows.append([cache.get(c) for c in chunks] if cache else None)
            if not keep:
                continue
            inputs = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                               truncation="only_second", max_length=max_length,
                               return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=amp_dtype, enabled=use_amp):
                logits = model(**inputs).logits
            logits = logits.float().view(len(keep), group_size)
            targets = torch.zeros(len(keep), dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            distilled = None
            if any(row is not None for row in teacher_rows):
                rows, mask = [], []
                for row in teacher_rows:
                    if row is None or any(v is None for v in row):
                        rows.append([0.0] * group_size)
                        mask.append(0.0)
                    else:
                        rows.append(row)
                        mask.append(1.0)
                soft = torch.tensor(rows, dtype=torch.float32, device=device)
                weight = torch.tensor(mask, dtype=torch.float32, device=device)
                target = torch.log_softmax(soft / distill_temperature, dim=1)
                student = torch.log_softmax(logits / distill_temperature, dim=1)
                divergence = torch.nn.functional.kl_div(student, target, log_target=True,
                                                        reduction="none").sum(dim=1)
                distilled = (divergence * weight).sum() / weight.sum().clamp(min=1.0)
                loss = loss + distill_weight * distilled
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
                accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
                history.append({"step": step, "loss": round(loss.item(), 4),
                                "top1": round(accuracy, 4),
                                "kl": round(float(distilled), 4) if distilled is not None else None})
                print(f"  epoch {epoch} step {step}/{steps} loss {loss.item():.4f} "
                      f"top1 {accuracy:.3f} {time.time() - started:.0f}s", flush=True)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.to(amp_dtype if device.startswith("cuda") else torch.float32)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    source = Path(model_path) / "provenance.json"
    if source.is_file():
        (output / "provenance.json").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(output / "training.json",
               {"base": str(model_path), "groups": len(keep) and len(groups), "epochs": epochs,
                "group_size": group_size, "groups_per_step": groups_per_step,
                "learning_rate": learning_rate, "max_length": max_length,
                "teacher": str(teacher_path) if teacher_path else None,
                "distill_weight": distill_weight if teacher_path else 0,
                "steps": step, "seconds": round(time.time() - started), "history": history[-50:]})
    store.close()
    print(f"Saved fine tuned reranker to {output}")
