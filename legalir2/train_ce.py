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


def derived_provenance(base_path, training):
    """Provenance for a checkpoint we trained, not one we downloaded.

    Copying the base model's provenance.json into a fine-tuned directory looks
    harmless and is not: it claims file hashes that training has just invalidated,
    so `verify-models` reports the trained weights as corrupt. Record the lineage
    instead, and leave out the `files` block that means "hash verifiable".
    """
    base = Path(base_path) / "provenance.json"
    return {"derived_from": (json.loads(base.read_text(encoding="utf-8"))
                             if base.is_file() else {"path": str(base_path)}),
            "training": training}


def group_targets(positive_count, group_size, device, torch):
    """Soft target distribution over one group: mass split evenly across the positives.

    With a single positive this is a one-hot vector and the loss is exactly the softmax
    cross entropy used before - which is the point. Multi-gold questions are 8.9% of the
    set but produce 34% of the lost recall, because the scorer gives partial credit for
    each gold document while the old objective could only name one of them as correct.
    Splitting the mass says "all of these belong at the top", which is what the metric
    actually rewards.
    """
    rows = torch.zeros(len(positive_count), group_size, device=device)
    for index, count in enumerate(positive_count):
        rows[index, :count] = 1.0 / count
    return rows


def recall_pairwise_loss(document_scores, positive_counts, gold_counts, torch,
                         cutoff=5):
    """Recall@k-aware pairwise loss over document scores.

    Positive documents never compete with one another.  A pair is weighted by the
    exact change in that question's Recall@k if the two current ranks were swapped.
    Ranks are detached so the selected boundary remains stable during backprop.
    """
    losses = []
    for raw, positives, gold_total in zip(document_scores, positive_counts, gold_counts):
        if positives >= len(raw) or gold_total <= 0:
            continue
        # Standardise this question's document scores before the pairwise term.
        #
        # softplus(s_negative - s_gold) is scale dependent, and a competent reranker
        # separates gold from mined negatives by 10-16 raw logits. softplus(-16) is
        # 1.1e-7: it prints as 0.0000 and back-propagates nothing. Two training runs
        # died this way, both looking like "the objective did not help" when the
        # objective had simply never received a gradient.
        #
        # Dividing by the standard deviation - NOT detached - makes the loss invariant
        # to the checkpoint's logit range, so the only way to reduce it is to reorder
        # the documents. Detaching the scale would instead let the model lower the loss
        # by inflating its spread, which changes no ranking at all.
        scores = (raw - raw.mean()) / raw.std().clamp(min=1e-3)
        order = torch.argsort(scores.detach(), descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(order), device=order.device)
        pairs = []
        for positive in range(positives):
            for negative in range(positives, len(scores)):
                weight = abs(int(ranks[positive] < cutoff) - int(ranks[negative] < cutoff)) / gold_total
                if weight:
                    pairs.append(weight * torch.nn.functional.softplus(scores[negative] - scores[positive]))
        if pairs:
            # Normalize by the total metric weight for this question.  Averaging by
            # pair count would dilute multi-gold questions simply because they have
            # more positive-negative pairs, despite each question contributing one
            # macro-recall observation.
            losses.append(torch.stack(pairs).sum() /
                          sum(abs(int(ranks[p] < cutoff) - int(ranks[n] < cutoff)) / gold_total
                              for p in range(positives) for n in range(positives, len(scores))))
    if not losses:
        # Keep a valid autograd graph when every group is already correctly separated.
        if document_scores:
            return torch.stack([scores.sum() * 0.0 for scores in document_scores]).sum()
        return torch.tensor(0.0, device=document_scores[0].device if document_scores else None,
                            requires_grad=True)
    return torch.stack(losses).mean()


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
          max_groups=None, max_positives=1, lora_rank=0, lora_alpha=None,
          document_loss=False):
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                              AutoTokenizer)

    from .models import INSTRUCTION, rerank_prompt, template_for
    from .rerank import head_for

    # A causal reranker is NOT a classifier with a different name. Prism and the Qwen
    # rerankers score sigmoid(logit_yes - logit_no) at the final position, using the
    # pretrained LM head. Loading one with AutoModelForSequenceClassification discards
    # that head, initialises a random one, and trains an objective the inference path
    # never uses - it would run, converge, and produce a model that scores nothing like
    # what rerank.py asks of it.
    head = head_for(model_path)
    causal = head == "causal"

    store = Store(store_path)
    groups = load_groups(groups_path)
    if max_groups:
        groups = groups[:max_groups]
    teacher = load_teacher(teacher_path)
    if document_loss and teacher:
        raise ValueError("--document-loss cannot be combined with teacher scores")
    if document_loss and not any(group.get("positives_by_gold") for group in groups):
        raise ValueError("--document-loss requires groups mined with positives_by_gold")
    rng = random.Random(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True,
        padding_side="left" if causal else "right")   # left, so [:, -1] is the real last token
    if causal:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, attn_implementation="sdpa",
            **dtype_kwarg(torch.float32)).to(device)
        template = template_for(model_path)
        prefix = tokenizer.encode(rerank_prompt(template)[0], add_special_tokens=False)
        suffix = tokenizer.encode(rerank_prompt(template)[1], add_special_tokens=False)
        label_ids = [tokenizer.convert_tokens_to_ids(x) for x in ("no", "yes")]
        if any(i is None or i < 0 for i in label_ids):
            raise ValueError(f"{model_path}: tokenizer has no single-token yes/no")
        instruction = template.get("instruction", INSTRUCTION)
        print(f"  causal reranker: scoring logit_yes - logit_no at the last position")
        print(f"  template: {template['system'][:60]}...")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, attn_implementation="sdpa", num_labels=1,
            **dtype_kwarg(torch.float32)).to(device)

    if lora_rank:
        # LoRA is not only a memory trick here: a full fine-tune of a 2B causal model
        # needs ~36 GB before activations, and the adapter is merged back on save so the
        # deployed artefact is a plain checkpoint with no peft dependency and the same
        # parameter count as the base.
        from peft import LoraConfig, get_peft_model
        targets = sorted({n.split(".")[-1] for n, _ in model.named_modules()
                          if n.split(".")[-1] in
                          ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")})
        if not targets:
            raise ValueError(f"{model_path}: no recognisable attention/MLP projections "
                             "to attach LoRA to")
        model = get_peft_model(model, LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha or 2 * lora_rank, lora_dropout=0.05,
            bias="none", target_modules=targets,
            task_type="CAUSAL_LM" if causal else "SEQ_CLS"))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA r={lora_rank} on {targets}")
        print(f"  trainable {trainable/1e6:.1f}M of {total/1e9:.2f}B "
              f"({100*trainable/total:.2f}%)")
    if checkpointing:
        model.gradient_checkpointing_enable()
        if lora_rank:
            model.enable_input_require_grads()   # checkpointing needs a grad-carrying input
    model.train()

    negatives = group_size - 1
    # cosine_schedule counts *optimizer* steps, and with gradient accumulation the
    # optimizer advances once per `accumulate` forwards. Sizing the schedule in
    # forwards silently stretches it: warmup runs `accumulate` times too long and the
    # cosine only reaches 1/accumulate of its arc, so the learning rate finishes near
    # its peak instead of near zero and the final weights are a mid-training snapshot.
    # Nothing in the loss curve shows this - it just trains to a worse checkpoint.
    forwards = (len(groups) // groups_per_step) * epochs
    steps = max(1, forwards // max(1, accumulate))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = cosine_schedule(optimizer, int(warmup_ratio * steps), max(1, steps))
    started, step, history = time.time(), 0, []
    use_amp, amp_dtype, scaler = amp_setup(torch, device)

    for epoch in range(epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        for start in range(0, len(order) - groups_per_step + 1, groups_per_step):
            batch = [groups[i] for i in order[start:start + groups_per_step]]
            pairs, teacher_rows, keep, positive_count = [], [], [], []
            document_batches, gold_counts = [], []
            for group in batch:
                # One passage per gold document, up to max_positives. The group stays
                # exactly group_size long either way, so the logits still reshape.
                by_gold = group.get("positives_by_gold") or {}
                if document_loss and not by_gold:
                    # A document loss cannot safely mix legacy flat groups with
                    # variable-length document bags in one flattened tokenizer batch.
                    continue
                if document_loss and by_gold:
                    in_pool = group.get("gold_ranks") or {}
                    gold_docs = [doc for doc in by_gold if in_pool.get(doc) is not None]
                    # Keep every available labelled document, then fill the remaining
                    # slots with the mined hard negatives.  Each document is a bag: all
                    # mined positive passages are scored and reduced by max below.
                    if len(gold_docs) >= group_size:
                        gold_docs = gold_docs[:group_size - 1]
                    docs = [(doc, list(by_gold[doc])) for doc in gold_docs]
                    for chunk in list(group["negatives"]):
                        if len(docs) >= group_size:
                            break
                        docs.append((None, [chunk]))
                    if len(docs) <= len(gold_docs):
                        continue
                    document_batches.append((len(gold_docs), len(group.get("gold", gold_docs)), docs))
                    keep.append(group)
                    for doc, chunks in docs:
                        pairs.extend((group["question"], store.chunk_text(c)) for c in chunks)
                    continue
                if max_positives > 1 and len(by_gold) > 1:
                    chosen = list(by_gold.values())[:max_positives]
                    chunk_positives = [rng.choice(v) for v in chosen]
                else:
                    chunk_positives = [rng.choice(group["positives"])]
                taken = set(chunk_positives)
                want = group_size - len(chunk_positives)
                pool = [c for c in group["negatives"] if c not in taken]
                rng.shuffle(pool)
                picked = pool[:want]
                if len(picked) < want:
                    if not picked:
                        continue
                    while len(picked) < want:
                        picked.append(rng.choice(pool))
                chunks = chunk_positives + picked
                positive_count.append(len(chunk_positives))
                keep.append(group)
                pairs.extend((group["question"], store.chunk_text(c)) for c in chunks)
                cache = teacher.get(group["id"], {})
                teacher_rows.append([cache.get(c) for c in chunks] if cache else None)
            if not keep:
                continue
            if causal:
                budget = max_length - len(prefix) - len(suffix)
                if budget < 64:
                    raise ValueError("max_length too short for this prompt template")
                texts = [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}"
                         for q, d in pairs]
                ids = tokenizer(texts, add_special_tokens=False, truncation=True,
                                max_length=budget)["input_ids"]
                inputs = tokenizer.pad(
                    {"input_ids": [prefix + x + suffix for x in ids]},
                    padding=True, return_tensors="pt")
            else:
                inputs = tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                                   padding=True, truncation="only_second",
                                   max_length=max_length, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=amp_dtype, enabled=use_amp):
                if causal:
                    # Only the final position is ever read, and a causal LM head over a
                    # ~150k vocab at 32 x 768 positions would allocate ~15 GB of logits
                    # per step in fp32. logits_to_keep slices before the head, which is
                    # the difference between fitting on any card and fitting on none.
                    try:
                        out = model(**inputs, logits_to_keep=1).logits
                    except TypeError:
                        out = model(**inputs, num_logits_to_keep=1).logits
                    pair = out[:, -1, label_ids].float()
                    raw = pair[:, 1] - pair[:, 0]      # logit_yes - logit_no
                else:
                    raw = model(**inputs).logits
            if document_loss and document_batches:
                cursor = 0
                document_scores, positive_counts, gold_totals = [], [], []
                for positives, gold_total, docs in document_batches:
                    values = []
                    for _, chunks in docs:
                        values.append(raw[cursor:cursor + len(chunks)].float().max())
                        cursor += len(chunks)
                    document_scores.append(torch.stack(values))
                    positive_counts.append(positives)
                    gold_totals.append(gold_total)
                loss = recall_pairwise_loss(document_scores, positive_counts, gold_totals,
                                            torch, cutoff=5)
                targets = None
                distilled = None
            else:
                logits = raw.float().view(len(keep), group_size)
                targets = group_targets(positive_count, group_size, device, torch)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                distilled = None
            if not document_loss and any(row is not None for row in teacher_rows):
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
                # "did the top-scoring candidate come from the positives" - the
                # multi-positive generalisation of top-1 against a one-hot target.
                if document_loss:
                    accuracy = sum(float(torch.argmax(s) < p) for s, p in
                                   zip(document_scores, positive_counts)) / max(1, len(document_scores))
                else:
                    accuracy = (targets.gather(1, logits.argmax(dim=1, keepdim=True))
                                > 0).float().mean().item()
                history.append({"step": step, "loss": round(loss.item(), 4),
                                "top1": round(accuracy, 4),
                                "kl": round(float(distilled), 4) if distilled is not None else None})
                print(f"  epoch {epoch} step {step}/{forwards} loss {loss.item():.4f} "
                      f"top1 {accuracy:.3f} {time.time() - started:.0f}s", flush=True)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if lora_rank:
        # Merge, so what ships is an ordinary checkpoint: no peft at inference, no
        # adapter to keep alongside the weights, and the same parameter count as the
        # base - which is what the 4B budget is audited against.
        print("  merging the LoRA adapter into the base weights")
        model = model.merge_and_unload()
    model.to(amp_dtype if device.startswith("cuda") else torch.float32)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    record = {"base": str(model_path), "groups": len(keep) and len(groups), "epochs": epochs,
                "group_size": group_size, "groups_per_step": groups_per_step,
                "max_positives": max_positives, "head": head,
                "document_loss": document_loss,
                "lora_rank": lora_rank,
                "learning_rate": learning_rate, "max_length": max_length,
                "teacher": str(teacher_path) if teacher_path else None,
                "distill_weight": distill_weight if teacher_path else 0,
                "steps": step, "optimizer_steps": steps, "seconds": round(time.time() - started), "history": history[-50:]}
    write_json(output / "training.json", record)
    write_json(output / "provenance.json", derived_provenance(model_path, record))
    store.close()
    print(f"Saved fine tuned reranker to {output}")
