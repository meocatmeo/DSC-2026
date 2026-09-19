# Candidate expansion — CPU measurement, both hypotheses

*17 September 2026. No GPU used. Reproduced from `artifacts2/store.sqlite`, the cached
lexical top-200 for all 1,050 validation questions, and `artifacts/splits/validation.json`.*

> **Verdict: neither change clears the gate you set.** Best coverage gain is **+0.0040**
> candidate recall against a **+0.0194** available from depth alone. Do not rerank the
> recovered candidates.

---

## 1. The mechanism is real — more real than D1 suggested

`PLAN-legalir4` D1 killed citations because only **0.5% of questions** name a document.
That measured the wrong direction. Passages citing documents is a different regime:

| | |
|---|---|
| documents with a parsed number field | 7,769 / 8,532 (91.1%) |
| chunks containing a citation-shaped token | 18.7% |
| **chunks citing a document that exists in this corpus** | **12.7%** |
| documents citing ≥1 in-corpus document | 5,442 (63.8%) |
| out-degree | mean 3.0, median 2, max 133 |
| documents cited by someone | 3,009 (35.3%) |

There is a real citation graph here, and `q60858` is exactly the structure you described:

```
Q: Chi cục trưởng Chi cục Thuế cần đảm bảo thêm các trình độ, chứng chỉ nào khác?

  gold 128674  558/QĐ-TCT        job standards for tax leadership    lexical rank 1
  gold 262254  01/2014/TT-BGDĐT  6-level language competency frame   NOT in top-200
  gold 280768  03/2014/TT-BTTTT  IT skills standard                  NOT in top-200
```

Both unreachable golds are **cited by the rank-1 document**. No text similarity will ever
retrieve a language-competency framework from a question about tax officials. One citation
hop reaches both. This is `PLAN-legalir4` D3's "multi-gold answers are related" finding
with the relation finally named: *the top hit cites the other answers.*

---

## 2. And it is rare

| | |
|---|---|
| multi-gold questions | 93 / 1,050 (8.9%) |
| missing-gold instances vs lexical top-50 | 32 |
| **... cited by a gold that WAS retrieved** | **5 (15.6%)** |

`q60858` is two of those five. Five instances across 1,050 questions.

---

## 3. Expansion against the depth control

Candidate recall over all 1,050 validation questions. `pool/q` is what the reranker would
have to score, so it is the cost axis.

| config | pool/q | recall | vs base@50 | recall per added doc |
|---|---|---|---|---|
| base @50 | 49.7 | 0.9600 | — | — |
| **base @100** *(depth)* | 97.3 | **0.9737** | **+0.0137** | 0.00029 |
| **base @200** *(depth)* | 162.8 | **0.9794** | **+0.0194** | 0.00017 |
| base @50 + cites of top-1 | 50.9 | 0.9610 | +0.0010 | **0.00083** |
| base @50 + cites of top-3 | 53.9 | 0.9616 | +0.0016 | 0.00023 |
| base @50 + cites of top-5 | 56.8 | 0.9616 | +0.0016 | 0.00023 |
| base @50 + cites of top-10 | 64.3 | 0.9640 | +0.0040 | 0.00027 |

Blanket expansion is worse still — at matched pool size depth wins everywhere up to 200:

| | pool/q | recall |
|---|---|---|
| base @100 | 97.3 | 0.9737 |
| cites of top-50 | 124.5 | 0.9706 |
| base @200 | 162.8 | 0.9794 |
| cites of top-100 | 229.8 | 0.9800 |

**Citation expansion is the most *efficient* thing available per added document — at
top-1, 0.00083 against depth's 0.00029 — and the least *consequential*.** +0.0010 is one
question. The detection floor on 1,050 questions is ~0.0094, so every row in the
expansion block is unmeasurable even in candidate recall, before any conversion loss.

---

## 4. Colloquial query expansion

The term you named does break lexical retrieval:

| question | gold in lexical top-200? |
|---|---|
| `q134012` Phạt nguội là gì? | **no** |
| `q164280` ... bị phạt nguội thì có thể nộp phạt qua đường bưu điện ... | **no** |
| `q1206` Dùng biển số giả để trốn phạt nguội ... | yes, rank 16 |

Two of three fail, against a 3% base rate. The mechanism is genuine.

The population is not. `phạt nguội` appears in **3** of 1,050 questions. Across nine
common colloquial terms (`sổ đỏ`, `căn cước công dân`, `bảo hiểm thất nghiệp`,
`tai nạn lao động`, …) the total is ~17 questions, most of which already work.

And the decisive check — **is the residual a vocabulary problem at all?**

> Of the 34 gold documents outside lexical top-200, **31 already share more than half of
> their question's tokens.** Only **3** are vocabulary mismatches.

The words match and the document still does not rank. That is discrimination among
similar text, not a lexicon gap, and no amount of query expansion touches it.

---

## 5. Two things that would change these numbers

**This is the lexical channel only,** because `artifacts2/dense` is not in this workspace.
It matters, and it cuts against both hypotheses:

| | candidate recall |
|---|---|
| lexical @200 (measured here) | 0.9794 |
| **v9 fused @50** (ANALYSIS-v7) | **0.9805** |

The fused pool at depth 50 already beats lexical at depth 200. Dense retrieval is
recovering much of what I measured as "unreachable", so **every number in §3 is an upper
bound on what expansion adds to v9** — the true figure is smaller, not larger.

To settle it exactly I need the fused top-200 per validation question, which the v11
cache already contains:

```bash
python - <<'EOF'
import gzip, json
out = {}
with gzip.open('validation.jsonl.gz', 'rt', encoding='utf-8') as s:
    for line in s:
        r = json.loads(line)
        out[r['id']] = {'fused': [d for d, _ in r['fused']], 'gold': r['gold']}
json.dump(out, open('fused_validation_top200.json', 'w'))
EOF
```

That is a ~5 MB file and rebuilding §3 against it is minutes of CPU.

---

## 6. Recommendation

**Do not rerank the recovered candidates.** Your own gate — *"rerank only if this actually
improves coverage"* — has not been passed: +0.0040 at best, against a 0.0094 floor, on a
baseline that overstates the opportunity.

Two further reasons to hold:

- **Conversion.** v11 changed 55 top-five sets and zero recalls. A recovered document
  enters at pool rank ~55 and is, in `q60858`'s case, a language-competency standard
  competing against 50 tax documents. Surviving to the top five is exactly the operation
  measured at ~0.
- **Depth is strictly better and free of new concepts.** If coverage is the goal,
  `candidates: 50 → 100` is +0.0137 for 2× rerank cost, no new code.

The one version of this idea not ruled out is **not** expanding the pool: give the stacker
a `cited_by_top1` feature, so a recovered document carries the evidence that retrieved it
instead of being thrown into a text-similarity contest it will lose. That is CPU-only on
an existing cache. But note it is a *ranking feature*, and §9 of `SUMMARY.md` is the
measurement of what ranking features buy here — so I would expect it to change the
ordering and not the score.

---

## Artifacts

| file | what |
|---|---|
| `analysis/v12/citations.pkl` | document→cited-documents and chunk→cited-documents maps |
| this file | the measurements above |

The citation graph took 15 s to build over all 341,401 chunks and is reusable for any
follow-up.
