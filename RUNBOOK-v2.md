# LegalIR v2 — what changed, why, and how to run it

`legalir2/` is a new package alongside the existing `legalir/`. The old BM25 path is
untouched, so the current submission route still works while v2 is being validated.

## Measured so far (CPU only, no neural stage yet)

Full held-out validation split, 1,050 questions, identical questions in both rows:

| System | Recall@5 | Precision@5 | candidate R@20 | candidate R@50 | candidate R@100 |
|---|---:|---:|---:|---:|---:|
| `legalir` BM25 only | 0.7379 | 0.1577 | — | — | — |
| `legalir` BM25 + phrase evidence (best baseline) | 0.7855 | 0.1676 | 0.9030 | 0.9332 | — (R@60 = 0.9344) |
| `legalir2` lexical stage, default settings | 0.8438 | 0.1806 | 0.9375 | 0.9600 | 0.9737 |
| `legalir2` lexical stage, tuned on training questions | **0.8483** | 0.1811 | 0.9337 | 0.9608 | 0.9752 |

Two things matter in that table. Recall@5 went up by **+0.063** before any neural
model runs at all. And the reranker's ceiling moved: the gold document is now inside
the top 50 for 96.1% of questions instead of 93.3%, and inside the top 100 for 97.5%.

The previous system's own numbers show where the remaining points are. BM25 candidate
recall@60 was 0.934 while final Recall@5 was 0.785, so about 0.15 recall was sitting
inside the candidate list and being ranked away. Zero-shot Qwen3-Reranker-0.6B
recovered almost none of it (public leaderboard 0.77 → 0.78). That gap is a
*ranking* problem, and a reranker fine-tuned on the organiser's own 7,000 labelled
questions is the direct fix.

## Why each change was made

**Article-aware chunking** (`legalir2/structure.py`). 91% of the 8,532 documents
contain an `Điều N.` article sequence and the median document is 4,869 words, so
fixed 180-word windows cut provisions in half and mixed unrelated articles. v2
reflows the hard-wrapped source text, recovers the Chương / Mục / Điều hierarchy, and
makes each article one retrieval unit carrying a breadcrumb of its parents. The
longest non-decreasing run of article numbers is kept, so `theo quy định tại Điều 5`
inside running text does not start a spurious unit. 341,401 chunks result (down from
521,955), and each one is a coherent legal provision a cross encoder can judge.

**The accented title is now indexed.** The supplied `name` field is an ASCII slug, so
accented questions could never match the document's own subject. v2 also extracts the
capitalised Vietnamese title block and the normalised document number (`110/2018/NĐ-CP`)
from the header — 98% and 91% coverage respectively — and indexes them as a separate
weighted field.

**A hashed word-bigram channel** (`legalir2/lexical.py`). This was the single largest
lexical gain. Vietnamese words span several syllables, so unigram BM25 over syllables
loses the phrase that carries the legal concept: `bảo hiểm xã hội` scores as three
unrelated tokens. Adding consecutive word pairs as a second BM25F channel, hashed into
2²² buckets so no word segmenter or external dictionary is needed, lifted Recall@5
from 0.767 to 0.825 and candidate R@20 from 0.895 to 0.951 on the first 300 validation
questions. It is the segmentation-free version of what a Vietnamese tokeniser buys.

**BM25F with real field weights and length normalisation**, replacing the flat
`"t1" OR "t2" OR …` FTS5 query where every term weighed the same and the title was
folded into the body. Postings are numpy arrays with the weights baked in, so a query
is gathers and adds: ~11 questions/second on two CPU cores, which is what made
parameter sweeping possible at all.

**Document scores aggregate several chunks.** The old code kept only a document's
single best chunk. Adding a decayed bonus from its next best chunks (`decay=0.15`,
`depth=8` after tuning) was worth about +0.03 Recall@5 on its own — a question is usually answered
by one article, but a document whose several articles all match is more likely to be
the right document.

**Hyperparameters are tuned on training questions, not the holdout** (`legalir2/tune.py`),
so the validation number above stays a clean check rather than something the settings
were fitted to. Coordinate descent over 600 training questions moved the composite
objective from 0.9231 to 0.9304 and, transferring to the untouched holdout, Recall@5
from 0.8438 to 0.8483. Those values are now the defaults in `legalir2/lexical.py`.
Note `crumb = 0.0`: the chapter/section path earns nothing as a lexical field, while it
still goes to the embedder and the cross encoder, where it says which part of a document
a passage came from.

## Start here: the CPU-only run

Everything measured above was produced on two CPU cores with no GPU, no model download
and no network access. Run this first on any laptop — it reproduces the numbers and
produces a submittable prediction file before any GPU time is spent.

**Requirements:** Python 3.11+ and **numpy** — that is the whole list. `torch` and
`transformers` are imported only by the dense, rerank and training modules, which this
path never touches.

**Resources:** ~1.4 GB of disk (652 MB chunk store, 700 MB of postings), ~1.9 GB peak RAM,
about 5 minutes of build plus 2 minutes for the full 1,050-question evaluation.

```powershell
Set-Location 'D:\DSC-2026'
py -c "import numpy; print(numpy.__version__)"    # if this fails: py -m pip install numpy

py -m unittest discover -s tests2 -t . -v         # 44 tests, 7 neural ones skip without torch
py -m legalir2 build-store                        # ~2 min  -> artifacts2\store.sqlite
py -m legalir2 index-lexical                      # ~3 min  -> artifacts2\lexical\*.npz

# Reproduce the headline number on the held-out split.
py -m legalir2 predict --no-dense --no-rerank --evaluate `
    --output 'D:\DSC-2026\outputs\v2-lexical-validation.json'
# expect: Recall 0.8483  Precision 0.1811
#         candidate recall {'@5': 0.8483, '@20': 0.9337, '@50': 0.9608, '@100': 0.9752}

# Then produce a real submission from the same CPU-only system.
py -m legalir2 predict --no-dense --no-rerank `
    --questions 'D:\DSC-2026\public-official.json' `
    --output 'D:\DSC-2026\outputs\submission-v2-lexical.json'
py -m legalir2 validate --predictions 'D:\DSC-2026\outputs\submission-v2-lexical.json'
py -m legalir2 pack --predictions 'D:\DSC-2026\outputs\submission-v2-lexical.json' `
    --output 'D:\DSC-2026\outputs\submission-v2-lexical.zip'
```

That ZIP is worth submitting on its own. On validation it beats the best configuration
you have run so far by +0.063 Recall@5, and it is a safe floor to measure the GPU work
against rather than a step on the way to it.

Two more things worth doing on CPU while the GPU is unavailable:

```powershell
# Re-derive the lexical settings yourself (~35 min). Runs on training questions, so it
# does not spend the validation split. It should land on the values already in
# legalir2/lexical.py; if it lands somewhere else, that is worth knowing.
py -m legalir2 tune --limit 600 --output 'D:\DSC-2026\outputs\v2-tuning.json'

# Read where the failures are. .diagnostics.json holds, per question, the rank of each
# gold document in the candidate list — the 13 questions with no rank at all are the
# ones the dense channel exists to fix.
py -c "import json; d=json.load(open(r'D:\DSC-2026\outputs\v2-lexical-validation.diagnostics.json',encoding='utf-8')); print([v['question'] for v in d.values() if all(v['candidate_ranks'].get(g) is None for g in v['relevant'])])"
```

`probe`, `index-dense`, `mine` (with `--no-dense` it is CPU-only but slow), `train-ce`
and `train-bi` all need `torch`; the training steps need a GPU to be practical.

## What still has to run on the GPU

Nothing neural has been executed yet. The code is written and unit-tested against tiny
random fixtures; it has not been run with real weights.

### Models (all from `[DSC@UIT 2026] Danh sách mô hình - Sheet1.csv`)

| Role | Checkpoint | Parameters | Ships? |
|---|---|---:|---|
| Bi-encoder | `AITeamVN/Vietnamese_Embedding_v2` | ~568M | yes |
| Cross encoder | `AITeamVN/Vietnamese_Reranker` | ~568M | yes |
| Distillation teacher | `Qwen/Qwen3-Reranker-4B` | ~4B | **no** — offline only |

Both deployed checkpoints are bge-m3 / bge-reranker-v2-m3 derivatives already tuned on
Vietnamese, which is why they are preferred over the Qwen3-0.6B pair currently in use.
Deployed total is ~1.14B parameters, and `python -m legalir2 audit` fails loudly if the
combination ever reaches 4B. Roughly 2.8B of headroom is left, so a larger second-stage
reranker (`infgrad/Prism-Qwen3.5-Reranker-2B`, ~2B, total ~3.14B) can be added later if
validation says it earns its place.

The teacher is used only to write a score cache to disk; the student is the sole model
in the submitted system, which is the arrangement your instructions allow.

### Runbook

```bash
cd /workspace/DSC-2026
source .venv/bin/activate
export HF_HUB_OFFLINE=0                      # only for `download`

python -m legalir2 build-store                # ~2 min,  341k chunks
python -m legalir2 index-lexical              # ~3 min,  730 MB of postings
python -m unittest discover -s tests2 -t . -v # 44 tests, neural ones included on GPU

python -m legalir2 download --which embedding reranker
python -m legalir2 verify-models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

# Screen checkpoints before committing an hour of indexing to one. `probe` ranks the
# gold passage against 23 hard distractors on 300 training questions: ~5 minutes each,
# enough to compare several sheet checkpoints and drop the weak ones.
python -m legalir2 probe --model models2/embedding --mode embedding
python -m legalir2 probe --model models2/reranker  --mode reranker

# 1. zero-shot baseline first, so every later number has something to beat
python -m legalir2 index-dense --batch-size 32            # ~1 h
python -m legalir2 predict --evaluate --limit 200 \
    --output outputs/v2-zeroshot-smoke.json               # smoke test, do not read quality into it
python -m legalir2 predict --evaluate \
    --output outputs/v2-zeroshot.json                     # full 1,050

# 2. mine training groups from the organiser's labels
python -m legalir2 mine                                   # ~20 min

# 3. train (reranker first: it carries most of the gain)
python -m legalir2 train-ce --epochs 2 --group-size 8 --groups-per-step 4   # ~2 h
python -m legalir2 predict --evaluate --reranker-model models2/reranker-tuned \
    --output outputs/v2-ce-tuned.json

python -m legalir2 train-bi --epochs 2 --batch-size 8 --negatives 7         # ~2 h
python -m legalir2 index-dense --embedding-model models2/embedding-tuned \
    --dense artifacts2/dense-tuned                                          # ~1 h
python -m legalir2 predict --evaluate \
    --embedding-model models2/embedding-tuned --dense artifacts2/dense-tuned \
    --reranker-model models2/reranker-tuned --output outputs/v2-both-tuned.json

# 4. optional: distil the 4B teacher into the same 568M student
python -m legalir2 download --which teacher
python -m legalir2 teacher --batch-size 16                # ~1 h, offline scores only
python -m legalir2 train-ce --teacher-scores artifacts2/groups/teacher.jsonl \
    --distill-weight 0.4 --output models2/reranker-distilled

# 5. submit the configuration that measured best
python -m legalir2 predict --questions public-official.json \
    --embedding-model models2/embedding-tuned --dense artifacts2/dense-tuned \
    --reranker-model models2/reranker-tuned --output outputs/submission-v2.json
python -m legalir2 validate --predictions outputs/submission-v2.json
python -m legalir2 pack --predictions outputs/submission-v2.json
```

Budget about 8–10 GPU hours for the whole sequence on a 24 GB card. Reduce
`--batch-size` on out-of-memory; the dense matrix is 0.7 GiB and is kept resident on
the card when it fits.

### Reading the results

`predict --evaluate` writes three sidecars next to the prediction: `.metrics.json`
(Recall and Precision computed exactly as `scoring.py` does, including its zero
penalty for empty or over-five answers, plus candidate recall at several depths),
`.diagnostics.json` (per question, where each gold document sat in the candidate list)
and `.run.json` (every setting, the parameter audit, and stage timings). Compare
candidate recall and final recall separately: the first says whether retrieval found
the document, the second whether ranking kept it.

Expected shape of the results, stated as a prediction to be checked rather than a
claim: the fine-tuned cross encoder should convert a large part of the gap between
candidate R@50 (0.960) and Recall@5 (0.844). If it lands near 0.90 on validation, that
is roughly 0.88–0.89 on the public leaderboard given the ~0.015 validation-to-public
gap your BM25 run showed. If the fine-tuned reranker does **not** beat the zero-shot
one, suspect the mined chunk-level positives first — inspect `artifacts2/groups/train.jsonl`
and `train.stats.json`, especially the `missed` and `pruned` counts.

### Choosing a GPU

The model is small — 568M parameters — so this workload does not reward a datacenter
card. What it needs is enough VRAM for AdamW in fp32 and bfloat16 support.

VRAM arithmetic for `train-ce`: 568M parameters at fp32 is 2.27 GB, gradients another
2.27 GB, and AdamW's two moment buffers 4.54 GB — **9.1 GB before a single activation**.
With gradient checkpointing and 32 pairs of 512 tokens per step, activations add roughly
1–3 GB, so the cross encoder fits in 16 GB. `train-bi` pushes 64 passages per step and
lands nearer 14 GB, which is tight on a 16 GB card.

| Target | Card | Fits |
|---|---|---|
| First submission (`train-ce` only) | 16 GB Ampere+ — RTX 4080, A4000 | yes |
| Full path including `train-bi` | **24 GB Ampere+ — RTX 4090, 3090, A5000** | comfortably |
| Overkill for this size | A100, H100 | 2–4x the price for well under 2x the throughput |

**Ampere or newer is a hard requirement in practice.** Turing cards (RTX 2080 Ti, T4) have
no bfloat16. The training loop now detects this and falls back to float16 with a gradient
scaler, so it will run — but float16 training of a 568M encoder is slower and less stable
than bfloat16, and the saving is not worth it.

Host specifications that matter beyond the GPU:

* **8+ vCPU.** Tokenisation in the training loop is CPU-side and single threaded; a fast
  GPU on 2 vCPUs will wait on it. `build-store`, `index-lexical` and `mine` are entirely CPU.
* **32 GB system RAM.** The lexical postings need about 2 GB resident; the rest is headroom.
* **50 GB disk.** CUDA torch is about 5 GB installed, each checkpoint 2.3 GB, the chunk
  store and postings 1.4 GB, dense vectors 0.7 GB. The common 20 GB default is not enough.
* **CUDA 12.x host driver**, since `gpu-setup.sh` installs the cu128 wheels.

### What runs on CPU, and what does not

Measured here on two cores with a 568M bge-m3-geometry encoder. Scale roughly linearly
with your core count — an 8-core laptop is about 4x these rates.

| Command | Two cores | Verdict on a laptop |
|---|---|---|
| `build-store`, `index-lexical`, `predict --no-dense --no-rerank`, `tune` | minutes | **Yes** — numpy only, no model at all |
| `probe --limit 25` (600 passages @ 768 tokens) | ~22 min | **Yes** — this is what it is for |
| `probe --limit 25 --max-length 256` | ~7 min | **Yes** — a faster screen, if you only need to spot a loser |
| `index-dense` (341,401 chunks @ 768) | ~206 h | **No.** GPU only |
| `train-ce` / `train-bi` | far worse — backward pass | **No.** GPU only |
| `predict` with a reranker, full 1,050 questions | ~90 h at 50x3 passages | **No**, but `--limit 20` is a viable smoke test |

Sequence length dominates: 768 → 512 tokens is about 1.5x faster, 768 → 256 about 3.3x.
Article chunks are 320 whitespace words, roughly 500–600 tokens, so `--max-length 512`
truncates a little and `256` truncates a lot — fine for a screen, not for a measurement.

### Context length differs by an order of magnitude across the sheet

bge-m3 and its Vietnamese derivatives extend the position table to 8192; plain XLM-R
stops at 512; PhoBERT based rerankers such as `itdainb/PhoRanker` stop at 256. Asking for
more than a checkpoint holds raises an index error inside the embedding layer — which on
a full corpus index surfaces an hour into the run. Every requested length is now clamped
to what the checkpoint actually accepts, with a printed note, so a short context model
degrades instead of crashing. The effective length is recorded in the dense manifest and
in `.run.json`.

### Environment note: transformers 4 vs 5

`pip install transformers` on a fresh environment now installs 5.x, while the GPU
environment pins 4.51.3. Version 5 renamed `torch_dtype` to `dtype` in
`from_pretrained`. The loader picks the right keyword at run time, so both work — but
if you hit any other 5.x incompatibility, pinning is the quick escape:

```powershell
& $py -m pip install "transformers==4.51.3"
```

## The path from 0.85 to 0.92

Public test measured **0.85** from the CPU-only lexical system, against **0.8483** on the
held-out validation split. That is close enough to 1:1 that validation can be trusted to
choose between configurations from here — a better position than the earlier BM25 run,
where validation 0.7379 became public 0.77.

Where the remaining points physically are, from `outputs/v2-lexical-validation.diagnostics.json`:

| Rank of the gold document | Share of answer mass |
|---|---:|
| 1–5 (already returned) | 0.8483 |
| **6–50** | **0.1125** |
| 51–200 | 0.0186 |
| never retrieved | 0.0206 |

So a reranker over the top 50 has a ceiling of 0.9608 and **0.92 means capturing 64% of
the gap between the current order and a perfect one**. Deepening the shortlist lowers the
bar: at 100 candidates the ceiling is 0.9752 and 0.92 needs 57%; at 200, 0.9794 and 55%.

That is demanding but not unreasonable for an in-domain fine-tuned cross encoder. It is
not, however, one step. Treat it as a budget:

| Step | Expected | Why |
|---|---|---|
| Fine-tuned cross encoder over top 50 | 0.885 – 0.910 | Captures 35–55% of the headroom. Everything else depends on this working. |
| Deepen to 100 candidates via two-stage reranking | +0.005 – 0.015 | Raises the ceiling from 0.9608 to 0.9752 at roughly the cost of the shallow pass. |
| Fine-tuned bi-encoder + dense channel | +0.005 – 0.015 | Attacks the 2.1% never retrieved and raises the ceiling further. |
| Round-two hard negatives, mined with the trained models | +0.005 – 0.015 | Standard second iteration; the first round's negatives are only as hard as the lexical stage. |
| Ensemble or distillation into the second stage | +0.005 – 0.015 | The 2.86B of unused budget, spent where it is measured to help. |
| Retrain the final model on all 7,000 questions | +0.003 – 0.010 | 18% more labelled data, after configuration is frozen on the 5,950 split. |

Midpoints land near 0.92–0.93. Every line has to roughly land; if the first one comes in
under 0.87, stop and inspect the mined groups rather than continuing down the list.

### Two-stage and ensembled reranking

`--reranker-model` accepts several checkpoints. The first is the wide pass over
`candidates` documents; the rest rescore only its best `stage2_candidates`, so a deep
shortlist costs little more than a shallow one. Scores from different rerankers are not
comparable across models, so each ranking is min-max normalised within the question before
being combined; `reranker_weights` sets the mix, and equal weights mean two models that
disagree exactly cancel to a deterministic tie.

```powershell
# Deep shortlist, one cheap wide pass, one careful pass over the survivors.
py -m legalir2 predict --evaluate `
    --reranker-model models2
eranker-tuned models2
eranker-distilled `
    --settings '{"candidates": 100, "stage2_candidates": 25, "passages": 3}' `
    --output 'D:\DSC-2026\outputs\v2-two-stage.json'
```

Tune `candidates`, `stage2_candidates` and `passages` together: more candidates raises the
ceiling, more passages per document reduces the chance the relevant article was never shown
to the reranker at all, and both cost linear GPU time.

### Round-two negative mining

The first mining round draws negatives from the lexical stage alone, so they are only as
hard as BM25F. Once a bi-encoder is trained, mine again — the negatives it surfaces are the
ones the model itself confuses, which is what the second round of training needs.

```powershell
py -m legalir2 mine --dense artifacts2\dense-tuned --embedding-model models2\embedding-tuned `
    --output 'D:\DSC-2026\artifacts2\groups\train-round2.jsonl'
py -m legalir2 train-ce --groups 'D:\DSC-2026\artifacts2\groups\train-round2.jsonl' `
    --output models2\reranker-round2
```

### Training on all 7,000 at the very end

`artifacts/splits/train.json` holds 5,950 questions; the organisers supplied 7,000. Freeze
every configuration choice using the 5,950/1,050 split, then — and only then — mine and
retrain on the full `train.json` for the submitted model. It adds 18% more supervision, and
from that point the validation split is inside training, so its score is no longer a valid
estimate of anything. Keep the split-trained numbers as the record.

## Things deliberately not done

* No external API of any kind, commercial or otherwise; `download` is the only command
  that opens a socket and it fetches static weight files at pinned commits.
* No model outside the approved sheet.
* No generated or pseudo-labelled training data. Positives are passages of documents the
  organiser labelled; negatives are passages of documents the organiser did not label.
* No quantisation is claimed as a parameter reduction — the audit counts every
  parameter, including embeddings and frozen weights.
* The public and private test questions are never used to fit anything.

## Known gaps and next levers, in order of expected value

1. **Train the reranker** — written, not yet run. Largest expected gain by far.
2. **Train the bi-encoder** — written, not yet run. Should push candidate R@50 from
   0.960 toward 0.98, raising the reranker's ceiling.
3. **Pseudo-relevance feedback (RM3)** on the lexical channel — cheap, untested,
   plausibly +0.01 candidate recall.
4. **Amendment and citation graph.** The corpus contains consolidated, amending and
   amended versions of the same law. Mining relationships from the supplied text only
   could propose related candidates and also improves negative denoising. Note the
   current mining already drops near-duplicate negatives, because such a "negative" is
   very often an equally correct unlisted answer.
5. **A second, larger reranker** over the top 15, using the remaining ~2.8B of budget.
6. **Grouped holdouts.** The split groups exactly duplicated questions, but not
   near-duplicates or questions sharing a gold document, so validation still overstates
   generalisation to unseen laws somewhat. Worth a document-grouped holdout before
   trusting any small difference between configurations.
