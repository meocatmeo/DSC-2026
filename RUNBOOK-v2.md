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
