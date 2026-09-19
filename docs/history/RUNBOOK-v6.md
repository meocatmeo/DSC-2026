# RUNBOOK v6 — historical 2B reranker experiment

**Superseded for the active pipeline. Do not run phase six or select the Qwen 2B
checkpoint. The active path uses the shipped 0.6B `models2/reranker-g32` checkpoint
and `gpu-phase4.sh multipos`, which now enables `--document-loss`.** The material below
is retained only as an audit trail for the previously measured, unshipped experiment.

Current shipped submission: **0.9493** public. This runbook takes a bare rented GPU to
either a better number or a clear answer that 2B does not help. Both outcomes are worth
the eight hours; only one of them is worth a submission slot.

**Box:** 1x RTX 6000 Ada, 48 GB, ~764 GB/s measured. Rent the listing with the **long
max duration**, not the cheap one — the full sequence is ~8 h and a 3 h ceiling
reclaims the instance mid-training.

---

## 0. Before you rent — one thing on the laptop

`aic-readwrite.conf` holds a Google OAuth refresh token with `scope = drive` (full Drive
access). It was committed to the public repo once. The history was purged and
force-pushed, but **the credential itself was never rotated**, so anyone who cloned the
repo in that window still has working full-Drive access.

Re-authorise with `scope = drive.file` — that limits the token to files rclone itself
created, which is all this project ever touches:

```
rclone config reconnect gdrive:
```

Do it before you paste that file onto a rented box you do not control.

---

## 1. Rent

Pick the listing by **Max Duration**, then bandwidth, then price. The full sequence:

| stage | what it does | est. on this card |
|---|---|---|
| preflight | loads the 2B, asserts it ranks, measures throughput | 25 min |
| train | LoRA fine-tune, merged on save | ~3.5 h |
| measure | validation + paired test vs 0.9493 | ~1.5 h |
| submit | public prediction + packed zip | ~1.5 h |

~8 h total. At $0.88/h that is about $7.

Disk: 60 GB is enough. The corpus, both indexes, three checkpoints and the 2B base come
to roughly 25 GB.

---

## 2. Bring the box up

SSH in, then — in this order, because each step assumes the last one finished.

```bash
# rclone first: everything else arrives through it
curl https://rclone.org/install.sh | sudo bash

mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<'EOF'
<paste the contents of aic-readwrite.conf here>
EOF
chmod 600 ~/.config/rclone/rclone.conf

rclone lsd gdrive:            # must list DSC-2026 and DSC-2026-results
```

If `rclone lsd` fails, stop. Nothing downstream works and the failure will resurface
four steps later disguised as a missing file.

```bash
# restore: code, corpus, every surviving checkpoint, both dense indexes, the outputs
rclone copy gdrive:DSC-2026/gpu-restore.sh /workspace/ 
cd /workspace && bash gpu-restore.sh
```

`gpu-restore.sh` is idempotent and does a lot: pulls code and corpus, restores every
`models2/*` directory it finds in Drive (not a hardcoded list — that bug silently left
`reranker-g32` behind on a fresh box), restores `artifacts2/dense` and `dense-tuned`
(saving 1.2 h of indexing each), repairs stale provenance files, builds the venv, builds
the chunk store and lexical postings, re-fetches the base checkpoints that fine-tuning
starts from, and runs the 97-test suite.

**Read its summary block.** It prints what survived:

```
  trained reranker : restored
  tuned embedder   : restored
  mined groups     : restored
  dense index      : restored
```

Anything saying LOST or missing changes the plan — come back and say which.

It also runs a lexical-only baseline that must print **Recall 0.8483**. That number is
the corpus integrity check. A different number means the corpus did not arrive intact,
and every later measurement is against a different dataset than the 0.9493 was.

---

## 3. Phase six

```bash
cd /workspace/DSC-2026
.venv/bin/pip install peft         # the LoRA path needs it; restore does not install it

tmux new -s p6                     # from OUTSIDE tmux — SSH drops kill foreground jobs
RERANK_BATCH=32 bash gpu-phase6.sh preflight
```

Detach with `ctrl-b d`, return with `tmux a -t p6`. From anywhere:

```bash
rclone cat gdrive:DSC-2026-results/logs/STATUS.txt
```

### Stop at preflight and read it

Four checks, twenty-five minutes:

1. **Download and model_type.** Confirms the checkpoint is what it claims to be.
2. **Architecture support.** A `KeyError` here means this `transformers` does not know
   Qwen3.5 — loud, cheap, and the whole answer.
3. **Parameter audit.** Must total under 4,000,000,000. Prism-2B plus the 0.6B embedder
   is ~2.6B, so this should pass with room, but it is the competition rule and the one
   failure that would invalidate a submission after the fact.
4. **Does it actually rank.** A known-relevant passage must outscore a known-irrelevant
   one.

Check 4 is the reason this script exists. A wrong prompt template **does not raise** —
Prism's system prompt and instruction genuinely differ from Qwen3-Reranker's, and using
the wrong one just makes every score quietly worse. You would find out after four and a
half hours of training, and the symptom would look like "the 2B did not help."

If check 4 fails: fix `TEMPLATES["prism"]` in `legalir2/models.py` against the model
card and re-run. Twenty-five minutes, not eight hours.

Preflight also prints **measured throughput** in passages/s and converts it into the
hours a full pass costs. Compare that against the table in section 1 before committing
— if it is far worse than estimated, that is a decision point, not a surprise.

### Train

```bash
bash gpu-phase6.sh train            # ~3.5 h
```

`LORA_RANK=16`, merged on save, so what ships is an ordinary 2B checkpoint rather than
an adapter the inference path would have to know about. `--group-size 16
--groups-per-step 1 --accumulate 8` keeps 128 pairs per optimiser step at half the VRAM
of 32x4.

48 GB is technically enough to attempt a full fine-tune, but AdamW on 2B params wants
~32 GB of optimiser state before activations, and full-vs-LoRA is not the thing being
tested. Leave the rank alone.

If it OOMs: halve `--group-size` and double `--accumulate`. The effective batch is what
matters, not the split.

It trains on the **already-mined groups**. The candidate distribution has not changed,
so those negatives are still the ones a 2B model faces at inference — that distribution
match is what was worth +0.019 back in phase two, and re-mining would only risk breaking
it.

### Measure

```bash
bash gpu-phase6.sh measure          # ~1.5 h
```

Prints a **paired** comparison against the 0.9493 submission: mean difference, a 95%
bootstrap interval, and an exact sign test. Then re-runs the parameter audit on the
trained checkpoint.

**Submit only if the interval clears zero.**

Four consecutive interventions have now measured as indistinguishable from no change at
this resolution. The detection floor on the 1,050-question holdout is about 0.0094, and
the remaining gains live in the 0.003–0.008 band, so "not significant" here means "too
small to see," not "definitely zero." But a new submission whose interval spans zero
spends a slot to learn nothing.

### Submit

```bash
bash gpu-phase6.sh submit           # ~1.5 h
```

Produces `outputs/submission-v6.zip`, validated and packed. It is also backed up to
Drive, so the box can die afterwards without losing it.

---

## 4. What to expect

The 2B has better prior evidence than anything else tried this session: cross-encoder
quality scales with parameter count, and you are deploying 1.2B of a 4B allowance, so
this is the one remaining change that **adds capacity** rather than re-weighting what is
already there.

Against that: the base rate this session is four interventions, four nulls. And a 2B
reranker costs roughly 3x the inference time of the 0.6B, permanently — every future
measurement pays it, and if the competition caps inference time it could be
disqualifying regardless of accuracy. Preflight prints the number so that is a decision
rather than a discovery.

Honest expectation: somewhere between +0.000 and +0.008, with real probability mass on
"not distinguishable."

---

## 5. Before you terminate the box

```bash
rclone copy /workspace/DSC-2026/outputs gdrive:DSC-2026-results/outputs -P
rclone copy /workspace/DSC-2026/models2/reranker-2b-tuned \
    gdrive:DSC-2026-results/models/reranker-2b-tuned -P --transfers 4
```

`gpu-phase6.sh` backs up after every stage, so this is belt-and-braces — but the trained
checkpoint is 4 GB and three and a half hours, and it is the one thing on that box that
cannot be rebuilt cheaply.

Then:

```bash
nvidia-smi                          # confirm nothing is still running
tmux kill-server
```

and terminate from the Vast console.
