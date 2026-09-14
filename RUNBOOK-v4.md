# GPU runbook — phase 4

From a freshly rented box to a packed submission. ~9 GPU hours, unattended after step 6.

Current best: **0.9433 public** (0.9371 validation), from `models2/reranker-g32`.

---

## 0. Before you rent

Nothing on the box is precious — everything lives in Drive:

| in Drive | size | rebuild cost if lost |
|---|---|---|
| `models/reranker-g32` | ~1.1 GB | 2 h — **this is the 0.9433 model** |
| `models/reranker-tuned`, `models/embedding-tuned` | ~2.2 GB | 2 h each |
| `dense/`, `dense-tuned/` | ~1.4 GB | 1.2 h each |
| `groups/`, `outputs/`, `logs/` | small | 11 min to re-mine |

Rent anything with **24 GB or more**. Nothing in phase 4 needs more.

---

## 1. Upload the current code from the laptop

```powershell
Set-Location 'D:\DSC-2026'
.\rclone.exe --config .\aic-readwrite.conf copy .\_upload gdrive:DSC-2026 --progress
```

Verify it actually went (this is where a dead token shows up):

```powershell
.\rclone.exe --config .\aic-readwrite.conf lsl gdrive:DSC-2026 --max-age 1h
```

Empty output here means the upload failed. Everything downstream depends on it.

---

## 2. Install rclone on the box

```bash
curl https://rclone.org/install.sh | sudo bash
mkdir -p ~/.config/rclone
```

Then paste your `aic-readwrite.conf` contents into `~/.config/rclone/rclone.conf`
(`nano ~/.config/rclone/rclone.conf`). Check it works before anything else:

```bash
rclone lsd gdrive:
```

---

## 3. Restore everything

```bash
mkdir -p /workspace/DSC-2026 && cd /workspace/DSC-2026
rclone copy gdrive:DSC-2026 . --progress
bash gpu-restore.sh
```

This pulls the code and corpus, restores **every** checkpoint it finds in Drive, brings
back both dense indexes, re-downloads the base models, rebuilds the environment, and
rebuilds the chunk store and lexical postings. 20–40 minutes, mostly transfer.

It ends with a table of what it restored. **You must see `reranker-g32` in it.**

---

## 4. Verify the phase-4 code arrived

The single most common failure in this project has been running stale code. Two checks:

```bash
cd /workspace/DSC-2026
ls legalir2/stack.py                                    # must exist
.venv/bin/python -m legalir2 stack --help | head -3     # must not error
.venv/bin/python -m legalir2 train-ce --help | grep max-positives
.venv/bin/python -m legalir2 tune --help | grep context-radius
```

Any of those missing → step 1 didn't land. Re-upload and re-pull; don't work around it.

---

## 5. Run the test suite on the box

```bash
.venv/bin/python -m unittest discover -s tests2 -t . 2>&1 | tail -3
```

Expect `Ran 77 tests ... OK`. On the box torch is installed, so fewer skips than on the
laptop. A failure here is worth ten minutes before nine GPU hours.

---

## 6. Launch

```bash
tmux new -s p4
```

**From outside tmux.** If you are already inside one, `Ctrl-b` then `c` for a new window
instead — `tmux new` inside tmux prints "sessions should be nested with care" and does
nothing.

```bash
cd /workspace/DSC-2026 && bash gpu-auto4.sh
```

Within the first two seconds it prints either

```
[hh:mm:ss] log channel OK - STATUS.txt is in gdrive:DSC-2026-results/logs
```

or a warning naming the rclone error. **Wait for that line before detaching.** If it
warns, the run still works but you will be blind to it from off the box.

Then `Ctrl-b` then `d` to detach. Close the laptop.

---

## 7. Watch from anywhere

```bash
rclone cat gdrive:DSC-2026-results/logs/STATUS.txt     # heartbeat, updated every 60 s
rclone cat gdrive:DSC-2026-results/logs/auto.log       # everything
```

`STATUS.txt` shows the current stage, elapsed time, GPU utilisation and the last 40 log
lines. GPU at 0% across several heartbeats means something is wedged; 95–100% means
training.

Back on the box: `tmux attach -t p4`.

---

## 8. What runs, and the gate on each

| # | stage | time | the number that matters |
|---|---|---|---|
| 1 | `multipos` | ~2.5 h | multi-gold recall, was **0.7527**; single-gold must not fall below **0.9551** |
| 2 | `stack` | ~1.0 h | printed feature weights — do the retrieval features carry real weight? |
| 3 | `sweep` | ~1.5 h | best `context_radius`; do **not** expect `candidates=100` to win |
| 4 | `ceiling` | ~4.0 h | does the tuned index beat the base one *after* re-mining? |
| 5 | `audit` | seconds | deployed total must be under 4,000,000,000 |

A failing stage does not stop the run. `multipos` gets one automatic retry with the
mined groups kept and only the checkpoint cleared.

**Stage 1's gate is the one to read first.** If multi-gold rises well above 0.7527 while
single-gold holds at 0.9551, the objective fix worked and the rest compounds on it. If
single-gold *fell*, the extra positives are crowding out negatives — rerun that stage
with `--max-positives 2`:

```bash
FORCE=1 bash gpu-phase4.sh multipos   # after editing the flag in the script
```

---

## 9. When it finishes

The closing summary prints a table of every scored configuration with the models that
produced it, which one was chosen, and whether the zip was packed. The autopilot picks
by validation Recall@5 from recorded provenance and refuses anything over 4B parameters.

```bash
ls -lh outputs/submission-v4.zip
rclone cat gdrive:DSC-2026-results/logs/auto.log | tail -40
```

Submit only if the chosen row beats **0.9371** validation — that is what sits behind your
0.9433 public score. Validation has run ~0.006 below public on this system.

---

## 10. Before terminating the box

```bash
cd /workspace/DSC-2026
rclone copy models2       gdrive:DSC-2026-results/models  -P --transfers 4
rclone copy artifacts2/dense       gdrive:DSC-2026-results/dense       -P
rclone copy artifacts2/dense-tuned gdrive:DSC-2026-results/dense-tuned -P
rclone copy outputs       gdrive:DSC-2026-results/outputs -P
rclone copy artifacts2/groups gdrive:DSC-2026-results/groups -P

rclone lsf -R gdrive:DSC-2026-results --dirs-only
rclone size gdrive:DSC-2026-results
```

The autopilot backs up after every stage, so this is belt-and-braces — but a listing you
have actually read is the only backup that counts. Compare the `model.safetensors` byte
counts for any new checkpoint against the local ones; a truncated upload looks identical
in a directory listing.

Then pull the small stuff to the laptop and terminate:

```powershell
.\rclone.exe --config .\aic-readwrite.conf copy gdrive:DSC-2026-results/outputs .\_results\outputs --progress
.\rclone.exe --config .\aic-readwrite.conf copy gdrive:DSC-2026-results/logs    .\_results\logs    --progress
```

---

## Troubleshooting, from failures this project has actually hit

| symptom | cause | fix |
|---|---|---|
| `unrecognized arguments: --max-positives` | stale code on the box | step 1, then step 4 |
| `HFValidationError: Repo id must be...` | a **local** model directory is missing, not a network problem | `python -m legalir2 download --which reranker` |
| `hash mismatch model.safetensors` on a fine-tuned dir | stale base provenance | `mv provenance.json provenance.base.json` |
| `ImportError: requires the protobuf library` | SentencePiece tokenizer, no `tokenizer.json` | `pip install protobuf sentencepiece` |
| nothing in `logs/` after 5 min | usually the autopilot isn't running, or is a stale copy without log shipping | `grep -c ship_logs gpu-auto4.sh` — 0 means stale |
| `tmux: sessions should be nested` | you were already inside tmux | `Ctrl-b` `c` instead |
| `Operation not permitted` deleting files | — | not applicable on the GPU box |

**Never edit a script while it is running.** Bash reads scripts incrementally, so
overwriting `gpu-phase4.sh` mid-run corrupts execution in confusing ways. Pull updates
between stages or after the run.
