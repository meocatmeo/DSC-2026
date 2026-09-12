# UIT DSC 2026 — Task 1 LegalIR

Local legal-document retrieval using **SQLite FTS5 BM25 + Qwen3-Embedding-0.6B +
Qwen3-Reranker-0.6B**. Output is ranked document IDs, not generated legal answers.

## Model choice and compliance

| Component | Checkpoint | Published size | License |
|---|---|---:|---|
| Dense retrieval | Qwen/Qwen3-Embedding-0.6B | 0.6B | Apache-2.0 |
| Reranking | Qwen/Qwen3-Reranker-0.6B | 0.6B | Apache-2.0 |
| Lexical retrieval | SQLite FTS5 BM25 | No neural parameters | Python-bundled SQLite |

Both checkpoints appear in `D:\DSC-2026\[DSC@UIT 2026] Danh sách mô hình - Sheet1.csv`.
Check the **current BTC approval list** before competition use. Model cards:

- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

Approximate combined size is 1.2B, not an exact audit. `audit-models` counts **all**
parameters, including frozen weights and embeddings, and checks the strict <4B
limit across both checkpoints. Inference also checks the combined loaded count.
Quantization is not used or treated as a parameter-count reduction.

Only BTC corpus text and titles are indexed. URLs are never followed. No external
datasets, generated examples, pseudo-labels, fine-tuning, or teacher APIs are used.
Chunking is a search-index representation of the unchanged supplied text, not
synthetic training examples. Confirm BTC's interpretation if their preprocessing
rules change. The training file is currently used for evaluation splits, not weight
updates. Do not claim this is a fine-tuned or competition-optimized model.

The explicit `download` command retrieves static model artifacts over HTTPS;
**inference loads local files only**, without remote code or hosted inference.
Downloads record immutable revisions and SHA-256 hashes. Archive these local model
directories for exact reproduction rather than redownloading a future `main`.

## Architecture

### Experimental legal evidence ranking

`--legal-weight 0.3` adds extractive 2–4-word query-phrase evidence, with
candidate-local rarity weights and accent-folded matching for the supplied ASCII
titles. It blends this evidence with normalized fusion scores. No extra queries,
training examples, models or external dictionaries are generated. Default weight
remains zero pending full validation. With neural reranking, this evidence only
affects the pre-reranker order/tie-breaking, not the neural score itself.

Reranker evidence now avoids near-identical overlapping passages when possible.
Evaluation writes a `.diagnostics.json` sidecar containing gold-document candidate
ranks, and reports candidate Recall separately from final Recall@5. This separates
retrieval misses from ranking misses; diagnostics are never used as inference input.

Measured CPU comparison (same five-output policy, 60 candidates):

| Validation slice | Original Recall | Phrase-aware Recall | Original Precision | Phrase-aware Precision |
|---|---:|---:|---:|---:|
| First 100 (errors inspected during development) | .718333 | .783333 | .152 | .166 |
| Next 100 (checked with unchanged weight) | .795 | .825 | .168 | .174 |

Candidate Recall was .958333 and .973333 respectively. Neither result is a
full-validation or hidden-test result. The full original BM25 run found in the
workspace has Recall .737857 and Precision .157714 on 1,050 questions.
The existing public prediction and ZIP were not overwritten by these experiments.

```powershell
Set-Location 'D:\DSC-2026'
py -m legalir predict --questions 'D:\DSC-2026\artifacts\splits\validation.json' --output 'D:\DSC-2026\outputs\legal-validation.json' --evaluate --legal-weight 0.3
```

### Higher-quality roadmap (not implemented or claimed as measured)

1. Preserve article/clause headings and parent provisions during indexing rather
   than relying solely on arbitrary word windows. Rebuild dense indices afterward.
2. Add separate title/citation candidate channels with accent-aware normalization.
   Distinguish a document's own number from numbers merely cited in its preamble.
3. Diagnose definitions versus procedures versus sanctions and evaluate evidence
   matching each question's requested legal function. Do not hard-filter by intent.
4. Train a reranker on BTC-provided labels with a clean held-out split, only after
   confirming how BTC permits negative selection and derived training pairs.
   Do not use generated teacher answers or external legal examples.
5. Extract amendment/reference relationships from the supplied corpus only; use
   them to propose related candidates, not automatically declare legal validity.
   Do not assume the newest document is the ground truth.
6. Test document-grouped holdouts and per-category regressions, and compare all
   changes on identical queries. Keep retrieval and ranking metrics separate.

The measured high candidate Recall suggests reranking/evidence selection deserves
priority over blindly increasing embedding-model size. No system can guarantee
zero retrieval errors or a specific leaderboard score.

1. Normalize Unicode to NFC and whitespace; preserve Vietnamese accents.
2. Split each passage into 180-whitespace-word chunks, with 40-word overlap.
   Include the supplied title (hyphens converted to spaces) in each chunk.
3. Retrieve up to 600 BM25 chunks and 600 cosine-similarity chunks.
4. Deduplicate each ranking to document IDs, then reciprocal-rank fuse (constant 60).
5. For the best 60 documents, select up to two retrieved passages per document.
6. Use local Qwen yes/no relevance scoring; document score is maximum passage score.
7. Return five unique document IDs in decreasing relevance order. Fusion breaks ties.

Without `--rerank`, use fusion order. Without `--mode hybrid`, use BM25 only.
Empty/unmatched queries use deterministic existing-document fallbacks. The 20
empty-passage documents remain in the corpus, with title-only retrieval where
possible. This fallback is valid output, not a guarantee of relevance.

The 8,532 supplied documents produce **521,955 chunks** with default settings.
Dense vectors use float32, 1024 dimensions: approximately **1.99 GiB** excluding
metadata/index files. Search uses a memory-mapped NumPy matrix; neural inference
runs on the selected device, while vector search runs on CPU. This favors simple,
portable reproduction over maximum search throughput. Allow several GB for SQLite,
several GB for model weights, and additional space for dependencies/build files.
Full GPU indexing can take hours depending on hardware; benchmark first.

## CPU quick start (this Windows PC)

All commands below assume the repository is at `D:\DSC-2026`.
Python is available through `py`; the machine's `python` shortcut may not work.
**The BM25 path needs no pip packages.** Python must include SQLite FTS5 (the
installed Python 3.13.7 was tested successfully).

```powershell
Set-Location 'D:\DSC-2026'
py -m unittest discover -s 'D:\DSC-2026\tests' -v
py -m legalir prepare
py -m legalir split
py -m legalir predict --questions 'D:\DSC-2026\artifacts\splits\validation.json' --output 'D:\DSC-2026\outputs\bm25-validation.json' --evaluate
py -m legalir predict --questions 'D:\DSC-2026\public-official.json' --output 'D:\DSC-2026\outputs\submission.json'
py -m legalir pack --predictions 'D:\DSC-2026\outputs\submission.json'
```

Preparation builds `D:\DSC-2026\artifacts\corpus.sqlite`. Do not rebuild it while
inference is running. Rebuilding invalidates any dense index associated with it.
The split creates 5,950 training and 1,050 validation records (seed 2026). Exact
normalized duplicate questions remain in the same split. This does not eliminate
all possible near-duplicate or same-document overlap; inspect those separately
before claiming generalization to unseen legal documents.

For a quick smoke test, add `--limit 10 --evaluate` to validation prediction and
use a separate output name. **A limited public prediction is NOT a submission**;
the packager rejects it because question coverage is incomplete.

## NVIDIA GPU setup

Use a recent NVIDIA driver and preferably 12–16 GB or more VRAM as a starting
point; actual needs depend on batch/sequence lengths. Reduce `--batch-size` on
out-of-memory errors. CPU neural execution is possible with `--device cpu`, but
not recommended for the full corpus. Tested software interfaces: Python 3.13.7,
PyTorch 2.7.1, Transformers 4.51.3, NumPy 2.2.6. Python 3.11/3.12 are also sensible
GPU deployment choices, but were not tested here.

On the NVIDIA Windows machine, copy this repository and the BTC corpus to
`D:\DSC-2026`, then create a **new** environment (do not copy a Windows venv):

```powershell
Set-Location 'D:\DSC-2026'
py -m venv 'D:\DSC-2026\.venv'
& 'D:\DSC-2026\.venv\Scripts\python.exe' -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
& 'D:\DSC-2026\.venv\Scripts\python.exe' -m pip install -r 'D:\DSC-2026\requirements.txt'
& 'D:\DSC-2026\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
& 'D:\DSC-2026\.venv\Scripts\python.exe' -m unittest discover -s 'D:\DSC-2026\tests' -v
& 'D:\DSC-2026\.venv\Scripts\python.exe' -m legalir download
& 'D:\DSC-2026\.venv\Scripts\python.exe' -m legalir audit-models
```

If this environment already has CPU PyTorch, uninstall torch before installing
the CUDA wheel, or use a fresh environment. CUDA 12.8 wheels require a compatible
driver/GPU; consult PyTorch's official installation selector for your hardware.
Linux GPU users can run the same Python modules from their absolute repository
path using the corresponding environment's Python; defaults resolve relative to
the installed repository, not the Windows path. No Linux GPU run was performed here.

Download retrieves roughly a few GB of weights. Interrupted artifact downloads
restart that file; verified completed files are reused when revisions match.
`audit-models` loads the models sequentially on CPU and verifies stored file hashes.
Run it before indexing and again before final reproduction if weights were moved.

## GPU indexing, validation, and final predictions

```powershell
Set-Location 'D:\DSC-2026'
$python = 'D:\DSC-2026\.venv\Scripts\python.exe'
# Explicitly prevent network access by Hugging Face during all model execution.
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
& $python -m legalir prepare
& $python -m legalir split
& $python -m legalir index-dense --batch-size 8
# First smoke-test execution. Do not infer quality from ten questions.
& $python -m legalir predict --mode hybrid --rerank --questions 'D:\DSC-2026\artifacts\splits\validation.json' --output 'D:\DSC-2026\outputs\hybrid-smoke.json' --limit 10 --evaluate
# Then compare full holdout scores with BM25 and hybrid without reranking.
& $python -m legalir predict --mode hybrid --questions 'D:\DSC-2026\artifacts\splits\validation.json' --output 'D:\DSC-2026\outputs\hybrid-validation.json' --evaluate
& $python -m legalir predict --mode hybrid --rerank --questions 'D:\DSC-2026\artifacts\splits\validation.json' --output 'D:\DSC-2026\outputs\reranked-validation.json' --evaluate
# Choose the best measured configuration; this command uses hybrid + reranker.
& $python -m legalir predict --mode hybrid --rerank --questions 'D:\DSC-2026\public-official.json' --output 'D:\DSC-2026\outputs\submission.json'
& $python -m legalir validate --predictions 'D:\DSC-2026\outputs\submission.json'
& $python -m legalir pack --predictions 'D:\DSC-2026\outputs\submission.json'
```

`index-dense` restarts from scratch if interrupted. It only writes its completion
manifest after all chunks are embedded. Predictions are saved after the full run;
there is currently no per-query resume. Run a smoke test before long jobs.
Models use left padding; embeddings use last-token pooling and L2 normalization.
Documents truncate at 1024 tokenizer tokens and reranking at 1536 tokens; word
chunking is not tokenizer-aware, so unusually token-dense chunks may truncate.
Qwen model-card query instructions and reranker prompt suffix are implemented.

Tune `--candidates`, `--chunk-candidates`, `--passages`, and `--top-k` on validation,
not hidden public labels. Recall is primary; precision only breaks recall ties.
More sophisticated retrieval or fine-tuning is not guaranteed to help: compare
measured results first. Hyperparameter selection on this holdout makes the result
a development score rather than an unbiased final-test estimate.

## Submission contract

Final artifact: **`D:\DSC-2026\outputs\submission.zip`**.
It contains exactly one UTF-8 file named `submission.json`, with no directories:

```json
{"question_id": {"answer": ["document_id_1", "document_id_2"]}}
```

The example IDs only illustrate the schema. Actual predictions use supplied IDs.
The validator rejects duplicate JSON keys, missing/extra question IDs, missing
`answer`, non-list answers, non-string IDs, duplicate IDs, unknown IDs, and >5 IDs.
An empty list is accepted by the metric but gets zero; normal inference returns
five where the corpus contains at least five documents. Rank order is preserved
through packaging. Rank quality itself cannot be verified by a format validator.
ZIP contents are reopened and validated after writing.

Macro Recall and Precision match `D:\DSC-2026\scoring.py` on valid inputs and on
the >5/empty-list penalty. Evaluation is deliberately stricter about mismatched
question sets and missing labels. `--evaluate` cannot score the unlabeled public
test. Per-question metrics and runtime/configuration manifests are sidecar files
and **are not included in the prediction ZIP**.

## Tests and reproduction hand-in

### Measured locally during implementation

- Indexed all 8,532 documents into 521,955 chunks.
- BM25, first 100 questions of the fixed validation split: Recall **0.718333**,
  Precision **0.152000**, approximately **121 seconds** on the current laptop.
- These are **subset development metrics**, not full-holdout or public-test scores.
- Actual pretrained Qwen weights and CUDA execution have **not** been tested here.
  Neural integration is checked with tiny local random model fixtures only.
- No complete public-test submission has been generated; use the full prediction
  and packaging commands above. Do not submit a smoke-test JSON.

Core tests cover chunking, duplicate grouping, document deduplication, fusion,
metrics, malformed submissions, Unicode JSON, and end-to-end ZIP creation.
Optional neural tests create tiny random Qwen fixtures locally to exercise pooling,
yes/no scoring, dense indexing and hybrid reranking without downloads. Fixtures
are temporary software-test artifacts; they are not training examples for the
competition and do not establish pretrained model quality.

Keep the organizer's scorer unchanged. For final BTC verification, provide:

- Source code and this README, plus exact command lines.
- Model directories, provenance manifests and `D:\DSC-2026\artifacts\model-audit.json`.
- BTC data or instructions identifying the exact BTC release; do not redistribute
  it publicly without permission.
- Corpus/dense index manifests, split manifest, validation metrics, prediction
  run manifest, hardware and driver details.
- A dependency lock from the **actual GPU environment**, using:

```powershell
& 'D:\DSC-2026\.venv\Scripts\python.exe' -m pip freeze | Set-Content -Encoding UTF8 'D:\DSC-2026\outputs\requirements-gpu-lock.txt'
```

Top-level dependencies are pinned, but transitive dependencies must be captured
with the deployment lock. GPU floating-point behavior may differ slightly across
hardware. Archive the exact index and model files for the strongest reproduction.
Docker is optional under the supplied BTC rules. Git ignores data artifacts,
models, environments and outputs; include required ignored files in the separate
private hand-in package rather than assuming a code-only clone contains them.