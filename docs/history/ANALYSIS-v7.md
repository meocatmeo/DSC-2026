Recommendation: document-level training with a Recall@5-weighted pairwise loss
==========================================================================

The best-supported next change is to train the existing Vietnamese reranker on
the same document evidence that inference uses, with an objective that concentrates
on getting labelled documents across the fifth-place cutoff. This targets a verified
training/inference mismatch. It is a proposed algorithm, not a measured improvement;
there is no evidence that can guarantee a hidden-test score above 0.9493 in advance.

This review used the six root Markdown files, the `legalir2` implementation, GPU
scripts, and saved validation predictions. The calculations are reproducible with
`python -m analysis.ranking_audit`; `analysis/v7-audit.json` contains the full error
inventory, source hashes, and comparisons. It does not run neural inference or fit
anything to validation labels.

**The actual baseline matters.**

The saved `submission-v4.run.json` and `v4-stacked.run.json` agree on the deployed
configuration: `models2/embedding`, `models2/reranker-g32`, `artifacts2/dense`,
`v4-stacker.json`, 50 reranked documents, three passages, and a 768-token inference
limit. The local embedding provenance identifies `AITeamVN/Vietnamese_Embedding_v2`;
the reranker training scripts start from `AITeamVN/Vietnamese_Reranker`. The saved
audit counts 567,754,752 embedding and 567,755,777 reranker parameters.

`PROCESS-v6.md` instead describes a tuned Qwen embedding, causal Qwen reranker,
and top-100 reranking. Those descriptions do not match the saved shipped manifest.
Use the manifest as the baseline for the next implementation.

Public 0.9493 is the reported leaderboard result. Recomputing the saved 1,050-query
validation predictions gives **0.9453968254**, exactly matching the saved metric.

| Quantity | Verified value |
|---|---:|
| Final validation Recall@5 | 0.945397 |
| Candidate recall at the actual rerank depth, 50 | 0.980476 |
| Candidate recall at 100 | 0.984524 |
| Gold documents missing from the final five | 86 |
| Missing gold documents already inside the rerank pool | 50 |
| Available ranking headroom at depth 50 | 0.035079 |
| Additional candidate headroom from 50 to 100 | 0.004048 |

Candidate recall is an oracle bound: it assumes perfect selection and is not a
prediction of what training will achieve. It also does not prove the three passages
shown to the reranker contain the evidence needed to identify the right document.

**Target both single-answer and multiple-answer questions.**

| Bucket | Questions | Current recall within bucket | Missed gold inside top 50 | Recoverable contribution to overall recall |
|---|---:|---:|---:|---:|
| One gold document | 957 | 0.961338 | 27 | 0.025714 |
| Multiple gold documents | 93 | 0.781362 | 23 | 0.009365 |

Multiple-answer questions account for about 35% of all recall loss, but only
26.7% of the loss recoverable by reordering the actual 50 candidates. Perfectly
fixing that bucket alone would move validation to about 0.954762. The larger
available improvement is in single-answer questions. This contradicts the conclusion
in `PROCESS-v6.md` that multiple-answer selection is the only worthwhile target.

Examples from the saved results illustrate the differences:

- Query `123452`, about types of construction contracts: the amendment `226545`
  was candidate 5, but dropped; the base decree `258342` was retained.
- Query `9060`, about an informatics certificate for a high-school teacher: the
  sole gold `154615` was candidate 14 and dropped. It never appears as a positive
  in the 5,950 training questions, so a document-frequency rule cannot fix it.
- Query `132594`, about the procedure for a tax-payment extension: gold `303485`
  was candidate 8; the output instead contained five related tax-extension documents.

There are also underspecified questions and closely related legal versions among
the errors. These examples do not establish that every unlabelled result is legally
wrong. Training must optimize the supplied retrieval labels without claiming to
resolve missing jurisdiction or date information.

**The remaining mismatch is below the candidate generator.**

`mine.py` now correctly calls `Pipeline.candidates()`. That important correction
does not make the whole training path match inference:

1. Mining takes the first two chunks per positive document from its own chunk
   ordering. It takes one chunk per negative document.
2. `train_ce.py` samples one of those positive chunks and applies a passage-level
   softmax loss. Its optional multiple-positive mode still supervises sampled
   passages; it does not train document bags.
3. Inference calls `Pipeline._passages()`, which selects three passages with
   duplicate suppression, and the stacker combines their maximum, second score,
   mean, spread, and other evidence.
4. The supplied g32 training command does not set `--max-length`; the current CLI
   default is 512, while the saved prediction uses 768. Restore `training.json`
   before treating 512 as a verified property of the historical weights.

A positive document label says that the document is relevant. It does not say that
either of its two lexically strongest chunks answers the question. Randomly assigning
the document label to a chunk creates weak passage labels. Conversely, an irrelevant
document may contain a second or third passage that fools inference, although training
only exposed its first chunk. That is a specific reason to change the training unit.

The prior `v4-multipos` run already tried spreading softmax target mass over several
positives. Its validation recall was 0.933016 versus g32's 0.937143, before stacking.
That result is not a reason to repeat the same intervention under another name.

**The proposed algorithm.**

Keep the shipped candidate generator and initialize from the shipped reranker.
For a training question q, let C(q) be its actual top-50 candidate documents and
G(q) be all organizer-labelled gold documents. No validation or public labels enter
training, mining, feature fitting, or checkpoint selection.

Select each candidate's passage bag P(q,d) through one shared implementation of
`Pipeline._passages()`: three distinct passages, identical text construction and
token limits in training and inference. Preserve chunk IDs and token truncation
metadata so the equality can be checked. Gold outside C(q) is logged as a retrieval
miss; it must not be silently inserted into evaluation candidates.

Use a document score based on the strongest evidence in the bag:

    S(q,d) = max over p in P(q,d) of reranker_logit(q,p)

This is multiple-instance learning. The positive label applies to the bag, allowing
its strongest supporting passage to carry the gradient. It does not declare every
passage in a relevant document positive. Use raw classification logits for the
shipped sequence classifier; a future causal model would use logit_yes - logit_no.

For each labelled document g in C(q) and unlabelled candidate n, compute their
current document ranks r_g and r_n. Use the exact magnitude of the change in Recall@5
that exchanging their positions would produce:

    w(g,n) = abs(1[r_g <= 5] - 1[r_n <= 5]) / |G(q)|

Then train with a LambdaRank-style weighted pairwise logistic loss:

    L(q) = sum over g in G(q) intersect C(q), n in C(q) minus G(q)
           of w(g,n) * softplus(S(q,n) - S(q,g))

Treat ranks and weights as fixed during each backward pass and refresh them as
scores change. Average over questions. Keep the denominator equal to the full
number of gold labels, including gold missing from the pool. Do not add an arbitrary
multiple-answer oversampling factor: macro recall already defines the contribution.

This loss uses only gold-versus-negative pairs. It never pushes one gold document
below another gold document. A gold document below fifth paired with a negative
inside the top five receives weight; a swap entirely below fifth receives none.
Correct gold documents inside the five also receive margin protection against
negatives outside. Softplus gives greater gradient to the confusions the current
model scores incorrectly, rather than spending equal effort on random easy negatives.

The swap weights are mathematically exact for the metric. The smooth training loss
is a surrogate, and optimizing it does not guarantee a validation or public gain.
This distinction is the boundary between a justified objective and an invented promise.

Preserve the useful lexical and dense evidence by fitting the final fusion against
the new reranker's document features on separate, out-of-fold training scores. Its
final training objective should use the same Recall@5 pair weights on the final
fused document scores. Evaluate the complete new reranker-plus-fusion pair against
the original complete pair. Reusing the old stacker without checking calibration
would repeat the unresolved confound in the 2B comparison.

This requires new mining records and a document-level training path; existing
`train-ce --max-positives` does not implement it. Scoring 50 x 3 passages per query
also costs more than the current sampled 32-passage groups. Microbatching bounds
memory, but does not remove that compute cost. No GPU-hour or score estimate is
justified before a throughput measurement with the actual implementation.

**Why a generic diversity algorithm is not the answer.**

Recall is additive over selected documents. For a set A of five documents:

    E[Recall(A) | q] = sum over d in A of E[1[d in G] / |G| | q]

Therefore selecting the five documents with the largest expected recall contributions
is already optimal in expectation. No independence assumption between documents is
needed. Relationships between laws can improve those estimates, but do not establish
an automatic bonus for adding an amendment or a penalty for two similar documents.
Document diversity can remove a second required gold document just as easily as it
can remove a distractor. Lower recall on multiple-answer questions alone does not
justify MMR, clustering, or forced legal-family coverage.

**Correct the measurement path before paying for another large run.**

The current cross-fit procedure needs two corrections before it can support a claim:
`cv --fit-stacker` fits and evaluates the stacker on the same cached questions, even
if the underlying reranker scores are out of fold. The stacker itself needs held-out
labels. Also, `cache.folds()` hashes question IDs; the local 7,000-question data has
16 exact duplicate-question groups, and 10 of those groups cross the current two
folds. Group equal normalized question text together before assigning folds.

The 1,050-question split has been used repeatedly for development. Report it as a
development comparison. For stronger evidence, use an outer grouped split whose
labels are excluded from every fitted stage, including the fusion; creating outer
predictions may require new reranker fits. Do not label in-sample stacker evaluation
as cross-validated generalization.

There is no universal 0.0094 detection floor. Precision depends on the paired
differences, not only the number of questions. As a mathematical example, seven
single-answer rescues with zero regressions among 1,050 questions give +0.006667
and a two-sided exact sign-test p-value of 0.015625. This is an example, not a new
model result. Repeatedly accepting the best of many nominal 95% intervals still
creates a multiple-comparison problem.

Finally, `compare.py`'s "identical" count means equal recall, not equal predictions.
The local g32-versus-stacker comparison changes 939 ordered top-five lists while
only 19 questions change recall. The phase-six count of 1,011 equal-recall questions
cannot establish that the larger model "barely got a vote." Its raw predictions
are not present locally, so that experiment cannot be independently reanalyzed here.

The proposed document objective should be one prespecified candidate, followed by
one complete paired comparison. Judge net macro-recall gain, wins and losses,
single/multiple-answer buckets, and seen/unseen gold documents. Do not translate
validation improvements into a fixed public-score offset. Once configuration is
settled, fitting on all 7,000 labels is a final deployment step; those same questions
then cannot serve as its holdout.

**What this session established.**

The audit reproduces the saved validation metric, checks split and label consistency,
verifies that error mass sums to 1 minus recall, and distinguishes prediction changes
from recall changes. All 86 missing gold documents are saved for inspection. The local
corpus has the same reported document/chunk counts and settings as the remote baseline,
but a different corpus fingerprint; local titles are auxiliary annotations, and exact
inference reproduction requires checking that difference first.

The document-level objective is now implemented behind `train-ce --document-loss` and
the phase-four runner points at the shipped `models2/reranker-g32` 0.6B checkpoint,
writing a separate `models2/reranker-doc` output. Mining is constrained to the same
50-document candidate depth used at inference, with three passages per positive
document. The Qwen 2B route is not selected by the runner. Training has not been
executed in this workspace because the local Python
runtime has no PyTorch or GPU; the existing 0.9493 checkpoint and submission remain
unchanged. The next GPU run should perform one prespecified training run and one paired
comparison before any public prediction is considered.
