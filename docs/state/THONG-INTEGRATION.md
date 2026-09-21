# Teammate integration, 20 September 2026

Local `main` fast-forwarded from `89fcd61` to `origin/thong` at
`e091868d0ad76ee49e6959fec746cc1ffaefd560`. Existing uncommitted work was
preserved. No push was performed.

The branch adds three CPU experiments: `scripts/rerank_consensus.py`,
`scripts/rerank_trifecta.py`, and `scripts/rerank_deep_tournament.py`.
No model quantization implementation was present. The other remote branch,
`nth` at `f6e8ada`, also contains no quantization implementation found by the
source audit. A quantized checkpoint or an unpushed change may exist elsewhere.

## Production integration

`legalir2/consensus.py` implements probability-weighted voting (Borda), PageRank,
actual win-minus-loss counts (Copeland), and direct head-to-head replacement
gates. Document IDs remain aligned with their scores. Every compared candidate
can compete, including the additional full-text retrieval candidates.

`legalir2/fulltext_prior.py` combines those signals with the full-text relevance
prior. It evaluates a fixed 19-policy grid on development only; ties keep the
baseline. The selected policy must improve the separate validation set before
private deployment. Otherwise the scored 0.9510 answer sets are preserved.
The current duo answers, rather than older V9 answers, anchor the consensus.
The new code also checks that no development/validation IDs or normalized text
aliases occur in the frozen question memory. Existing models load sequentially,
and the private pairwise comparisons are reused.

## Findings in the original experiments

The imported `scripts/rerank_*.py` files are retained as teammate research
references. They are not production entry points and are not shipped in the
4090 release. Use the calibrated runner linked below.

- Trifecta builds memory from all 7,000 records in `data/train.json`, which
  includes validation labels. Excluding only each query's own match does not
  remove other held-out labels from memory.
- DMLT builds its allowed answer set from those same labels and forces removal
  of answers absent from that set. An unseen training document can still be a
  correct test answer. This hard filter is excluded from the integration.
- DMLT's `sign(diff) * abs(diff)` equals `diff`, so its standardized "Copeland"
  duplicated Borda. The integrated version counts wins minus losses.
- DMLT zips a filtered document list with an unfiltered score slice. The
  integration always indexes scores by document ID.
- DMLT's manifest contains fixed 0.964286 validation metrics regardless of the
  executed configuration. These are not accepted as measured results.
- Its validation path uses the same short-passage cache twice and a V9 anchor;
  its private path can fuse distinct full/short caches with a different anchor.
  That validation does not establish performance of the private configuration.
- Signal-cache filenames alone do not establish input/model identity. The
  integrated runner binds caches to datasets, models, evidence and source code.

Multi-view fusion and memory wildcard injection are not enabled in the combined
release: aligned full/short labeled evidence and a separate evaluation would be
needed before deploying them. Existing filtered memory remains in candidate
generation. No additional model or parameter count is introduced.

CPU replay: `reports/integration/thong-e091868/replay.json` (local report).
It uses existing short-passage labeled caches, not new full-text GPU inference.
The original DMLT is replayed both with its all-label input and with the frozen
training-only memory to expose sensitivity to held-out labels. Neither replay
creates private answers or a submission ZIP.

Measured CPU replay:

| Policy | Development (429) | Validation (1,050) |
| --- | ---: | ---: |
| Current short-passage duo | 0.951437 | 0.948254 |
| Integrated Borda, existing prior | 0.950660 | 0.950635 |
| Integrated graph, existing prior | 0.951437 | 0.950635 |
| Original DMLT, all 7,000 labels | 0.955517 | 0.964286 |
| Original DMLT, filtered 5,518 training records | 0.699106 | 0.728016 |

Development retains the baseline in this CPU-only comparison. The validation
improvement of the voting variants is descriptive and was not used to override
that selection. The DMLT filter's apparent gain relies on the all-label answer
space; its result is unsuitable as evidence of generalization. The combined
full-text experiment still needs GPU execution and has no measured private score.

Run instructions: [4090 runbook](../runbooks/RUNBOOK-fulltext-prior-4090.md).
