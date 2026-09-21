# Retrieval investigation after the full-text-prior experiment

Status: CPU investigation implemented and a separate 4090 runner published.
The new ranking has not run on GPU or produced a measured submission score.
The private best remains the user-reported **0.9510**.
All models remain **BF16, without quantization**, per the user's instruction.

Follow-up: the GPU rescue experiment retained baseline. See
[the completed failure audit and next experiment](AFTER-RESCUE-AUDIT.md).

## Verified source and failures

Downloaded the completed `private-fulltext-prior-4090-v1` development caches
from the DSC rclone path. Verified all 429 question identities, frozen dataset
hashes, comparison fits, evidence token traces, relevance/base hashes and the
exact score digest recorded by calibration.

Baseline development recall: **0.9502719503**. The alpha=0.25, strength=0.25
relevance blend changes 184 answer sets, but only one question improves and two
worsen. No relevance logits hit the probability-clipping boundary. The failure
is not explained by clipping or corrupt caches.

There are **32 missed gold-document instances**: 24 outside the candidate pool
and eight present but ranked out. Four gold instances were introduced by the
earlier full-text expansion, but none was selected into the final top five.
The current pool's oracle coverage is **0.964258**. Oracle coverage assumes a
perfect final ranker and must not be reported as attainable model recall.

Audit report: `reports/private/20260920-prior-development/audit.json`.

## Implemented experiments

### Supporting-document retrieval

`legalir2/related_documents.py` builds a citation graph using exact document
numbers in the supplied corpus. Ambiguous ownership is excluded. It proposes
up to eight related documents from the current top five without deleting any
existing candidate. Citation frequency discounts ubiquitous references.
References in the already-selected passages receive extra weight when available.

Also evaluated training co-answer relationships using only the frozen 5,518
training records; all development/validation IDs and normalized aliases were
excluded. This channel recovered no additional development gold documents and
was not selected. It is not permission to use all 7,000 labels.

| Candidate retrieval | Development coverage | Validation coverage |
| --- | ---: | ---: |
| Existing pool | 0.964258 | 0.975714 |
| Citation expansion | 0.967754 | 0.977619 |

Citation expansion recovered three gold documents on development. The policy
was then frozen and recovered four gold documents across three validation
questions. Validation uses the saved short-passage duo pool; it is not a new
full-text model validation. These are candidate coverage results only.

A separate CPU test applied small citation score bonuses to existing compared
candidates. Development selected weight zero; no ranking change was promoted.

Reports: `reports/private/20260920-prior-development/related-v1/`.

### Focused passage selection

`legalir2/focused_retrieval.py` keeps the frozen evidence module unchanged.
It combines full-text unigram and bigram retrieval, document-local term rarity,
and passage diversity. Grammatical filler cannot dominate legal concepts;
negation tokens remain eligible. The final prompt still obeys the existing
2,048-token pair budget and uses the real checkpoint tokenizer.

Ten gold-document bundles from six deliberately selected development questions
were inspected. The final version exposes the previously missing housing
allowance clause in question 29338/document 285041 and the overdue-interest
clause in question 27654/document 132797. Individual document bundles used
782–961 tokens; every checked paired prompt fit the existing limit.
This targeted inspection is not a held-out ranking measurement. The model may
still choose incorrectly even when a better clause is visible.

Probe: `reports/private/20260920-prior-development/focused-fulltext-v2/evidence-probe.json`.

A broader document-level keyword expansion recovered only one additional gold
instance on development (oracle 0.965423). It is not selected for deployment.
The coverage-only audit skips already-covered questions because preserving
their candidates mathematically preserves oracle recall; that shortcut must
never be used in inference or to claim measured runtime/precision.

## Verification and next gate

- Eight unit tests pass for retrieval, citation normalization, known-document
  membership, frozen-index identity, deterministic packing and token budgets.
- Local indexes use the exact frozen store hash
  `bf3799fe1af0ba58f61ba9861ff35dc6c9d352f2a6d7b822dbd7f63873cb5f7a`.
- No private labels, answer edits, quantization, checkpoint modification or
  remote changes outside DSC were performed.
- Published release `20260920-fulltext-prior-4090-v1` is unchanged. These new
  modules are not connected to its launcher and must not be copied into a live
  signed run.

Before another full private inference run, the improved evidence and supporting
documents need development GPU scoring and then a fixed-policy validation
check. A larger candidate pool can increase cost and introduce false positives;
candidate recovery alone is not sufficient to recommend the full GPU run.
No significant private-score increase has been demonstrated by this work.

## Isolated 4090 release

`20260920-retrieval-rescue-4090-v1` uses `legalir2/retrieval_rescue.py` and
`scripts/gpu/gpu-retrieval-rescue-4090.sh`. Instructions:
[4090 runbook](../runbooks/RUNBOOK-retrieval-rescue-4090.md).

It reuses the verified 429 full-text development baselines and compares focused
contenders against the original top five. Development selects one of four
conservative replacement policies or baseline; baseline wins ties. Only a
positive fixed-policy validation difference allows private inference. Validation
first computes its missing full-text baseline, rather than using the earlier
short-passage scores. Both V9 and duo submission ZIPs are always produced; a
rejected policy falls back to the exact scored duo answers.

Checks passed: scoring equivalence against the frozen comparator, one-swap and
confidence rules, corrupt-cache rejection, resume without model loading,
development-only selection, validation regression rejection, submission routing,
real Bash stage sequencing/failure backup, and existing auto-stop checks. Real
source replay verified all 2,080 private answers and 429 development rows. Three
sample questions produced 300 real-tokenizer sequences, at most 2,033 tokens.
The local release smoke test mocks only GPU asset loading; GPU inference and
speed remain untested. See `reports/private/retrieval-rescue-release-check/verification.json`.

The launcher restores through the existing DSC asset paths, uploads logs every
60 seconds, resumes signed shards and includes the submit stage. With
`AUTO_STOP=1`, the existing Vast watcher verifies successful exit and cloud
backup before stopping compute. Setup precedes arming; failures leave the
instance running. The previous release and scored ZIP are unchanged.

Published under `gdrive:DSC-2026-results/releases/private/20260920-retrieval-rescue-4090-v1`.
ZIP SHA-256: `361e91286cbd1983e79d8876835eb8e5268cde6d64163c6d1e058ececc600424`.
Remote verification downloaded and compared both release files: two matching,
zero differences. A local `published.json` prevents accidental release overwrite.
