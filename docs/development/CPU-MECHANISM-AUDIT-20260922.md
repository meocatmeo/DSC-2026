# CPU mechanism audit — 22 September 2026

The frozen 0.9523 agreement submission remains the best measured private result.
Neither experiment below produced a candidate worth replacing it with. No private
answers were edited, no new submission was produced, and no GPU scoring was run.
Development has already been examined repeatedly: these are diagnostic results,
not independent evidence of generalization. Validation was not scored.

## 1. Supervised question-to-document retrieval

`legalir2/learned_question_retrieval.py` fits a fixed kernel-ridge multilabel
retriever from the frozen training questions. It uses question vocabulary to
learn associations with gold document IDs. Neither development nor validation
questions enter the fit (ID and normalized-text checks).

Of 5,518 source rows, duplicate cleanup retains 5,506: eight identical duplicates
are removed and four rows with conflicting labels for identical text are excluded.
The fit covers 2,656 document labels. It proposes up to four documents outside
the original candidate pool; it does not alter the ranking.

| Development coverage, 429 questions | Recall | Change |
| --- | ---: | ---: |
| Original candidate pool | 0.964258 | — |
| Add supervised retrieval | 0.966589 | +0.002331 |
| Add nearest-neighbor retrieval control | 0.967754 | +0.003497 |

The supervised model recovers two gold documents but underperforms the control.
This is **candidate coverage**, not final top-five recall. Documents never labeled
in training cannot be recovered through a learned document-label association.
Do not spend GPU hours promoting this version.

Reproduce with `scripts/analysis/evaluate_learned_questions.py`; frozen settings,
results and proposals are in `reports/learned-question-retrieval-v2/`.

## 2. Isolate new-candidate admission from deep-prior changes

`legalir2/append_only_deep.py` keeps the current agreement baseline's first four
answers. It considers only candidates added by the saved 512 deep-pool run and
ignores that run's changed prior and fitted abilities. A newcomer must beat the
current fifth answer with saved direct probability at least 0.95, and have a
positive mean preference advantage against the four protected answers.

| Development top-five recall | Value |
| --- | ---: |
| Current agreement baseline | 0.952603 |
| Append-only admission | 0.950272 |
| Difference | −0.002331 |
| Better / worse / changed sets | 0 / 1 / 14 |
| Paired bootstrap 95% interval | [−0.006993, 0] |
| New gold documents available / admitted | 5 / 0 |

Question 4580 loses gold document 288898 to 105901 despite a 0.98287 direct
comparator preference. Strong comparator confidence alone is not a safe admission
signal. This result does **not** establish the cause of the earlier private
deep-pool regression; it shows that freezing the baseline and removing the new
prior do not, by themselves, solve this development failure.

Reproduce with `scripts/analysis/evaluate_append_only_deep.py`. Source comparison
rows are checked against the completed 512 run's calibration digest; original
comparison matrices and baseline/retrained identities are also verified.

## 3. What the five gold candidates actually showed the models

`scripts/analysis/audit_deep_evidence.py` exports the exact saved passages and token
IDs after checking each raw row against its comparison digest. Output:
`reports/append-only-deep-v1/evidence-audit.json`. These cases were selected after
looking at errors; they must not become question-specific submission rules.

| Question / document | Observed evidence problem |
| --- | --- |
| 61600 / 77960 | Empty document label; packed evidence consists mainly of water-quality tables. The question asks testing frequency. |
| 163810 / 97249 | Issuing circular refers to attached QCVN 11-1:2012/BYT. Saved passages contain the issuing text rather than the requested lead limit. The top answer, 148668, is an unnamed infant-formula table document; a parent/annex relationship is plausible but not verified. |
| 52892 / 237914 | Correct fisheries-inspection authority clause is present. Existing second pointwise reranker scores it 0.99409, but the comparator assigns only about 0.029 against the current fifth answer. This is not simply a missing-clause failure. |
| 51092 / 258168 | Saved evidence is mainly a procedure index and citations to other decisions. Second pointwise score is 0.98409; comparator preference against the current fifth is about 0.3165. High pointwise score alone also needs calibration. |
| 159820 / 279667 | The consultation restriction is in passage two. Passage one discusses appointments and asset verification. Existing first pointwise reranker scores passage two −4.6875 versus passage one's −9.9375. Lexical passage ordering placed the less relevant passage first. |

The raw corpus has only `id`, `name`, and `passage`; explicit attachment ownership
is absent. Do not infer a parent link solely from file adjacency or shared terms.
The audit distinguishes the displayed saved passages from claims about complete
legal coverage; it does not assert that the gold labeling is wrong.

## Next experiment worth designing

Test **passage relevance before pair comparison**, using the existing pointwise
reranker's passage scores to select/order evidence, rather than spending tokens
first on lexical overlap. Keep candidate pools, checkpoint, prior, answer count,
and admission policy fixed so only evidence selection changes. Training must use
the same evidence construction as inference if a new checkpoint is fitted.

This is a hypothesis, not an implemented production change or promised increase.
The saved scores can demonstrate ordering errors, but cannot predict the new
duo comparisons: a GPU rescore is required to measure the effect. Predeclare the
policy, compare all available labeled examples before private inference if
authorized, and report paired differences and uncertainty. Do not tune on these
five examples or interpret the reused development score as a fresh holdout.

Separately, validate explicit corpus-supported attachment links before attempting
parent/annex evidence sharing. Copying text between loosely related documents
would create false evidence and can make the existing failures worse.

## Checks

The learned-retrieval and append-only modules each have four targeted tests;
the four existing agreement tests also pass (12 total). These checks validate
implementation invariants, not competition-score improvement. The frozen release
and its cloud backup were not modified.
