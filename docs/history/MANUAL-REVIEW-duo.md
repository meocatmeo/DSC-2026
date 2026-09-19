# Manual review of comparator mistakes

Direct corpus review can supply better explanations and testable ranking hypotheses. It does not reveal the hidden public answer IDs, and editing a JSON file does not automatically train the local model.

## Three confirmed validation failures

- **64354 — practice after imprisonment:** gold document `234757` contains the conditions involving criminal proceedings, serving a sentence and bans on practice. The promoted `70582` is scoped to licensing in the military. The lesson is to match the applicable population and decisive condition, rather than the shared medical-practice topic.
- **152976 — foreign-element adoption:** `35408`, which both rankings retain, directly defines adoption involving foreigners permanently resident in Vietnam. The additional gold document `77650` supplies residence terminology. The promoted `244431` describes the adoption department's functions. This illustrates that multiple labeled documents may provide complementary evidence, while a topical administrative document need not supply the requested definition.
- **41690 — admission-eligibility penalties:** retained `278875` has Article 9 on the exact issue. Removed gold document `242728` explicitly amends clauses of Article 9. Promoted `50885` does contain admission penalties, but for driving schools. This is a scope and amendment-link failure, not simply an unrelated keyword match. A title-only inspection missed that nuance.

These are observations from the frozen competition corpus and its labels, not current legal advice. We have not yet reconstructed the exact token windows the model saw, so we cannot determine whether it missed the evidence or judged it incorrectly after reading it.

## One manually reviewed public candidate

Question **71086** asks about disciplining/dismissing a company employee caring for a child under twelve months. Starting with the original V9 submission, replace `109796` (civil-service disciplinary rules) with `143446` (the 2012 Labor Code), whose text explicitly mentions that employee condition. Preserve the other four documents, including the later Labor Code already returned by V9.

This is a plausible corpus-supported relevance edit, **not a confirmed public-label correction**. The question does not name a legal version; whether the hidden labels credit the historical code is unknown. The comparator made this same individual swap, but hundreds of other changes prevent its aggregate public score from evaluating this one edit.

Files in `analysis/impact/duo-20260918T110105Z/manual-review/`:

- `submission-manual-01.zip`: separate experimental submission, exactly one question changed from original V9; all 1,000 questions and document IDs validated.
- `review.json`: exact before/after IDs, supporting excerpt, limitations and three diagnostic validation counterexamples.

No claim of a score increase is made. The original V9 file remains intact. The candidate has not been uploaded or submitted.

## How the model can learn the lesson

Use these cases to specify general training and retrieval improvements: preserve amendment links, compare document scope, and retain complementary evidence for multi-document answers. Mine analogous examples from the **training split**, annotate evidence spans and difficult alternatives, then fine-tune the local model and evaluate on untouched questions.

Do not train on these reviewed validation questions and then report their improved validation score as generalization. Public manual hypotheses also must not become unquestioned positive labels. Three post-hoc validation corrections would improve that file by construction; they would not demonstrate a better retrieval system.

The review is an initial seven-case inspection (three validation losses and four related public changes), not a completed audit of all 454 changed public answer sets. Only the one described public edit has been packaged.
