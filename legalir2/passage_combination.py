"""Frozen winner with a fifth-slot challenge supported by both evidence orders."""
from . import agreement_097, duo_agreement

POLICY = {'name': 'passage-combination-fifth-v1', 'anchor': agreement_097.POLICY,
          'protected_prefix': 4, 'maximum_replacements': 1,
          'new_minimum_pair_preference': .97, 'old_minimum_pair_preference_exclusive': .5,
          'ability_support': 'positive challenger-minus-current-fifth in all four score rows',
          'tie_break': 'new minimum preference descending, four-row minimum ability margin descending, document ID ascending'}


def predict(original, control, ordered_original, ordered_control):
    """Never read labels; the challenger must beat the actual winning fifth."""
    rows = (original, control, ordered_original, ordered_control)
    for row in rows[1:]:
        if any(row[k] != original[k] for k in ('id', 'question', 'docs', 'prior')):
            raise ValueError('Combination requires identical questions, candidate order and priors')
    scores = [duo_agreement.scores(row) for row in rows]
    answer, _ = agreement_097.predict(original, control)
    docs = original['docs']; victim = docs.index(answer[4]); chosen = set(answer)
    eligible = []; rejected = {'new_preference': 0, 'old_preference': 0, 'ability': 0}
    for i, doc in enumerate(docs):
        if doc in chosen:
            continue
        old_pairs = [float(p[i, victim]) for _, _, p in scores[:2]]
        new_pairs = [float(p[i, victim]) for _, _, p in scores[2:]]
        margins = [float(ability[i]-ability[victim]) for _, ability, _ in scores]
        if min(new_pairs) < POLICY['new_minimum_pair_preference']:
            rejected['new_preference'] += 1
        elif min(old_pairs) <= POLICY['old_minimum_pair_preference_exclusive']:
            rejected['old_preference'] += 1
        elif min(margins) <= 0:
            rejected['ability'] += 1
        else:
            eligible.append((min(new_pairs), min(margins), doc, old_pairs, new_pairs, margins))
    trace = {'policy': POLICY['name'], 'replaced': False, 'eligible': len(eligible),
             'rejected': rejected, 'anchor_fifth': answer[4]}
    if eligible:
        _, _, doc, old_pairs, new_pairs, margins = min(eligible, key=lambda r: (-r[0], -r[1], r[2]))
        trace.update(replaced=True, removed=answer[4], added=doc, old_pairs=old_pairs,
                     new_pairs=new_pairs, ability_margins=margins)
        answer = list(answer); answer[4] = doc
    return answer, trace
