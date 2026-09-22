"""Three fixed deadline experiments around the scored agreement submission.

No fitting, gold access, new inference or mutation of the original policy.
"""
from . import duo_agreement

POLICIES = {
    'fifth-wide': {'slot': 4, 'minimum_pair_preference': .8},
    'fifth-strict': {'slot': 4, 'minimum_pair_preference': .98},
    'fourth-rescue': {'slot': 3, 'minimum_pair_preference': .95},
}


def predict(base, candidate, name):
    if name not in POLICIES:
        raise ValueError('Unknown deadline policy')
    current, _ = duo_agreement.predict(base, candidate)
    prior, old_ability, old_p = duo_agreement.scores(base)
    _, new_ability, new_p = duo_agreement.scores(candidate)
    docs = base['docs']
    original = sorted(range(len(docs)), key=lambda i: (-float(.75*prior[i]+.25*old_ability[i]),docs[i]))[:5]
    policy = POLICIES[name]
    answer = current.copy() if name == 'fourth-rescue' else [docs[i] for i in original]
    victim = original[policy['slot']]
    blocked = set(original) | {docs.index(d) for d in answer}
    eligible = []
    for i, doc in enumerate(docs):
        if i in blocked:
            continue
        confidence = min(old_p[i,victim],new_p[i,victim])
        margin = min(old_ability[i]-old_ability[victim],new_ability[i]-new_ability[victim])
        if confidence >= policy['minimum_pair_preference'] and margin > 0:
            eligible.append((float(confidence),float(margin),doc))
    trace = {'policy':name,'eligible':len(eligible),'replaced':False}
    if eligible:
        confidence, margin, doc = min(eligible,key=lambda r:(-r[0],-r[1],r[2]))
        answer[policy['slot']] = doc
        trace.update(replaced=True,removed=docs[victim],added=doc,
                     minimum_pair_preference=confidence,minimum_ability_margin=margin)
    if len(set(answer))!=5 or answer[:3]!=current[:3]:
        raise ValueError('Invalid protected answer')
    if name=='fourth-rescue' and answer[4]!=current[4]:
        raise ValueError('Current fifth answer changed')
    if name!='fourth-rescue' and answer[:4]!=current[:4]:
        raise ValueError('Protected first four changed')
    return answer,trace
