"""User-requested fixed 0.97 agreement trial using the original saved scores."""
from . import duo_agreement

POLICY = {'name': 'fifth-097', 'minimum_pair_preference': .97,
          'parent': 'original-0.90-agreement', 'protected_prefix': 4, 'maximum_changes': 1}


def predict(base, candidate):
    # The 0.90 rule chooses the highest minimum pair preference first. If that
    # winner fails 0.97, every other eligible challenger also fails it.
    answer, trace = duo_agreement.predict(base, candidate)
    result = dict(trace, policy=POLICY['name'], reverted=False)
    if trace['replaced'] and min(trace['old_pair'], trace['new_pair']) < POLICY['minimum_pair_preference']:
        answer = list(answer); answer[4] = trace['removed']
        result.update(reverted=True, restored_document=trace['removed'])
    return answer, result
