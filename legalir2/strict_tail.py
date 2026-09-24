"""One fixed 0.99 agreement experiment; no new comparisons or label access."""
from . import deadline_agreement

POLICY = {'name': 'fifth-099', 'minimum_pair_preference': .99,
          'parent': 'fifth-strict', 'protected_prefix': 4, 'maximum_changes': 1}


def predict(base, candidate):
    answer, trace = deadline_agreement.predict(base, candidate, 'fifth-strict')
    result = dict(trace, policy=POLICY['name'], reverted=False)
    if trace['replaced'] and trace['minimum_pair_preference'] < POLICY['minimum_pair_preference']:
        answer = list(answer)
        answer[4] = trace['removed']
        result.update(reverted=True, reverted_document=trace['added'], restored_document=trace['removed'])
    return answer, result
