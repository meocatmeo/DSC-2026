"""CPU sensitivity controls for an extra vote; these are not ViRanker predictions."""
from fractions import Fraction
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.current_fusion import FIXED_POLICY, fuse
from legalir2.fourth_voter import pool
from legalir2.h100 import guard
from legalir2.labeled_gate import summarize
from legalir2.store import file_hash, read_json, write_json


def measure(rows, before, after):
    return summarize([(Fraction(len(set(r['gold']) & set(a)), len(r['gold'])),
                       Fraction(len(set(r['gold']) & set(b)), len(r['gold'])), set(a) != set(b))
                      for r, a, b in zip(rows, before, after)])


def main():
    source = ROOT/'reports/current-fusion/20260921-current-v2/development-rankings.json'
    out = ROOT/'reports/current-fusion/20260921-fourth-sensitivity'
    controls = ('duplicate_v9', 'duplicate_duo', 'duplicate_fulltext_duo', 'uninformative_hash')
    weights = (.1, .25, 1.)
    guard(out/'registered-plan.json', {'source': file_hash(source), 'code': file_hash(Path(__file__)),
        'controls': list(controls), 'weights': list(weights),
        'interpretation': 'Sensitivity only; no model predictions, policy selection or deployment'})
    rows = read_json(source)['rows']
    base = [r['rankings']['fulltext_duo'][:5] for r in rows]
    three = [fuse(r['rankings'])[:5] for r in rows]
    trials = []
    for control in controls:
        orderings = [r['rankings'][control.removeprefix('duplicate_')] if control.startswith('duplicate_')
            else sorted(pool(r), key=lambda d: hashlib.sha256(('fourth-control-v1:'+r['id']+':'+d).encode()).digest())
            for r in rows]
        for weight in weights:
            policy = dict(FIXED_POLICY, weights=dict(FIXED_POLICY['weights'], fourth=weight))
            predictions = [fuse(dict(r['rankings'], fourth=fourth), policy)[:5] for r, fourth in zip(rows, orderings)]
            trials.append({'control': control, 'weight': weight, 'vs_three': measure(rows, three, predictions),
                           'vs_fulltext': measure(rows, base, predictions)})
    result = {'questions': len(rows), 'trials': trials, 'submission_allowed': False,
        'conclusion_scope': 'Measures susceptibility to duplicate/no-information votes, not actual ViRanker quality.'}
    write_json(out/'report.json', result)
    for r in trials:
        print(r['control'], r['weight'], r['vs_three'], flush=True)


if __name__ == '__main__':
    main()
