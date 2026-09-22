"""Check actual paired predictions against the registered FTS coverage and score gates."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.h100 import digest, guard
from legalir2.labeled_gate import evaluate
from legalir2.store import file_hash, read_json, write_json
from audit_prior_development import DATASET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, default=ROOT/'reports/private/20260921-fts-coverage')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.coverage/'registered-plan.json')
    coverage = read_json(args.coverage/'report.json')
    if coverage['plan'] != digest(plan) or plan['coverage_gain_required'] != .008 or plan['score_deadband'] != .0052:
        raise ValueError('Coverage plan differs')
    if not coverage['coverage_gate_cleared'] or coverage['best_exploratory']['pooled']['delta'] <= .008:
        raise ValueError('Coverage gate did not clear; stop before admission evaluation')
    records = {s: read_json(DATASET/(s+'.json')) for s in ('development', 'validation')}
    if {s: len(r) for s, r in records.items()} != {'development': 429, 'validation': 1050}:
        raise ValueError('Expected all 1479 labeled questions')
    if {s: digest(r) for s, r in records.items()} != plan['datasets']:
        raise ValueError('Labeled set identity changed')
    bundle = read_json(args.predictions)
    if (bundle['coverage_plan'] != digest(plan) or bundle['baseline_method'] != 'duo-fulltext-v2'
            or bundle['policy'] != coverage['best_exploratory']['policy']):
        raise ValueError('Predictions do not identify the swept policy and full-text baseline')
    # Identities must refer to the same algorithms on both splits. Exporters must
    # verify source cache hashes before writing this bundle; this gate cannot
    # reconstruct neural predictions or independently certify a supplied cache.
    if not bundle['baseline_identity'] or not bundle['proposal_identity']:
        raise ValueError('Missing inference identities')
    known = set(read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')['known_docs'])
    guard(args.out/'inputs.json', {'predictions_sha256': file_hash(args.predictions),
        'coverage_plan': digest(plan), 'baseline_identity': bundle['baseline_identity'],
        'proposal_identity': bundle['proposal_identity'], 'gate_code': file_hash(Path(__file__)),
        'evaluation_code': file_hash(ROOT/'legalir2/labeled_gate.py')})
    report = evaluate(records, bundle['splits'], known)
    report['score_gate_passed'] = report['pooled']['decision'] == 'promote'
    report['submission_performed'] = False
    write_json(args.out/'report.json', report)
    print(report)


if __name__ == '__main__':
    main()
