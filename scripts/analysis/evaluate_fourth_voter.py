"""Evaluate a fixed fourth vote against both full-text duo and three-voter fusion."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.current_fusion import FIXED_POLICY, fuse
from legalir2.fourth_voter import validate_result
from legalir2.h100 import digest, guard
from legalir2.store import file_hash, read_json, write_json
from evaluate_current_fusion import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--fourth-run', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--split', choices=('development', 'validation'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    policy = dict(FIXED_POLICY, weights=dict(FIXED_POLICY['weights'], fourth=1.))
    guard(args.out/'registered-policy.json', {'policy': policy, 'split': args.split,
        'code': file_hash(Path(__file__)), 'fusion_code': file_hash(ROOT/'legalir2/current_fusion.py'),
        'selection': 'Fixed equal fourth vote; no weight search on evaluated labels'})
    source, fourth = read_json(args.rankings), read_json(args.fourth_run/'rankings.json')
    manifest = read_json(args.fourth_run/'inputs.json')
    if fourth['inputs'] != digest(manifest) or fourth['source_rankings'] != file_hash(args.rankings):
        raise ValueError('Fourth-voter cache identity differs')
    if manifest['questions'] != file_hash(args.questions) or manifest['rankings'] != file_hash(args.rankings):
        raise ValueError('Labeled source identities differ')
    rows, records = source['rows'], read_json(args.questions)
    lookup = {r['id']: r for r in fourth['rows']}
    expected = 429 if args.split == 'development' else 1050
    if len(rows) != expected or len(lookup) != len(fourth['rows']) or set(lookup) != set(records) or {r['id'] for r in rows} != set(records):
        raise ValueError('Incomplete labeled fourth-voter evaluation')
    base, three, four = [], [], []
    for row in rows:
        record = records[row['id']]
        if not row['gold'] or row['gold'] != record['answer'] or row['question'] != record['question']:
            raise ValueError('Labels or question differ')
        result = lookup[row['id']]
        validate_result(result, row, fourth['inputs'])
        base.append(row['rankings']['fulltext_duo'][:5])
        three.append(fuse(row['rankings'])[:5])
        four.append(fuse(dict(row['rankings'], fourth=result['ordering']), policy)[:5])
    guard(args.out/'inputs.json', {'base': file_hash(args.rankings), 'fourth': file_hash(args.fourth_run/'rankings.json')})
    report = {'split': args.split, 'policy': policy, 'vs_fulltext': compare(rows, base, four),
        'vs_three_voters': compare(rows, three, four), 'submission_allowed': False,
        'pooled_gate': 'Not evaluated here; require complete paired predictions for both labeled splits'}
    write_json(args.out/'predictions.json', {r['id']: {'baseline': a, 'proposed': b} for r, a, b in zip(rows, base, four)})
    write_json(args.out/'report.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
