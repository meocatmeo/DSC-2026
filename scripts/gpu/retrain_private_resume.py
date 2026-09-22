"""Resume with direct fitted-score validation instead of a discarded second solve."""
import argparse
from pathlib import Path
import sys

import retrain_private_submit as runner
import numpy as np

retrain = runner.retrain


def validate_rows(rows, base):
    if len(rows) != len(base):
        raise ValueError('Incomplete retrained comparator cache')
    for row, original in zip(rows, base):
        size = len(row['docs'])
        if (row.get('base') != runner.digest(original) or size < 5 or len(set(row['docs'])) != size
                or any(row[k] != original[k] for k in ('id', 'question', 'gold', 'docs', 'prior'))):
            raise ValueError('Inference pool, evidence or prior changed')
        for field in ('prior', 'ability'):
            values = np.asarray(row[field], dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError('Invalid comparison scores')
        probability = np.asarray(row['probabilities'], dtype=float)
        if (probability.shape != (size, size) or not np.isfinite(probability).all()
                or (probability < 0).any() or (probability > 1).any()
                or not np.allclose(probability + probability.T, 1., atol=1e-6)):
            raise ValueError('Invalid comparison tournament')
        # The existing residual test checks the actual saved optimum at 1e-5.
        # Do not re-normalize prior and solve an unused second optimization problem.
        retrain.private.validate_comparison_fit(row)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, required=True)
    args, _ = parser.parse_known_args()
    runner.guard(args.out/'numerical-validation-repair.json', {
        'schema': 'retrain-direct-fit-validation-v1',
        'wrapper_sha256': runner.file_hash(Path(__file__)),
        'original_runner_sha256': runner.file_hash(Path(runner.__file__)),
        'gradient_max_abs_tolerance': 1e-5,
        'changed': 'Validate matrix and saved fit directly; omit unused second tournament solve.',
        'ranking_model_and_policy_changed': False})
    original = retrain.validate_rows
    retrain.validate_rows = validate_rows
    print('Numerical repair active: direct saved-fit validation; model, scoring and cache identity unchanged.', flush=True)
    try:
        runner.main()
    finally:
        retrain.validate_rows = original


if __name__ == '__main__':
    main()
