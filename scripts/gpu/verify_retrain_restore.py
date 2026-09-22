"""Verify all restored source shards before spending GPU time on private inference."""
from pathlib import Path
from retrain_private_submit import options, retrain


def main():
    args = options(Path('outputs/retrain/duo-fulltext-retrain-4090-v1'))
    retrain.manifest(args)
    directory, signature = retrain.arm_inputs(args, 'control')
    retrain.duo.verify_model(directory/'model', signature)
    count = sum(len(rows) for _, rows in retrain.prior.bases(retrain.reference(args), 'private'))
    expected = len(retrain.prior.records(retrain.reference(args), 'private'))
    if count != expected:
        raise ValueError('Restored private evidence is incomplete')
    print(f'Restored control model and all {count} private evidence rows verified. No GPU inference performed.', flush=True)


if __name__ == '__main__':
    main()
