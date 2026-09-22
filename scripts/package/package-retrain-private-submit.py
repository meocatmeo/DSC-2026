"""Publishable additive private-inference tools; frozen model code is not edited."""
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    names = ['launch_retrain.py', 'vast_stop_retrain.py', 'retrain_private_submit.py', 'gpu-retrain-private-submit.sh']
    if args.fresh:
        names += ['gpu-retrain-private-fresh.sh', 'verify_retrain_restore.py']
    payload = {name: (ROOT/'scripts/gpu'/name).read_bytes().replace(b'\r\n', b'\n') for name in names}
    document = 'RUNBOOK-retrain-private-fresh.md' if args.fresh else 'RUNBOOK-retrain-private-submit.md'
    payload['README.md'] = (ROOT/'docs/runbooks'/document).read_bytes().replace(b'\r\n', b'\n')
    payload['files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    directory = ROOT/'releases/tools'/('20260921-retrain-private-submit-v2' if args.fresh else '20260921-retrain-private-submit-v1')
    for name, data in payload.items():
        target = directory/name
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Use a new release version; existing releases are immutable')
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        (directory/name).write_bytes(data)
    print(directory)
    print('Files:', len(payload), 'Bytes:', sum(map(len, payload.values())))


if __name__ == '__main__':
    main()
