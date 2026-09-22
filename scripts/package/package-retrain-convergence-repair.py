"""Immutable v3 tools based on v2, with only a private row-validator wrapper."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT/'releases/tools/20260921-retrain-private-submit-v2'
    payload = {}
    for line in (source/'files.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        data = (source/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Frozen v2 tool changed: '+name)
        payload[name] = data
    payload['retrain_private_resume.py'] = (ROOT/'scripts/gpu/retrain_private_resume.py').read_bytes().replace(b'\r\n', b'\n')
    old = b'"$TOOL_DIR/retrain_private_submit.py"'
    shell = payload['gpu-retrain-private-submit.sh']
    if shell.count(old) != 1:
        raise ValueError('Unexpected v2 dispatcher')
    payload['gpu-retrain-private-submit.sh'] = shell.replace(old, b'"$TOOL_DIR/retrain_private_resume.py"')
    payload['README.md'] = (ROOT/'docs/runbooks/RUNBOOK-retrain-convergence-repair.md').read_bytes().replace(b'\r\n', b'\n')
    payload['files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    out = ROOT/'releases/tools/20260922-retrain-private-submit-v3'
    for name, data in payload.items():
        target = out/name
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Use a new immutable release ID')
    out.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        (out/name).write_bytes(data)
    print(out)
    print('Files:', len(payload))


if __name__ == '__main__':
    main()
