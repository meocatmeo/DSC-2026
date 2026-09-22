"""Add verified missing training imports without changing any frozen release files."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

FILES = ('legalir2/mine.py', 'legalir2/train_bi.py', 'legalir2/train_ce.py')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def install(root, source):
    manifest = json.loads((source/'repair-manifest.json').read_text())
    if (manifest['schema'] != 'retrain-missing-imports-v1' or set(manifest['files']) != set(FILES)
            or sha(root/'private-files.sha256') != manifest['base_manifest_sha256']):
        raise RuntimeError('This repair requires the unchanged fulltext-retrain-4090-v1 release')
    originals = [line.split('  ', 1) for line in (root/'private-files.sha256').read_text().splitlines()]
    for expected, name in originals:
        if name.endswith('.py') and sha(root/name) != expected:
            raise RuntimeError('Frozen Python source changed: '+name)
    # Check every source and destination before writing any new module.
    for name, expected in manifest['files'].items():
        target = root/name
        if sha(source/name) != expected:
            raise RuntimeError('Repair checksum mismatch: '+name)
        if target.is_symlink() or root not in target.resolve().parents:
            raise RuntimeError('Dependency path escapes the workspace')
        if target.exists() and sha(target) != expected:
            raise RuntimeError('Refusing to replace an existing different dependency: '+name)
    for name, expected in manifest['files'].items():
        target = root/name
        if not target.exists():
            temporary = target.with_suffix('.repair.tmp')
            temporary.write_bytes((source/name).read_bytes())
            temporary.replace(target)
        if sha(target) != expected:
            raise RuntimeError('Installed dependency checksum mismatch: '+name)
        print('Verified '+name, flush=True)
    marker = root/'training-dependencies-repair.json'
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(marker)


def active_training(root):
    scripts = {'gpu-fulltext-retrain-4090.sh', 'gpu-private-setup.sh', 'gpu-private-h100.sh'}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if (process/'cwd').resolve(strict=True) != root:
                continue
            argv = [part.decode(errors='replace') for part in (process/'cmdline').read_bytes().split(b'\0') if part]
            if any(Path(arg).name in scripts for arg in argv[1:]) or (
                    '-m' in argv and 'legalir2.fulltext_retrain' in argv):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/DSC-2026-fulltext-retrain-4090-v1'))
    args = parser.parse_args()
    root = args.root.resolve()
    if sys.platform != 'linux':
        raise RuntimeError('Run this installer on the Linux GPU instance')
    if active_training(root):
        raise RuntimeError('A GPU job is still active. Wait for it to exit before repairing imports.')
    import fcntl
    with (root/'.dsc-bootstrap.lock').open('a') as bootstrap, (root/'.ranking-h100.lock').open('a') as workspace:
        for lock in (bootstrap, workspace):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Workspace is busy; no dependency files were changed') from None
        install(root, Path(__file__).resolve().parent)
        python = root/'.venv-h100/bin/python'
        subprocess.run([str(python), '-c',
            'from legalir2.train_ce import derived_provenance; '
            'from legalir2.train_bi import cosine_schedule; from legalir2.mine import load_groups; '
            'from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict; '
            'from legalir2.duo import train; print("Training imports passed in the GPU environment.")'],
            cwd=root, check=True)
    print('Repair complete. Prepared inputs, cached comparisons and checkpoints are unchanged. Resume with the v2 launcher.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('REPAIR ERROR: '+str(error), file=sys.stderr, flush=True)
        sys.exit(1)
