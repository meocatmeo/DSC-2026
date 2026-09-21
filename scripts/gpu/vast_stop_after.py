"""Arm a separate Vast stop watcher for a successful, fully backed-up DSC run."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

DEFAULT_RUN = 'private-duo-fulltext-v2-4090-fast1'


def api(instance, key, stop=False):
    request = urllib.request.Request(
        f'https://console.vast.ai/api/v0/instances/{instance}/',
        data=b'{"state":"stopped"}' if stop else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
        method='PUT' if stop else 'GET')
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Vast API HTTP {error.code}; check instance ID and API-key permissions') from None
    except urllib.error.URLError:
        raise RuntimeError('Cannot reach Vast API; instance has not been confirmed stopped') from None
    if stop:
        if result.get('success') is not True:
            raise RuntimeError('Vast did not confirm acceptance of the stop request')
    else:
        record = result.get('instances', {})
        if not isinstance(record, dict) or str(record.get('id')) != instance:
            raise RuntimeError('Vast response does not identify the requested instance')
    return result


def ready(root, run):
    out = root/'outputs/private'/run
    if (out/'.stage').read_text().strip() != 'exited status=0; final backup':
        raise RuntimeError('Run did not finish successfully. Instance remains running; inspect run.log.')
    questions = json.loads((root/'data/private-official.json').read_text(encoding='utf-8'))
    for model in ('v9', 'duo'):
        with zipfile.ZipFile(out/f'submission-private-{model}.zip') as archive:
            if archive.namelist() != ['submission.json'] or archive.testzip() is not None:
                raise RuntimeError('Invalid submission ZIP; instance remains running')
            predictions = json.loads(archive.read('submission.json'))
        if set(predictions) != set(questions):
            raise RuntimeError('Incomplete submission; instance remains running')
        if any(len(r['answer']) != 5 or len(set(r['answer'])) != 5 for r in predictions.values()):
            raise RuntimeError('Invalid answer count; instance remains running')
    return out


def verify_backup(out, run):
    # Match the existing runtime's artifact exclusions. Checking the whole run
    # also covers candidate caches, evidence traces and reproducibility files.
    remote = f'gdrive:DSC-2026-results/runs/private/{run}/artifacts'
    flags = ['--exclude', '/.*', '--exclude', '**/.ship.*/**', '--exclude', '*.tmp',
             '--exclude', '*.partial', '--exclude', 'run.log', '--exclude', 'transfer-errors.log',
             '--exclude', 'STATUS.txt']
    common = ['rclone', '--contimeout', '20s', '--timeout', '90s', '--retries', '2']
    subprocess.run(common + ['check', str(out), remote, '--one-way'] + flags,
                   check=True, timeout=1200)


def watch(config):
    import fcntl
    settings = json.loads(config.read_text())
    root, run = Path(settings['root']), settings['run']
    # Do not compete with or change the inference process. Once obtained, hold
    # its workspace lock through verification so a new run cannot start here.
    print('Waiting for the inference workspace lock to be released.', flush=True)
    with (root/'.ranking-h100.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(30)
        out = ready(root, run)
        print('Run exited successfully. Verifying cloud backup.', flush=True)
        verify_backup(out, run)
        print('Backup verified. Requesting Vast STOP (disk is preserved).', flush=True)
        for attempt in range(3):
            try:
                api(settings['instance'], settings['key'], stop=True)
                print('Vast accepted STOP. Check the instance becomes Inactive in the console.', flush=True)
                return
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', default=DEFAULT_RUN)
    parser.add_argument('--watch', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watch:
        watch(args.watch)
        return
    if sys.platform != 'linux':
        raise RuntimeError('Run this helper in a SECOND terminal on the Linux GPU instance')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run):
        raise ValueError('Invalid DSC run ID')
    root = args.root.resolve()
    out = root/'outputs/private'/args.run
    if not (out/'.stage').is_file() or not (root/'.ranking-h100.lock').is_file():
        raise RuntimeError('Expected existing DSC run not found; check --root and --run')
    instance = os.environ.get('CONTAINER_ID', '') or os.environ.get('VAST_CONTAINERLABEL', '').removeprefix('C.')
    if not instance:
        instance = input('Vast INSTANCE ID (not machine ID): ').strip()
    if not re.fullmatch(r'[0-9]+', instance):
        raise ValueError('Invalid instance ID')
    key = os.environ.get('CONTAINER_API_KEY', '')
    if not key:
        key = getpass.getpass('Vast API key (hidden; from cloud.vast.ai/manage-keys/): ').strip()
    if not key:
        raise ValueError('An API key is required to stop this instance')
    api(instance, key)  # Read-only identity/authentication check before arming.
    directory = Path.home()/'.config/dsc-vast-stop'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    config = directory/f'{instance}-{args.run}-{time.time_ns()}.json'
    with config.open('x', encoding='utf-8') as stream:
        config.chmod(0o600)
        json.dump({'root': str(root), 'run': args.run, 'instance': instance, 'key': key}, stream)
    log = directory/(config.stem + '.log')
    with log.open('w') as stream:
        log.chmod(0o600)
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--watch', str(config)],
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    print(f'AUTO-STOP ARMED for instance {instance}, run {args.run}. Watcher PID: {process.pid}')
    print(f'Watcher log: {log}')
    print('Inference continues unchanged. STOP follows successful exit and verified backup.')
    print('On run/upload/API failure, the instance stays running. STOP preserves disk; storage charges continue.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'AUTO-STOP ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
