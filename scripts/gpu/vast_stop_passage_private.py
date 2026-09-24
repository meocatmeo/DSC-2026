"""Stop the specified Vast instance after the passage-order evaluation and backup."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_RUN = 'passage-order-private-4090-v1'
DEFAULT_ROOT = '/workspace/DSC-passage-order-private-4090-v1'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def api(instance, key, stop=False):
    # Preserve the API operation used by the previously tested private-run watcher.
    request = urllib.request.Request(
        f'https://console.vast.ai/api/v0/instances/{instance}/',
        data=b'{"state":"stopped"}' if stop else None,
        headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'},
        method='PUT' if stop else 'GET')
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Vast API HTTP {error.code}; check instance ID and API-key permissions') from None
    except urllib.error.URLError:
        raise RuntimeError('Cannot reach Vast API; STOP has not been confirmed') from None
    if stop:
        if result.get('success') is not True:
            raise RuntimeError('Vast did not confirm acceptance of STOP')
    else:
        record = result.get('instances', {})
        if not isinstance(record, dict) or str(record.get('id')) != instance:
            raise RuntimeError('Vast response does not identify the requested instance')
    return result


EXPECTED_PLAN = 'dc873a0f7d3f2a385dd39e462f5ee0f79d35fb1736fd6eebfa3b5470023f43c3'


def ready(root, run):
    out = root/'outputs/passage-order'/run
    if (out/'.stage').read_text().strip() != 'exited status=0; final backup':
        raise RuntimeError('Run failed or is incomplete; instance remains running.')
    plan, prepared = read_json(out/'plan.json'), read_json(out/'prepared.json')
    identity = digest(plan)
    if identity != EXPECTED_PLAN or prepared['plan'] != identity or plan['questions'] != 2080:
        raise RuntimeError('Evaluation belongs to a different prepared experiment')
    if len(prepared['files']) != 2080:
        raise RuntimeError('Incomplete prepared question inventory')
    report = read_json(out/'submission-choice.json')
    if report['plan'] != identity or report['questions'] != 2080:
        raise RuntimeError('Missing or foreign completed evaluation report')
    for name, expected in plan['data_files'].items():
        path = (out/name).resolve()
        if not path.is_relative_to(out.resolve()) or file_hash(path) != expected:
            raise RuntimeError('Submission reference data checksum differs')
    fingerprints = {}
    for name, expected in prepared['files'].items():
        path = (out/name).resolve()
        if not path.is_relative_to(out.resolve()) or file_hash(path) != expected:
            raise RuntimeError('Prepared question checksum differs')
        for arm in ('original-duo', 'retrained-control'):
            cache = out/'scores'/arm
            marker = {'plan': identity, 'model': digest(plan['models'][arm])}
            if read_json(cache/'resume.json') != marker:
                raise RuntimeError('Comparator cache belongs to another run')
            replay = read_json(cache/'replay.json')
            if (any(replay[k] != value for k, value in marker.items())
                    or replay['baseline_predictions_identical'] is not True
                    or not 0 <= replay['maximum_probability_error'] <= plan['replay_tolerance']):
                raise RuntimeError('Unmodified-input GPU replay did not pass')
            score_path = cache/path.name
            row = read_json(score_path)
            if row['source'] != digest(read_json(path)):
                raise RuntimeError('Scored question input differs')
            fingerprints[f'{arm}/{path.name}'] = file_hash(score_path)
    if report['scores'] != digest(fingerprints):
        raise RuntimeError('Incomplete or changed scored evaluation')
    import zipfile
    if report.get('private_inference_performed') is not True or report['zip_sha256'] != file_hash(out/'submission-private-passage-order.zip'):
        raise RuntimeError('Final private submission is missing or changed')
    if report['alternate_zip_sha256'] != file_hash(out/'submission-private-passage-order-098.zip'):
        raise RuntimeError('Alternative submission is missing or changed')
    questions = read_json(out/'data/questions.json')
    known = set(read_json(out/'data/known-docs.json'))
    with zipfile.ZipFile(out/'submission-private-passage-order.zip') as archive:
        if archive.namelist() != ['submission.json'] or archive.testzip() is not None:
            raise RuntimeError('Corrupt submission ZIP')
        predictions = json.loads(archive.read('submission.json'))
    if len(predictions) != 2080 or set(predictions) != set(questions):
        raise RuntimeError('Incomplete submission questions')
    if any(len(r['answer']) != 5 or len(set(r['answer'])) != 5 or not set(r['answer']) <= known for r in predictions.values()):
        raise RuntimeError('Invalid submission answer')
    return out


def verify_backup(out, run):
    remote = f'gdrive:DSC-2026-results/runs/passage-order/{run}/artifacts'
    flags = ['--exclude', '/.*', '--exclude', '**/.ship.*/**', '--exclude', '*.tmp',
             '--exclude', '*.partial', '--exclude', 'run.log', '--exclude', 'transfer-errors.log',
             '--exclude', 'STATUS.txt']
    # Google Drive supplies content hashes. Check every local artifact, including
    # all prepared inputs and both comparator score caches; remote extra files cannot hide a missing local artifact.
    subprocess.run(['rclone', '--contimeout', '20s', '--timeout', '90s', '--retries', '2',
                    'check', str(out), remote, '--one-way']+flags, check=True, timeout=1200)


def watcher_settings(directory, root, run, instance):
    """Find this helper's live watcher without displaying or passing its saved key."""
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = [p.decode() for p in (process/'cmdline').read_bytes().split(b'\0') if p]
            if '--watch' not in argv or not any(Path(p).name == 'vast_stop_passage_private.py' for p in argv):
                continue
            config = Path(argv[argv.index('--watch')+1]).resolve()
            if config.parent != directory.resolve():
                continue
            settings = read_json(config)
            if (Path(settings['root']).resolve() == root and settings['run'] == run
                    and settings['instance'] == instance):
                return settings
        except (OSError, ValueError, KeyError, IndexError):
            continue
    return None


def watch(config):
    import fcntl
    settings = read_json(config)
    root, run = Path(settings['root']), settings['run']
    stage = root/'outputs/passage-order'/run/'.stage'
    print('Waiting for the target passage-order run to exit. Setup may still be running.', flush=True)
    with (root/'.ranking-h100.lock').open('a') as lock:
        while True:
            # Setup/restore release this same lock between stages. A free lock
            # alone is not a completed passage-order job; require the target exit marker.
            if (stage.is_file() and stage.stat().st_mtime_ns >= settings.get('after_ns', 0)
                    and stage.read_text().strip().startswith('exited status=')):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
            time.sleep(30)
        out = ready(root, run)  # Re-read under the lock; failed exits never STOP.
        print('Run succeeded. Verifying all cloud artifacts before STOP.', flush=True)
        verify_backup(out, run)
        print('Backup verified. Requesting Vast STOP; disk is preserved.', flush=True)
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
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--run', default=DEFAULT_RUN)
    parser.add_argument('--instance', required=False, help='Explicit Vast instance ID; must match this terminal')
    parser.add_argument('--after-ns', type=int, default=0,
                        help='Ignore completion markers older than this launch timestamp')
    parser.add_argument('--watch', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.watch:
        watch(args.watch)
        return
    if sys.platform != 'linux':
        raise RuntimeError('Arm this from a SECOND terminal on the Linux GPU instance')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run):
        raise ValueError('Invalid DSC run ID')
    root = args.root.resolve()
    if not (root/'scripts/gpu/gpu-passage-private-4090.sh').is_file():
        raise RuntimeError('Passage-order workspace is missing; wait for extraction or check --root')
    detected = [os.environ.get('CONTAINER_ID', ''),
                os.environ.get('VAST_CONTAINERLABEL', '').removeprefix('C.')]
    detected = [s for s in detected if s]
    if len(set(detected)) > 1:
        raise RuntimeError('Conflicting Vast instance IDs in the environment; check this terminal')
    instance = args.instance or (detected[0] if detected else input('Vast INSTANCE ID (not machine ID): ').strip())
    if detected and instance != detected[0]:
        raise RuntimeError('Requested instance ID differs from this GPU terminal; STOP was not armed')
    if not re.fullmatch(r'[0-9]+', instance):
        raise ValueError('Invalid instance ID')
    directory = Path.home()/'.config/dsc-vast-stop'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    import fcntl
    # Hold one watcher slot through the child's lifetime. Re-arming cannot create
    # two active watchers for the same instance and run.
    with (directory/f'{instance}-{args.run}-passage.lock').open('a') as slot:
        try:
            fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            settings = watcher_settings(directory, root, args.run, instance)
            if settings is None:
                raise RuntimeError('Watcher lock belongs to an unverified process/workspace; no duplicate watcher or job started.') from None
            stage = root/'outputs/passage-order'/args.run/'.stage'
            exited = stage.is_file() and stage.read_text().strip().startswith('exited status=')
            ignores_old_exit = exited and settings.get('after_ns', 0) > stage.stat().st_mtime_ns
            if not args.after_ns or not exited or ignores_old_exit:
                print(f'AUTO-STOP ALREADY ARMED for instance {instance}, run {args.run}.')
                return
            # A previous failed attempt may still have a watcher asleep for 30s.
            # Let it observe that failed exit and release its slot, then re-arm.
            print('Waiting for the previous attempt watcher to exit before re-arming.', flush=True)
            for _ in range(40):
                time.sleep(1)
                try:
                    fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    continue
            else:
                raise RuntimeError('Previous watcher is still active; inspect its log before restarting.') from None
        key = os.environ.get('CONTAINER_API_KEY', '') or getpass.getpass(
            'Vast API key (hidden; from cloud.vast.ai/manage-keys/): ').strip()
        if not key:
            raise ValueError('Vast API key is required; never paste it into chat')
        api(instance, key)  # Read-only authentication/instance check.
        config = directory/f'{instance}-{args.run}-{time.time_ns()}.json'
        descriptor = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump({'root': str(root), 'run': args.run, 'instance': instance, 'key': key,
                       'after_ns': args.after_ns}, stream)
        log = config.with_suffix('.log')
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--watch', str(config)], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True, pass_fds=(slot.fileno(),))
        time.sleep(1)
        status = process.poll()
        if status == 0:
            print(f'Watcher already completed; inspect {log} and check the Vast console.')
            return
        if status is not None:
            raise RuntimeError(f'Watcher exited before arming completed; inspect {log}')
        print(f'AUTO-STOP ARMED for instance {instance}, run {args.run}. Watcher PID: {process.pid}')
        print(f'Watcher log: {log}')
        print('You can close this terminal. Scoring continues unchanged.')
        print('STOP requires successful exit, a complete 2080-question submission and verified cloud artifacts.')
        print('On failure the instance stays running. STOP preserves disk; storage charges continue.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'AUTO-STOP ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
