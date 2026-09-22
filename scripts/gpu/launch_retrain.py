"""Visible, resumable 4090 launcher; auto-stop is armed before a new training job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/DSC-2026-fulltext-retrain-4090-v1')
RUN = 'duo-fulltext-retrain-4090-v1'
REMOTE = 'gdrive:DSC-2026-results/releases/retrain/20260921-fulltext-retrain-4090-v1'
ZIP_SHA = 'b0dbcf9ff954133d55aa554c53f8117a8b74ad72bb5f0814a7742611356d9d0d'
SESSION = 'dsc-retrain-v2'
JOB_SCRIPTS = {'gpu-fulltext-retrain-4090.sh', 'gpu-private-setup.sh', 'gpu-private-h100.sh', 'gpu-retrain-private-submit.sh', 'gpu-retrain-private-fresh.sh'}


def say(text):
    print(text, flush=True)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def is_job(argv):
    if any(Path(a).name in JOB_SCRIPTS for a in argv[1:]):
        return True
    if '-m' in argv and 'legalir2.fulltext_retrain' in argv:
        return True
    return any(Path(a).name == 'launch_retrain.py' for a in argv[1:]) and '--job' in argv


def running(root):
    result = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cwd = (proc/'cwd').resolve(strict=True)
            if cwd != root and root not in cwd.parents:
                continue
            argv = [a.decode(errors='replace') for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            if is_job(argv):
                result.append(int(proc.name))
        except (OSError, RuntimeError):
            continue
    return result


def restore(root):
    """Fill missing release files; never overwrite modified code or any checkpoint."""
    archive = root/'fulltext-retrain.zip'
    if not archive.exists() or sha(archive) != ZIP_SHA:
        say('Downloading the frozen training ZIP (154 MiB).')
        temporary = archive.with_suffix('.zip.download')
        subprocess.run(['rclone', 'copyto', REMOTE+'/fulltext-retrain.zip', str(temporary),
                        '--progress', '--contimeout', '20s', '--timeout', '90s', '--retries', '2'], check=True)
        if sha(temporary) != ZIP_SHA:
            raise RuntimeError('Downloaded ZIP failed SHA256 verification')
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        files = [line.split('  ', 1) for line in bundle.read('private-files.sha256').decode().splitlines()]
        if {name for _, name in files} != set(bundle.namelist())-{'private-files.sha256'}:
            raise RuntimeError('Release manifest membership differs')
        files.append((hashlib.sha256(bundle.read('private-files.sha256')).hexdigest(), 'private-files.sha256'))
        for expected, name in files:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise RuntimeError('Unsafe archive entry')
            target = root/relative
            if target.is_symlink() or root not in target.resolve().parents:
                raise RuntimeError('Release path escapes the workspace')
            data = bundle.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError('Release file checksum differs: '+name)
            if target.exists():
                if sha(target) != expected:
                    raise RuntimeError('Existing file was modified; refusing to overwrite: '+str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.download')
            temporary.write_bytes(data)
            temporary.replace(target)
    say('All frozen release files verified. Existing models, caches and checkpoints retained.')


def mode(root):
    python = root/'.venv-h100/bin/python'
    if not python.is_file():
        return 'all'
    code = '''import importlib.metadata as m
from pathlib import Path
for line in Path('config/requirements-private-h100.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        name, version = line.split('==')
        assert m.version(name) == version, name
import torch
from flash_attn import flash_attn_func
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
'''
    checked = subprocess.run([str(python), '-c', code], cwd=root, capture_output=True, text=True, timeout=120)
    if checked.returncode:
        say('Pinned environment needs setup; the installer will check and repair packages.')
        return 'all'
    return 'run'


def preflight(root, selected):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('Use Linux x86_64 with Python 3.10–3.12 (Python 3.12 recommended).')
    tools = ['rclone', 'tmux', 'flock', 'timeout', 'setsid', 'sha256sum', 'nvidia-smi']
    if selected == 'all':
        tools += ['nvcc', 'g++']
    missing = [name for name in tools if not shutil.which(name)]
    if missing:
        raise RuntimeError('Missing tools: '+', '.join(missing)+'; use a CUDA 12.8 devel image.')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    say('GPU: '+gpu)
    if 'RTX 4090' not in gpu:
        raise RuntimeError('This release expects an RTX 4090.')
    if selected == 'all' and 'release 12.8,' not in subprocess.check_output(['nvcc', '--version'], text=True):
        raise RuntimeError('CUDA toolkit mismatch: the pinned installer requires CUDA 12.8 devel.')
    free = shutil.disk_usage(root).free/2**30
    needed = 80 if selected == 'all' else 30
    say(f'Free disk: {free:.1f} GiB; required for {selected}: {needed} GiB.')
    if free < needed:
        raise RuntimeError(f'Insufficient disk: need {needed} GiB free before starting {selected}.')


def arm(root, run, after_ns=0):
    say('Arming automatic STOP. Enter the Vast API key only at the hidden terminal prompt.')
    subprocess.run([sys.executable, str(Path(__file__).with_name('vast_stop_retrain.py')),
                    '--root', str(root), '--run', run, '--after-ns', str(after_ns)], check=True)


def retire_training_watcher(root):
    """Under the workspace lock, retire only this completed source run's watcher."""
    directory = (Path.home()/'.config/dsc-vast-stop').resolve()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = [a.decode() for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            if '--watch' not in argv or not any(Path(a).name == 'vast_stop_retrain.py' for a in argv):
                continue
            config = Path(argv[argv.index('--watch')+1]).resolve()
            if config.parent != directory:
                continue
            settings = read(config)
            if Path(settings['root']).resolve() != root or settings['run'] != RUN:
                continue
        except (OSError, ValueError, KeyError, IndexError):
            continue
        say('Retiring the completed training run watcher before arming private inference.')
        try:
            os.kill(int(proc.name), signal.SIGTERM)
        except ProcessLookupError:
            continue
        for _ in range(30):
            if not proc.exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Previous training watcher has not exited; no new job started.')


def latest_log(root, run):
    paths = list((root/'outputs/launcher'/run).glob('*/launcher.log'))
    paths += list((root/'outputs/private-setup').glob('*/run.log'))
    paths += [root/'outputs/retrain'/run/'run.log', root/'outputs/private'/('assets-'+run)/'run.log']
    return max((p for p in paths if p.is_file()), key=lambda p: p.stat().st_mtime_ns, default=None)


def follow(log, status=None, root=None, run=None):
    say('Live log: '+str(log))
    say('Ctrl+C ends this log view only. The tmux job and auto-stop watcher continue.')
    stream = log.open()
    try:
        while True:
            chunk = stream.read()
            if chunk:
                print(chunk, end='', flush=True)
            if status is not None and status.exists():
                state = read(status)
                if state.get('state') == 'exited':
                    print(stream.read(), end='', flush=True)
                    say('JOB EXIT STATUS: '+str(state['exit_code']))
                    return state['exit_code']
            if root is not None:
                next_log = latest_log(root, run)
                if next_log is not None and next_log != log:
                    stream.close()
                    log, stream = next_log, next_log.open()
                    say('\nFollowing next stage log: '+str(log))
                stage = root/'outputs/retrain'/run/'.stage'
                if stage.exists() and stage.read_text().strip().startswith('exited status=') and not running(root):
                    value = stage.read_text().strip()
                    say(value)
                    return 0 if value == 'exited status=0; final backup' else 1
            time.sleep(1)
    except KeyboardInterrupt:
        say('\nLog view closed. The GPU job has not been interrupted.')
        return 0
    finally:
        stream.close()


def job(args):
    root, attempt = args.root.resolve(), args.attempt.resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    status = attempt/'status.json'
    env = dict(os.environ, ROOT=str(root), RETRAIN_RUN_ID=args.run, BASE_RESULTS='gdrive:DSC-2026-results')
    write(status, {'state': 'running', 'pid': os.getpid(), 'mode': args.mode})
    code = 1
    with (attempt/'launcher.log').open('a', buffering=1) as log:
        print(f'Launching {args.mode}; root={root}; run={args.run}', file=log, flush=True)
        try:
            private_scripts = {'private-submit': 'gpu-retrain-private-submit.sh', 'fresh-private': 'gpu-retrain-private-fresh.sh'}
            command = (['bash', str(Path(__file__).with_name(private_scripts[args.mode]))]
                       if args.mode in private_scripts else ['bash', 'scripts/gpu/gpu-fulltext-retrain-4090.sh', args.mode])
            code = subprocess.run(command,
                                  cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        except Exception as error:
            print('LAUNCH ERROR: '+str(error), file=log, flush=True)
        finally:
            print(f'JOB EXIT STATUS: {code}. Failure leaves the instance running; inspect this log.', file=log, flush=True)
            write(status, {'state': 'exited', 'exit_code': code, 'mode': args.mode})
            try:
                subprocess.run(['rclone', 'copyto', str(attempt/'launcher.log'),
                    f'gdrive:DSC-2026-results/runs/retrain/{args.run}/logs/launcher/{attempt.name}.log',
                    '--contimeout', '20s', '--timeout', '60s', '--retries', '1'],
                    stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
            except Exception as error:
                print('Launcher log backup failed: '+str(error), file=log, flush=True)
    return code


def launch(args):
    import fcntl
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    say(f'Workspace: {root}\nRun: {args.run}')
    # Unlike tmux session names, live runner processes distinguish work from an idle shell.
    with (root/'.dsc-bootstrap.lock').open('a') as bootstrap:
        try:
            fcntl.flock(bootstrap, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another launcher is preparing this workspace. Use its terminal.') from None
        active = running(root)
        if active:
            say('Existing job is running (PIDs '+', '.join(map(str, active))+'). No duplicate job will start.')
            stage = root/'outputs/retrain'/args.run/'.stage'
            stale_exit = stage.is_file() and stage.read_text().strip().startswith('exited status=')
            arm(root, args.run, time.time_ns() if stale_exit else 0)
            log = latest_log(root, args.run)
            if log is None:
                say('No log created yet. Inspect the existing tmux pane; rerun this launcher to check again.')
                return 0
            status = log.with_name('status.json') if log.name == 'launcher.log' else None
        else:
            stage = root/'outputs/retrain'/args.run/'.stage'
            if stage.is_file() and stage.read_text().strip() == 'exited status=0; final backup':
                say('This run has already completed. Verifying backup and arming STOP instead of retraining.')
                arm(root, args.run)
                return 0
            with (root/'.ranking-h100.lock').open('a') as workspace:
                try:
                    fcntl.flock(workspace, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('Workspace is busy. No files changed and no second job started.') from None
                restore(root)
                selected = ('fresh-private' if getattr(args, 'fresh_private', False) else
                            'private-submit' if getattr(args, 'private_submit', False) else mode(root))
                preflight(root, 'all' if selected == 'fresh-private' else selected)
                if selected == 'private-submit':
                    source = root/'outputs/retrain/duo-fulltext-retrain-4090-v1'
                    if not (source/'arms/control/model/duo.json').is_file():
                        raise RuntimeError('The trained control checkpoint is missing. Resume the original GPU instance.')
                    if mode(root) != 'run':
                        raise RuntimeError('The saved inference environment is incomplete; this submission command will not retrain or reinstall it.')
                    retire_training_watcher(root)
                elif selected == 'fresh-private':
                    retire_training_watcher(root)
                after_ns = time.time_ns()
            arm(root, args.run, after_ns)
            attempt = root/'outputs/launcher'/args.run/str(after_ns)
            attempt.mkdir(parents=True)
            log, status = attempt/'launcher.log', attempt/'status.json'
            log.touch()
            command = shlex.join([sys.executable, str(Path(__file__).resolve()), '--job',
                '--root', str(root), '--run', args.run, '--mode', selected, '--attempt', str(attempt)])
            env_args = ['-e', 'RCLONE_CONFIG='+os.environ.get('RCLONE_CONFIG', str(Path.home()/'.config/rclone/rclone.conf'))]
            existing = subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True).returncode == 0
            if existing:
                invocation = ['tmux', 'new-window', '-d', '-P', '-F', '#{pane_id}', '-t', SESSION,
                              '-n', 'run-'+str(after_ns), '-c', str(root)]+env_args+[command]
            else:
                invocation = ['tmux', 'new-session', '-d', '-P', '-F', '#{pane_id}', '-s', SESSION,
                              '-c', str(root)]+env_args+[command]
            pane = subprocess.check_output(invocation, text=True).strip()
            say(f'Job started in tmux {SESSION}, pane {pane}. Following its log here; no client switch is needed.')
    return follow(log, status, root if active and status is None else None, args.run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--job', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=('all', 'run', 'private-submit', 'fresh-private'), default='all', help=argparse.SUPPRESS)
    parser.add_argument('--private-submit', action='store_true', help='Use saved control checkpoint for private submission without validation')
    parser.add_argument('--fresh-private', action='store_true', help='Install environment and restore cloud checkpoint on a new GPU, then submit without validation')
    parser.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.private_submit or args.fresh_private:
        args.run = 'duo-fulltext-retrain-private-control-4090-v1'
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run):
        parser.error('Invalid run ID')
    if args.job:
        if args.attempt is None or args.root.resolve() not in args.attempt.resolve().parents:
            parser.error('Job attempt must be inside its workspace')
        return job(args)
    return launch(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nLauncher interrupted. Check the printed job/watcher status before leaving the instance running.', flush=True)
        sys.exit(130)
    except Exception as error:
        print(f'LAUNCHER ERROR: {error}\nNo successful GPU start is being claimed. Fix the error and rerun this launcher.', file=sys.stderr, flush=True)
        sys.exit(1)
