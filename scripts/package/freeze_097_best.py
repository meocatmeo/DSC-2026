"""Preserve the reported 0.9533 ZIP and independently replay its strict policy."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.store import read_json, file_hash, write_json
from legalir2 import agreement_097
from freeze_best import put, encoded

EXPECTED = '34167af0e86aa5478992320319f319c45130d3ea894fb5b161deae661a4b4fc6'


def main():
    previous = ROOT/'releases/handoff/20260922-best-09523-v1'
    out = ROOT/'releases/handoff/20260923-best-09533-v2'
    candidate = ROOT/'submissions/private/20260923-fifth-097-v1'
    source = candidate/'submission-private-fifth-097.zip'
    if file_hash(source) != EXPECTED:
        raise ValueError('Scored ZIP changed')
    record = {'score': .9533, 'reported_by': 'user', 'date': '2026-09-23',
        'policy': 'fifth-097', 'zip_sha256': EXPECTED, 'previous_best': .9525,
        'delta': .0008, 'source_zip': str(source.relative_to(ROOT))}
    put(candidate/'leaderboard-result.json', encoded(record))
    put(ROOT/'submissions/ready/submission-best-0.9533.zip', source.read_bytes())
    put(out/'reference/submission-best-0.9533.zip', source.read_bytes())
    put(out/'reference/leaderboard-result.json', encoded(record))
    for name in ('review.json', 'changes.json', 'submission-private-fifth-097.run.json'):
        put(out/'reference'/name, (candidate/name).read_bytes())
    manifest = read_json(previous/'recovery/manifest.json')
    for name, expected in manifest['files'].items():
        path = previous/'recovery'/name
        if file_hash(path) != expected:
            raise ValueError('Previous frozen recovery changed: '+name)
        if name != 'recover.py':
            put(out/'recovery'/name, path.read_bytes())
    policy = Path(agreement_097.__file__).read_bytes()
    put(out/'recovery/legalir2/agreement_097.py', policy)
    original_runtime = (previous/'recovery/recover.py').read_text(encoding='utf8')
    old_import = 'from legalir2.duo_agreement import predict'
    new_import = "from legalir2.agreement_097 import predict as strict_predict\n\ndef predict(base, candidate):\n    answer, trace = strict_predict(base, candidate)\n    return answer, dict(trace, replaced=trace['replaced'] and not trace.get('reverted', False))"
    if original_runtime.count(old_import) != 1 or original_runtime.count('changes!=306') != 1:
        raise ValueError('Unexpected original recovery implementation')
    # Only the policy, expected count, and output name differ from the tested
    # previous recovery. The saved model scores remain identical.
    replayed = {}; replacements = 0
    for name in manifest['score_shards']:
        for pair in read_json(out/'recovery'/name):
            answer, trace = agreement_097.predict(pair['baseline'], pair['candidate'])
            qid = pair['baseline']['id']
            if qid in replayed:
                raise ValueError('Duplicate recovery question')
            replayed[qid] = {'answer': answer}
            replacements += trace['replaced'] and not trace.get('reverted', False)
    with zipfile.ZipFile(source) as archive:
        payload = archive.read('submission.json')
        if json.loads(payload) != replayed or len(replayed) != 2080:
            raise ValueError('New best does not replay from frozen model scores')
        entry = archive.getinfo('submission.json')
        metadata = {k: getattr(entry, k) for k in manifest['zip_metadata']}
    runtime = original_runtime.replace(old_import, new_import).replace('changes!=306', f'changes!={replacements}')
    runtime = runtime.replace('submission-private-agreement.zip', 'submission-private-fifth-097.zip')
    put(out/'recovery/recover.py', runtime.encode())
    manifest.update(schema='best-private-09533-recovery-v1', reported_private_score=.9533,
        best_zip_sha256=EXPECTED, policy=agreement_097.POLICY,
        submission_json_sha256=hashlib.sha256(payload).hexdigest(), zip_metadata=metadata)
    manifest['files'] = {p.relative_to(out/'recovery').as_posix(): file_hash(p)
        for p in sorted((out/'recovery').rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    put(out/'recovery/manifest.json', encoded(manifest))
    put(out/'source/freeze_097_best.py', Path(__file__).read_bytes())
    for name in ('scripts/analysis/submit_agreement_097.py', 'tests2/test_agreement_097.py'):
        put(out/'source'/name, (ROOT/name).read_bytes())
    put(out/'README.md', ('# Best private result: 0.9533\n\n'
        'User-reported 23 September 2026, strict fifth-slot agreement.\n'
        'This snapshot replays 2,080 answers from frozen comparator scores. It is not fresh GPU inference.\n'
        'Run `python recovery/recover.py --out /path/to/output` with NumPy installed.\n'
        'The unchanged model/index assets and Docker image remain in the separately preserved '
        '`20260922-best-09523-v1` release. That Docker image still reproduces 0.9523; '
        'a 0.9533 image has not been built/tested in this snapshot.\n').encode())
    report_dir = ROOT/'reports/best-09533-recovery'
    subprocess.run([sys.executable, '-B', str(out/'recovery/recover.py'), '--out', str(report_dir)], check=True)
    report = read_json(report_dir/'recovery-report.json')
    if report['zip_sha256'] != EXPECTED or not report['exact_zip_match']:
        raise ValueError('Standalone recovery failed')
    put(out/'verification/recovery-report.json', encoded(report))
    files = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in ('RELEASE.sha256', 'source-code.zip') and '__pycache__' not in p.parts}
    archive_path = out/'source-code.zip'
    temporary = out/'source-code.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            info = zipfile.ZipInfo(name, (2026, 9, 23, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (out/name).read_bytes())
    if archive_path.exists() and file_hash(archive_path) != file_hash(temporary):
        raise ValueError('Existing archive differs')
    temporary.replace(archive_path)
    files['source-code.zip'] = file_hash(archive_path)
    put(out/'RELEASE.sha256', ''.join(f'{sha}  {name}\n' for name, sha in sorted(files.items())).encode())
    write_json(ROOT/'submissions/ready/current-best.json', {**record,
        'ready_zip': 'submissions/ready/submission-best-0.9533.zip',
        'recovery_release': str(out.relative_to(ROOT)), 'exact_recovery_verified': True,
        'new_docker_image_verified': False})
    print(json.dumps({'release': str(out), 'score': .9533, 'replacements': replacements,
                      'exact_zip_recovery': True, 'sha256': EXPECTED}))


if __name__ == '__main__':
    main()
