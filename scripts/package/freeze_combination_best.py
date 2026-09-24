"""Freeze the reported 0.9538 result and verify standalone score-cache recovery."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legalir2.store import read_json, write_json, file_hash
from legalir2.h100 import digest
from legalir2 import passage_combination
from freeze_best import put, encoded

EXPECTED = 'c179e33fddeab08f0a8d9cd11d8657845f3d3cbd4415df9e3ba4cc15e9d0f4bb'


def main():
    candidate = ROOT/'submissions/private/passage-combination-v1'
    source = candidate/'submission-private-passage-combination.zip'
    previous = ROOT/'releases/handoff/20260923-best-09533-v2'
    out = ROOT/'releases/handoff/20260923-best-09538-v1'
    review, plan = read_json(candidate/'review.json'), read_json(candidate/'plan.json')
    if file_hash(source) != EXPECTED or review['plan'] != digest(plan):
        raise ValueError('Reported candidate identity changed')
    for name, expected in plan['code'].items():
        if file_hash(ROOT/'legalir2'/name) != expected:
            raise ValueError('Combination source changed: '+name)
    if file_hash(candidate/'recovery-scores.json') != review['recovery_scores_sha256']:
        raise ValueError('Recovery scores changed')
    record = {'score': .9538, 'reported_by': 'user', 'date': '2026-09-23',
        'policy': passage_combination.POLICY['name'], 'zip_sha256': EXPECTED,
        'previous_best': .9533, 'delta': .0005, 'changed_sets_vs_previous_best': 43,
        'source_zip': source.relative_to(ROOT).as_posix()}
    put(candidate/'leaderboard-result.json', encoded(record))
    put(ROOT/'submissions/ready/submission-best-0.9538.zip', source.read_bytes())
    put(out/'reference/submission-best-0.9538.zip', source.read_bytes())
    put(out/'reference/leaderboard-result.json', encoded(record))
    for name in ('review.json', 'plan.json', 'changes.json', 'submission-private-passage-combination.run.json'):
        put(out/'reference'/name, (candidate/name).read_bytes())
    for name in ('development.json', 'development-changes.json', 'development-change-audit.json'):
        put(out/'reference'/name, (ROOT/'reports/passage-combination-v1'/name).read_bytes())
    put(out/'reference/gpu-submission-choice.json',
        (ROOT/'reports/passage-private-submission-review/submission-choice.json').read_bytes())
    context = out/'recovery'
    put(context/'legalir2/__init__.py', b'')
    for name in ('duo_agreement.py', 'agreement_097.py', 'passage_combination.py'):
        put(context/'legalir2'/name, (ROOT/'legalir2'/name).read_bytes())
    for name in ('questions.json', 'known-docs.json'):
        src = ROOT/'reports/passage-private-v2/data'/name
        expected = read_json(ROOT/'reports/passage-private-v2/plan.json')['data_files']['data/'+name]
        if file_hash(src) != expected: raise ValueError('Frozen data changed: '+name)
        put(context/'data'/name, src.read_bytes())
    put(context/'data/scores.json', (candidate/'recovery-scores.json').read_bytes())
    runtime = (previous/'recovery/recover.py').read_text(encoding='utf8')
    start, end = runtime.index('from legalir2.agreement_097'), runtime.index('def sha(')
    runtime = runtime[:start]+'from legalir2.passage_combination import predict\n\n\n'+runtime[end:]
    replacements = {
        "base,candidate=pair['baseline'],pair['candidate']": "base,candidate=pair['baseline'],pair['control']",
        'answer,trace=predict(base,candidate)': "answer,trace=predict(base,candidate,pair['ordered_original'],pair['ordered_control'])",
        'changes!=138': 'changes!=43',
        'submission-private-fifth-097.zip': 'submission-private-passage-combination.zip'}
    for old, new in replacements.items():
        if old not in runtime: raise ValueError('Unexpected recovery template: '+old)
        runtime = runtime.replace(old, new)
    put(context/'recover.py', runtime.encode())
    old_manifest = read_json(previous/'recovery/manifest.json')
    with zipfile.ZipFile(source) as archive:
        payload = archive.read('submission.json'); entry = archive.getinfo('submission.json')
        metadata = {k: getattr(entry, k) for k in old_manifest['zip_metadata']}
    manifest = {'schema': 'best-private-09538-recovery-v1', 'reported_private_score': .9538,
        'mode': 'Saved-score recovery, not fresh model inference', 'best_zip_sha256': EXPECTED,
        'policy': passage_combination.POLICY, 'source_plan': digest(plan),
        'score_shards': ['data/scores.json'], 'questions_sha256': file_hash(context/'data/questions.json'),
        'submission_json_sha256': hashlib.sha256(payload).hexdigest(), 'zip_metadata': metadata,
        'files': {p.relative_to(context).as_posix(): file_hash(p) for p in sorted(context.rglob('*'))
                  if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}}
    put(context/'manifest.json', encoded(manifest))
    for name in ('scripts/package/freeze_combination_best.py', 'scripts/analysis/combine_passage_scores.py',
                 'tests2/test_passage_combination.py', 'tests2/test_agreement_097.py',
                 'tests2/test_duo_agreement.py', 'docs/development/PASSAGE-COMBINATION.md'):
        put(out/'source'/name, (ROOT/name).read_bytes())
    put(out/'README.md', ('# Best private result: 0.9538\n\n'
        'User-reported result for the fixed CPU passage combination. The original 0.9533 '
        'winner remains separate. This snapshot recomputes 2,080 predictions from four '
        'saved score rows per question (two models under two passage orders).\n\n'
        'Run `python recovery/recover.py --out /path/to/output` with NumPy installed. '
        'Only the frozen private questions are accepted. This is not fresh GPU inference.\n\n'
        'Model assets remain in `20260922-best-09523-v1/model-vault`. The passage-order '
        'GPU source release is `20260923-passage-private-4090-v2`. The old tested Docker '
        'image reproduces 0.9523; no 0.9538 Docker image is built or verified here.\n').encode())
    report_dir = ROOT/'reports/best-09538-recovery'
    subprocess.run([sys.executable, '-B', str(context/'recover.py'), '--out', str(report_dir)],
                   cwd=out, check=True)
    report = read_json(report_dir/'recovery-report.json')
    if report['zip_sha256'] != EXPECTED or report['exact_zip_match'] is not True:
        raise ValueError('Standalone recovery did not reproduce the scored ZIP')
    put(out/'verification/recovery-report.json', encoded(report))
    files = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.name not in ('RELEASE.sha256', 'source-code.zip', 'source-code.zip.tmp')
             and '__pycache__' not in p.parts}
    temporary = out/'source-code.zip.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            info = zipfile.ZipInfo(name, (2026, 9, 23, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (out/name).read_bytes())
    target = out/'source-code.zip'
    if target.exists() and file_hash(target) != file_hash(temporary): raise ValueError('Frozen archive differs')
    temporary.replace(target); files['source-code.zip'] = file_hash(target)
    put(out/'RELEASE.sha256', ''.join(f'{sha}  {name}\n' for name, sha in sorted(files.items())).encode())
    write_json(ROOT/'submissions/ready/current-best.json', {**record,
        'ready_zip': 'submissions/ready/submission-best-0.9538.zip',
        'recovery_release': out.relative_to(ROOT).as_posix(), 'exact_recovery_verified': True,
        'new_docker_image_verified': False})
    print({'release': str(out), 'score': .9538, 'exact_zip_recovery': True, 'sha256': EXPECTED})


if __name__ == '__main__':
    main()
