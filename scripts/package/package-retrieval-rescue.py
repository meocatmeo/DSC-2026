"""Build the isolated BF16 rescue release from the verified published package."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/analysis'))
from audit_prior_development import load
from legalir2 import fulltext_prior as prior
from legalir2.h100 import digest
from legalir2.store import file_hash, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace-unpublished', action='store_true')
    args = parser.parse_args()
    original = ROOT/'releases/private/20260920-fulltext-prior-4090-v1/private-inference-overlay.zip'
    if file_hash(original) != '8326ceb744e0fe885a80c55502d7b748615046ef64fc00a3efd4d4497858d4cb':
        raise ValueError('Published source release changed')
    with zipfile.ZipFile(original) as archive:
        payload = {name: archive.read(name) for name in archive.namelist() if name != 'private-files.sha256'}
    signature = read_json(ROOT/'analysis/impact/duo-20260918T110105Z/inputs.json')
    corpus = ROOT/'analysis/v11/baseline/corpus.jsonl.gz'
    saved = read_json(ROOT/'reports/private/20260920-prior-development/related-v1/graph.json')
    if (saved['inputs']['code'] != file_hash(ROOT/'legalir2/related_documents.py')
            or saved['inputs']['corpus'] != file_hash(corpus)
            or saved['inputs']['store'] != signature['source_inputs']['files']['artifacts-ranking-h100/store.sqlite']):
        raise ValueError('Citation graph provenance differs')
    rows = [{k: v for k, v in row.items() if k != 'relevance'} for row in load()]
    asset_dir = 'analysis/retrieval-rescue/'
    payload[asset_dir+'development-baseline.json.gz'] = gzip.compress(json.dumps(rows, ensure_ascii=False).encode(), mtime=0)
    payload[asset_dir+'citation-graph.json'] = json.dumps(saved['graph'], ensure_ascii=False).encode()
    baseline_marker = read_json(ROOT/'reports/private/20260920-prior-development/remote/inputs.json')
    asset = {'schema': 'rescue-assets-v1', 'files': {name: hashlib.sha256(payload[asset_dir+name]).hexdigest()
        for name in ('development-baseline.json.gz', 'citation-graph.json')},
        'duo_inputs': digest(signature), 'store_sha256': saved['inputs']['store'],
        'corpus_sha256': baseline_marker['assets']['evidence']['metadata']['corpus_sha256'],
        'development_recall': float(prior.recall_vector(rows, prior.BASELINE).mean()),
        'source_run': 'private-fulltext-prior-4090-v1',
        'source_calibration_sha256': file_hash(ROOT/'reports/private/20260920-prior-development/remote/calibration.json')}
    payload[asset_dir+'manifest.json'] = json.dumps(asset, ensure_ascii=False).encode()
    for name in ('legalir2/retrieval_rescue.py', 'legalir2/focused_retrieval.py', 'legalir2/related_documents.py',
                 'scripts/gpu/gpu-retrieval-rescue-4090.sh', 'tests2/test_retrieval_rescue.py',
                 'tests2/test_rescue_shell.py', 'tests2/test_focused_retrieval.py', 'tests2/test_related_documents.py',
                 'docs/runbooks/RUNBOOK-retrieval-rescue-4090.md'):
        payload[name] = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
    for name, expected in signature['code'].items():
        if hashlib.sha256(payload['legalir2/'+name]).hexdigest() != expected:
            raise ValueError('Frozen comparator code changed')
    payload['private-files.sha256'] = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
        for name, data in sorted(payload.items())).encode()
    output = ROOT/'releases/private/20260920-retrieval-rescue-4090-v1/private-inference-overlay.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and (not args.replace_unpublished or (output.parent/'published.json').exists()):
        raise ValueError('Do not overwrite a release; use a new version')
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid archive')
        for name, data in payload.items():
            if archive.read(name) != data:
                raise ValueError('Archive bytes changed')
    temporary.replace(output)
    output.with_suffix('.zip.sha256').write_text(f'{file_hash(output)}  {output.name}\n', encoding='utf-8')
    print(f'Verified {len(payload)} files, {len(rows)} reusable development rows; BF16 only.')
    print(output, file_hash(output))


if __name__ == '__main__':
    main()
