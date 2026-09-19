"""Restore the public Qwen partner only when its bytes match the saved run."""
import argparse
from pathlib import Path

from .download import fetch
from .store import file_hash, read_json, write_json

REPO = 'Qwen/Qwen3-Reranker-0.6B'
COMMIT = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'
PREFIX = 'models2/partner_qwen/'


def restore(signature, directory, check=False):
    expected = {name.removeprefix(PREFIX): digest
                for name, digest in signature['source_inputs']['files'].items()
                if name.startswith(PREFIX) and not name.endswith('/provenance.json')}
    if not {'config.json', 'model.safetensors', 'tokenizer.json'}.issubset(expected):
        raise ValueError('Source signature is missing required Qwen model files')
    if any(Path(name).name != name or name in ('.', '..') for name in expected):
        raise ValueError('Unexpected Qwen model filename')
    records = {}
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or file_hash(path) != digest:
            if check:
                raise ValueError(f'Qwen file missing/changed: {path}')
            # Download to a staging path; never replace a model with unchecked bytes.
            pending = directory / '.restore' / name
            fetched, _ = fetch(REPO, COMMIT, name, pending)
            if fetched != digest or file_hash(pending) != digest:
                raise ValueError(f'Official Qwen file differs from saved run: {name}')
            pending.replace(path)
        records[name] = {'sha256': digest, 'bytes': path.stat().st_size}
        print(f'Qwen verified: {name}', flush=True)
    if not check:
        write_json(directory / 'provenance.json', {
            'repository': REPO, 'revision': COMMIT, 'commit': COMMIT, 'files': records,
            'restored_against': 'saved duo source-input SHA256 hashes'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signature', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    restore(read_json(args.signature), args.directory, args.check)


if __name__ == '__main__':
    main()
