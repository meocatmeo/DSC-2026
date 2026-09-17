"""Small CPU checks for interrupted model/index downloads; no model is loaded."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from .models import checkpoint_identity
from .store import Store, file_hash, read_json


def model_ready(directory):
    directory = Path(directory)
    read_json(directory / 'config.json')
    index = directory / 'model.safetensors.index.json'
    names = set(read_json(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if not names:
        raise ValueError('Empty weight map')
    for name in names:
        path = directory / name
        with path.open('rb') as stream:
            size = path.stat().st_size
            header_size = struct.unpack('<Q', stream.read(8))[0]
            if header_size > min(size - 8, 100_000_000):
                raise ValueError(f'Incomplete safetensors header: {name}')
            header = json.loads(stream.read(header_size))
        offsets = [v['data_offsets'][1] for k, v in header.items() if k != '__metadata__']
        if not offsets or size != 8 + header_size + max(offsets):
            raise ValueError(f'Incomplete safetensors weights: {name}')


def dense_ready(directory, store_path, model_path):
    directory, model_path = Path(directory), Path(model_path)
    manifest = read_json(directory / 'manifest.json')
    if manifest['store_sha256'] != file_hash(store_path):
        raise ValueError('Dense/store hash mismatch')
    provenance = model_path / 'provenance.json'
    identity = checkpoint_identity(read_json(provenance) if provenance.exists() else {'path': str(model_path)})
    expected = manifest.get('model_identity') or checkpoint_identity(manifest.get('model'))
    if identity != expected:
        raise ValueError('Dense/embedding identity mismatch')
    vectors = np.load(directory / 'vectors.npy', mmap_mode='r')
    try:
        ids = np.load(directory / 'chunk_ids.npy', allow_pickle=False)
        if vectors.shape != (manifest['chunks'], manifest['dimension']) or ids.shape != (manifest['chunks'],):
            raise ValueError('Dense array shape mismatch')
        store = Store(store_path)
        try:
            actual = np.fromiter((r[0] for r in store.db.execute('SELECT id FROM chunks ORDER BY id')), dtype=np.int64)
            if not np.array_equal(ids, actual):
                raise ValueError('Dense chunk IDs differ from the store')
        finally:
            store.close()
    finally:
        vectors._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['model', 'dense'])
    parser.add_argument('path', nargs='?')
    args = parser.parse_args()
    try:
        if args.kind == 'model':
            model_ready(args.path)
        else:
            dense_ready('artifacts2/dense', 'artifacts2/store.sqlite', 'models2/embedding')
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f'{args.kind} needs restore/rebuild: {error}', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
