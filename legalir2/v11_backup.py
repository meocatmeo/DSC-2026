"""Freeze complete index lines before uploading active V11 checkpoints.

No model imports and no changes to scoring or the resume signature.
"""
import os
from pathlib import Path
import sys

SPLITS = ('train', 'validation', 'public', 'benchmark/train')


def snapshot_indexes(output, destination):
    output, destination = Path(output), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        if (output / (split + '.jsonl.gz')).is_file() and (output / (split + '.manifest.json')).is_file():
            continue
        try:
            with (output / (split + '.rows/index.jsonl')).open('rb') as stream:
                # Freeze the byte boundary before reading. Never follow appends or
                # decode a potentially half-written UTF-8 character at that boundary.
                data = stream.read(os.fstat(stream.fileno()).st_size)
        except FileNotFoundError:
            continue
        end = data.rfind(b'\n') + 1
        (destination / (split.replace('/', '-') + '.jsonl')).write_bytes(data[:end])


if __name__ == '__main__':
    snapshot_indexes(*sys.argv[1:])
