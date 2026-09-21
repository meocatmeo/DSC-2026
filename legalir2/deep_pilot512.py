"""Extend the fixed pilot to 512 training questions, reusing verified raw scores."""
import argparse
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from . import deep_pool as deep, deep_pilot as pilot
from .h100 import digest, guard
from .store import file_hash, read_json, write_json

PILOT_INPUTS = '15817574110d6a82026fbc0ec3d2cd501620ef0994c91b08f384b55a19263783'
PILOT_FUSION = '7a2cf9fba9b857e9ec2834ab80146ddc024530955bf19321ceaed510f046879c'


def original_args(args):
    return SimpleNamespace(**dict(vars(args), out=args.pilot_out, deep_assets=Path('analysis/deep-pool')))


def identity(args):
    old = original_args(args)
    source, target = deep.manifest(old), deep.manifest(args)
    if digest(source) != PILOT_INPUTS or digest(deep.fitted(old)) != PILOT_FUSION:
        raise ValueError('Expected the completed 256-question pilot')
    if {k: v for k, v in source.items() if k != 'assets'} != {k: v for k, v in target.items() if k != 'assets'}:
        raise ValueError('Raw-score inputs differ beyond the training asset extension')
    before, after = deep.datasets(old), deep.datasets(args)
    ids = list(before['records'])
    if len(ids) != 256 or len(after['records']) != 512 or list(after['records'])[:256] != ids:
        raise ValueError('Expected a nested fixed 256-to-512 training extension')
    if any(after['records'][q] != r for q, r in before['records'].items()) or after['v9'][:256] != before['v9']:
        raise ValueError('Pilot training examples changed')
    return {'source': digest(source), 'target': digest(target), 'source_fusion': PILOT_FUSION,
            'code': file_hash(Path(__file__)), 'validation_evaluated': False}


def prepared(args):
    if read_json(args.out/'extension.json') != identity(args):
        raise ValueError('512-question extension identity changed')


def import_raw(args, split):
    """Import full shards; return the one partial training shard's rows for reuse."""
    old = original_args(args)
    questions, source_questions = deep.records(args, split), deep.records(old, split)
    items = list(questions.items())
    if items[:len(source_questions)] != list(source_questions.items()):
        raise ValueError('Raw reuse requires an unchanged question prefix')
    directory = args.out/'cache/deep'/split
    guard(directory/'resume.json', {'inputs': digest(deep.manifest(args)), 'split': split})
    signature = read_json(args.duo_run/'inputs.json')
    v9 = {r['id']: r for r in deep.v9_rows(args, split)}
    preserved = {r['id']: r for r in deep.base_rows(args, split)} if split != 'train' else None
    count, tail = 0, {}
    for path, rows in deep.raw_parts(old, split):
        expected = items[count:count+5]
        if len(rows) == len(expected):
            deep.validate_raw(rows, expected, v9, set(signature['known_docs']), np.asarray(signature['weights']), preserved)
            target = directory/path.name
            if target.exists() and read_json(target) != rows:
                raise ValueError('Imported raw shard differs from completed pilot')
            if not target.exists():
                write_json(target, rows)
        else:
            if count+len(rows) != len(source_questions):
                raise ValueError('Partial pilot shard is not final')
            tail.update({r['id']: r for r in rows})
        count += len(rows)
    if count != len(source_questions):
        raise ValueError('Incomplete pilot raw cache')
    print(f'Reused {count} verified {split} raw scores; {len(tail)} retained for a mixed shard.', flush=True)
    return tail


def prepare(args):
    deep.prepare(args)
    old = original_args(args)
    deep.choice(old)
    if deep.fitted(old)['training'] != digest(deep.training_rows(old)):
        raise ValueError('Pilot training cache changed')
    guard(args.out/'extension.json', identity(args))
    import_raw(args, 'train')
    import_raw(args, 'development')


def training(args):
    tail = import_raw(args, 'train')
    questions = deep.records(args, 'train')
    signature = read_json(args.duo_run/'inputs.json')
    weights, known = np.asarray(signature['weights']), set(signature['known_docs'])
    v9 = {r['id']: r for r in deep.v9_rows(args, 'train')}
    directory = args.out/'cache/deep/train'
    model = evidence = memory = None
    started, computed = time.time(), 0
    try:
        for start, items in deep.private.shard_items(questions, 5):
            path = directory/f'part-{start//5:05d}.json'
            if path.exists():
                deep.validate_raw(read_json(path), items, v9, known, weights)
                continue
            rows = []
            for q, r in items:
                if q in tail:
                    rows.append(tail[q])
                    continue
                if model is None:
                    model = deep.pipeline(args, {'attention': 'flash_attention_2', 'batch': args.batch}, 'retrieval')
                    for local in [model.dense.embedder.local]+[r.local for r in model.rerankers]:
                        deep.require_bf16(local)
                    evidence = deep.Evidence(args.index, deep.duo.PairTokenizer(model.rerankers[1].local.tokenizer, 2048), args.evidence_index)
                    memory = deep.QuestionMemory(signature['memory'])
                rows.append(deep.score_raw(args, model, evidence, memory, q, r, v9[q], None, weights, known))
                computed += 1
            deep.validate_raw(rows, items, v9, known, weights)
            write_json(path, rows)
            deep.private.progress('deep-pool-512/new-training', start+len(items), len(questions), computed, started)
    finally:
        if evidence is not None:
            evidence.close()
        if model is not None:
            model.close()


def calibrate(args):
    deep.calibrate(args)
    old, new = deep.compared(original_args(args), 'development'), deep.compared(args, 'development')
    if [r['id'] for r in old] != [r['id'] for r in new]:
        raise ValueError('Development comparison membership changed')
    before, after = deep.recall(old, 'deep'), deep.recall(new, 'deep')
    delta = after-before
    report = {'questions': len(new), 'training_256_recall': float(before.mean()),
              'training_512_recall': float(after.mean()), 'delta': float(delta.mean()),
              'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum()),
              'prefer_512': bool(delta.mean() > 1e-12), 'private_inference_performed': False,
              'validation_evaluated': False}
    write_json(args.out/'training-size-comparison.json', report)
    print('512 vs 256 development:', report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'training', 'fit', 'development', 'calibrate', 'submit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pilot-out', type=Path, default=Path('outputs/private/private-deep-pilot-4090-v1'))
    parser.add_argument('--duo-run', type=Path, default=Path('outputs/duo/duo-20260918T110105Z'))
    parser.add_argument('--index', type=Path, default=Path('artifacts-ranking-h100'))
    parser.add_argument('--evidence-index', type=Path, default=Path('artifacts-ranking-h100/duo-fulltext-v2.sqlite'))
    parser.add_argument('--baseline', type=Path, default=Path('analysis/private/baseline'))
    parser.add_argument('--models', type=Path, default=Path('models2'))
    parser.add_argument('--questions', type=Path, default=Path('data/private-official.json'))
    parser.add_argument('--dataset', type=Path, default=Path('analysis/fulltext-prior'))
    parser.add_argument('--deep-assets', type=Path, default=Path('analysis/deep-pool512'))
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch != 4 or args.out.resolve() in (args.source.resolve(), args.pilot_out.resolve(), args.duo_run.resolve(),
            Path('outputs/private/private-deep-pilot-submit-4090-v1').resolve()):
        parser.error('Keep batch 4 and preserve all existing runs')
    if args.stage != 'prepare':
        prepared(args)
    {'prepare': prepare, 'training': training, 'fit': deep.fit,
     'development': lambda a: deep.compare(a, 'development'),
     'calibrate': calibrate, 'submit': pilot.submit}[args.stage](args)


if __name__ == '__main__':
    main()
