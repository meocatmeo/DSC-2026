"""Private-test inference from frozen V9 and duo checkpoints; never trains or fits."""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from . import duo, h100
from .models import Local, audit
from .predict import load_questions, pack
from .question_memory import QuestionMemory
from .store import Store, file_hash, read_json, write_json

SCHEMA = 'private-v9-duo-v1'
SOURCE_MODULES = ('cache.py', 'compare.py', 'dense.py', 'fuse.py', 'h100.py',
                  'lexical.py', 'models.py', 'pipeline.py', 'predict.py',
                  'rerank.py', 'stack.py', 'store.py', 'structure.py', 'v11_preflight.py')


def question_records(path):
    records = load_questions(path)
    for qid, record in records.items():
        if not record['question'].strip() or record.get('answer') not in (None, []):
            raise ValueError(f'Expected an unlabelled private question: {qid}')
    # Scoring helpers expect an iterable. Never modify the official null-answer file.
    return {q: dict(r, answer=[]) for q, r in records.items()}


def verify_assets(args, signature):
    """Use source hashes for inference assets, not historical question files."""
    source = signature['source_inputs']
    directory = Path(__file__).parent
    for name in SOURCE_MODULES:
        if file_hash(directory / name) != source['files']['legalir2/' + name]:
            raise ValueError(f'Frozen V9 inference code changed: {name}')
    for name, expected in signature['code'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Frozen duo inference code changed: {name}')
    checked = {}
    for name, expected in source['files'].items():
        path = None
        if name.startswith('models2/'):
            # A restored provenance wrapper may differ. Actual weights/config/tokenizer
            # bytes must match the trained run, and are checked individually below.
            if not name.endswith('/provenance.json'):
                path = args.models / name.removeprefix('models2/')
        elif name.startswith('artifacts-ranking-h100/'):
            path = args.index / name.removeprefix('artifacts-ranking-h100/')
        elif name in ('analysis/ranking-h100/baseline/v9-ensemble.run.json',
                      'analysis/ranking-h100/baseline/v9-stacker-ens.json'):
            path = args.baseline / Path(name).name
        if path is not None:
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f'Frozen inference asset missing/changed: {path}; restore the exact source asset')
            checked[name] = expected
    required = ['artifacts-ranking-h100/store.sqlite',
                'artifacts-ranking-h100/lexical/unigram.npz',
                'artifacts-ranking-h100/lexical/bigram.npz',
                'artifacts-ranking-h100/dense/vectors.npy',
                'artifacts-ranking-h100/dense/chunk_ids.npy',
                'artifacts-ranking-h100/dense/manifest.json']
    required += ['models2/' + name + '/config.json' for name in h100.MODELS]
    if any(name not in checked for name in required):
        raise ValueError('Source manifest does not describe every required inference asset')
    return checked


def preflight(args):
    import torch
    records = question_records(args.questions)
    signature = read_json(args.duo_run / 'inputs.json')
    if signature['schema'] != duo.SCHEMA:
        raise ValueError('Unsupported duo checkpoint schema')
    choice = read_json(args.duo_run / 'calibration.json')
    duo.check_choice(SimpleNamespace(out=args.duo_run), choice)
    strength = choice['chosen']['strength']
    if strength not in duo.STRENGTHS or not math.isfinite(strength) or strength <= 0:
        raise ValueError('Expected the saved positive development-selected duo strength')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Private inference requires a CUDA GPU with BF16 support')
    packages = dict(signature['source_inputs']['packages'], **signature['packages'])
    for name, expected in packages.items():
        if expected is not None and importlib.metadata.version(name) != expected:
            raise ValueError(f'Frozen package mismatch: {name}; expected {expected}')
    print('Checking saved model/index/code hashes before private inference.', flush=True)
    assets = verify_assets(args, signature)
    policy = getattr(args, 'duo_policy', 'original')
    evidence_marker = {'policy': policy}
    if policy == 'fulltext-v2':
        from . import duo_evidence
        metadata = duo_evidence.index_metadata(args.evidence_index)
        if metadata['store_sha256'] != assets['artifacts-ranking-h100/store.sqlite']:
            raise ValueError('Full-text evidence index belongs to another frozen store')
        evidence_marker.update(index=file_hash(args.evidence_index), metadata=metadata,
                               code=file_hash(duo_evidence.__file__))
    checkpoint = duo.verify_model(args.duo_run / 'model', signature)
    baseline = read_json(args.baseline / 'v9-ensemble.run.json')
    weights = read_json(args.baseline / 'v9-stacker-ens.json')
    if weights.get('version', 1) != 1 or weights['weights'] != signature['weights']:
        raise ValueError('V9 and duo must use the original version-1 stacker weights')
    components = dict(baseline['audit']['components'], comparator=checkpoint['parameters'])
    parameter_audit = audit(components)
    store = Store(args.index / 'store.sqlite')
    try:
        if set(store.doc_ids) != set(signature['known_docs']):
            raise ValueError('Comparator corpus document IDs differ')
    finally:
        store.close()
    marker = {'schema': SCHEMA, 'questions_sha256': file_hash(args.questions),
              'questions': h100.digest(records), 'question_ids': list(records),
              'duo_inputs': h100.digest(signature), 'duo_model': file_hash(args.duo_run / 'model/duo.json'),
              'calibration': file_hash(args.duo_run / 'calibration.json'), 'strength': strength,
              'runtime': signature['runtime'], 'comparator_attention': signature['attention'],
              'infer_batch': args.infer_batch, 'shard_size': args.shard_size,
              'assets': assets, 'packages': packages, 'code': file_hash(__file__),
              'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
              'parameter_audit': parameter_audit}
    marker['evidence'] = evidence_marker
    # A different question file/model/backend cannot adopt any existing private shards.
    h100.guard(args.out / 'inputs.json', marker)
    write_json(args.out / 'audit.json', parameter_audit)
    print(f'Private input: {len(records)} questions; duo strength={strength}; '
          f'V9 attention={signature["runtime"]["attention"]}; duo attention={signature["attention"]}', flush=True)
    return records, signature, marker


def shard_items(records, size):
    items = list(records.items())
    for start in range(0, len(items), size):
        yield start, items[start:start+size]


def validate_v9(rows, items, known):
    h100.validate_rows(rows, items, 'retrieval')
    if any(len(r['docs']) < 5 or not set(r['docs']).issubset(known) for r in rows):
        raise ValueError('Invalid V9 candidate document IDs')


def validate_duo(rows, items, known, base, weights):
    duo.validate_scores(rows, items)
    for row, old in zip(rows, base):
        validate_comparison_fit(row)
        if not set(row['docs']).issubset(known) or row['baseline'] != h100.order(old, weights, 1):
            raise ValueError('Duo shard differs from its V9 private baseline')


def validate_comparison_fit(row):
    """Reject finite but stale scores that do not fit the saved comparisons."""
    prior, ability = np.asarray(row['prior']), np.asarray(row['ability'])
    probabilities = np.asarray(row['probabilities'])
    fitted = 1 / (1 + np.exp(-np.clip(ability[:, None] - ability[None, :], -30, 30)))
    # The saved tournament uses precision=1. Its strictly convex objective has
    # one optimum; check that optimum directly without normalizing prior again.
    gradient = (fitted - probabilities).sum(axis=1) + ability - prior
    if not np.isfinite(gradient).all() or np.max(np.abs(gradient)) > 1e-5:
        raise ValueError('Cached duo abilities do not match the saved pairwise comparisons')


def cache_row(path, items, validate, compute):
    if path.exists():
        rows = read_json(path)
        validate(rows, items)  # Fail closed on corrupt/wrong-question shards.
        return rows, False
    rows = compute(items)
    validate(rows, items)
    write_json(path, rows)
    return rows, True


def progress(label, committed, total, computed, started):
    rate = (time.time() - started) / max(1, computed)
    print(f'private/{label}: committed {committed}/{total}; {rate:.2f}s/question; '
          f'eta={(total-committed)*rate/60:.1f} min', flush=True)


def collect_v9(args, records, signature, marker):
    directory = args.out / 'cache/v9'
    h100.guard(directory / 'resume.json', {'inputs': h100.digest(marker), 'stage': 'v9'})
    source = getattr(args, 'v9_cache_from', None)
    if source is not None:
        import_v9(args, records, signature, marker, source)
    model, result, computed, started = None, [], 0, time.time()
    known = set(signature['known_docs'])
    try:
        def compute(items):
            nonlocal model
            if model is None:
                model = h100.pipeline(args, signature['runtime'], 'retrieval')
            return [h100.score_row(model, q, r, 'retrieval') for q, r in items]
        for start, items in shard_items(records, args.shard_size):
            rows, fresh = cache_row(directory / f'part-{start//args.shard_size:05d}.json', items,
                                    lambda r, i: validate_v9(r, i, known), compute)
            result.extend(rows)
            if fresh:
                computed += len(items)
                progress('v9', len(result), len(records), computed, started)
    finally:
        if model is not None:
            model.close()
    return result


def import_v9(args, records, signature, marker, source):
    """Reuse compatible frozen V9 results; never import old duo comparisons."""
    source = Path(source)
    if source.resolve() == args.out.resolve():
        raise ValueError('V9 cache source must be a different run')
    previous = read_json(source / 'inputs.json')
    for key in ('schema', 'questions_sha256', 'questions', 'question_ids', 'duo_inputs',
                'runtime', 'assets', 'packages'):
        if previous.get(key) is None or previous[key] != marker[key]:
            raise ValueError(f'Incompatible V9 cache source: {key}')
    imported = read_v9(SimpleNamespace(out=source, shard_size=previous['shard_size']),
                       records, signature, previous)
    for start, items in shard_items(records, args.shard_size):
        rows = imported[start:start + len(items)]
        target = args.out / 'cache/v9' / f'part-{start//args.shard_size:05d}.json'
        if target.exists():
            if read_json(target) != rows:
                raise ValueError('Existing V9 shard differs from imported baseline')
        else:
            write_json(target, rows)
    write_json(args.out / 'v9-import.json', {'source': str(source),
               'inputs_sha256': file_hash(source / 'inputs.json'), 'questions': len(imported)})


def read_v9(args, records, signature, marker):
    directory = args.out / 'cache/v9'
    if read_json(directory / 'resume.json') != {'inputs': h100.digest(marker), 'stage': 'v9'}:
        raise ValueError('V9 private cache identity mismatch')
    rows = []
    for start, items in shard_items(records, args.shard_size):
        part = read_json(directory / f'part-{start//args.shard_size:05d}.json')
        validate_v9(part, items, set(signature['known_docs']))
        rows.extend(part)
    return rows


def collect_duo(args, records, signature, marker):
    old = {r['id']: r for r in read_v9(args, records, signature, marker)}
    directory = args.out / 'cache/duo'
    base_hashes = {p.name: file_hash(p) for p in sorted((args.out / 'cache/v9').glob('part-*.json'))}
    h100.guard(directory / 'resume.json', {'inputs': h100.digest(marker), 'stage': 'duo', 'v9': base_hashes})
    local, evidence, codec, memory = None, None, None, None
    weights, known = np.asarray(signature['weights']), set(signature['known_docs'])
    result, computed, started = [], 0, time.time()
    try:
        def compute(items):
            nonlocal local, evidence, codec, memory
            if local is None:
                os.environ['LEGALIR_CAUSAL_ATTN'] = signature['attention']
                local = Local(args.duo_run / 'model', kind='causal', device='cuda')
                codec = duo.PairTokenizer(local.tokenizer, signature['max_length'])
                if getattr(args, 'duo_policy', 'original') == 'fulltext-v2':
                    from .duo_evidence import Evidence
                    evidence = Evidence(args.index, codec, args.evidence_index)
                else:
                    evidence = duo.Evidence(args.index, codec)
                memory = QuestionMemory(signature['memory'])
            rows = []
            for qid, record in items:
                docs, base = duo.pool(old[qid], weights, memory, known)
                if getattr(args, 'duo_policy', 'original') == 'fulltext-v2':
                    docs, base = evidence.expand(record['question'], docs, base, known)
                probabilities = duo.compare_documents(local, codec, record['question'],
                    evidence.encode(record['question'], docs), docs, args.infer_batch)
                prior, ability = duo.tournament(base, probabilities)
                rows.append({'id': qid, 'question': record['question'], 'gold': [], 'docs': docs,
                             'prior': prior.tolist(), 'ability': ability.tolist(),
                             'probabilities': probabilities.tolist(), 'baseline': h100.order(old[qid], weights, 1)})
                if getattr(args, 'duo_policy', 'original') == 'fulltext-v2':
                    rows[-1]['evidence'] = evidence.trace
            return rows
        for start, items in shard_items(records, args.shard_size):
            rows, fresh = cache_row(directory / f'part-{start//args.shard_size:05d}.json', items,
                lambda r, i: validate_duo(r, i, known, [old[q] for q, _ in i], weights), compute)
            result.extend(rows)
            if fresh:
                computed += len(items)
                progress('duo', len(result), len(records), computed, started)
    finally:
        if evidence is not None:
            evidence.close()
        if local is not None:
            local.close()
    return result


def write_submission(args, predictions, label, known, marker):
    records = question_records(args.questions)
    if set(predictions) != set(records):
        raise ValueError('Incomplete private question coverage')
    for qid, row in predictions.items():
        answer = row['answer']
        if len(answer) != 5 or len(set(answer)) != 5 or not set(answer).issubset(known):
            raise ValueError(f'Expected five distinct corpus document IDs for {qid}')
    path = args.out / f'submission-private-{label}.json'
    write_json(path, predictions)
    target = path.with_suffix('.zip')
    temporary = target.with_suffix('.zip.tmp')
    pack(path, SimpleNamespace(doc_ids=known), args.questions, temporary)
    with zipfile.ZipFile(temporary) as archive:
        import json
        if archive.namelist() != ['submission.json'] or json.loads(archive.read('submission.json')) != predictions:
            raise ValueError('Packed private answers differ from predictions')
    temporary.replace(target)
    write_json(path.with_suffix('.run.json'), {'schema': SCHEMA, 'method': label,
               'inputs': h100.digest(marker), 'questions': len(predictions),
               'questions_sha256': file_hash(args.questions), 'zip_sha256': file_hash(target),
               'duo_strength': marker['strength'] if label == 'duo' else None,
               'evidence': marker.get('evidence', {'policy': 'original'})})
    print(f'PRIVATE {label.upper()} ZIP READY: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'v9', 'duo', 'submit', 'all'])
    parser.add_argument('--questions', type=Path, default=Path('private-official.json'))
    parser.add_argument('--duo-run', type=Path, default=Path('outputs/duo/duo-20260918T110105Z'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('analysis/private/baseline'))
    parser.add_argument('--index', type=Path, default=Path('artifacts-ranking-h100'))
    parser.add_argument('--models', type=Path, default=Path('models2'))
    parser.add_argument('--infer-batch', type=int, default=8)
    parser.add_argument('--shard-size', type=int, default=10)
    parser.add_argument('--duo-policy', choices=['original', 'fulltext-v2'], default='original')
    parser.add_argument('--evidence-index', type=Path)
    parser.add_argument('--v9-cache-from', type=Path)
    args = parser.parse_args()
    if min(args.infer_batch, args.shard_size) < 1:
        parser.error('Batches and shards must be positive')
    if args.duo_policy == 'fulltext-v2' and args.evidence_index is None:
        parser.error('--evidence-index is required for fulltext-v2')
    if args.out.resolve() == args.duo_run.resolve():
        parser.error('Private outputs must not overwrite the trained duo run')
    records, signature, marker = preflight(args)
    known = set(signature['known_docs'])
    if args.stage == 'preflight':
        return
    if args.stage in ('all', 'v9'):
        rows = collect_v9(args, records, signature, marker)
        write_submission(args, {r['id']: {'answer': h100.order(r, signature['weights'], 1)} for r in rows}, 'v9', known, marker)
    if args.stage in ('all', 'duo'):
        rows = collect_duo(args, records, signature, marker)
        result = duo.diagnostics(rows, marker['strength'])
        write_submission(args, {q: {'answer': r['predicted']} for q, r in result.items()}, 'duo', known, marker)
    if args.stage == 'submit':
        # Read/validate all shards, but never invoke models in the packaging stage.
        v9 = read_v9(args, records, signature, marker)
        directory = args.out / 'cache/duo'
        base_hashes = {p.name: file_hash(p) for p in sorted((args.out / 'cache/v9').glob('part-*.json'))}
        expected = {'inputs': h100.digest(marker), 'stage': 'duo', 'v9': base_hashes}
        if read_json(directory / 'resume.json') != expected:
            raise ValueError('Duo private cache identity mismatch')
        rows = []
        for start, items in shard_items(records, args.shard_size):
            part = read_json(directory / f'part-{start//args.shard_size:05d}.json')
            validate_duo(part, items, known, v9[start:start+len(items)], signature['weights'])
            rows.extend(part)
        result = duo.diagnostics(rows, marker['strength'])
        write_submission(args, {r['id']: {'answer': h100.order(r, signature['weights'], 1)} for r in v9}, 'v9', known, marker)
        write_submission(args, {q: {'answer': r['predicted']} for q, r in result.items()}, 'duo', known, marker)


if __name__ == '__main__':
    main()
