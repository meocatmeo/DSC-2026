"""Matched evidence-training ablation, with fixed inference and a pooled release gate."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import zipfile

import numpy as np

from . import duo, fulltext_prior as prior, private_infer as private
from .cache import normalise_question
from .h100 import digest, guard
from .labeled_gate import evaluate, summarize
from .models import Local, audit
from .store import file_hash, read_json, write_json

SCHEMA = 'duo-fulltext-retrain-v1'
ARMS = ('old-evidence-control', 'control')
DATA_ID = 'c1d9515e8d27dc3b8ccefabd1dba2ab9d5065088089d23817d35386f2e4170a5'


def reference(args):
    return SimpleNamespace(**dict(vars(args), out=args.out/'reference'))


def code_identity():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('fulltext_retrain.py', 'labeled_gate.py', 'fulltext_training.py', 'duo.py')}


def check_groups(old, control, extra, heldout, roles):
    """The evidence contrast must retain identical ordered pairs and split membership."""
    blocked = {normalise_question(r['question']) for r in heldout.values()}
    ids = [g['id'] for g in old]
    if len(set(ids)) != len(ids) or set(ids) != set(roles):
        raise ValueError('Training group membership differs')
    seen = set()
    for variant in (control, extra):
        if [g['id'] for g in variant] != ids:
            raise ValueError('Training arm membership/order differs')
    for a, b, c in zip(old, control, extra):
        qid = a['id']
        text = normalise_question(a['question'])
        if qid in heldout or text in blocked or text in seen:
            raise ValueError('Held-out or duplicate training question')
        seen.add(text)
        for group in (a, b, c):
            if any(group[k] != a[k] for k in ('id', 'question', 'gold', 'role', 'development')):
                raise ValueError('Training labels or split changed between arms')
            if group['role'] != roles[qid] or group['development'] != (roles[qid] == 'monitor'):
                raise ValueError('Monitor assignment changed')
        if a['pairs'] != b['pairs'] or c['pairs'][:len(a['pairs'])] != a['pairs']:
            raise ValueError('Evidence arm altered original pairs')
        for group in (a, b, c):
            pairs = group['pairs']
            if (not pairs or len({tuple(p) for p in pairs}) != len(pairs)
                    or any(p not in group['gold'] or n in group['gold'] for p, n in pairs)
                    or set(group['evidence']) != {d for pair in pairs for d in pair}):
                raise ValueError('Invalid training pair/evidence labels')


def verify_data(args):
    plan = read_json(args.training/'inputs.json')
    verified = read_json(args.training/'verified.json')
    if digest(plan) != DATA_ID or verified['inputs'] != DATA_ID:
        raise ValueError('Expected the verified v3 training rebuild')
    hashes = {k: v['sha256'] for k, v in verified['archives'].items()}
    hashes['old-evidence-control'] = verified['aligned_old_control']['sha256']
    for name, expected in hashes.items():
        if file_hash(args.training/(name+'.json.gz')) != expected:
            raise ValueError('Training archive changed: '+name)
    heldout = {}
    for split, size in (('development', 429), ('validation', 1050)):
        path = args.dataset/(split+'.json')
        records = read_json(path)
        if digest(records) != plan['heldout'][split]:
            raise ValueError('Training exclusion set differs from evaluation')
        if len(records) != size:
            raise ValueError('Incomplete external labeled split')
        heldout.update(records)
    check_groups(*(duo.read_compressed(args.training/(name+'.json.gz')) for name in
                   (*ARMS, 'hard-negatives')), heldout, plan['roles'])
    print('Training archives, matched pairs and external split exclusions verified.', flush=True)
    return plan, hashes


def prepare(args):
    plan, hashes = verify_data(args)
    for name, expected in plan['tokenizer'].items():
        if file_hash(args.models/'partner_qwen'/name) != expected:
            raise ValueError('Training tokenizer changed: '+name)
    # This verifies model/package/index provenance and reproduces the scored source ZIP.
    prior.prepare(reference(args))
    seed = read_json(args.training/'development-cache/seed.json')
    current = prior.manifest(reference(args))
    for key in ('duo_inputs', 'duo_model', 'calibration', 'assets', 'packages', 'comparator_attention', 'strength'):
        if seed['assets'][key] != current['assets'][key]:
            raise ValueError('Cached development baseline is incompatible: '+key)
    for key in ('policy', 'metadata', 'code'):
        if seed['assets']['evidence'][key] != current['assets']['evidence'][key]:
            raise ValueError('Cached development evidence differs: '+key)
    if seed['datasets'] != current['datasets']:
        raise ValueError('Cached development uses different labeled data')
    signature = read_json(args.duo_run/'inputs.json')
    if digest(signature) != plan['signature']:
        raise ValueError('Training source comparator identity differs')
    from .duo_evidence import index_metadata
    if index_metadata(args.evidence_index) != plan['index']:
        raise ValueError('Inference and rebuilt training use different evidence indexes')
    marker = {'schema': SCHEMA, 'code': code_identity(), 'training': hashes, 'rebuild': DATA_ID,
        'reference': digest(prior.manifest(reference(args))), 'arms': ARMS,
        'epochs': 2, 'batch': 2, 'accumulate': 8, 'learning_rate': 2e-5,
        'max_length': 2048, 'attention': 'flash_attention_2', 'seed': 2026,
        'infer_batch': args.batch, 'primary': 'control', 'strength': .25, 'development_cache': digest(seed),
        'policy': 'Matched old/new evidence; added negatives exploratory only; fixed inference; pooled gate.'}
    # JSON normalizes tuples to lists; guards and subsequent checks use that representation.
    marker['arms'] = list(ARMS)
    guard(args.out/'inputs.json', marker)
    target = args.out/'reference/cache/base/development'
    guard(target/'resume.json', {'inputs': digest(current), 'split': 'development'})
    for name, expected in seed['files'].items():
        source = args.training/'development-cache'/name
        if file_hash(source) != expected:
            raise ValueError('Packaged baseline cache changed: '+name)
        if (target/name).exists() and file_hash(target/name) != expected:
            raise ValueError('Existing baseline cache differs: '+name)
        if not (target/name).exists():
            temporary = (target/name).with_suffix('.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(target/name)
    base = [r for _, rows in prior.bases(reference(args), 'development') for r in rows]
    if digest(base) != seed['rows']:
        raise ValueError('Packaged baseline membership/order changed')
    print('Prepared matched arms: 1427 fit / 131 checkpoint-monitor; external 429+1050 excluded. Reused 429 verified baseline comparisons.', flush=True)


def manifest(args):
    marker = read_json(args.out/'inputs.json')
    if (marker['schema'] != SCHEMA or marker['code'] != code_identity()
            or marker['infer_batch'] != args.batch or marker['rebuild'] != DATA_ID
            or marker['reference'] != digest(prior.manifest(reference(args)))):
        raise ValueError('Experiment identity changed; use a new run')
    return marker


def arm_inputs(args, arm):
    marker = manifest(args)
    if arm not in (*ARMS, 'hard-negatives'):
        raise ValueError('Unknown training arm')
    source = args.training/(arm+'.json.gz')
    if file_hash(source) != marker['training'][arm]:
        raise ValueError('Training archive changed')
    value = {k: marker[k] for k in ('epochs', 'batch', 'accumulate', 'learning_rate', 'max_length', 'attention')}
    value.update(experiment=digest(marker), arm=arm, training=marker['training'][arm])
    directory = args.out/'arms'/arm
    guard(directory/'inputs.json', value)
    return directory, value


def train(args):
    directory, signature = arm_inputs(args, args.arm)
    for name, expected in prior.manifest(reference(args))['assets']['assets'].items():
        if name.startswith('models2/partner_qwen/') and file_hash(args.models/Path(name).relative_to('models2')) != expected:
            raise ValueError('Starting model changed: '+name)
    target = directory/'training.json.gz'
    if not target.exists():
        temporary = target.with_suffix('.tmp')
        shutil.copyfile(args.training/(args.arm+'.json.gz'), temporary)
        temporary.replace(target)
    if file_hash(target) != signature['training']:
        raise ValueError('Run-local training archive changed')
    os.environ['LEGALIR_CAUSAL_ATTN'] = signature['attention']
    options = SimpleNamespace(**signature, out=directory, partner=args.models/'partner_qwen')
    duo.train(options)
    checkpoint = duo.verify_model(directory/'model', signature)
    components = dict(prior.manifest(reference(args))['assets']['parameter_audit']['components'])
    components['comparator'] = checkpoint['parameters']
    write_json(directory/'audit.json', audit(components))


def validate_rows(rows, base):
    duo.validate_scores(rows, [(r['id'], {'question': r['question'], 'answer': r['gold']}) for r in base])
    if len(rows) != len(base):
        raise ValueError('Incomplete retrained comparator cache')
    for row, original in zip(rows, base):
        if row.get('base') != digest(original) or any(row[k] != original[k] for k in
                                                   ('id', 'question', 'gold', 'docs', 'prior')):
            raise ValueError('Inference pool, evidence or prior changed')
        private.validate_comparison_fit(row)


def collect(args, split, arm, compute=True):
    marker = manifest(args)
    directory, signature = arm_inputs(args, arm)
    model_info = duo.verify_model(directory/'model', signature)
    cache = args.out/'cache'/arm/split
    identity = {'inputs': digest(marker), 'model': digest(model_info), 'split': split}
    if compute:
        guard(cache/'resume.json', identity)
    elif read_json(cache/'resume.json') != identity:
        raise ValueError('Evaluation cache identity changed')
    local = None
    result, count, computed, started = [], 0, 0, time.time()
    try:
        for path, base in prior.bases(reference(args), split):
            target = cache/path.name
            if target.exists():
                rows = read_json(target)
            else:
                if not compute:
                    raise ValueError('Incomplete evaluation; missing '+str(target))
                if local is None:
                    os.environ['LEGALIR_CAUSAL_ATTN'] = signature['attention']
                    local = Local(directory/'model', kind='causal', device='cuda')
                    codec = duo.PairTokenizer(local.tokenizer, 2048)
                rows = []
                for original in base:
                    evidence = {d: [original['evidence']['documents'][d]['input_ids']] for d in original['docs']}
                    probability = duo.compare_documents(local, codec, original['question'], evidence, original['docs'], args.batch)
                    row = {k: original[k] for k in ('id', 'question', 'gold', 'docs', 'prior')}
                    row.update(base=digest(original), probabilities=probability.tolist(),
                               ability=prior.fit(original['prior'], probability).tolist())
                    rows.append(row)
                validate_rows(rows, base)
                write_json(target, rows)
                computed += len(rows)
            validate_rows(rows, base)
            result.extend(rows)
            count += len(rows)
            if compute:
                private.progress('evidence-retrain/'+arm+'/'+split, count,
                                 len(prior.records(reference(args), split)), computed, started)
    finally:
        if local is not None:
            local.close()
    return result


def paired(before, after):
    if [r['id'] for r in before] != [r['id'] for r in after]:
        raise ValueError('Paired evaluation membership/order changed')
    return {a['id']: {'baseline': prior.predict(a, prior.BASELINE), 'proposed': prior.predict(b, prior.BASELINE)}
            for a, b in zip(before, after)}


def describe(records, predictions):
    return summarize([(Fraction(len(set(row['baseline']) & set(records[q]['answer'])), len(records[q]['answer'])),
                       Fraction(len(set(row['proposed']) & set(records[q]['answer'])), len(records[q]['answer'])),
                       set(row['baseline']) != set(row['proposed'])) for q, row in sorted(predictions.items())])


def screen(args):
    base = [r for _, rows in prior.bases(reference(args), 'development') for r in rows]
    old, new = (collect(args, 'development', arm, compute=False) for arm in ARMS)
    records = prior.records(reference(args), 'development')
    versus_base = describe(records, paired(base, new))
    versus_old = describe(records, paired(old, new))
    return {'inputs': digest(manifest(args)), 'scores': digest([base, old, new]),
        'baseline_comparison': versus_base, 'evidence_comparison': versus_old,
        'continue_validation': versus_base['delta'] > 0 and versus_old['delta'] > 0,
        'meaning': 'Cost-saving screen only. No promotion decision and no policy/weight selection.'}


def gate(args):
    screened = screen(args)
    if not screened['continue_validation']:
        return {'inputs': digest(manifest(args)), 'screen': screened, 'deploy': False,
                'reason': 'Development screen failed; preserve the scored baseline.'}
    datasets, baseline_pairs, evidence_pairs = {}, {}, {}
    for split in ('development', 'validation'):
        datasets[split] = prior.records(reference(args), split)
        base = [r for _, rows in prior.bases(reference(args), split) for r in rows]
        old, new = (collect(args, split, arm, compute=False) for arm in ARMS)
        baseline_pairs[split], evidence_pairs[split] = paired(base, new), paired(old, new)
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    baseline_report = evaluate(datasets, baseline_pairs, known)
    evidence_report = evaluate(datasets, evidence_pairs, known)
    # A baseline win alone cannot be attributed to rebuilt evidence.
    deploy = (baseline_report['pooled']['decision'] == 'promote'
              and evidence_report['pooled']['paired_bootstrap_ci95'][0] > 0)
    return {'inputs': digest(manifest(args)), 'screen': screened, 'deploy': deploy,
        'baseline_comparison': baseline_report, 'evidence_comparison': evidence_report,
        'prediction_hashes': digest([baseline_pairs, evidence_pairs]),
        'reason': 'Pooled baseline deadband + interval, and positive paired evidence interval required.'}


def submit(args):
    report = gate(args)
    write_json(args.out/'gate.json', report)
    if report['deploy']:
        rows = collect(args, 'private', 'control', compute=False)
        predictions = {r['id']: {'answer': prior.predict(r, prior.BASELINE)} for r in rows}
    else:
        path = args.source/'submission-private-duo.zip'
        if file_hash(path) != prior.SCORED_ZIP:
            raise ValueError('Scored baseline fallback changed')
        with zipfile.ZipFile(path) as archive:
            predictions = json.loads(archive.read('submission.json'))
    known = set(read_json(args.duo_run/'inputs.json')['known_docs'])
    private.write_submission(args, predictions, 'duo', known,
        {'strength': .25, 'evidence': {'policy': SCHEMA, 'deployed': report['deploy']}, 'gate': report})
    write_json(args.out/'submission-choice.json', {'candidate_deployed': report['deploy'],
        'baseline_private_score_user_reported': .9510, 'candidate_private_score': None,
        'zip_sha256': file_hash(args.out/'submission-private-duo.zip'), 'gate': report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify-data', 'prepare', 'train', 'development', 'screen', 'validation', 'gate', 'private', 'submit'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=(*ARMS, 'hard-negatives'), default='control')
    for name, default in {'training': 'analysis/fulltext-training', 'dataset': 'analysis/fulltext-prior',
            'source': 'outputs/private/private-duo-fulltext-v2-4090-fast1',
            'duo-run': 'outputs/duo/duo-20260918T110105Z', 'models': 'models2',
            'index': 'artifacts-ranking-h100', 'evidence-index': 'artifacts-ranking-h100/duo-fulltext-v2.sqlite',
            'baseline': 'analysis/private/baseline', 'questions': 'data/private-official.json'}.items():
        parser.add_argument('--'+name, type=Path, default=Path(default))
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.batch < 1 or args.out.resolve() in (args.source.resolve(), args.duo_run.resolve()):
        parser.error('Use a positive inference batch and a separate experiment output')
    if args.stage == 'verify-data':
        verify_data(args)
    elif args.stage in ('prepare', 'train', 'submit'):
        globals()[args.stage](args)
    elif args.stage in ('screen', 'gate'):
        report = globals()[args.stage](args)
        write_json(args.out/(args.stage+'.json'), report)
        print(args.stage+':', report, flush=True)
    elif args.stage in ('development', 'validation'):
        if args.stage == 'validation' and not screen(args)['continue_validation']:
            print('Development screen failed; skipping validation.', flush=True)
            return
        prior.collect_base(reference(args), args.stage)
        for arm in ARMS:
            collect(args, args.stage, arm)
    elif args.stage == 'private':
        report = gate(args)
        write_json(args.out/'gate.json', report)
        if report['deploy']:
            collect(args, 'private', 'control')
        else:
            print('Gate retained baseline; skipping private GPU scoring.', flush=True)


if __name__ == '__main__':
    main()
