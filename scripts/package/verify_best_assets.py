"""Verify preserved checkpoint and retrieval bytes against the scored run."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from legalir2.store import read_json,file_hash,write_json
from legalir2.h100 import digest


def main():
    directory=ROOT/'releases/handoff/20260922-best-09523-v1'
    baseline=read_json(directory/'provenance/baseline/inputs.json')
    source=read_json(directory/'provenance/duo-inputs.json')
    checked={};missing=[]
    for role,expected_inputs in (('original-duo',digest(source)),
            ('retrained-control',digest(read_json(directory/'provenance/retraining/arms/control/inputs.json')))):
        folder=directory/'model-vault'/role
        marker=read_json(folder/'duo.json')
        if marker['inputs']!=expected_inputs:
            raise ValueError('Model trained-run identity differs: '+role)
        if role=='retrained-control' and digest(marker)!=read_json(directory/'provenance/candidate/request.json')['model']:
            raise ValueError('Retrained checkpoint marker differs')
        for name,expected in marker['files'].items():
            target=folder/name
            if not target.is_file():
                missing.append(str(target));continue
            if file_hash(target)!=expected:
                raise ValueError('Checkpoint differs: '+str(target))
            checked[target.relative_to(directory).as_posix()]=expected
    for name,expected in source['source_inputs']['files'].items():
        if name.startswith('models2/') and not name.endswith('/provenance.json'):
            target=directory/'model-vault'/name.removeprefix('models2/')
        elif name.startswith('artifacts-ranking-h100/'):
            target=directory/'index-vault'/name.removeprefix('artifacts-ranking-h100/')
        else:
            continue
        if not target.is_file():
            missing.append(str(target));continue
        if file_hash(target)!=expected:
            raise ValueError('Frozen retrieval asset differs: '+str(target))
        checked[target.relative_to(directory).as_posix()]=expected
    report={'complete':not missing,'verified_files':checked,'missing':missing,
            'fresh_gpu_inference_tested':False,'reported_best_score':.9523,
            'parameter_components':baseline['parameter_audit']['components'],
            'additional_retrained_comparator_parameters':595776512}
    write_json(directory/'verification/model-assets.json',report)
    print(json.dumps({'complete':not missing,'verified_files':len(checked),'missing':missing},indent=2),flush=True)
    if missing:
        raise SystemExit(1)


if __name__=='__main__':
    main()
