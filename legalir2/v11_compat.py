"""Explicitly audited cache compatibility for the September 17 I/O repair.

Only listed code hashes are equivalent. Unknown scoring changes still reject
resume. The evidence repair only defines title-only scoring where the old code
raised, so it changes no previously committed row.
"""
import copy

EQUIVALENT = {
    "v11.py": [
        "13e140c58d73396355a60d3ea9e1c971fbe9479b072ed1a06d42e018f64432f1",
        "14c056ef57aa09b2640bd65e42bdb9b831fb6b6e2cbfa8b7a6d9f260423cc2e0",
        "2ccdd019e4b2bd2d951ee4e367f834d1534c543b382c934cad9b33402a747675",
        "4b7ab77366867655c8dcaed4224bb7612b13a3a7ec3dea84987591bc108959d9",
        "c8d6b26f1edb771853920cd62188383ea106e8b0cbf66fce5a83efeda88bbddf"
    ],
    "evidence.py": [
        "1158d4f6e74d418672135c6578be38b47b70dbde60b36a27375d55a6f657f8f3",
        "4c4e4daf2b2f5bb179d740f1483a2cdd0e8dd79d7c2ba24c650a5fd18b70bf2a",
        "777be005148359919878f122ad801468eeb5a12e3e4fac3de39a86579ddd4233"
    ]
}


def compatible_signature(signature):
    result = copy.deepcopy(signature)
    for name, hashes in EQUIVALENT.items():
        if result.get('code', {}).get(name) in hashes:
            result['code'][name] = 'v11-september17-compatible-' + name
    return result
