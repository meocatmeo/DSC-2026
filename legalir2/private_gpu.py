"""Check the rental GPU before installing or running private inference."""
import argparse
import json


def check(torch, profile):
    if profile not in ('h100', '4090'):
        raise ValueError('Unknown GPU profile')
    if torch.__version__ != '2.7.1+cu128':
        raise ValueError('Use the saved torch==2.7.1+cu128 environment')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA/BF16 unavailable; check the host driver')
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 2**30
    if profile == 'h100' and (capability != (9, 0) or 'H100' not in name.upper()):
        raise ValueError('The h100 profile requires an H100 GPU')
    if profile == '4090' and (capability != (8, 9) or '4090' not in name):
        raise ValueError('The 4090 profile requires an Ada RTX 4090 GPU')
    if memory < 22:
        raise ValueError('Need a GPU with at least 22 GiB visible VRAM')
    return {'profile': profile, 'gpu': name, 'capability': capability,
            'vram_gib': round(memory, 2), 'torch': torch.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('h100', '4090'), required=True)
    args = parser.parse_args()
    import torch
    print(json.dumps(check(torch, args.profile)), flush=True)


if __name__ == '__main__':
    main()
