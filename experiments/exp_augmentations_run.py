#!/usr/bin/env python3
"""
EXPERIMENT: exp_augmentations
Image augmentations x binary splits x dataset.

'identity' x 'odd_even' is a double sanity check: must reproduce baseline.

Usage
-----
    python experiments/exp_augmentations_run.py
    python experiments/exp_augmentations_run.py --aug-filter identity,hflip,noise_010
    python experiments/exp_augmentations_run.py --split-filter odd_even,footwear_vs_rest
    python experiments/exp_augmentations_run.py --workers 4
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import BASE_ARGS, DEFAULT_DEVICE, run_items
from datasets import DATASETS, dataset_args
from splits import SPLITS_BY_DATASET, SANITY_CHECK_SPLIT
from augmentations import augmentations_for, get_augmentation_fn

H_VALUES     = [100, 300, 1000]
ALPHA_VALUES = [
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3,
    1.0, 3.0, 10, 30, 100, 300, 1000, 3000, 10000,
]
SEEDS = [0, 1, 2]


def run_one(item):
    # __init__.py in the project root is not importable by name.
    # Load it by file path and register under a stable name so workers can find it.
    import importlib.util as _ilu, sys as _sys
    _mod_name = '_project_dataset'
    if _mod_name not in _sys.modules:
        _spec = _ilu.spec_from_file_location(
            _mod_name, Path(__file__).parent.parent / '__init__.py')
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules[_mod_name] = _mod
        _spec.loader.exec_module(_mod)
    dataset_module = _sys.modules[_mod_name]
    import torchvision
    import torch
    import torch.nn.functional as F
    import functools
    from main import execute

    aug_name   = item['aug']
    split_name = item['split']
    split_map  = item['split_map']
    dataset    = item['dataset']
    h, alpha, seed = item['h'], item['alpha'], item['seed']

    print(f"    {dataset}/{aug_name}/{split_name} h={h} alpha={alpha:.2e} seed={seed}  [{item['device']}]")

    # ── Inline augmentation functions ────────────────────────────────────────
    def _identity(x): return x
    def _hflip(x): return x.flip(-1)
    def _vflip(x): return x.flip(-2)
    def _rot90_cw(x): return torch.rot90(x, k=3, dims=(-2, -1))
    def _rot90_ccw(x): return torch.rot90(x, k=1, dims=(-2, -1))
    def _rot180(x): return torch.rot90(x, k=2, dims=(-2, -1))
    def _transpose(x): return x.transpose(-2, -1)
    def _cc(x, sz):
        N, C, H, W = x.shape
        p = (H - sz) // 2
        cr = x[:, :, p:p+sz, p:p+sz]
        return F.interpolate(cr.float(), size=(H, W), mode='bilinear',
                             align_corners=False).double()
    def _center_crop_20(x): return _cc(x, 20)
    def _center_crop_24(x): return _cc(x, 24)
    def _noise(sigma): return lambda x: (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)
    def _invert(x): return 1.0 - x
    def _brightness_up(x): return (x + 0.15).clamp(0.0, 1.0)
    def _brightness_down(x): return (x - 0.15).clamp(0.0, 1.0)
    def _blur_3x3(x):
        N, C, H, W = x.shape
        k = torch.ones(1, 1, 3, 3, dtype=x.dtype, device=x.device) / 9.0
        return F.conv2d(x.reshape(N*C, 1, H, W), k, padding=1).reshape(N, C, H, W)

    AUG_FNS = {
        'identity': _identity, 'hflip': _hflip, 'vflip': _vflip,
        'rot90_cw': _rot90_cw, 'rot90_ccw': _rot90_ccw, 'rot180': _rot180,
        'transpose': _transpose, 'center_crop_20': _center_crop_20,
        'center_crop_24': _center_crop_24,
        'noise_005': _noise(0.05), 'noise_010': _noise(0.10), 'noise_020': _noise(0.20),
        'invert': _invert, 'brightness_up': _brightness_up,
        'brightness_down': _brightness_down, 'blur_3x3': _blur_3x3,
    }
    aug_fn = AUG_FNS[aug_name]

    # ── Patch normalized dataset with augmentation ────────────────────────────
    from __init__ import dataset_to_tensors, intertwine_labels, center_normalize, intertwine_split
    tr_map = torchvision.transforms.ToTensor()

    def _load(raw_items):
        x, y, idx = intertwine_labels(*dataset_to_tensors(raw_items))
        x = aug_fn(x)
        x = center_normalize(x)
        return x, y, idx

    def make_patched_norm(ds_name):
        def patched_norm(ds, ps, seeds, d=0, params=None):
            torch.manual_seed(seeds[0])
            if ds != ds_name:
                return dataset_module.get_normalized_dataset.__wrapped__(ds, ps, seeds, d, params)
            loaders = {
                'fashion': lambda: (
                    list(torchvision.datasets.FashionMNIST(
                        '~/.torchvision/datasets/FashionMNIST',
                        train=True, download=True, transform=tr_map)) +
                    list(torchvision.datasets.FashionMNIST(
                        '~/.torchvision/datasets/FashionMNIST',
                        train=False, transform=tr_map))),
                'mnist': lambda: (
                    list(torchvision.datasets.MNIST(
                        '~/.torchvision/datasets/MNIST',
                        train=True, download=True, transform=tr_map)) +
                    list(torchvision.datasets.MNIST(
                        '~/.torchvision/datasets/MNIST',
                        train=False, transform=tr_map))),
                'kmnist': lambda: (
                    list(torchvision.datasets.KMNIST(
                        '~/.torchvision/datasets/KMNIST',
                        train=True, download=True, transform=tr_map)) +
                    list(torchvision.datasets.KMNIST(
                        '~/.torchvision/datasets/KMNIST',
                        train=False, transform=tr_map))),
                'emnist-letters': lambda: (
                    list(torchvision.datasets.EMNIST(
                        '~/.torchvision/datasets/EMNIST',
                        train=True, download=True, transform=tr_map, split='letters')) +
                    list(torchvision.datasets.EMNIST(
                        '~/.torchvision/datasets/EMNIST',
                        train=False, transform=tr_map, split='letters'))),
                'cifar10': lambda: (
                    list(torchvision.datasets.CIFAR10(
                        '~/.torchvision/datasets/CIFAR10',
                        train=True, download=True, transform=tr_map)) +
                    list(torchvision.datasets.CIFAR10(
                        '~/.torchvision/datasets/CIFAR10',
                        train=False, transform=tr_map))),
            }
            if ds in loaders:
                x, y, idx = _load(loaders[ds]())
            elif ds == 'cifar_catdog':
                raw = [(xi, yi) for xi, yi in torchvision.datasets.CIFAR10(
                    '~/.torchvision/datasets/CIFAR10', train=True, download=True,
                    transform=tr_map) if yi in [3, 5]]
                raw += [(xi, yi) for xi, yi in torchvision.datasets.CIFAR10(
                    '~/.torchvision/datasets/CIFAR10', train=False, transform=tr_map)
                    if yi in [3, 5]]
                x, y, idx = _load(raw)
            elif ds == 'cifar_shipbird':
                raw = [(xi, yi) for xi, yi in torchvision.datasets.CIFAR10(
                    '~/.torchvision/datasets/CIFAR10', train=True, download=True,
                    transform=tr_map) if yi in [8, 2]]
                raw += [(xi, yi) for xi, yi in torchvision.datasets.CIFAR10(
                    '~/.torchvision/datasets/CIFAR10', train=False, transform=tr_map)
                    if yi in [8, 2]]
                x, y, idx = _load(raw)
            elif ds == 'cifar_catplane':
                raw = [(xi, yi) for xi, yi in torchvision.datasets.CIFAR10(
                    '~/.torchvision/datasets/CIFAR10', train=True, download=True,
                    transform=tr_map) if yi in [3, 0]]
                raw += [(xi, yi) for xi, yi in torchvision.datasets.CIFAR10(
                    '~/.torchvision/datasets/CIFAR10', train=False, transform=tr_map)
                    if yi in [3, 0]]
                x, y, idx = _load(raw)
            elif ds == 'cifar_animal':
                raw = [(xi, 0 if yi in [0, 1, 8, 9] else 1)
                       for xi, yi in torchvision.datasets.CIFAR10(
                           '~/.torchvision/datasets/CIFAR10', train=True, download=True,
                           transform=tr_map)]
                raw += [(xi, 0 if yi in [0, 1, 8, 9] else 1)
                        for xi, yi in torchvision.datasets.CIFAR10(
                            '~/.torchvision/datasets/CIFAR10', train=False, transform=tr_map)]
                x, y, idx = _load(raw)
            else:
                return dataset_module.get_normalized_dataset.__wrapped__(ds, ps, seeds, d, params)
            return intertwine_split(x, y, idx, ps, seeds, y.unique())
        return functools.lru_cache(maxsize=2)(patched_norm)

    orig_norm = dataset_module.get_normalized_dataset
    orig_bin  = dataset_module.get_binary_dataset
    try:
        try:
            orig_norm.cache_clear()
        except AttributeError:
            pass
        dataset_module.get_normalized_dataset = make_patched_norm(dataset)

        def make_bin_patch(smap):
            def patched_bin(ds, ps, seeds, d, params=None, device=None, dtype=None):
                sets = dataset_module.get_normalized_dataset(ds, ps, seeds, d, params)
                outs = []
                for x, y, idx in sets:
                    x = x.to(device=device, dtype=dtype)
                    b = x.new_zeros(len(y), dtype=dtype)
                    for cls, sign in smap.items():
                        b[y == cls] = sign
                    outs.append((x, b, idx))
                return outs
            return patched_bin

        dataset_module.get_binary_dataset = make_bin_patch(split_map)

        args = dataset_args(dataset, {
            **BASE_ARGS, 'h': h, 'alpha': alpha,
            'seed_init': seed, 'device': item['device'],
        })
        run = None
        for run in execute(args):
            pass
    finally:
        dataset_module.get_normalized_dataset = orig_norm
        dataset_module.get_binary_dataset = orig_bin
        try:
            orig_norm.cache_clear()
        except AttributeError:
            pass

    if not run or 'regular' not in run or not run['regular'].get('dynamics'):
        return None
    f = run['regular']['dynamics'][-1]
    return {
        'experiment': 'exp_augmentations',
        'augmentation': aug_name,
        'split': split_name,
        'is_double_sanity': aug_name == 'identity' and split_name == SANITY_CHECK_SPLIT,
        'dataset': dataset,
        'h': h, 'alpha': alpha,
        'alpha_paper': alpha / math.sqrt(h),
        'seed_init': seed,
        'final_test_err':  f['test']['err'],
        'final_train_err': f['train']['err'],
        'wall_time':       f['wall'],
        'timed_out':       f['wall'] >= 590,
        'converged':       f['train']['err'] < 0.01,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset',      default='fashion',
                    help="Dataset name, or 'all'")
    ap.add_argument('--aug-filter',   default='',
                    help='Comma-separated augmentation names (empty = all)')
    ap.add_argument('--split-filter', default='',
                    help='Comma-separated split names (empty = all)')
    ap.add_argument('--workers',      type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(__file__).parent / 'runs' / 'exp_augmentations'
    out_dir.mkdir(parents=True, exist_ok=True)

    active    = DATASETS if args.dataset == 'all' \
                else [d for d in DATASETS if d['name'] == args.dataset]
    if not active:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")
    aug_filt  = set(args.aug_filter.split(','))   if args.aug_filter   else set()
    sp_filt   = set(args.split_filter.split(',')) if args.split_filter else set()

    todo = []
    for ds in active:
        ds_name = ds['name']
        for aug_name in augmentations_for(ds_name):
            if aug_filt and aug_name not in aug_filt:
                continue
            for split_name, split_map in SPLITS_BY_DATASET.get(ds_name, {}).items():
                if sp_filt and split_name not in sp_filt:
                    continue
                int_map = {int(k): int(v) for k, v in split_map.items()}
                for h in H_VALUES:
                    for alpha in ALPHA_VALUES:
                        for seed in SEEDS:
                            key = (f"{ds_name}__{aug_name}__{split_name}"
                                   f"_h{h}_alpha{alpha:.2e}_seed{seed}.json")
                            if not (out_dir / key).exists():
                                todo.append({
                                    'aug': aug_name, 'split': split_name,
                                    'dataset': ds_name, 'split_map': int_map,
                                    'h': h, 'alpha': alpha, 'seed': seed,
                                    'device': args.device,
                                })

    print('=' * 60)
    print(f"exp_augmentations | dataset(s): {[d['name'] for d in active]}")
    print(f"aug filter:   {args.aug_filter or 'all'}")
    print(f"split filter: {args.split_filter or 'all'}")
    print(f"device: {args.device} | workers: {args.workers} | to run: {len(todo)}")
    print(f"'identity' x '{SANITY_CHECK_SPLIT}' = double sanity check")
    print('=' * 60)

    if not todo:
        print('All complete!')
        return

    start = time.time()
    ok = fail = 0
    for item, metrics in run_items(run_one, todo, workers=args.workers,
                                    desc='exp_augmentations'):
        key = (f"{item['dataset']}__{item['aug']}__{item['split']}"
               f"_h{item['h']}_alpha{item['alpha']:.2e}_seed{item['seed']}.json")
        if metrics:
            json.dump(metrics, open(out_dir / key, 'w'), indent=2)
            ok += 1
        else:
            fail += 1

    print(f"\nDone in {(time.time()-start)/60:.1f}m -- {ok} ok, {fail} failed")
    print("Next: python experiments/exp_augmentations_plot.py")


if __name__ == '__main__':
    main()
