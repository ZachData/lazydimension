#!/usr/bin/env python3
"""
EXPERIMENT: exp_binary_split
All named binary splits for the chosen dataset.

The 'odd_even' split is a sanity check: it must reproduce the baseline
exactly (it is mathematically identical to what get_binary_dataset() does).

Usage
-----
    python experiments/exp_binary_split_run.py
    python experiments/exp_binary_split_run.py --dataset cifar10
    python experiments/exp_binary_split_run.py --split-filter odd_even,footwear_vs_rest
    python experiments/exp_binary_split_run.py --workers 4
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
    from main import execute
    import torch

    dataset    = item['dataset']
    split_name = item['split']
    split_map  = item['split_map']
    h, alpha, seed = item['h'], item['alpha'], item['seed']
    is_sanity  = item['is_sanity']

    print(f"    {dataset}/{split_name} h={h} alpha={alpha:.2e} seed={seed}"
          + (" [SANITY]" if is_sanity else "") + f"  [{item['device']}]")

    # Patch get_binary_dataset with this split's label map
    original = dataset_module.get_binary_dataset

    def patched(ds, ps, seeds, d, params=None, device=None, dtype=None):
        sets = dataset_module.get_normalized_dataset(ds, ps, seeds, d, params)
        outs = []
        for x, y, idx in sets:
            x = x.to(device=device, dtype=dtype)
            b = x.new_zeros(len(y), dtype=dtype)
            for cls, sign in split_map.items():
                b[y == cls] = sign
            outs.append((x, b, idx))
        return outs

    dataset_module.get_binary_dataset = patched
    try:
        args = dataset_args(dataset, {
            **BASE_ARGS, 'h': h, 'alpha': alpha,
            'seed_init': seed, 'device': item['device'],
        })
        run = None
        for run in execute(args):
            pass
    finally:
        dataset_module.get_binary_dataset = original

    if not run or 'regular' not in run or not run['regular'].get('dynamics'):
        return None
    f = run['regular']['dynamics'][-1]
    return {
        'experiment': 'exp_binary_split',
        'split': split_name,
        'is_sanity_check': is_sanity,
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
    ap.add_argument('--split-filter', default='',
                    help='Comma-separated split names to run (empty = all)')
    ap.add_argument('--workers',      type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(__file__).parent / 'runs' / 'exp_binary_split'
    out_dir.mkdir(parents=True, exist_ok=True)

    active = DATASETS if args.dataset == 'all' \
             else [d for d in DATASETS if d['name'] == args.dataset]
    if not active:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    sp_filter = set(args.split_filter.split(',')) if args.split_filter else set()

    todo = []
    for ds in active:
        ds_name = ds['name']
        for split_name, split_map in SPLITS_BY_DATASET.get(ds_name, {}).items():
            if sp_filter and split_name not in sp_filter:
                continue
            is_sanity = (split_name == SANITY_CHECK_SPLIT)
            int_map   = {int(k): int(v) for k, v in split_map.items()}
            for h in H_VALUES:
                for alpha in ALPHA_VALUES:
                    for seed in SEEDS:
                        key = f"{ds_name}__{split_name}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
                        if not (out_dir / key).exists():
                            todo.append({
                                'dataset': ds_name, 'split': split_name,
                                'split_map': int_map, 'is_sanity': is_sanity,
                                'h': h, 'alpha': alpha, 'seed': seed,
                                'device': args.device,
                            })

    print('=' * 60)
    print(f"exp_binary_split | dataset(s): {[d['name'] for d in active]}")
    for ds in active:
        splits = list(SPLITS_BY_DATASET.get(ds['name'], {}).keys())
        if sp_filter:
            splits = [s for s in splits if s in sp_filter]
        print(f"  {ds['name']}: {splits}")
    print(f"device: {args.device} | workers: {args.workers} | to run: {len(todo)}")
    print(f"'{SANITY_CHECK_SPLIT}' = sanity check, must match baseline exactly")
    print('=' * 60)

    if not todo:
        print('All complete!')
        return

    start = time.time()
    ok = fail = 0
    for item, metrics in run_items(run_one, todo, workers=args.workers,
                                    desc='exp_binary_split'):
        key = (f"{item['dataset']}__{item['split']}"
               f"_h{item['h']}_alpha{item['alpha']:.2e}_seed{item['seed']}.json")
        if metrics:
            json.dump(metrics, open(out_dir / key, 'w'), indent=2)
            ok += 1
        else:
            fail += 1

    print(f"\nDone in {(time.time()-start)/60:.1f}m -- {ok} ok, {fail} failed")
    print("Next: python experiments/exp_binary_split_plot.py")


if __name__ == '__main__':
    main()
