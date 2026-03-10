#!/usr/bin/env python3
"""
EXPERIMENT: exp_depth
Tests whether β = -0.5 holds across network depths L ∈ {1, 2, 3, 4, 6}.

NTK theory predicts the exponent is depth-independent. The baseline
only tests L=3.

Usage
-----
    python experiments/exp_depth_run.py
    python experiments/exp_depth_run.py --dataset mnist --workers 4
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import BASE_ARGS, DEVICE, extract_last_metrics, run_items
from datasets import DATASETS, dataset_args

L_VALUES     = [1, 2, 3, 4, 6]
H_VALUES     = [100, 300, 1000]
ALPHA_VALUES = [
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3,
    1.0, 3.0, 10, 30, 100, 300, 1000, 3000, 10000,
]
SEEDS = [0, 1, 2]


def run_one(item):
    from main import execute
    L, dataset, h, alpha, seed = (item['L'], item['dataset'],
                                   item['h'], item['alpha'], item['seed'])
    print(f"    L={L} {dataset} h={h} alpha={alpha:.2e} seed={seed}  [{DEVICE}]")
    args = dataset_args(dataset, {
        **BASE_ARGS, 'L': L,
        'h': h, 'alpha': alpha,
        'seed_init': seed, 'device': DEVICE,
    })
    run = None
    for run in execute(args):
        pass
    return extract_last_metrics(run,
        experiment='exp_depth',
        L=L, dataset=dataset, h=h, alpha=alpha,
        alpha_paper=alpha / math.sqrt(h),
        seed_init=seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='fashion',
                    help="Dataset name, or 'all'")
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(__file__).parent / 'runs' / 'exp_depth'
    out_dir.mkdir(parents=True, exist_ok=True)

    active = DATASETS if args.dataset == 'all' \
             else [d for d in DATASETS if d['name'] == args.dataset]
    if not active:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    todo = []
    skipped = 0
    for L in L_VALUES:
        for ds in active:
            for h in H_VALUES:
                for alpha in ALPHA_VALUES:
                    for seed in SEEDS:
                        key = f"L{L}_{ds['name']}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
                        f = out_dir / key
                        if f.exists():
                            try:
                                m = json.load(open(f))
                                if m.get('converged') is not None:
                                    skipped += 1
                                    continue
                            except Exception:
                                f.unlink()
                        todo.append({'L': L, 'dataset': ds['name'],
                                     'h': h, 'alpha': alpha, 'seed': seed})

    total = len(L_VALUES) * len(active) * len(H_VALUES) * len(ALPHA_VALUES) * len(SEEDS)
    print('=' * 60)
    print(f"exp_depth | dataset(s): {[d['name'] for d in active]}")
    print(f"L values: {L_VALUES}  (L=3 is baseline depth)")
    print(f"device: {DEVICE} | workers: {args.workers}")
    print(f"Total: {total} | done: {skipped} | to run: {len(todo)}")
    print('=' * 60)

    if not todo:
        print('All complete!')
        return

    start = time.time()
    ok = fail = 0
    for item, metrics in run_items(run_one, todo, workers=args.workers,
                                    desc='exp_depth'):
        key = (f"L{item['L']}_{item['dataset']}_h{item['h']}"
               f"_alpha{item['alpha']:.2e}_seed{item['seed']}.json")
        if metrics:
            json.dump(metrics, open(out_dir / key, 'w'), indent=2)
            ok += 1
        else:
            fail += 1

    print(f"\nDone in {(time.time()-start)/60:.1f}m -- {ok} ok, {fail} failed")


if __name__ == '__main__':
    main()
