#!/usr/bin/env python3
"""
EXPERIMENT: exp_seeds
Tests whether 3 seeds (baseline) gives reliable error bands.

Phase A: 30 seeds at the transition point (h=300, alpha=1.0) to get
         the true variance distribution.
Phase B: 10-seed full grid at baseline h values to measure collapse
         spread vs n_seeds.

Usage
-----
    python experiments/exp_seeds_run.py
    python experiments/exp_seeds_run.py --phase A
    python experiments/exp_seeds_run.py --phase B
    python experiments/exp_seeds_run.py --dataset mnist --workers 4
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

# Phase A: dense seed sampling at the transition
PHASE_A_H      = 300
PHASE_A_ALPHA  = 1.0
PHASE_A_SEEDS  = list(range(30))

# Phase B: 10-seed grid
H_VALUES       = [100, 300, 1000]
ALPHA_VALUES   = [
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3,
    1.0, 3.0, 10, 30, 100, 300, 1000, 3000, 10000,
]
PHASE_B_SEEDS  = list(range(10))


def run_one(item):
    from main import execute
    phase   = item['phase']
    dataset = item['dataset']
    h, alpha, seed = item['h'], item['alpha'], item['seed']
    print(f"    [{phase}] {dataset} h={h} alpha={alpha:.2e} seed={seed}  [{DEVICE}]")
    args = dataset_args(dataset, {
        **BASE_ARGS, 'h': h, 'alpha': alpha,
        'seed_init': seed, 'device': DEVICE,
    })
    run = None
    for run in execute(args):
        pass
    return extract_last_metrics(run,
        experiment='exp_seeds', phase=phase,
        dataset=dataset, h=h, alpha=alpha,
        alpha_paper=alpha / math.sqrt(h),
        seed_init=seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='fashion',
                    help="Dataset name, or 'all'")
    ap.add_argument('--phase',   default='both',
                    choices=['A', 'B', 'both'])
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(__file__).parent / 'runs' / 'exp_seeds'
    out_dir.mkdir(parents=True, exist_ok=True)

    active = DATASETS if args.dataset == 'all' \
             else [d for d in DATASETS if d['name'] == args.dataset]
    if not active:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    todo = []
    if args.phase in ('A', 'both'):
        for ds in active:
            for seed in PHASE_A_SEEDS:
                key = f"A_{ds['name']}_h{PHASE_A_H}_alpha{PHASE_A_ALPHA:.2e}_seed{seed}.json"
                if not (out_dir / key).exists():
                    todo.append({'phase': 'A', 'dataset': ds['name'],
                                 'h': PHASE_A_H, 'alpha': PHASE_A_ALPHA, 'seed': seed})

    if args.phase in ('B', 'both'):
        for ds in active:
            for h in H_VALUES:
                for alpha in ALPHA_VALUES:
                    for seed in PHASE_B_SEEDS:
                        key = f"B_{ds['name']}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
                        if not (out_dir / key).exists():
                            todo.append({'phase': 'B', 'dataset': ds['name'],
                                         'h': h, 'alpha': alpha, 'seed': seed})

    nA = len(active) * len(PHASE_A_SEEDS)
    nB = len(active) * len(H_VALUES) * len(ALPHA_VALUES) * len(PHASE_B_SEEDS)
    print('=' * 60)
    print(f"exp_seeds | dataset(s): {[d['name'] for d in active]}")
    print(f"Phase A: {nA} total | Phase B: {nB} total | to run: {len(todo)}")
    print(f"device: {DEVICE} | workers: {args.workers}")
    print('=' * 60)

    if not todo:
        print('All complete!')
        return

    start = time.time()
    ok = fail = 0
    for item, metrics in run_items(run_one, todo, workers=args.workers,
                                    desc='exp_seeds'):
        phase = item['phase']
        key = (f"{phase}_{item['dataset']}_h{item['h']}"
               f"_alpha{item['alpha']:.2e}_seed{item['seed']}.json")
        if metrics:
            json.dump(metrics, open(out_dir / key, 'w'), indent=2)
            ok += 1
        else:
            fail += 1

    print(f"\nDone in {(time.time()-start)/60:.1f}m -- {ok} ok, {fail} failed")
    print("Next: python experiments/exp_seeds_plot.py")


if __name__ == '__main__':
    main()
