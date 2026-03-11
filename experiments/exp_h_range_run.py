#!/usr/bin/env python3
"""
EXPERIMENT: exp_h_range
h ∈ [10, 30, 100, 300, 1000, 3000, 10000], 25 alpha, 3 seeds.

Recommended: workers=10, device=cpu (RTX 3080 fp64 is slower than 10 CPU cores).
~0.7h wall-clock at workers=10.

Usage
-----
    python experiments/exp_h_range_run.py --workers 10
    python experiments/exp_h_range_run.py --workers 10 --dataset all
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import BASE_ARGS, DEFAULT_DEVICE, extract_last_metrics, run_items
from datasets import DATASETS, dataset_args

H_VALUES = [10, 30, 100, 300, 1000, 3000, 10000]
ALPHA_VALUES = [
    1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
    0.1, 0.3, 1.0, 3.0, 10, 30, 100, 300, 1000,
    3000, 10000, 30000, 100000, 300000, 1e6, 3e6, 1e7,
]
SEEDS = [0, 1, 2]


def run_one(item):
    from main import execute
    device = item['device']
    dataset, h, alpha, seed = item['dataset'], item['h'], item['alpha'], item['seed']
    print(f"    {dataset} h={h} alpha={alpha:.2e} seed={seed}  [{device}]")
    args = dataset_args(dataset, {
        **BASE_ARGS, 'h': h, 'alpha': alpha,
        'seed_init': seed, 'device': device,
    })
    run = None
    for run in execute(args):
        pass
    dyn = (run or {}).get('regular', {}).get('dynamics', [])
    converged_loose = dyn[-1].get('train', {}).get('err', 1.0) < 0.5 if dyn else False
    return extract_last_metrics(run,
        experiment='exp_h_range',
        dataset=dataset, h=h, alpha=alpha,
        alpha_paper=alpha / math.sqrt(h),
        seed_init=seed,
        converged_loose=converged_loose,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='fashion')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--device',  default=DEFAULT_DEVICE,
                    help=f'cpu or cuda (default: {DEFAULT_DEVICE}; '
                         f'fp64 on consumer Nvidia is slower than CPU)')
    args = ap.parse_args()

    out_dir = Path(__file__).parent / 'runs' / 'exp_h_range'
    out_dir.mkdir(parents=True, exist_ok=True)

    active = DATASETS if args.dataset == 'all' \
             else [d for d in DATASETS if d['name'] == args.dataset]
    if not active:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    todo, skipped = [], 0
    for ds in active:
        for h in H_VALUES:
            for alpha in ALPHA_VALUES:
                for seed in SEEDS:
                    key = f"{ds['name']}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
                    f = out_dir / key
                    if f.exists():
                        try:
                            if json.load(open(f)).get('converged_loose') is not None:
                                skipped += 1; continue
                        except Exception:
                            f.unlink()
                    todo.append({'dataset': ds['name'], 'h': h,
                                 'alpha': alpha, 'seed': seed,
                                 'device': args.device})

    total = len(active) * len(H_VALUES) * len(ALPHA_VALUES) * len(SEEDS)
    print('=' * 60)
    print(f"exp_h_range | dataset(s): {[d['name'] for d in active]}")
    print(f"h: {H_VALUES}")
    print(f"device: {args.device} | workers: {args.workers}")
    print(f"total: {total} | done: {skipped} | to run: {len(todo)}")
    print('=' * 60)

    if not todo:
        print('All complete!')
        return

    start = time.time()
    ok = fail = 0
    for item, metrics in run_items(run_one, todo, workers=args.workers,
                                    desc='exp_h_range'):
        key = (f"{item['dataset']}_h{item['h']}"
               f"_alpha{item['alpha']:.2e}_seed{item['seed']}.json")
        if metrics:
            json.dump(metrics, open(out_dir / key, 'w'), indent=2)
            ok += 1
        else:
            fail += 1

    print(f"\nDone in {(time.time()-start)/60:.1f}m -- {ok} ok, {fail} failed")
    print(f"Results in: {out_dir}")
    print("Next: python experiments/exp_h_range_plot.py")


if __name__ == '__main__':
    main()
