#!/usr/bin/env python3
"""
EXPERIMENT: exp_wall_time
Extends max_wall from 600s to 3600s for runs near the regime boundary.

Tests whether the baseline's 600s timeout biased alpha* by cutting off
small-alpha (feature-training) runs before convergence.

Usage
-----
    python experiments/exp_wall_time_run.py
    python experiments/exp_wall_time_run.py --dataset mnist --workers 2
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

# Pairs near the transition where timeouts are most likely
SENSITIVE_PAIRS = [
    (100, 0.01), (100, 0.03), (100, 0.1), (100, 0.3),
    (300, 0.01), (300, 0.03), (300, 0.1), (300, 0.3), (300, 1.0),
    (1000, 0.01), (1000, 0.03), (1000, 0.1), (1000, 0.3),
    (1000, 1.0), (1000, 3.0),
    (100, 1.0), (100, 3.0),
    (300, 3.0), (300, 10.0),
    (1000, 10.0),
]
SEEDS = [0, 1, 2]
EXTENDED_WALL = 3600


def run_one(item):
    from main import execute
    dataset, h, alpha, seed = item['dataset'], item['h'], item['alpha'], item['seed']
    print(f"    {dataset} h={h} alpha={alpha:.2e} seed={seed}  [{item['device']}] wall={EXTENDED_WALL}s")
    args = dataset_args(dataset, {
        **BASE_ARGS,
        'h': h, 'alpha': alpha, 'seed_init': seed,
        'device': item['device'],
        'max_wall': EXTENDED_WALL,
    })
    run = None
    for run in execute(args):
        pass
    return extract_last_metrics(run,
        experiment='exp_wall_time',
        dataset=dataset, h=h, alpha=alpha,
        alpha_paper=alpha / math.sqrt(h),
        seed_init=seed,
        max_wall=EXTENDED_WALL,
        timed_out=(run or {}).get('regular', {}).get('dynamics', [{}])[-1]
            .get('wall', 0) >= EXTENDED_WALL - 10,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='fashion',
                    help="Dataset name, or 'all'")
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--device',  default=DEFAULT_DEVICE,
                    help='cpu or cuda (default: cpu; fp64 on consumer Nvidia is slower)')
    args = ap.parse_args()

    out_dir = Path(__file__).parent / 'runs' / 'exp_wall_time'
    out_dir.mkdir(parents=True, exist_ok=True)

    active = DATASETS if args.dataset == 'all' \
             else [d for d in DATASETS if d['name'] == args.dataset]
    if not active:
        raise ValueError(f"Unknown dataset '{args.dataset}'.")

    todo = []
    for ds in active:
        for h, alpha in SENSITIVE_PAIRS:
            for seed in SEEDS:
                key = f"{ds['name']}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
                if not (out_dir / key).exists():
                    todo.append({'dataset': ds['name'],
                                 'h': h, 'alpha': alpha, 'seed': seed,
                                 'device': item['device']})

    total = len(active) * len(SENSITIVE_PAIRS) * len(SEEDS)
    print('=' * 60)
    print(f"exp_wall_time | dataset(s): {[d['name'] for d in active]}")
    print(f"{len(SENSITIVE_PAIRS)} sensitive pairs x {len(SEEDS)} seeds = {total} total")
    print(f"max_wall: {EXTENDED_WALL}s  (baseline: 600s)")
    print(f"device: {args.device} | workers: {args.workers}")
    print(f"to run: {len(todo)}")
    print('=' * 60)

    if not todo:
        print('All complete!')
        return

    start = time.time()
    ok = fail = 0
    for item, metrics in run_items(run_one, todo, workers=args.workers,
                                    desc='exp_wall_time'):
        key = (f"{item['dataset']}_h{item['h']}"
               f"_alpha{item['alpha']:.2e}_seed{item['seed']}.json")
        if metrics:
            json.dump(metrics, open(out_dir / key, 'w'), indent=2)
            ok += 1
        else:
            fail += 1

    print(f"\nDone in {(time.time()-start)/60:.1f}m -- {ok} ok, {fail} failed")
    print("Next: python experiments/exp_wall_time_plot.py")


if __name__ == '__main__':
    main()
