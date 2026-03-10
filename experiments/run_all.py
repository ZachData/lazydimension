#!/usr/bin/env python3
"""
run_all.py — Master runner for all fidelity experiments.

Runs each sub-experiment sequentially by calling its run script directly
with Python. Each experiment can also be run individually.

Usage
-----
    python experiments/run_all.py
    python experiments/run_all.py --dataset mnist
    python experiments/run_all.py --dataset all
    python experiments/run_all.py --skip exp_seeds,exp_augmentations
    python experiments/run_all.py --only exp_h_range,exp_depth
    python experiments/run_all.py --workers 4
    python experiments/run_all.py --dry-run

Notes
-----
exp_augmentations is excluded by default — its full matrix is very large.
Pass --include-aug to enable it, or run it directly with filters:

    python experiments/exp_augmentations_run.py \\
        --aug-filter identity,hflip,noise_010 \\
        --split-filter odd_even,footwear_vs_rest
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── Experiment registry ───────────────────────────────────────────────────────
# (name, script, extra_args)
EXPERIMENTS = [
    ('exp_h_range',       'exp_h_range_run.py',       []),
    ('exp_seeds',         'exp_seeds_run.py',          ['--phase=both']),
    ('exp_wall_time',     'exp_wall_time_run.py',      []),
    ('exp_binary_split',  'exp_binary_split_run.py',   []),
    ('exp_depth',         'exp_depth_run.py',          []),
    ('exp_augmentations', 'exp_augmentations_run.py',  [
        '--aug-filter=identity,hflip,noise_010,invert,blur_3x3',
        '--split-filter=odd_even,footwear_vs_rest,random_seed42',
    ]),
]


def run_experiment(name, script, extra_args, dataset, workers, dry_run):
    exp_dir = Path(__file__).parent
    cmd = [sys.executable, str(exp_dir / script),
           f'--dataset={dataset}',
           f'--workers={workers}'] + extra_args

    print(f"\n{'='*60}")
    print(f"  {name}  (dataset={dataset})")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry-run: skipped)")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(exp_dir))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  OK  {name} completed in {elapsed/60:.1f}m")
        return True
    else:
        print(f"\n  FAILED  {name} (exit {result.returncode}) after {elapsed/60:.1f}m")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset',     default='fashion',
                    help="Dataset name, or 'all' (default: fashion)")
    ap.add_argument('--only',        default='',
                    help='Comma-separated experiment names to run exclusively')
    ap.add_argument('--skip',        default='',
                    help='Comma-separated experiment names to skip')
    ap.add_argument('--include-aug', action='store_true',
                    help='Include exp_augmentations (excluded by default)')
    ap.add_argument('--workers',     type=int, default=1,
                    help='--workers to pass to each run script (default: 1)')
    ap.add_argument('--dry-run',     action='store_true',
                    help='Print commands without running them')
    args = ap.parse_args()

    only_set = set(args.only.split(',')) if args.only else set()
    skip_set = set(args.skip.split(',')) if args.skip else set()

    to_run = []
    for name, script, extra_args in EXPERIMENTS:
        if name == 'exp_augmentations' and not args.include_aug and name not in only_set:
            continue
        if only_set and name not in only_set:
            continue
        if name in skip_set:
            continue
        to_run.append((name, script, extra_args))

    print('=' * 60)
    print(f"run_all | dataset={args.dataset}  workers={args.workers}"
          + ('  [DRY RUN]' if args.dry_run else ''))
    print(f"Experiments ({len(to_run)}): {[n for n, *_ in to_run]}")
    if not args.include_aug and not only_set:
        print("  (exp_augmentations excluded -- pass --include-aug to enable)")
    print('=' * 60)

    if not to_run:
        print('Nothing to run.')
        return

    results = {}
    overall_start = time.time()

    for name, script, extra_args in to_run:
        ok = run_experiment(name, script, extra_args,
                            args.dataset, args.workers, args.dry_run)
        results[name] = 'OK' if ok else 'FAILED'

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"SUMMARY -- {elapsed/60:.1f}m total")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = 'OK' if status == 'OK' else 'FAILED'
        print(f"  {icon}  {name}")

    n_failed = sum(1 for s in results.values() if s == 'FAILED')
    if n_failed:
        print(f"\n{n_failed} experiment(s) failed.")
        sys.exit(1)
    else:
        print("\nAll done.")
        print("Next: python experiments/plot_all.py")


if __name__ == '__main__':
    main()
