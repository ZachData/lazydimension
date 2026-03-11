#!/usr/bin/env python3
"""
run_all.py — Master runner for all fidelity experiments.

Defaults:
  --workers 10   (optimal for Ryzen 5600X 6c/12t)
  --device  cpu  (RTX 3080 fp64 is ~1/32 its fp32 speed; CPU is faster)

Excluded by default (opt-in via --include-wall-time / --include-aug):
  exp_wall_time    -- requires max_wall=3600s per run; ~1.6h even at workers=10
  exp_augmentations -- very large matrix; run separately with filters

Usage
-----
    python experiments/run_all.py                        # fashion, workers=10, cpu
    python experiments/run_all.py --dataset mnist
    python experiments/run_all.py --dataset all
    python experiments/run_all.py --workers 4
    python experiments/run_all.py --skip exp_depth
    python experiments/run_all.py --only exp_h_range,exp_seeds
    python experiments/run_all.py --include-wall-time
    python experiments/run_all.py --dry-run

Estimated wall-clock at workers=10, cpu, fashion:
    exp_h_range        ~0.7h
    exp_seeds          ~0.7h
    exp_binary_split   ~1.4h
    exp_depth          ~1.0h
    ──────────────────────────
    Total              ~3.8h
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    ('exp_h_range',       'exp_h_range_run.py',       []),
    ('exp_seeds',         'exp_seeds_run.py',          ['--phase=both']),
    ('exp_binary_split',  'exp_binary_split_run.py',   []),
    ('exp_depth',         'exp_depth_run.py',          []),
    # Opt-in only:
    ('exp_wall_time',     'exp_wall_time_run.py',      []),
    ('exp_augmentations', 'exp_augmentations_run.py',  [
        '--aug-filter=identity,hflip,noise_010,invert,blur_3x3',
        '--split-filter=odd_even,footwear_vs_rest,random_seed42',
    ]),
]

OPT_IN = {'exp_wall_time', 'exp_augmentations'}


def run_experiment(name, script, extra_args, dataset, workers, device, dry_run):
    exp_dir = Path(__file__).parent
    cmd = [sys.executable, str(exp_dir / script),
           f'--dataset={dataset}',
           f'--workers={workers}',
           f'--device={device}'] + extra_args

    print(f"\n{'='*60}")
    print(f"  {name}  (dataset={dataset}  workers={workers}  device={device})")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  (dry-run: skipped)")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(exp_dir))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  OK  {name} in {elapsed/60:.1f}m")
        return True
    else:
        print(f"\n  FAILED  {name} (exit {result.returncode}) after {elapsed/60:.1f}m")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset',           default='fashion')
    ap.add_argument('--workers',           type=int, default=10,
                    help='Parallel workers (default: 10 for Ryzen 5600X)')
    ap.add_argument('--device',            default='cpu',
                    help='cpu or cuda (default: cpu — fp64 on consumer Nvidia is slower)')
    ap.add_argument('--only',              default='',
                    help='Comma-separated names to run exclusively')
    ap.add_argument('--skip',              default='',
                    help='Comma-separated names to skip')
    ap.add_argument('--include-wall-time', action='store_true',
                    help='Include exp_wall_time (opt-in: 3600s/run)')
    ap.add_argument('--include-aug',       action='store_true',
                    help='Include exp_augmentations (opt-in: large matrix)')
    ap.add_argument('--dry-run',           action='store_true')
    args = ap.parse_args()

    only_set = set(args.only.split(',')) if args.only else set()
    skip_set = set(args.skip.split(',')) if args.skip else set()

    to_run = []
    for name, script, extra_args in EXPERIMENTS:
        if name in OPT_IN:
            included = (
                (name == 'exp_wall_time'     and args.include_wall_time) or
                (name == 'exp_augmentations' and args.include_aug) or
                name in only_set
            )
            if not included:
                continue
        if only_set and name not in only_set:
            continue
        if name in skip_set:
            continue
        to_run.append((name, script, extra_args))

    print('=' * 60)
    print(f"run_all | dataset={args.dataset} workers={args.workers} device={args.device}"
          + ('  [DRY RUN]' if args.dry_run else ''))
    print(f"Running ({len(to_run)}): {[n for n, *_ in to_run]}")
    excluded = [n for n in OPT_IN
                if not any(n == r[0] for r in to_run)]
    if excluded:
        print(f"Excluded (opt-in): {excluded}")
    print('=' * 60)

    if not to_run:
        print('Nothing to run.')
        return

    results = {}
    overall_start = time.time()
    for name, script, extra_args in to_run:
        ok = run_experiment(name, script, extra_args,
                            args.dataset, args.workers, args.device, args.dry_run)
        results[name] = 'OK' if ok else 'FAILED'

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"SUMMARY -- {elapsed/60:.1f}m total")
    print('=' * 60)
    for name, status in results.items():
        print(f"  {'OK  ' if status=='OK' else 'FAIL'}  {name}")

    if any(s == 'FAILED' for s in results.values()):
        sys.exit(1)
    else:
        print("\nAll done. Next: python experiments/plot_all.py")


if __name__ == '__main__':
    main()
