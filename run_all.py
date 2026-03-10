#!/usr/bin/env python3
"""
run_all.py — Master runner for all fidelity experiments.

Runs each sub-experiment sequentially using Modal. Each experiment is
independent and can also be run individually (see its own run_experiments.py).

Usage
-----
# Default: run all experiments on Fashion-MNIST
modal run experiments/run_all.py

# Run on a different single dataset
modal run experiments/run_all.py --dataset mnist

# Run on all datasets
modal run experiments/run_all.py --dataset all

# Skip specific experiments (comma-separated)
modal run experiments/run_all.py --skip exp_seeds,exp_augmentations

# Run only specific experiments
modal run experiments/run_all.py --only exp_h_range,exp_depth

# Dry run: print what would be run without launching anything
modal run experiments/run_all.py --dry-run

Experiment order
----------------
1. exp_h_range        — expanded width range
2. exp_seeds          — seed count effect
3. exp_wall_time      — extended timeout
4. exp_binary_split   — binary split variants
5. exp_depth          — network depth variants
6. exp_augmentations  — image augmentations  (heaviest; skipped unless --include-aug)

exp_augmentations is excluded from the default run because its full matrix
(16 augmentations × 7 splits × 25 alpha × 3 h × 3 seeds) is very large.
Pass --include-aug to include it, or run it directly with filters:

    modal run experiments/exp_augmentations/run_experiments.py \\
        --aug-filter identity,hflip,noise_010,invert \\
        --split-filter odd_even,footwear_vs_rest
"""

import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("lazy-run-all")

# ── Experiment registry ───────────────────────────────────────────────────────
# Each entry: (name, script_path, extra_modal_args)
# extra_modal_args are appended after --dataset=<dataset>

EXPERIMENTS = [
    ('exp_h_range',      'exp_h_range/run_experiments.py',      []),
    ('exp_seeds',        'exp_seeds/run_experiments.py',         ['--phase=both']),
    ('exp_wall_time',    'exp_wall_time/run_experiments.py',     []),
    ('exp_binary_split', 'exp_binary_split/run_experiments.py',  []),
    ('exp_depth',        'exp_depth/run_experiments.py',         []),
    # exp_augmentations excluded by default — add --include-aug to enable
    ('exp_augmentations','exp_augmentations/run_experiments.py', [
        '--aug-filter=identity,hflip,noise_010,invert,blur_3x3',
        '--split-filter=odd_even,footwear_vs_rest,random_seed42',
    ]),
]

EXPERIMENTS_BY_NAME = {name: (script, args) for name, script, args in EXPERIMENTS}


def _run_experiment(name: str, script: str, extra_args: list, dataset: str, dry_run: bool):
    exp_dir = Path(__file__).parent
    script_path = str(exp_dir / script)
    cmd = ['modal', 'run', script_path, f'--dataset={dataset}'] + extra_args

    print(f"\n{'='*70}")
    print(f"  [{name}]  dataset={dataset}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  (dry-run: skipped)")
        return True

    start = time.time()
    result = subprocess.run(cmd, cwd=str(exp_dir))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✓ {name} completed in {elapsed/60:.1f}m")
        return True
    else:
        print(f"\n  ✗ {name} FAILED (exit {result.returncode}) after {elapsed/60:.1f}m")
        return False


@app.local_entrypoint()
def main(
    dataset: str = 'fashion',
    only: str = '',
    skip: str = '',
    include_aug: bool = False,
    dry_run: bool = False,
):
    """
    Args:
        dataset      : dataset to run on (default: 'fashion', or 'all')
        only         : comma-separated experiment names to run exclusively
        skip         : comma-separated experiment names to skip
        include_aug  : include exp_augmentations (excluded by default — it's large)
        dry_run      : print commands without running them
    """
    only_set = set(only.split(',')) if only else set()
    skip_set = set(skip.split(',')) if skip else set()

    to_run = []
    for name, script, extra_args in EXPERIMENTS:
        if name == 'exp_augmentations' and not include_aug and name not in only_set:
            continue
        if only_set and name not in only_set:
            continue
        if name in skip_set:
            continue
        to_run.append((name, script, extra_args))

    print("=" * 70)
    print(f"run_all.py — dataset={dataset}  dry_run={dry_run}")
    print(f"Experiments to run ({len(to_run)}):")
    for name, script, _ in to_run:
        print(f"  {name}")
    if not include_aug and not only_set:
        print("  (exp_augmentations excluded — pass --include-aug to enable)")
    print("=" * 70)

    if not to_run:
        print("Nothing to run.")
        return

    results = {}
    overall_start = time.time()

    for name, script, extra_args in to_run:
        ok = _run_experiment(name, script, extra_args, dataset, dry_run)
        results[name] = 'OK' if ok else 'FAILED'

    elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"SUMMARY — {elapsed/60:.1f}m total")
    print(f"{'='*70}")
    for name, status in results.items():
        icon = '✓' if status == 'OK' else '✗'
        print(f"  {icon}  {name}: {status}")

    n_failed = sum(1 for s in results.values() if s == 'FAILED')
    if n_failed:
        print(f"\n{n_failed} experiment(s) failed. Re-run individually to investigate.")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully.")
        print(f"\nNext: run the plot.py in each experiment's folder, or")
        print(f"      run experiments/plot_all.py for a combined report.")
