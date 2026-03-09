#!/usr/bin/env python3
"""Run lazy training experiments locally (no Modal required).

Experiments run sequentially on whatever device is available (CUDA if present,
otherwise CPU).  Results are saved incrementally so the run is fully resumable.

Usage:
    python run_experiments_local.py
    python run_experiments_local.py --workers 4   # parallel processes (CPU)
    python run_experiments_local.py --device cpu  # force CPU
    python run_experiments_local.py --h 100 300   # only specific widths
    python run_experiments_local.py --dry-run     # print experiment list only
"""

import argparse
import json
import math
import multiprocessing
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment grid (mirrors run_experiments.py)
# ---------------------------------------------------------------------------
H_VALUES = [100, 300, 1000]
ALPHA_VALUES = [
    1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0,
    10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000,
    3000000, 10000000,
]
SEEDS = [0, 1, 2]

EXPERIMENT_ARGS_TEMPLATE = {
    'dtype': 'float64',
    'seed_testset': 0,
    'seed_kernelset': 0,
    'seed_trainset': 0,
    'dataset': 'fashion',
    'ptr': 10000,
    'ptk': 0,
    'pte': 50000,
    'd': None,
    'data_param1': None,
    'data_param2': None,
    'arch': 'fc',
    'act': 'swish',
    'act_beta': 1.0,
    'bias': 0,
    'last_bias': 0,
    'var_bias': 0,
    'L': 3,
    'mix_angle': 45,
    'cv_L1': 2,
    'cv_L2': 2,
    'cv_h_base': 1,
    'cv_fsz': 5,
    'cv_pad': 1,
    'cv_stride_first': 1,
    'init_kernel': 0,
    'init_kernel_ptr': 0,
    'regular': 1,
    'running_kernel': None,
    'final_kernel': 0,
    'final_kernel_ptr': 0,
    'final_headless': 0,
    'final_headless_ptr': 0,
    'init_features_ptr': 0,
    'final_features': 0,
    'final_features_ptr': 0,
    'train_kernel': 1,
    'store_kernel': 0,
    'delta_kernel': 0,
    'stretch_kernel': 0,
    'save_outputs': 0,
    'save_state': 0,
    'save_weights': 0,
    'f0': 1,
    'tau_over_h': 1e-3,
    'tau_over_h_kernel': 1e-3,
    'tau_alpha_crit': 1e3,
    'temperature': 0.0,
    'batch_min': 1,
    'batch_max': None,
    'dt_amp': 1.1,
    'dt_dam': 1.1 ** 3,
    'max_wall': 600,
    'max_wall_kernel': 600,
    'wall_max_early_stopping': None,
    'chunk': 100000,
    'max_dgrad': 1e-4,
    'max_dout': 1e-1,
    'loss': 'softhinge',
    'loss_beta': 20.0,
    'loss_margin': 1.0,
    'stop_margin': 1.0,
    'stop_frac': 1.0,
    'bs': None,
    'ckpt_step': 100,
    'ckpt_tau': 1e4,
}


# ---------------------------------------------------------------------------
# Metrics extraction (identical to run_experiments.py)
# ---------------------------------------------------------------------------
def extract_metrics(run: dict, h: int, alpha: float, seed: int) -> dict | None:
    """Extract final metrics from a training run."""
    if 'regular' not in run or 'dynamics' not in run['regular']:
        return None
    dynamics = run['regular']['dynamics']
    if not dynamics:
        return None
    final = dynamics[-1]
    return {
        'h': h,
        'alpha': alpha,
        'seed_init': seed,
        'dataset': 'fashion',
        'ptr': 10000,
        'pte': 50000,
        'final_train_err': final['train']['err'],
        'final_train_loss': final['train']['loss'],
        'final_train_aloss': final['train']['aloss'],
        'final_train_margin': final['train']['mind'],
        'final_train_nd': final['train']['nd'],
        'final_test_err': final['test']['err'],
        'final_test_loss': final['test']['loss'],
        'final_test_aloss': final['test']['aloss'],
        'final_test_margin': final['test']['mind'],
        'n_steps': final['step'],
        'final_t': final['t'],
        'wall_time': final['wall'],
        'converged': final['train']['err'] < 0.5,
        'init_train_err': dynamics[0]['train']['err'],
        'init_test_err': dynamics[0]['test']['err'],
    }


# ---------------------------------------------------------------------------
# Single-experiment runner (called in-process or by worker)
# ---------------------------------------------------------------------------
def run_one(h: int, alpha: float, seed: int, device: str, output_dir: Path) -> dict | None:
    """Run a single experiment and save result to JSON.  Returns metrics or None."""
    from main import execute  # imported here so multiprocessing workers get it

    args = {
        **EXPERIMENT_ARGS_TEMPLATE,
        'h': h,
        'alpha': alpha,
        'seed_init': seed,
        'device': device,
    }

    run = None
    try:
        for run in execute(args):
            pass
        metrics = extract_metrics(run, h, alpha, seed) if run else None
    except Exception as e:
        print(f"  FAILED h={h} α={alpha:.2e} seed={seed}: {e}", flush=True)
        metrics = None

    if metrics is not None:
        key = output_dir / f"h{h}_alpha{alpha:.2e}_seed{seed}.json"
        with open(key, 'w') as f:
            json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Worker entry-point for multiprocessing
# ---------------------------------------------------------------------------
def _worker(task):
    """Unpack task tuple and call run_one (needed for Pool.map)."""
    h, alpha, seed, device, output_dir = task
    label = f"h={h} α={alpha:.2e} seed={seed}"
    t0 = time.time()
    metrics = run_one(h, alpha, seed, device, output_dir)
    elapsed = time.time() - t0
    status = "ok" if metrics else "FAILED"
    print(f"  [{status}] {label}  ({elapsed:.0f}s)", flush=True)
    return metrics is not None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=str, default=None,
                        help='Torch device: "cuda", "cpu", etc. '
                             'Auto-detected if omitted.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel worker processes. '
                             'Use 1 (default) for GPU runs to avoid OOM. '
                             'Values >1 are useful for CPU-only runs.')
    parser.add_argument('--h', type=int, nargs='+', default=None,
                        help='Restrict to specific width(s), e.g. --h 100 300')
    parser.add_argument('--alpha', type=float, nargs='+', default=None,
                        help='Restrict to specific alpha value(s)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Restrict to specific seed(s), e.g. --seeds 0')
    parser.add_argument('--output-dir', type=str, default='runs',
                        help='Directory for JSON result files (default: runs/)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the experiment list and exit without running')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    # ── output dir ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── build experiment list ───────────────────────────────────────────────
    h_values     = args.h     or H_VALUES
    alpha_values = args.alpha or ALPHA_VALUES
    seeds        = args.seeds or SEEDS

    experiments = []
    skipped = 0
    for h in h_values:
        for alpha in alpha_values:
            for seed in seeds:
                json_file = output_dir / f"h{h}_alpha{alpha:.2e}_seed{seed}.json"
                if json_file.exists():
                    try:
                        with open(json_file) as f:
                            m = json.load(f)
                        if m.get('converged') is not None:
                            skipped += 1
                            continue
                    except Exception:
                        json_file.unlink()  # corrupt file — re-run it
                experiments.append((h, alpha, seed))

    total = len(experiments) + skipped
    print("=" * 60)
    print(f"Total experiments : {total}")
    print(f"Already complete  : {skipped}")
    print(f"To run            : {len(experiments)}")
    print(f"Workers           : {args.workers}")
    print("=" * 60)

    if args.dry_run:
        for h, alpha, seed in experiments:
            print(f"  h={h:4d}  α={alpha:.2e}  seed={seed}")
        return

    if not experiments:
        print("All experiments already complete!")
        return

    # ── run ─────────────────────────────────────────────────────────────────
    start = time.time()
    completed = failed = 0

    if args.workers == 1:
        # Sequential — simplest, best for GPU
        for i, (h, alpha, seed) in enumerate(experiments, 1):
            label = f"h={h} α={alpha:.2e} seed={seed}"
            print(f"[{i}/{len(experiments)}] {label}", flush=True)
            t0 = time.time()
            metrics = run_one(h, alpha, seed, args.device, output_dir)
            elapsed = time.time() - t0
            if metrics:
                completed += 1
                print(f"  ✓ test_err={metrics['final_test_err']:.4f}  ({elapsed:.0f}s)", flush=True)
            else:
                failed += 1
                print(f"  ✗ FAILED  ({elapsed:.0f}s)", flush=True)

            # ETA estimate
            done = completed + failed
            avg = (time.time() - start) / done
            remaining = avg * (len(experiments) - done)
            print(f"  ETA: {remaining/60:.1f} min remaining", flush=True)

    else:
        # Parallel — useful for CPU-only runs
        tasks = [(h, alpha, seed, args.device, output_dir)
                 for h, alpha, seed in experiments]
        with multiprocessing.Pool(processes=args.workers) as pool:
            for i, ok in enumerate(pool.imap_unordered(_worker, tasks), 1):
                if ok:
                    completed += 1
                else:
                    failed += 1
                done = completed + failed
                pct = 100 * done / len(experiments)
                elapsed = time.time() - start
                avg = elapsed / done
                remaining = avg * (len(experiments) - done)
                print(f"  Progress: {done}/{len(experiments)} ({pct:.0f}%)  "
                      f"ETA {remaining/60:.1f} min", flush=True)

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print(f"Done: {completed} succeeded, {failed} failed in {elapsed/60:.1f} min")
    print(f"Results saved in: {output_dir}/")
    print("Next step: python3 plot.py")
    print("=" * 60)


if __name__ == '__main__':
    # Required for multiprocessing on macOS / Windows
    multiprocessing.freeze_support()
    main()
