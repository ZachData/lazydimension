"""
_common.py — Shared helpers for all experiment run scripts.

Provides:
  BASE_ARGS     — default training arguments matching the paper's baseline
  DEVICE        — 'cuda' if available, else 'cpu'
  run_items()   — local parallel/sequential runner replacing Modal's .map()
  project_root  — Path to the codebase root (parent of experiments/)
"""

import math
import multiprocessing
import sys
import traceback
from pathlib import Path

import torch

# ── Project root ──────────────────────────────────────────────────────────────
# experiments/ sits directly inside the project root
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Baseline training arguments ───────────────────────────────────────────────
BASE_ARGS = {
    'dtype': 'float64',
    'seed_testset': 0, 'seed_kernelset': 0, 'seed_trainset': 0,
    'd': None, 'data_param1': None, 'data_param2': None,
    'arch': 'fc', 'act': 'swish', 'act_beta': 1.0,
    'bias': 0, 'last_bias': 0, 'var_bias': 0, 'L': 3,
    'mix_angle': 45, 'cv_L1': 2, 'cv_L2': 2, 'cv_h_base': 1,
    'cv_fsz': 5, 'cv_pad': 1, 'cv_stride_first': 1,
    'init_kernel': 0, 'init_kernel_ptr': 0, 'regular': 1,
    'running_kernel': None, 'final_kernel': 0, 'final_kernel_ptr': 0,
    'final_headless': 0, 'final_headless_ptr': 0, 'init_features_ptr': 0,
    'final_features': 0, 'final_features_ptr': 0, 'train_kernel': 1,
    'store_kernel': 0, 'delta_kernel': 0, 'stretch_kernel': 0,
    'save_outputs': 0, 'save_state': 0, 'save_weights': 0,
    'f0': 1,
    'tau_over_h': 1e-3, 'tau_over_h_kernel': 1e-3, 'tau_alpha_crit': 1e3,
    'temperature': 0.0, 'batch_min': 1, 'batch_max': None,
    'dt_amp': 1.1, 'dt_dam': 1.1 ** 3,
    'max_wall': 600, 'max_wall_kernel': 600,
    'wall_max_early_stopping': None,
    'max_dgrad': 1e-4, 'max_dout': 1e-1,
    'loss': 'softhinge', 'loss_beta': 20.0,
    'loss_margin': 1.0, 'stop_margin': 1.0, 'stop_frac': 1.0,
    'bs': None, 'ckpt_step': 100, 'ckpt_tau': 1e4,
    'ptk': 0,
}


# ── Local runner ──────────────────────────────────────────────────────────────

def _worker(args):
    """Top-level function for multiprocessing (must be picklable)."""
    fn, item = args
    try:
        return fn(item)
    except Exception:
        traceback.print_exc()
        return None


def run_items(fn, items, workers=1, desc=''):
    """
    Run fn(item) for each item, optionally in parallel.

    Parameters
    ----------
    fn      : callable(item) -> result
    items   : list of items
    workers : number of parallel processes (1 = sequential)
    desc    : label for progress output

    Yields
    ------
    (item, result) pairs in completion order.
    """
    if not items:
        return

    total = len(items)
    label = f'[{desc}] ' if desc else ''

    if workers <= 1:
        for i, item in enumerate(items, 1):
            print(f'  {label}{i}/{total}', flush=True)
            try:
                result = fn(item)
            except Exception:
                traceback.print_exc()
                result = None
            yield item, result
    else:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(workers) as pool:
            pairs = [(fn, item) for item in items]
            for i, result in enumerate(pool.imap_unordered(_worker, pairs), 1):
                print(f'  {label}{i}/{total}', flush=True)
                # result here is the return value of _worker(fn, item)
                # We don't have item back from imap_unordered easily,
                # so use a wrapper approach below
            # Re-do with starmap to keep item association
        # Use pool.starmap for ordered results with item tracking
        with ctx.Pool(workers) as pool:
            results = pool.starmap(_worker, pairs)
        for item, result in zip(items, results):
            yield item, result


def extract_last_metrics(run, **extra):
    """Extract final-step metrics from a completed run dict."""
    if not run or 'regular' not in run or 'dynamics' not in run['regular']:
        return None
    dynamics = run['regular']['dynamics']
    if not dynamics:
        return None
    f = dynamics[-1]
    return {
        'final_test_err':  f['test']['err'],
        'final_train_err': f['train']['err'],
        'wall_time':       f['wall'],
        'timed_out':       f['wall'] >= 590,
        'converged':       f['train']['err'] < 0.01,
        **extra,
    }
