"""
_common.py — Shared helpers for all experiment run scripts.

Provides:
  BASE_ARGS     — default training arguments matching the paper's baseline
  DEFAULT_DEVICE — 'cpu' by default (see note below)
  run_items()   — local parallel/sequential runner
  project_root  — Path to the codebase root (parent of experiments/)

Device note
-----------
These experiments use float64 throughout. Consumer Nvidia GPUs (including
the RTX 3080) throttle fp64 to 1/32 of fp32 throughput (~0.3 TFLOPS),
which is slower than 10 CPU cores for the small FC networks used here.
Default is therefore 'cpu'. Pass --device cuda only if you have a
data-centre GPU (A100, H100) where fp64 is not throttled.
"""

import multiprocessing
import sys
import traceback
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ── Device default ────────────────────────────────────────────────────────────
DEFAULT_DEVICE = 'cpu'

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

def _worker_fn(args):
    fn, item = args
    try:
        return item, fn(item)
    except Exception:
        traceback.print_exc()
        return item, None


def run_items(fn, items, workers=1, desc=''):
    """
    Run fn(item) for each item in items, optionally in parallel.

    Yields (item, result) pairs as they complete.
    workers=1  → sequential (safe for debugging)
    workers>1  → multiprocessing.Pool with spawn context
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
        pairs = [(fn, item) for item in items]
        completed = 0
        with ctx.Pool(workers) as pool:
            for item, result in pool.imap_unordered(_worker_fn, pairs):
                completed += 1
                print(f'  {label}{completed}/{total}', flush=True)
                yield item, result


def extract_last_metrics(run, **extra):
    """Extract final-step metrics from a completed run dict."""
    if not run or 'regular' not in run or not run['regular'].get('dynamics'):
        return None
    f = run['regular']['dynamics'][-1]
    return {
        'final_test_err':  f['test']['err'],
        'final_train_err': f['train']['err'],
        'wall_time':       f['wall'],
        'timed_out':       f['wall'] >= 590,
        'converged':       f['train']['err'] < 0.01,
        **extra,
    }
