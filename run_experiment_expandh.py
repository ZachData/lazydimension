#!/usr/bin/env python3
"""
EXPERIMENT: exp_h_range
Expanded width range across ~3 orders of magnitude, on all supported datasets.

MOTIVATION
----------
Baseline uses h ∈ {100, 300, 1000} — one order of magnitude. With only a 10×
spread in h you cannot distinguish the claimed α* ∝ h^{-1/2} from h^{-0.4} or
h^{-0.6}. We expand to h ∈ {10, 30, 100, 300, 1000, 3000, 10000} (3 decades
feasible at float64 on H200) and repeat for every dataset in the codebase.

Testing multiple datasets simultaneously answers whether the exponent β = -0.5
is universal across data modalities (grayscale digits, colour images, character
recognition) or specific to Fashion-MNIST.

CHANGES FROM BASELINE
---------------------
| Parameter  | Baseline            | This experiment                    |
|------------|---------------------|------------------------------------|
| H_VALUES   | [100, 300, 1000]    | [10, 30, 100, 300, 1000, 3000, 10000] |
| datasets   | fashion only        | all 9 supported datasets           |
| seeds      | 3                   | 3 (unchanged)                      |
| everything else | —              | unchanged                          |

NOTE on h > 10000
-----------------
At float64 with L=3, a hidden layer weight matrix for h=100000 is
100000×100000×8 bytes = 80 GB — infeasible on a single H200. Upper bound is
roughly h ≈ 10000 per GPU. For larger h, switch dtype to float32 in BASE_ARGS
and set RUN_LARGE_H=True; this halves memory and allows h up to ~30000–50000.
"""

import json
import math
import sys
import time
from pathlib import Path

import modal

# Allow importing datasets.py from the parent experiments/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets import DATASETS, dataset_args

app = modal.App("lazy-exp-h-range")

lazy_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy", "scipy")
    .add_local_dir(
        str(Path(__file__).parent.parent.parent),
        remote_path="/workspace",
        copy=True,
        ignore=["__pycache__", "*.egg-info", ".venv", "runs", "figures", "experiments"],
    )
)

# ── Width sweep ─────────────────────────────────────────────────────────────
H_VALUES = [10, 30, 100, 300, 1000, 3000, 10000]
RUN_LARGE_H = False  # set True + change dtype to float32 to attempt h > 10000

# ── Alpha sweep (identical to baseline) ─────────────────────────────────────
ALPHA_VALUES = [
    1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3,
    1.0, 3.0, 10, 30, 100, 300, 1000, 3000, 10000, 30000,
    100000, 300000, 1000000, 3000000, 10000000,
]

SEEDS = [0, 1, 2]

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


def extract_metrics(run, dataset, h, alpha, seed):
    if not run or 'regular' not in run or 'dynamics' not in run['regular']:
        return None
    dynamics = run['regular']['dynamics']
    if not dynamics:
        return None
    f = dynamics[-1]
    return {
        'experiment': 'exp_h_range',
        'dataset': dataset,
        'h': h,
        'alpha': alpha,
        'alpha_paper': alpha / math.sqrt(h),
        'seed_init': seed,
        'final_test_err': f['test']['err'],
        'final_train_err': f['train']['err'],
        'wall_time': f['wall'],
        'timed_out': f['wall'] >= 590,
        'converged': f['train']['err'] < 0.01,
        'converged_loose': f['train']['err'] < 0.5,
        'n_steps': f['step'],
    }


@app.function(image=lazy_image, gpu="H200", timeout=4 * 60 * 60, concurrency_limit=20)
def run_batch(batch):
    import os
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    from main import execute

    results = []
    for item in batch:
        dataset, h, alpha, seed = item['dataset'], item['h'], item['alpha'], item['seed']
        key = f"{dataset}_h{h}_alpha{alpha:.2e}_seed{seed}.json"

        if h > 10000:
            print(f"  SKIP h={h}: requires float32 / multi-GPU")
            results.append((key, None))
            continue

        print(f"  {dataset} h={h} α={alpha:.2e} seed={seed}")
        try:
            args = dataset_args(dataset, {
                **BASE_ARGS, 'h': h, 'alpha': alpha,
                'seed_init': seed, 'device': 'cuda',
            })
            run = None
            for run in execute(args):
                pass
            metrics = extract_metrics(run, dataset, h, alpha, seed)
            results.append((key, metrics))
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append((key, None))

    return results


@app.local_entrypoint()
def main(batch_size: int = 6, dataset: str = 'fashion'):
    """
    Args:
        dataset: dataset name to run (default: 'fashion').
                 Pass 'all' to run every supported dataset.
    """
    out_dir = Path(__file__).parent / 'runs'
    out_dir.mkdir(exist_ok=True)

    active = DATASETS if dataset == 'all' else [d for d in DATASETS if d['name'] == dataset]
    if not active:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Options: {[d['name'] for d in DATASETS]} or 'all'")

    todo = []
    skipped = 0
    for ds in active:
        for h in H_VALUES:
            for alpha in ALPHA_VALUES:
                for seed in SEEDS:
                    key = f"{ds['name']}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
                    f = out_dir / key
                    if f.exists():
                        try:
                            m = json.load(open(f))
                            if m.get('converged_loose') is not None:
                                skipped += 1
                                continue
                        except Exception:
                            f.unlink()
                    todo.append({'dataset': ds['name'], 'h': h, 'alpha': alpha, 'seed': seed})

    total = len(active) * len(H_VALUES) * len(ALPHA_VALUES) * len(SEEDS)
    print("=" * 70)
    print("exp_h_range | datasets:", [d['name'] for d in active])
    print(f"h values: {H_VALUES}")
    print(f"Total: {total} | done: {skipped} | to run: {len(todo)}")
    print("=" * 70)

    if not todo:
        print("All complete!")
        return

    batches = [todo[i:i+batch_size] for i in range(0, len(todo), batch_size)]
    start = time.time()
    handles = list(enumerate(run_batch.map([b for b in batches], return_exceptions=True)))

    ok = fail = 0
    for i, results in handles:
        if isinstance(results, Exception):
            fail += len(batches[i])
            print(f"  batch {i} ERROR: {results}")
            continue
        for key, metrics in results:
            if metrics:
                json.dump(metrics, open(out_dir / key, 'w'), indent=2)
                ok += 1
            else:
                fail += 1

    print(f"\nDone in {(time.time()-start)/60:.1f}m: {ok} ok, {fail} failed")
    print("Next: python plot.py")
