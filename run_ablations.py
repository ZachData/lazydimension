#!/usr/bin/env python3
"""Run minimal ablation tests against the fixed baseline.

Three tests:
  extra_seed  - Seed 42 at baseline config to verify reproducibility
  relu        - ReLU instead of Swish
  mnist       - MNIST instead of Fashion-MNIST

Each uses a small grid: 2 widths × 6 alphas = 12 runs per test.
Total: 36 runs, ~6 GPU-hours worst case, ~2 wall-hours at concurrency 10.

Usage:
    modal run run_ablations.py --test extra_seed
    modal run run_ablations.py --test relu,mnist
    modal run run_ablations.py --test all
"""

import json
import math
import time
from pathlib import Path
from collections import defaultdict

import modal

app = modal.App("lazy-ablations")

lazy_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy", "scipy")
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/workspace",
        copy=True,
        ignore=["__pycache__", "*.egg-info", ".venv", "runs*", "figures"],
    )
)

# ── Fixed baseline config (tau=0, pure gradient flow) ────────────────────
BASE_ARGS = {
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
    'tau_over_h': 0.0,
    'tau_over_h_kernel': 0.0,
    'tau_alpha_crit': None,
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

# 6 alpha_code values spanning both regimes and the boundary
ALPHA_GRID = [1e-2, 1.0, 1e2, 1e4, 1e5, 1e6]
H_GRID = [100, 1000]

TESTS = {
    'extra_seed': {
        'desc': 'Seed 42 to check result variance',
        'h': H_GRID,
        'alpha': ALPHA_GRID,
        'seeds': [42],
        'overrides': {},
    },
    'relu': {
        'desc': 'ReLU activation (Jacot et al. used ReLU)',
        'h': H_GRID,
        'alpha': ALPHA_GRID,
        'seeds': [0],
        'overrides': {'act': 'relu'},
    },
    'mnist': {
        'desc': 'MNIST instead of Fashion-MNIST',
        'h': H_GRID,
        'alpha': ALPHA_GRID,
        'seeds': [0],
        'overrides': {'dataset': 'mnist'},
    },
}


def extract_metrics(run: dict, h: int, alpha: float, seed: int, test_name: str) -> dict | None:
    if 'regular' not in run or 'dynamics' not in run['regular']:
        return None
    dynamics = run['regular']['dynamics']
    if not dynamics:
        return None
    final = dynamics[-1]
    return {
        'test': test_name,
        'h': h,
        'alpha': alpha,
        'seed_init': seed,
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


def _patch_arch_inplace_mul():
    """Fix inplace .mul_() in arch/__init__.py that breaks ReLU backward pass."""
    import subprocess
    subprocess.run(
        ["sed", "-i", "s/.mul_(factor \\/ b)/* (factor \\/ b)/g",
         "/workspace/arch/__init__.py"],
        check=True,
    )


@app.function(image=lazy_image, gpu="H200", timeout=2 * 60 * 60, concurrency_limit=10)
def run_batch(batch: list[tuple], test_name: str, overrides: dict) -> list[tuple[str, dict | None]]:
    import os, sys
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    _patch_arch_inplace_mul()
    from main import execute

    results = []
    for i, (h, alpha, seed) in enumerate(batch):
        key = f"{test_name}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
        print(f"  [{i+1}/{len(batch)}] {test_name}: h={h} α={alpha:.2e} seed={seed}")
        try:
            args = {**BASE_ARGS, **overrides, 'h': h, 'alpha': alpha,
                    'seed_init': seed, 'device': 'cuda'}
            run = None
            for run in execute(args):
                pass
            metrics = extract_metrics(run, h, alpha, seed, test_name) if run else None
            results.append((key, metrics))
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append((key, None))
    return results


@app.local_entrypoint()
def main(test: str = "all", batch_size: int = 12):
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / 'runs_ablations'
    output_dir.mkdir(exist_ok=True)

    if test == 'all':
        test_names = list(TESTS.keys())
    else:
        test_names = [t.strip() for t in test.split(',')]
        for t in test_names:
            if t not in TESTS:
                print(f"Unknown test '{t}'. Available: {', '.join(TESTS.keys())}")
                return

    all_experiments = []
    skipped = 0

    for name in test_names:
        cfg = TESTS[name]
        n = len(cfg['h']) * len(cfg['alpha']) * len(cfg['seeds'])
        print(f"  {name}: {cfg['desc']} ({n} runs)")

        for h in cfg['h']:
            for alpha in cfg['alpha']:
                for seed in cfg['seeds']:
                    key = f"{name}_h{h}_alpha{alpha:.2e}_seed{seed}.json"
                    json_file = output_dir / key
                    if json_file.exists():
                        try:
                            with open(json_file) as f:
                                m = json.load(f)
                                if m.get('converged') is not None:
                                    skipped += 1
                                    continue
                        except Exception:
                            json_file.unlink()
                    all_experiments.append((name, h, alpha, seed))

    total = len(all_experiments) + skipped
    print(f"\n{'='*60}")
    print(f"Total: {total} | Already done: {skipped} | To run: {len(all_experiments)}")
    print(f"Worst case: {len(all_experiments)*600/3600:.1f} GPU-hours")
    print(f"{'='*60}")

    if not all_experiments:
        print("All done!")
        return

    by_test = defaultdict(list)
    for name, h, alpha, seed in all_experiments:
        by_test[name].append((h, alpha, seed))

    handles = []
    for name, exps in by_test.items():
        overrides = TESTS[name]['overrides']
        batches = [exps[i:i+batch_size] for i in range(0, len(exps), batch_size)]
        for batch in batches:
            handles.append((name, run_batch.spawn(batch, name, overrides)))

    print(f"Spawned {len(handles)} containers")

    start = time.time()
    completed = failed = 0
    for name, handle in handles:
        try:
            batch_results = handle.get()
            for key, metrics in batch_results:
                if metrics is not None:
                    with open(output_dir / key, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    completed += 1
                else:
                    failed += 1
            ok = sum(1 for _, m in batch_results if m)
            nok = sum(1 for _, m in batch_results if not m)
            print(f"  {name}: {ok} ok, {nok} failed")
        except Exception as e:
            print(f"  {name}: ERROR: {e}")

    elapsed = time.time() - start
    print(f"\nDone: {completed} ok, {failed} failed in {elapsed/60:.1f}m")
    print(f"Results: {output_dir}/")
