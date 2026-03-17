#!/usr/bin/env python3
"""
Smoke test: one 60s run per ablation config to catch failures before real runs.

Usage:
    modal run run_ablation_smoke.py
"""

import time
from pathlib import Path

import modal

app = modal.App("lazy-smoke")

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
    'max_wall': 60,             # 60s timeout for smoke test
    'max_wall_kernel': 60,
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

SMOKE_TESTS = {
    'extra_seed': {},
    'relu':       {'act': 'relu'},
    'mnist':      {'dataset': 'mnist'},
}


def _patch_arch_inplace_mul():
    """Fix inplace .mul_() in arch/__init__.py that breaks ReLU backward pass."""
    import subprocess
    subprocess.run(
        ["sed", "-i", "s/.mul_(factor \\/ b)/* (factor \\/ b)/g",
         "/workspace/arch/__init__.py"],
        check=True,
    )


@app.function(image=lazy_image, gpu="T4", timeout=10 * 60)
def smoke_one(test_name: str, overrides: dict) -> dict:
    import os, sys, traceback
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    _patch_arch_inplace_mul()

    result = {'test': test_name, 'status': 'unknown', 'error': None,
              'steps': None, 'wall': None, 'train_err': None, 'test_err': None}
    try:
        from main import execute

        seed = 42 if test_name == 'extra_seed' else 0
        args = {**BASE_ARGS, **overrides,
                'h': 100, 'alpha': 100.0, 'seed_init': seed, 'device': 'cuda'}

        run = None
        for run in execute(args):
            pass

        if run is None or 'regular' not in run:
            result['status'] = 'FAIL'
            result['error'] = 'no output'
            return result

        dynamics = run['regular']['dynamics']
        if not dynamics:
            result['status'] = 'FAIL'
            result['error'] = 'empty dynamics'
            return result

        final = dynamics[-1]
        result['status'] = 'PASS'
        result['steps'] = final['step']
        result['wall'] = round(final['wall'], 1)
        result['train_err'] = round(final['train']['err'], 4)
        result['test_err'] = round(final['test']['err'], 4)

    except Exception as e:
        result['status'] = 'FAIL'
        result['error'] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"

    return result


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("SMOKE TEST: 1 run per ablation (h=100, α=100, 60s)")
    print("=" * 60)

    start = time.time()
    handles = []
    for name, overrides in SMOKE_TESTS.items():
        print(f"  Spawning: {name}")
        handles.append((name, smoke_one.spawn(name, dict(overrides))))

    print(f"\nWaiting for {len(handles)} results...\n")

    all_pass = True
    for name, handle in handles:
        try:
            r = handle.get()
        except Exception as e:
            r = {'test': name, 'status': 'FAIL', 'error': str(e)}

        passed = r['status'] == 'PASS'
        icon = '✓' if passed else '✗'
        all_pass = all_pass and passed

        print(f"  {icon} {r['test']:<14}", end="")
        if passed:
            print(f"  {r['steps']:>5} steps  {r['wall']:>5.1f}s  "
                  f"train_err={r['train_err']:.4f}  test_err={r['test_err']:.4f}")
        else:
            print(f"  ERROR: {(r.get('error') or 'unknown')[:200]}")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"{'ALL PASSED' if all_pass else 'SOME FAILED'} in {elapsed:.0f}s")
    print(f"{'='*60}")
