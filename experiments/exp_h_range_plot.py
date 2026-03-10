#!/usr/bin/env python3
"""
ANALYSIS: exp_h_range
Plots collapse figures and fits the scaling exponent β for every dataset.

Output per dataset:
  results/{dataset}_collapse.png   — left: test err vs α_paper, right: vs α√h
  results/exponents.json           — β ± σ and R² for every dataset
  results/exponent_comparison.png  — β vs dataset (should be flat at -0.5)

Usage:
    python plot.py [--runs-dir runs]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent))
from datasets import DATASETS


def load(directory):
    rows = []
    for f in Path(directory).glob('*.json'):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    return rows


def group(rows, dataset):
    """Return {h: {alpha_paper: [test_err, ...]}} for one dataset."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get('dataset') != dataset:
            continue
        err = r.get('final_test_err')
        if err is None or err > 0.5:
            continue
        by[r['h']][r['alpha_paper']].append(err)
    return by


def find_boundary(alphas, errs, frac=0.5):
    """Interpolate α* at which error crosses frac of its range."""
    alphas, errs = np.array(alphas), np.array(errs)
    idx = np.argsort(alphas)
    alphas, errs = alphas[idx], errs[idx]
    lo, hi = errs.min(), errs.max()
    if hi - lo < 0.002:
        return None
    thresh = lo + frac * (hi - lo)
    for i in range(len(alphas) - 1):
        if (errs[i] - thresh) * (errs[i+1] - thresh) <= 0:
            t = (thresh - errs[i]) / (errs[i+1] - errs[i] + 1e-30)
            return math.exp(math.log(alphas[i]+1e-30) + t * (math.log(alphas[i+1]+1e-30) - math.log(alphas[i]+1e-30)))
    return None


def fit_exponent(h_arr, a_arr):
    """Fit log α* = β log h + c. Returns (β, β_std, R²)."""
    lh = np.log(h_arr)
    la = np.log(a_arr)
    A = np.vstack([lh, np.ones_like(lh)]).T
    (beta, c), *_ = np.linalg.lstsq(A, la, rcond=None)
    pred = A @ np.array([beta, c])
    ss_res = np.sum((la - pred)**2)
    ss_tot = np.sum((la - la.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    resid = la - pred
    se = np.sqrt(np.sum(resid**2) / max(len(lh)-2, 1))
    beta_std = se / np.sqrt(np.sum((lh - lh.mean())**2))
    return float(beta), float(beta_std), float(r2)


def plot_dataset(rows, dataset_name, h_values, out_dir):
    by = group(rows, dataset_name)
    h_vals = sorted(h for h in by if h in h_values)
    if not h_vals:
        print(f"  {dataset_name}: no data")
        return None

    colors = cm.viridis(np.linspace(0, 1, len(h_vals)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{dataset_name} — Test Error Collapse', fontsize=13, fontweight='bold')

    alpha_stars = {}
    for ci, h in enumerate(h_vals):
        data = by[h]
        aps = sorted(data)
        avgs = [np.mean(data[a]) for a in aps]
        stds = [np.std(data[a]) for a in aps]
        codes = [a * math.sqrt(h) for a in aps]
        c = colors[ci]
        lbl = f'h={h}'

        axes[0].loglog(aps, avgs, 'o-', color=c, label=lbl, lw=2, ms=4)
        axes[0].fill_between(aps,
                             np.array(avgs)-np.array(stds),
                             np.array(avgs)+np.array(stds),
                             alpha=0.15, color=c)
        axes[1].loglog(codes, avgs, 'o-', color=c, label=lbl, lw=2, ms=4)
        axes[1].fill_between(codes,
                             np.array(avgs)-np.array(stds),
                             np.array(avgs)+np.array(stds),
                             alpha=0.15, color=c)

        astar = find_boundary(aps, avgs)
        if astar:
            alpha_stars[h] = astar

    for ax, xlabel, title in [
        (axes[0], 'α  (= α_code / √h)', 'vs α_paper  (should be horizontally offset)'),
        (axes[1], 'α√h  (= α_code)',    'vs α√h  (should COLLAPSE if β = −½)')
    ]:
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Test Error', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / f'{dataset_name}_collapse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {dataset_name}_collapse.png")

    if len(alpha_stars) < 3:
        print(f"  {dataset_name}: only {len(alpha_stars)} boundary points — skipping exponent fit")
        return None

    h_arr = np.array(sorted(alpha_stars))
    a_arr = np.array([alpha_stars[h] for h in h_arr])
    beta, beta_std, r2 = fit_exponent(h_arr, a_arr)
    dev = abs(beta - (-0.5)) / (beta_std + 1e-9)
    print(f"  {dataset_name}: β = {beta:.4f} ± {beta_std:.4f}  R²={r2:.4f}  "
          f"deviation from -0.5: {dev:.1f}σ")
    return {'dataset': dataset_name, 'beta': beta, 'beta_std': beta_std,
            'r2': r2, 'deviation_sigma': dev,
            'h_vals': h_arr.tolist(), 'alpha_stars': a_arr.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default=str(Path(__file__).parent / 'runs' / 'exp_h_range'))
    args = ap.parse_args()

    rows = load(args.runs_dir)
    print(f"Loaded {len(rows)} runs")
    out = Path(__file__).parent / 'results' / 'exp_h_range'
    out.mkdir(exist_ok=True)

    H_VALUES = [10, 30, 100, 300, 1000, 3000, 10000]
    present_datasets = sorted(set(r.get('dataset') for r in rows if r.get('dataset')))
    print(f"Datasets found: {present_datasets}\n")

    exponent_results = []
    for ds_name in present_datasets:
        result = plot_dataset(rows, ds_name, H_VALUES, out)
        if result:
            exponent_results.append(result)

    if not exponent_results:
        print("No exponent results to summarise.")
        return

    # ── Cross-dataset summary plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r['dataset'] for r in exponent_results]
    betas = [r['beta'] for r in exponent_results]
    errs  = [r['beta_std'] for r in exponent_results]
    x = np.arange(len(names))
    ax.bar(x, betas, yerr=errs, capsize=4, color='steelblue', alpha=0.8)
    ax.axhline(-0.5, color='red', ls='--', lw=2, label='Paper claim: β = −0.5')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Fitted scaling exponent β', fontsize=11)
    ax.set_title('Scaling Exponent β per Dataset\n(α* ∝ h^β; paper claims β = −0.5)', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'exponent_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ exponent_comparison.png")

    json.dump(exponent_results, open(out / 'exponents.json', 'w'), indent=2)
    print("✓ exponents.json\n")

    print("=" * 60)
    print(f"{'Dataset':<20} {'β':>8} {'±σ':>8} {'R²':>6}  {'|dev|/σ':>8}")
    print("-" * 60)
    for r in exponent_results:
        flag = " ← SUSPECT" if r['deviation_sigma'] > 2 else ""
        print(f"{r['dataset']:<20} {r['beta']:>8.4f} {r['beta_std']:>8.4f} "
              f"{r['r2']:>6.4f}  {r['deviation_sigma']:>8.1f}{flag}")


if __name__ == '__main__':
    main()
