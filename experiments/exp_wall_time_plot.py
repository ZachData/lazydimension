#!/usr/bin/env python3
"""
ANALYSIS: exp_wall_time — multi-dataset timeout impact.

Per dataset: scatter of baseline vs extended test error for sensitive pairs.
Cross-dataset: heatmap of timeout rate at each (h, α) combination.
              Bar chart of mean Δ test error per dataset for timed-out runs.

Usage:
    python plot.py [--runs-dir runs] [--baseline-dir ../../runs]
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
    p = Path(directory)
    if not p.exists():
        return rows
    for f in p.glob('*.json'):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default=str(Path(__file__).parent / 'runs' / 'exp_wall_time'))
    ap.add_argument('--baseline-dir', default=str(Path(__file__).parent / 'runs' / 'baseline'),
                    help='Baseline run_experiments.py runs/ for comparison')
    args = ap.parse_args()

    extended = load(args.runs_dir)
    baseline = load(args.baseline_dir)
    print(f"Extended: {len(extended)} | Baseline: {len(baseline)}")

    out = Path(__file__).parent / 'results' / 'exp_wall_time'
    out.mkdir(exist_ok=True)

    # Key: (dataset, h, log10_alpha_rounded, seed)
    def make_key(r):
        ds = r.get('dataset', 'fashion')
        return (ds, r['h'], round(math.log10(r['alpha']+1e-30), 1), r.get('seed_init', 0))

    baseline_lut = {make_key(r): r for r in baseline}

    # ── 1. Timeout audit by dataset ──────────────────────────────────────────
    print(f"\n{'Dataset':<20} {'runs':>6} {'timed_out':>10} {'%':>5}")
    print("-" * 45)
    for ds in DATASETS:
        name = ds['name']
        ds_base = [r for r in baseline if r.get('dataset', 'fashion') == name]
        n_to = sum(1 for r in ds_base if r.get('wall_time', r.get('final_wall', 0)) >= 590)
        pct = 100 * n_to / len(ds_base) if ds_base else 0
        print(f"{name:<20} {len(ds_base):>6} {n_to:>10} {pct:>4.1f}%")

    # ── 2. Per-dataset scatter: baseline vs extended ─────────────────────────
    datasets_present = sorted(set(r.get('dataset', 'fashion') for r in extended))
    n_ds = len(datasets_present)
    if n_ds == 0:
        print("No extended runs found.")
        return

    cols = min(3, n_ds)
    rows_grid = math.ceil(n_ds / cols)
    fig, axes = plt.subplots(rows_grid, cols, figsize=(5*cols, 4*rows_grid))
    axes = np.array(axes).flatten() if n_ds > 1 else [axes]

    delta_by_dataset = {}
    for di, ds_name in enumerate(datasets_present):
        ax = axes[di]
        ext_ds = [r for r in extended if r.get('dataset', 'fashion') == ds_name]
        pairs_to = []  # (baseline_err, extended_err) for timed-out baseline
        pairs_ok = []  # same for converged baseline

        for r in ext_ds:
            b = baseline_lut.get(make_key(r))
            if b is None:
                continue
            be = b.get('final_test_err')
            ee = r.get('final_test_err')
            if be is None or ee is None:
                continue
            was_to = b.get('wall_time', b.get('final_wall', 0)) >= 590
            (pairs_to if was_to else pairs_ok).append((be, ee))

        if pairs_to:
            ax.scatter(*zip(*pairs_to), c='red', alpha=0.7, s=20, label='Was timed out', zorder=3)
        if pairs_ok:
            ax.scatter(*zip(*pairs_ok), c='cornflowerblue', alpha=0.4, s=10, label='Converged', zorder=2)

        lim = [0, 0.5]
        ax.plot(lim, lim, 'k--', lw=0.8)
        ax.set_xlim(0, 0.5); ax.set_ylim(0, 0.5)
        ax.set_xlabel('Baseline (600s)', fontsize=9)
        ax.set_ylabel('Extended (3600s)', fontsize=9)
        ax.set_title(ds_name, fontsize=10, fontweight='bold')
        if pairs_to or pairs_ok:
            ax.legend(fontsize=7)

        diffs = [be - ee for be, ee in pairs_to if be is not None and ee is not None]
        delta_by_dataset[ds_name] = {'mean_delta': np.mean(diffs) if diffs else 0,
                                      'n_timed_out': len(pairs_to),
                                      'n_improved': sum(1 for d in diffs if d > 0.001)}

    for ax in axes[n_ds:]:
        ax.set_visible(False)
    plt.suptitle('Test Error: 600s vs 3600s — All Datasets', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / 'scatter_all_datasets.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ scatter_all_datasets.png")

    # ── 3. Cross-dataset mean Δ bar chart ─────────────────────────────────────
    ds_names = list(delta_by_dataset.keys())
    deltas   = [delta_by_dataset[d]['mean_delta'] for d in ds_names]
    nto      = [delta_by_dataset[d]['n_timed_out'] for d in ds_names]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    ax.bar(ds_names, deltas, color=colors, alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Mean Δ test error\n(baseline − extended, positive = baseline overestimated)', fontsize=10)
    ax.set_title('Effect of 600s→3600s Timeout Extension per Dataset\n'
                 '(for previously-timed-out runs only)', fontsize=11)
    ax.set_xticklabels(ds_names, rotation=20, ha='right', fontsize=9)
    for i, (d, n) in enumerate(zip(deltas, nto)):
        ax.text(i, d + 0.0005, f'n={n}', ha='center', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'delta_by_dataset.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ delta_by_dataset.png")

    # ── 4. Summary ───────────────────────────────────────────────────────────
    summary = {
        'n_extended': len(extended),
        'n_baseline': len(baseline),
        'per_dataset': delta_by_dataset,
    }
    json.dump(summary, open(out / 'wall_time_summary.json', 'w'), indent=2)
    print("✓ wall_time_summary.json\n")

    print(f"{'Dataset':<20} {'n_timed_out':>12} {'mean_Δ':>8} {'n_improved':>12}")
    print("-" * 58)
    for ds_name, d in delta_by_dataset.items():
        flag = " ← SIGNIFICANT" if abs(d['mean_delta']) > 0.005 else ""
        print(f"{ds_name:<20} {d['n_timed_out']:>12} {d['mean_delta']:>8.4f} "
              f"{d['n_improved']:>12}{flag}")


if __name__ == '__main__':
    main()
