#!/usr/bin/env python3
"""
Compare original (sabotaged, with momentum) vs fixed (pure gradient flow) results.

Produces a 2x2 grid:
  Top row:    original results (tau_over_h=1e-3)
  Bottom row: fixed results    (tau_over_h=0.0)
  Left col:   test error vs alpha_paper  (= alpha_code / sqrt(h))
  Right col:  test error vs alpha_code   (= alpha_paper * sqrt(h))

If only one set of results exists, plots a single 1x2 row.

Usage:
    python3 plot_comparison.py
    python3 plot_comparison.py --original-dir runs --fixed-dir runs_fixed
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(directory):
    """Load all results from JSON files in the directory."""
    results = []
    dirpath = Path(directory)
    if not dirpath.exists():
        return results
    for json_file in dirpath.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Error loading {json_file}: {e}")
    return results


def group_by_h(results):
    """Group results by h, then by alpha_paper, returning sorted averages."""
    h_values = sorted(set(r['h'] for r in results))
    grouped = {}
    for h in h_values:
        h_data = defaultdict(list)
        for r in results:
            if r['h'] != h:
                continue
            test_err = r.get('final_test_err')
            if test_err is None or test_err > 0.5:
                continue
            alpha_code = r['alpha']
            alpha_paper = alpha_code / np.sqrt(h)
            h_data[alpha_paper].append(test_err)

        alphas = sorted(h_data.keys())
        if not alphas:
            continue
        grouped[h] = {
            'alpha_paper': np.array(alphas),
            'alpha_code': np.array([a * np.sqrt(h) for a in alphas]),
            'mean': np.array([np.mean(h_data[a]) for a in alphas]),
            'std': np.array([np.std(h_data[a]) for a in alphas]),
        }
    return grouped


COLORS = {100: 'purple', 300: 'teal', 1000: 'orange'}


def plot_row(axes, grouped, row_label):
    """Plot one row: left = vs alpha_paper, right = vs alpha_code."""
    ax_left, ax_right = axes

    for h, data in sorted(grouped.items()):
        color = COLORS.get(h, 'gray')
        ax_left.loglog(data['alpha_paper'], data['mean'], 'o-',
                       color=color, label=f'h={h}', linewidth=2, markersize=5)
        ax_left.fill_between(data['alpha_paper'],
                             data['mean'] - data['std'],
                             data['mean'] + data['std'],
                             alpha=0.15, color=color)

        ax_right.loglog(data['alpha_code'], data['mean'], 'o-',
                        color=color, label=f'h={h}', linewidth=2, markersize=5)
        ax_right.fill_between(data['alpha_code'],
                              data['mean'] - data['std'],
                              data['mean'] + data['std'],
                              alpha=0.15, color=color)

    ax_left.set_ylabel('Test Error', fontsize=11)
    ax_left.set_title(f'{row_label}: Test Error vs α', fontsize=11, fontweight='bold')
    ax_left.legend(fontsize=9)
    ax_left.grid(True, alpha=0.3)

    ax_right.set_title(f'{row_label}: Test Error vs α√h', fontsize=11, fontweight='bold')
    ax_right.legend(fontsize=9)
    ax_right.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-dir', default='runs',
                        help='Directory with original (momentum) results')
    parser.add_argument('--fixed-dir', default='runs_fixed',
                        help='Directory with fixed (gradient flow) results')
    args = parser.parse_args()

    original = load_results(args.original_dir)
    fixed = load_results(args.fixed_dir)

    have_original = len(original) > 0
    have_fixed = len(fixed) > 0

    if not have_original and not have_fixed:
        print("No results found in either directory.")
        return

    nrows = have_original + have_fixed

    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]  # keep 2D indexing

    row = 0
    if have_original:
        grouped = group_by_h(original)
        plot_row(axes[row], grouped,
                 f'Original (τ/h=1e-3, {len(original)} runs)')
        row += 1

    if have_fixed:
        grouped = group_by_h(fixed)
        plot_row(axes[row], grouped,
                 f'Fixed (τ/h=0, pure gradient flow, {len(fixed)} runs)')

    # shared x-labels on bottom row
    axes[-1, 0].set_xlabel('α  (= alpha_code / √h)', fontsize=11)
    axes[-1, 1].set_xlabel('α√h  (= alpha_code)', fontsize=11)

    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    out = 'results/plot_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")

    # Print numerical summary
    for label, data in [('Original', original), ('Fixed', fixed)]:
        if not data:
            continue
        grouped = group_by_h(data)
        print(f"\n{'='*70}")
        print(f"{label} results")
        print(f"{'='*70}")
        print(f"{'h':>6} {'alpha_code':>12} {'alpha_paper':>12} {'test_err':>10} {'±std':>8}")
        for h in sorted(grouped):
            g = grouped[h]
            for i in range(len(g['alpha_paper'])):
                print(f"{h:>6} {g['alpha_code'][i]:>12.2e} {g['alpha_paper'][i]:>12.2e} "
                      f"{g['mean'][i]:>10.4f} {g['std'][i]:>8.4f}")


if __name__ == '__main__':
    main()
