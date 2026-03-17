#!/usr/bin/env python3
"""
Plot ablation results. Loads runs_ablations/ and runs_fixed/ (as baseline reference).

Usage:
    python3 plot_ablations.py
    python3 plot_ablations.py --ablation-dir runs_ablations --baseline-dir runs_fixed
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

COLORS = {100: 'purple', 300: 'teal', 1000: 'orange'}


def load_ablation_results(directory):
    """Load ablation JSONs, grouped by test name."""
    by_test = defaultdict(list)
    for f in Path(directory).glob('*.json'):
        try:
            with open(f) as fh:
                d = json.load(fh)
                name = d.get('test', f.stem.split('_h')[0])
                by_test[name].append(d)
        except Exception:
            pass
    return dict(by_test)


def load_baseline(directory):
    """Load baseline (runs_fixed) results."""
    results = []
    for f in Path(directory).glob('*.json'):
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except Exception:
            pass
    return results


def group(runs):
    """Returns {h: {alpha_code: [test_errs]}}."""
    out = defaultdict(lambda: defaultdict(list))
    for r in runs:
        err = r.get('final_test_err')
        if err is None or err > 0.5:
            continue
        out[r['h']][r['alpha']].append(err)
    return out


def plot_on_ax(ax, grouped, linestyle='-', label_suffix='', alpha_opacity=1.0):
    for h in sorted(grouped):
        alphas = sorted(grouped[h])
        if not alphas:
            continue
        means = [np.mean(grouped[h][a]) for a in alphas]
        color = COLORS.get(h, 'gray')
        ax.loglog(alphas, means, 'o' + linestyle, color=color,
                  label=f'h={h}{label_suffix}', linewidth=2, markersize=4,
                  alpha=alpha_opacity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation-dir', default='runs_ablations')
    parser.add_argument('--baseline-dir', default='runs_fixed')
    args = parser.parse_args()

    ablations = load_ablation_results(args.ablation_dir)
    baseline_runs = load_baseline(args.baseline_dir)
    baseline_grouped = group(baseline_runs) if baseline_runs else None

    test_names = [t for t in ['extra_seed', 'relu', 'mnist'] if t in ablations]
    if not test_names:
        print(f"No ablation results found in {args.ablation_dir}/")
        return

    fig, axes = plt.subplots(1, len(test_names), figsize=(6 * len(test_names), 5),
                             squeeze=False)

    for idx, name in enumerate(test_names):
        ax = axes[0][idx]

        if baseline_grouped:
            plot_on_ax(ax, baseline_grouped, linestyle='--',
                       label_suffix=' (baseline)', alpha_opacity=0.4)

        test_grouped = group(ablations[name])
        plot_on_ax(ax, test_grouped)

        ax.set_xlabel('alpha_code', fontsize=10)
        ax.set_ylabel('Test Error', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    out = 'results/plot_ablations.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved {out}")

    # Numerical table
    for name in test_names:
        runs = ablations[name]
        grouped = group(runs)
        print(f"\n{'='*50}")
        print(f"{name} ({len(runs)} runs)")
        print(f"{'='*50}")
        print(f"{'h':>6} {'alpha':>12} {'test_err':>10} {'n':>4}")
        for h in sorted(grouped):
            for a in sorted(grouped[h]):
                vals = grouped[h][a]
                print(f"{h:>6} {a:>12.2e} {np.mean(vals):>10.4f} {len(vals):>4}")


if __name__ == '__main__':
    main()
