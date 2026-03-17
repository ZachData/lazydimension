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

    # Build baseline lookup
    baseline_lookup = {}
    if baseline_grouped:
        for h in baseline_grouped:
            for a in baseline_grouped[h]:
                baseline_lookup[(h, a)] = np.mean(baseline_grouped[h][a])

    # Structured summary
    lines = []
    s = lines.append

    s("=" * 80)
    s("ABLATION RESULTS SUMMARY")
    s("=" * 80)
    s("")
    s("Context: This codebase studies the lazy-to-feature training transition in")
    s("neural networks. The key claim is that the regime boundary scales as")
    s("α* = O(h^{-1/2}), where α is a scaling parameter and h is network width.")
    s("The baseline uses tau=0 (pure gradient flow), Swish activation, Fashion-MNIST.")
    s("")
    s("Each ablation changes ONE thing from the baseline. 'delta' is the difference")
    s("from the baseline at the same (h, alpha_code).")
    s("")

    for name in test_names:
        g = group(ablations[name])
        n_runs = len(ablations[name])
        s("-" * 80)
        s(f"TEST: {name} ({n_runs} runs)")
        s("")
        s(f"  {'h':>6} {'alpha_code':>12} {'test_err':>10} {'±std':>8} {'baseline':>10} {'delta':>8} {'n':>4}")

        for h in sorted(g):
            for a in sorted(g[h]):
                vals = g[h][a]
                mean = np.mean(vals)
                std = np.std(vals)
                bl = baseline_lookup.get((h, a))
                bl_str = f"{bl:.4f}" if bl is not None else "    n/a"
                delta_str = f"{mean - bl:+.4f}" if bl is not None else "    n/a"
                s(f"  {h:>6} {a:>12.2e} {mean:>10.4f} {std:>8.4f} {bl_str:>10} {delta_str:>8} {len(vals):>4}")
        s("")

    if baseline_grouped:
        s("-" * 80)
        s("BASELINE REFERENCE (tau=0, Swish, Fashion-MNIST)")
        s("")
        s(f"  {'h':>6} {'alpha_code':>12} {'test_err':>10} {'±std':>8} {'n':>4}")
        for h in sorted(baseline_grouped):
            for a in sorted(baseline_grouped[h]):
                vals = baseline_grouped[h][a]
                s(f"  {h:>6} {a:>12.2e} {np.mean(vals):>10.4f} {np.std(vals):>8.4f} {len(vals):>4}")
        s("")

    s("=" * 80)
    s("KEY QUESTIONS FOR REVIEW:")
    s("  1. Does extra_seed land within ±std of the baseline? (variance check)")
    s("  2. Does relu show the same transition shape? (activation robustness)")
    s("  3. Does mnist show the same transition shape? (dataset robustness)")
    s("  4. Do any deltas exceed 0.005? (potential concern)")
    s("  5. Is the qualitative pattern preserved: lower error at small alpha,")
    s("     peak near alpha_code~1, lower error at large alpha?")
    s("=" * 80)

    output = "\n".join(lines)
    print(output)

    summary_file = Path(args.ablation_dir) / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write(output + "\n")
    print(f"\nSaved to {summary_file}")


if __name__ == '__main__':
    main()
