#!/usr/bin/env python3
"""
ANALYSIS: exp_seeds — multi-dataset seed count effect.

Per dataset:
  Phase A: distribution of test error at transition point across 30 seeds.
           Reports mean, std, SEM(n=3), SEM(n=10), SEM(n=30).
  Phase B: collapse spread at n=3 vs n=10.

Cross-dataset:
  results/variance_by_dataset.png  — bar chart of σ at transition per dataset
  results/spread_vs_seeds.png      — collapse spread vs n_seeds per dataset
  results/seeds_summary.json       — all numerical results
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


def collapse_spread(by_h_alpha, h_values, target_codes):
    """Std of mean test errors across h at each α√h value (lower = better collapse)."""
    spreads = []
    for code in target_codes:
        pts = []
        for h in h_values:
            ap = code / math.sqrt(h)
            bucket = by_h_alpha[h]
            closest = min(bucket, key=lambda a: abs(math.log(a+1e-30) - math.log(ap+1e-30)), default=None)
            if closest and abs(math.log(closest+1e-30) - math.log(ap+1e-30)) < 0.3:
                pts.append(np.mean(bucket[closest]))
        if len(pts) >= 2:
            spreads.append(np.std(pts))
    return float(np.mean(spreads)) if spreads else float('nan')


def analyse_phase_a(rows, dataset):
    errs = [r['final_test_err'] for r in rows
            if r.get('phase') == 'A' and r.get('dataset') == dataset
            and r.get('final_test_err') is not None]
    if not errs:
        return None
    return {
        'dataset': dataset, 'n': len(errs),
        'mean': float(np.mean(errs)),
        'std': float(np.std(errs)),
        'sem_3': float(np.std(errs) / math.sqrt(3)),
        'sem_10': float(np.std(errs) / math.sqrt(10)),
        'sem_30': float(np.std(errs) / math.sqrt(30)),
        'min': float(min(errs)), 'max': float(max(errs)),
    }


def analyse_phase_b(rows, dataset, n_seeds_list=(3, 5, 10)):
    phase_b = [r for r in rows
               if r.get('phase') == 'B' and r.get('dataset') == dataset]
    if not phase_b:
        return None
    all_seeds = sorted(set(r['seed_init'] for r in phase_b))
    H_VALUES = [100, 300, 1000]
    TARGET_CODES = [1e-3, 1e-2, 0.1, 1.0, 10, 100, 1000]

    spreads = {}
    for n in n_seeds_list:
        use = set(all_seeds[:n])
        by = defaultdict(lambda: defaultdict(list))
        for r in phase_b:
            if r['seed_init'] in use and r.get('final_test_err', 1) <= 0.5:
                by[r['h']][r['alpha_paper']].append(r['final_test_err'])
        spreads[n] = collapse_spread(by, H_VALUES, TARGET_CODES)
    return {'dataset': dataset, 'spreads': spreads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default=str(Path(__file__).parent / 'runs' / 'exp_seeds'))
    args = ap.parse_args()

    rows = load(args.runs_dir)
    print(f"Loaded {len(rows)} runs")
    out = Path(__file__).parent / 'results' / 'exp_seeds'
    out.mkdir(exist_ok=True)

    datasets = sorted(set(r.get('dataset') for r in rows if r.get('dataset')))

    # ── Phase A: variance at transition ──────────────────────────────────────
    phase_a_results = [r for ds in datasets if (r := analyse_phase_a(rows, ds))]

    if phase_a_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Phase A: Test Error Distribution at Transition Point (h=300, α=1.0)',
                     fontsize=12, fontweight='bold')

        names = [r['dataset'] for r in phase_a_results]
        stds  = [r['std']  for r in phase_a_results]
        means = [r['mean'] for r in phase_a_results]
        x = np.arange(len(names))

        axes[0].bar(x, stds, color='steelblue', alpha=0.8)
        axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=25, ha='right', fontsize=8)
        axes[0].set_ylabel('Std of test error across 30 seeds')
        axes[0].set_title('Initialization variance at transition point')
        axes[0].grid(axis='y', alpha=0.3)

        n_seeds_range = list(range(2, 31))
        colors = cm.tab10(np.linspace(0, 1, len(phase_a_results)))
        for ri, r in enumerate(phase_a_results):
            sems = [r['std'] / math.sqrt(n) for n in n_seeds_range]
            axes[1].plot(n_seeds_range, sems, 'o-', color=colors[ri],
                         label=r['dataset'], lw=1.5, ms=3)
        axes[1].axvline(3,  color='red',    ls='--', lw=1.5, label='n=3 (baseline)')
        axes[1].axvline(10, color='orange', ls='--', lw=1.5, label='n=10')
        axes[1].set_xlabel('Number of seeds'); axes[1].set_ylabel('SEM of mean test error')
        axes[1].set_title('SEM vs seed count per dataset')
        axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out / 'variance_by_dataset.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ variance_by_dataset.png")

        print(f"\n{'Dataset':<20} {'n':>4} {'mean':>7} {'std':>7}  "
              f"{'SEM(3)':>8} {'SEM(10)':>8} {'SEM(30)':>8}")
        print("-" * 75)
        for r in phase_a_results:
            print(f"{r['dataset']:<20} {r['n']:>4} {r['mean']:>7.4f} {r['std']:>7.4f}  "
                  f"{r['sem_3']:>8.4f} {r['sem_10']:>8.4f} {r['sem_30']:>8.4f}")

    # ── Phase B: collapse spread vs n_seeds ──────────────────────────────────
    n_seeds_list = (3, 5, 10)
    phase_b_results = [r for ds in datasets if (r := analyse_phase_b(rows, ds, n_seeds_list))]

    if phase_b_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = cm.tab10(np.linspace(0, 1, len(phase_b_results)))
        x = np.arange(len(n_seeds_list))
        w = 0.8 / len(phase_b_results)
        for ri, r in enumerate(phase_b_results):
            spreads = [r['spreads'].get(n, float('nan')) for n in n_seeds_list]
            ax.bar(x + ri*w, spreads, width=w*0.9, color=colors[ri],
                   alpha=0.8, label=r['dataset'])
        ax.set_xticks(x + w*(len(phase_b_results)-1)/2)
        ax.set_xticklabels([f'n={n}' for n in n_seeds_list])
        ax.set_ylabel('Collapse spread (lower = better)')
        ax.set_title('Collapse Spread vs Seed Count per Dataset\n'
                     '(std of mean test errors at fixed α√h across h values)', fontsize=11)
        ax.legend(fontsize=8, ncol=2); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'spread_vs_seeds.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ spread_vs_seeds.png")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        'phase_a': phase_a_results,
        'phase_b': phase_b_results,
    }
    json.dump(summary, open(out / 'seeds_summary.json', 'w'), indent=2)
    print("✓ seeds_summary.json")


if __name__ == '__main__':
    main()
