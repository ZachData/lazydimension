#!/usr/bin/env python3
"""
ANALYSIS: exp_binary_split

Three outputs:
1. SANITY CHECK — 'odd_even' vs baseline
   For every (dataset, h, alpha, seed) pair where both an odd_even result
   and a baseline result exist, compare test errors. They must be identical
   (within float rounding). Any divergence is a bug in the split injection.

2. PER-DATASET SPLIT COMPARISON
   For each dataset, plot test error vs α√h for all named splits.
   Fits β per split and reports the table.

3. CROSS-SPLIT EXPONENT SUMMARY
   Bar chart: β per (dataset, split). Should be flat at -0.5 everywhere.

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
from splits import SPLITS_BY_DATASET, SANITY_CHECK_SPLIT


def load(directory):
    rows = []
    for f in Path(directory).glob('*.json'):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    return rows


def find_boundary(alphas, errs, frac=0.5):
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
            la = math.log(alphas[i]+1e-30) + t * (math.log(alphas[i+1]+1e-30) - math.log(alphas[i]+1e-30))
            return math.exp(la)
    return None


def fit_beta(h_vals, a_stars):
    lh = np.log(h_vals)
    la = np.log(a_stars)
    A = np.vstack([lh, np.ones_like(lh)]).T
    (beta, c), *_ = np.linalg.lstsq(A, la, rcond=None)
    pred = A @ np.array([beta, c])
    ss_res = np.sum((la - pred)**2)
    ss_tot = np.sum((la - la.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    se = np.sqrt(np.sum((la - pred)**2) / max(len(lh)-2, 1))
    beta_std = se / np.sqrt(np.sum((lh - lh.mean())**2))
    return float(beta), float(beta_std), float(r2)


def group_by_split_and_h(rows, dataset, split):
    """Return {h: {alpha_paper: [test_err]}} for one (dataset, split)."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get('dataset') != dataset or r.get('split') != split:
            continue
        err = r.get('final_test_err')
        if err is None or err > 0.5:
            continue
        by[r['h']][r['alpha_paper']].append(err)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default=str(Path(__file__).parent / 'runs' / 'exp_binary_split'))
    ap.add_argument('--baseline-dir', default=str(Path(__file__).parent / 'runs' / 'baseline'),
                    help='Baseline runs/ for odd_even sanity comparison')
    args = ap.parse_args()

    rows = load(args.runs_dir)
    baseline = load(args.baseline_dir)
    print(f"Loaded {len(rows)} split runs, {len(baseline)} baseline runs")

    out = Path(__file__).parent / 'results' / 'exp_binary_split'
    out.mkdir(exist_ok=True)

    H_VALUES = [100, 300, 1000]

    # ── 1. Sanity check: odd_even == baseline ─────────────────────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECK: odd_even split vs baseline")
    print("=" * 60)

    # Index baseline by (dataset, h, log_alpha, seed)
    def bkey(r):
        return (r.get('dataset', 'fashion'), r['h'],
                round(math.log10(r['alpha']+1e-30), 1),
                r.get('seed_init', 0))

    baseline_lut = {bkey(r): r for r in baseline}
    sanity_rows = [r for r in rows if r.get('split') == SANITY_CHECK_SPLIT]

    max_diff = 0.0
    n_checked = 0
    mismatches = []
    for r in sanity_rows:
        b = baseline_lut.get(bkey(r))
        if b is None:
            continue
        be = b.get('final_test_err')
        re = r.get('final_test_err')
        if be is None or re is None:
            continue
        diff = abs(be - re)
        max_diff = max(max_diff, diff)
        n_checked += 1
        if diff > 1e-6:
            mismatches.append((r['dataset'], r['h'], r['alpha'], r.get('seed_init'), be, re, diff))

    if n_checked == 0:
        print("  No overlap between odd_even runs and baseline runs.")
        print("  Copy baseline runs into the baseline-dir and rerun.")
    elif mismatches:
        print(f"  *** FAILURES: {len(mismatches)} mismatches (max Δ={max_diff:.6f}) ***")
        for ds, h, alpha, seed, be, re, diff in mismatches[:10]:
            print(f"    {ds} h={h} α={alpha:.2e} seed={seed}: baseline={be:.5f} odd_even={re:.5f} Δ={diff:.2e}")
    else:
        print(f"  ✓ All {n_checked} odd_even results match baseline (max Δ={max_diff:.2e})")

    sanity_result = {
        'n_checked': n_checked,
        'n_mismatches': len(mismatches),
        'max_diff': max_diff,
        'status': 'PASS' if n_checked > 0 and not mismatches else ('FAIL' if mismatches else 'NO_DATA'),
    }

    # ── 2. Per-dataset collapse + exponent per split ──────────────────────────
    print("\n" + "=" * 60)
    print("EXPONENT FITS PER (DATASET, SPLIT)")
    print("=" * 60)
    print(f"{'Dataset':<20} {'Split':<25} {'β':>8} {'±σ':>8} {'R²':>6}  {'|dev|/σ':>8}")
    print("-" * 80)

    all_results = []

    for ds in DATASETS:
        ds_name = ds['name']
        splits_for_ds = list(SPLITS_BY_DATASET.get(ds_name, {}).keys())
        if not splits_for_ds:
            continue

        # One collapse figure per dataset with all splits overlaid
        n_splits = len(splits_for_ds)
        colors = cm.tab10(np.linspace(0, 1, n_splits))
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        fig.suptitle(f'{ds_name} — collapse under α√h per split', fontsize=12, fontweight='bold')

        for si, split_name in enumerate(splits_for_ds):
            by = group_by_split_and_h(rows, ds_name, split_name)
            h_vals_present = [h for h in H_VALUES if h in by and by[h]]
            if not h_vals_present:
                continue

            # For collapse plot: aggregate over h, plot vs alpha_code
            all_alpha_codes = defaultdict(list)
            for h in h_vals_present:
                for ap, errs in by[h].items():
                    code = ap * math.sqrt(h)
                    all_alpha_codes[round(math.log10(code+1e-30), 2)].append(np.mean(errs))

            codes_sorted = sorted(10**lc for lc in sorted(all_alpha_codes.keys()))
            avgs = [np.mean(all_alpha_codes[round(math.log10(c+1e-30), 2)]) for c in codes_sorted]
            ls = '--' if split_name == SANITY_CHECK_SPLIT else '-'
            ax.loglog(codes_sorted, avgs, marker='o', ms=3, lw=1.5,
                      color=colors[si], label=split_name, ls=ls)

            # Fit exponent
            alpha_stars = {}
            for h in h_vals_present:
                aps = sorted(by[h].keys())
                avg_errs = [np.mean(by[h][a]) for a in aps]
                astar = find_boundary(aps, avg_errs)
                if astar:
                    alpha_stars[h] = astar

            if len(alpha_stars) >= 3:
                h_arr = np.array(sorted(alpha_stars))
                a_arr = np.array([alpha_stars[h] for h in h_arr])
                beta, beta_std, r2 = fit_beta(h_arr, a_arr)
                dev = abs(beta - (-0.5)) / (beta_std + 1e-9)
                flag = " ← SUSPECT" if dev > 2 else ""
                sanity_tag = " [SANITY]" if split_name == SANITY_CHECK_SPLIT else ""
                print(f"{ds_name:<20} {split_name:<25} {beta:>8.4f} {beta_std:>8.4f} "
                      f"{r2:>6.4f}  {dev:>8.1f}{flag}{sanity_tag}")
                all_results.append({
                    'dataset': ds_name, 'split': split_name,
                    'is_sanity': split_name == SANITY_CHECK_SPLIT,
                    'beta': beta, 'beta_std': beta_std, 'r2': r2,
                    'deviation_sigma': dev,
                })
            else:
                print(f"{ds_name:<20} {split_name:<25} {'—':>8} (insufficient data)")

        ax.set_xlabel('α√h', fontsize=11)
        ax.set_ylabel('Test Error', fontsize=11)
        ax.set_title(f'Collapse under α√h — all splits\n(dashed = sanity check / odd_even = baseline)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f'{ds_name}_splits_collapse.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {ds_name}_splits_collapse.png")

    # ── 3. Summary bar chart ──────────────────────────────────────────────────
    if all_results:
        labels = [f"{r['dataset']}\n{r['split']}" for r in all_results]
        betas  = [r['beta'] for r in all_results]
        errs   = [r['beta_std'] for r in all_results]
        colors = ['gold' if r['is_sanity'] else 'steelblue' for r in all_results]

        fig, ax = plt.subplots(figsize=(max(12, len(labels)*0.6), 5))
        x = np.arange(len(labels))
        ax.bar(x, betas, yerr=errs, capsize=3, color=colors, alpha=0.85)
        ax.axhline(-0.5, color='red', ls='--', lw=2, label='Paper claim: β = −0.5')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Fitted β', fontsize=11)
        ax.set_title('Scaling Exponent β per (Dataset, Split)\n'
                     'Gold bars = odd_even (sanity check, must match baseline)', fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'exponent_all_splits.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ exponent_all_splits.png")

    # ── 4. Save JSON ──────────────────────────────────────────────────────────
    summary = {'sanity_check': sanity_result, 'exponent_fits': all_results}
    json.dump(summary, open(out / 'split_exponents.json', 'w'), indent=2)
    print("✓ split_exponents.json")


if __name__ == '__main__':
    main()
