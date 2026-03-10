#!/usr/bin/env python3
"""
ANALYSIS: exp_augmentations

Four outputs:
  1. DOUBLE SANITY CHECK — identity × odd_even vs unaugmented baseline
  2. β PER AUGMENTATION (collapsed over splits) — does augmentation change the exponent?
  3. β PER (AUGMENTATION, SPLIT) for Fashion-MNIST — full heatmap
  4. TASK DIFFICULTY vs augmentation — how does baseline test error shift?

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
from augmentations import AUGMENTATIONS_FOR_SPATIAL, SANITY_AUG
from splits import SANITY_CHECK_SPLIT


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
            la = (math.log(alphas[i]+1e-30)
                  + t * (math.log(alphas[i+1]+1e-30) - math.log(alphas[i]+1e-30)))
            return math.exp(la)
    return None


def fit_beta(h_vals, a_stars):
    lh = np.log(np.array(h_vals))
    la = np.log(np.array(a_stars))
    A = np.vstack([lh, np.ones_like(lh)]).T
    (beta, c), *_ = np.linalg.lstsq(A, la, rcond=None)
    pred = A @ np.array([beta, c])
    ss_res = np.sum((la - pred)**2)
    ss_tot = np.sum((la - la.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    se = np.sqrt(np.sum((la - pred)**2) / max(len(lh)-2, 1))
    beta_std = se / (np.sqrt(np.sum((lh - lh.mean())**2)) + 1e-30)
    return float(beta), float(beta_std), float(r2)


def compute_beta(rows, dataset, aug, split, H_VALUES=(100, 300, 1000)):
    by_h = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r.get('dataset') != dataset or r.get('augmentation') != aug
                or r.get('split') != split):
            continue
        err = r.get('final_test_err')
        if err is None or err > 0.5:
            continue
        by_h[r['h']][r['alpha_paper']].append(err)

    alpha_stars = {}
    for h in H_VALUES:
        if h not in by_h:
            continue
        aps = sorted(by_h[h])
        avgs = [np.mean(by_h[h][a]) for a in aps]
        astar = find_boundary(aps, avgs)
        if astar:
            alpha_stars[h] = astar

    if len(alpha_stars) < 3:
        return None
    h_arr = sorted(alpha_stars)
    a_arr = [alpha_stars[h] for h in h_arr]
    beta, beta_std, r2 = fit_beta(h_arr, a_arr)
    dev = abs(beta - (-0.5)) / (beta_std + 1e-9)
    return {'beta': beta, 'beta_std': beta_std, 'r2': r2, 'deviation_sigma': dev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-dir', default=str(Path(__file__).parent / 'runs' / 'exp_augmentations'))
    ap.add_argument('--baseline-dir', default='../../runs')
    args = ap.parse_args()

    rows = load(args.runs_dir)
    baseline = load(args.baseline_dir)
    print(f"Loaded {len(rows)} augmentation runs, {len(baseline)} baseline runs")

    out = Path(__file__).parent / 'results' / 'exp_augmentations'
    out.mkdir(exist_ok=True)

    H_VALUES = (100, 300, 1000)

    # ── 1. Double sanity check ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"DOUBLE SANITY CHECK: aug='{SANITY_AUG}' × split='{SANITY_CHECK_SPLIT}' vs baseline")
    print("=" * 60)

    def bkey(r):
        ds = r.get('dataset', 'fashion')
        return (ds, r['h'],
                round(math.log10(r.get('alpha', 1e-30)+1e-30), 1),
                r.get('seed_init', 0))

    bl_lut = {bkey(r): r for r in baseline}
    sanity = [r for r in rows
              if r.get('augmentation') == SANITY_AUG
              and r.get('split') == SANITY_CHECK_SPLIT]

    max_diff = 0.0
    n_checked = 0
    mismatches = []
    for r in sanity:
        b = bl_lut.get(bkey(r))
        if b is None:
            continue
        be, re = b.get('final_test_err'), r.get('final_test_err')
        if be is None or re is None:
            continue
        diff = abs(be - re)
        max_diff = max(max_diff, diff)
        n_checked += 1
        if diff > 1e-6:
            mismatches.append((r['dataset'], r['h'], r.get('alpha'), r.get('seed_init'), be, re, diff))

    if n_checked == 0:
        print("  No overlap. Copy baseline runs to --baseline-dir and rerun.")
        sanity_status = 'NO_DATA'
    elif mismatches:
        print(f"  *** FAIL: {len(mismatches)} mismatches (max Δ={max_diff:.2e}) ***")
        for ds, h, alpha, seed, be, re, d in mismatches[:10]:
            print(f"    {ds} h={h} α={alpha:.2e} seed={seed}: bl={be:.5f} aug={re:.5f} Δ={d:.2e}")
        sanity_status = 'FAIL'
    else:
        print(f"  ✓ All {n_checked} pairs match baseline (max Δ={max_diff:.2e})")
        sanity_status = 'PASS'

    # ── 2. β per augmentation (Fashion-MNIST, odd_even split) ─────────────────
    print("\n" + "=" * 60)
    print("β PER AUGMENTATION  (fashion, odd_even split, H={100,300,1000})")
    print("=" * 60)

    aug_betas = {}
    for aug in AUGMENTATIONS_FOR_SPATIAL:
        res = compute_beta(rows, 'fashion', aug, SANITY_CHECK_SPLIT, H_VALUES)
        if res:
            aug_betas[aug] = res
            flag = " ← SUSPECT" if res['deviation_sigma'] > 2 else ""
            sanity_tag = " [SANITY]" if aug == SANITY_AUG else ""
            print(f"  {aug:<20} β={res['beta']:+.4f} ±{res['beta_std']:.4f}  "
                  f"R²={res['r2']:.4f}  {res['deviation_sigma']:.1f}σ{flag}{sanity_tag}")

    if aug_betas:
        fig, ax = plt.subplots(figsize=(12, 4))
        names = list(aug_betas.keys())
        betas = [aug_betas[n]['beta'] for n in names]
        errs  = [aug_betas[n]['beta_std'] for n in names]
        colors = ['gold' if n == SANITY_AUG else 'steelblue' for n in names]
        x = np.arange(len(names))
        ax.bar(x, betas, yerr=errs, capsize=3, color=colors, alpha=0.85)
        ax.axhline(-0.5, color='red', ls='--', lw=2, label='β = −0.5')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Fitted β')
        ax.set_title('Scaling Exponent β per Augmentation\n'
                     'fashion × odd_even split  (gold = identity/sanity)', fontsize=11)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'beta_per_augmentation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ beta_per_augmentation.png")

    # ── 3. β heatmap: augmentation × split (Fashion-MNIST) ───────────────────
    from splits import SPLITS_BY_DATASET
    fashion_splits = list(SPLITS_BY_DATASET.get('fashion', {}).keys())
    augs_with_data = [a for a in AUGMENTATIONS_FOR_SPATIAL if a in aug_betas or True]

    heat_betas = np.full((len(AUGMENTATIONS_FOR_SPATIAL), len(fashion_splits)), float('nan'))
    for ai, aug in enumerate(AUGMENTATIONS_FOR_SPATIAL):
        for si, split in enumerate(fashion_splits):
            res = compute_beta(rows, 'fashion', aug, split, H_VALUES)
            if res:
                heat_betas[ai, si] = res['beta']

    if not np.all(np.isnan(heat_betas)):
        fig, ax = plt.subplots(figsize=(max(8, len(fashion_splits)*1.2),
                                        max(5, len(AUGMENTATIONS_FOR_SPATIAL)*0.5)))
        im = ax.imshow(heat_betas, aspect='auto', cmap='RdYlGn',
                       vmin=-0.7, vmax=-0.3)
        ax.set_xticks(range(len(fashion_splits)))
        ax.set_xticklabels(fashion_splits, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(AUGMENTATIONS_FOR_SPATIAL)))
        ax.set_yticklabels(AUGMENTATIONS_FOR_SPATIAL, fontsize=8)
        plt.colorbar(im, ax=ax, label='β (target: -0.5)')
        ax.set_title('β heatmap: augmentation × split  (fashion)\n'
                     'Green = -0.5 (good), Red = deviates from -0.5', fontsize=11)
        # Annotate cells
        for ai in range(len(AUGMENTATIONS_FOR_SPATIAL)):
            for si in range(len(fashion_splits)):
                v = heat_betas[ai, si]
                if not math.isnan(v):
                    ax.text(si, ai, f'{v:.2f}', ha='center', va='center',
                            fontsize=6, color='black')
        plt.tight_layout()
        plt.savefig(out / 'beta_heatmap_fashion.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ beta_heatmap_fashion.png")

    # ── 4. Task difficulty per augmentation (mean test error at large α) ──────
    # "Large α" = NTK/lazy regime where test error reflects task difficulty
    LARGE_ALPHA_THRESHOLD = 100.0  # alpha_paper

    diff_by_aug = {}
    for aug in AUGMENTATIONS_FOR_SPATIAL:
        high_alpha_errs = [
            r['final_test_err'] for r in rows
            if r.get('dataset') == 'fashion'
            and r.get('augmentation') == aug
            and r.get('split') == SANITY_CHECK_SPLIT
            and r.get('alpha_paper', 0) >= LARGE_ALPHA_THRESHOLD
            and r.get('final_test_err') is not None
        ]
        if high_alpha_errs:
            diff_by_aug[aug] = np.mean(high_alpha_errs)

    if diff_by_aug:
        fig, ax = plt.subplots(figsize=(12, 4))
        names = sorted(diff_by_aug, key=lambda a: diff_by_aug[a])
        vals  = [diff_by_aug[n] for n in names]
        colors = ['gold' if n == SANITY_AUG else 'salmon' for n in names]
        x = np.arange(len(names))
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Mean test error at α_paper ≥ 100 (NTK regime)')
        ax.set_title('Task Difficulty by Augmentation\n'
                     '(fashion, odd_even split; higher = harder task in lazy regime)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'task_difficulty_by_aug.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ task_difficulty_by_aug.png")

    # ── 5. Save summary JSON ──────────────────────────────────────────────────
    summary = {
        'sanity_check': {'status': sanity_status, 'n_checked': n_checked,
                         'max_diff': max_diff, 'n_mismatches': len(mismatches)},
        'beta_per_aug_fashion_odd_even': {k: v for k, v in aug_betas.items()},
        'task_difficulty_by_aug': diff_by_aug,
    }
    json.dump(summary, open(out / 'augmentation_summary.json', 'w'), indent=2)
    print("✓ augmentation_summary.json")


if __name__ == '__main__':
    main()
