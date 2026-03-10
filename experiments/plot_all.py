#!/usr/bin/env python3
"""
plot_all.py — Unified analysis across all fidelity experiments.

Reads the results/ JSON from each sub-experiment (produced by their
individual plot.py scripts) and synthesises everything into:

  report/summary_table.png    — β table: every condition vs paper claim
  report/beta_overview.png    — multi-panel bar chart, one panel per experiment
  report/collapse_quality.png — collapse spread and seed-count effects
  report/timeout_bias.png     — regime-boundary shift from wall-time extension
  report/VERDICT.md           — plain-English verdict per paper claim

The paper's sole key finding is:
  "The boundary between lazy and feature training regimes scales as
   α* = O(h^{-1/2}), confirmed by collapse of test error curves when
   plotted against α√h."

This decomposes into three testable sub-claims:
  C1. The scaling exponent is β = −0.5  (β in α* ∝ h^β)
  C2. Collapse quality is high  (curves actually align on a single line)
  C3. The result is not an artifact of the experimental setup
      (timeout bias, split choice, depth, augmentation)

Usage
-----
  # Run individual plot.py first, then:
  python experiments/plot_all.py

  # Point at non-default results directories:
  python experiments/plot_all.py \\
    --h-range-results   exp_h_range/results \\
    --seeds-results     exp_seeds/results \\
    --wall-results      exp_wall_time/results \\
    --split-results     exp_binary_split/results \\
    --depth-results     exp_depth/results \\
    --aug-results       exp_augmentations/results \\
    --out               report
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ── Paper reference values ────────────────────────────────────────────────────
PAPER_BETA        = -0.5          # claimed exponent
PAPER_DATASET     = 'fashion'     # baseline dataset
PAPER_H_VALUES    = [100, 300, 1000]
PAPER_L           = 3
PAPER_MAX_WALL    = 600           # seconds
CONFIRM_THRESHOLD = 2.0           # |deviation| < 2σ → consistent with claim
SUSPECT_THRESHOLD = 2.0           # |deviation| ≥ 2σ → suspect


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path):
    p = Path(path)
    if p.exists():
        try:
            return json.load(open(p))
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    return None


def _beta_status(beta, beta_std):
    """Return ('PASS'|'WARN'|'FAIL'|'NO DATA', colour, symbol)."""
    if beta is None or beta_std is None:
        return 'NO DATA', '#aaaaaa', '?'
    dev = abs(beta - PAPER_BETA) / (beta_std + 1e-9)
    if dev < CONFIRM_THRESHOLD:
        return 'PASS', '#2ecc71', '✓'
    elif dev < 4.0:
        return 'WARN', '#f39c12', '~'
    else:
        return 'FAIL', '#e74c3c', '✗'


def _fmt_beta(beta, beta_std):
    if beta is None:
        return '—'
    return f'{beta:+.3f} ± {beta_std:.3f}'


def _dev(beta, beta_std):
    if beta is None or beta_std is None:
        return float('nan')
    return abs(beta - PAPER_BETA) / (beta_std + 1e-9)


# ── Load all results ──────────────────────────────────────────────────────────

def load_all(args):
    root = Path(__file__).parent
    def rp(sub, filename):
        return root / sub / filename

    return {
        'h_range':   _load(args.h_range_results   / 'exponents.json'),
        'seeds':     _load(args.seeds_results      / 'seeds_summary.json'),
        'wall':      _load(args.wall_results       / 'wall_time_summary.json'),
        'split':     _load(args.split_results      / 'split_exponents.json'),
        'aug':       _load(args.aug_results        / 'augmentation_summary.json'),
        # depth has no plot.py yet — we read raw runs directly
        'depth_dir': args.depth_results,
    }


# ── Section builders ──────────────────────────────────────────────────────────

def section_h_range(data):
    """C1: does β = -0.5 with a proper h range?"""
    rows = []
    if data is None:
        return rows
    for r in data:
        ds = r.get('dataset', '?')
        beta, beta_std, r2 = r.get('beta'), r.get('beta_std'), r.get('r2')
        status, colour, sym = _beta_status(beta, beta_std)
        rows.append({
            'experiment': 'exp_h_range',
            'condition': f'h∈[10…10k]  {ds}',
            'dataset': ds,
            'beta': beta, 'beta_std': beta_std, 'r2': r2,
            'deviation_sigma': _dev(beta, beta_std),
            'status': status, 'colour': colour, 'symbol': sym,
            'note': f'R²={r2:.3f}' if r2 else '',
        })
    return rows


def section_seeds(data):
    """C2: did extra seeds change collapse spread?"""
    rows = []
    if data is None:
        return rows
    for r in (data.get('phase_b') or []):
        ds = r.get('dataset', '?')
        spreads = r.get('spreads', {})
        s3  = spreads.get(3)  or spreads.get('3')
        s10 = spreads.get(10) or spreads.get('10')
        if s3 is None or s10 is None:
            continue
        rel_change = (s10 - s3) / (s3 + 1e-9)
        # negative = better collapse at 10 seeds; >15% worse = suspect
        if abs(rel_change) < 0.05:
            status, colour, sym = 'PASS', '#2ecc71', '✓'
            note = f'spread unchanged n=3→10 ({rel_change:+.1%})'
        elif rel_change < -0.15:
            status, colour, sym = 'WARN', '#f39c12', '~'
            note = f'spread improved {rel_change:+.1%} — 3 seeds underestimated variance'
        else:
            status, colour, sym = 'PASS', '#2ecc71', '✓'
            note = f'spread Δ={rel_change:+.1%}'
        rows.append({
            'experiment': 'exp_seeds',
            'condition': f'n=3→10 seeds  {ds}',
            'dataset': ds,
            'beta': None, 'beta_std': None, 'r2': None,
            'deviation_sigma': float('nan'),
            'status': status, 'colour': colour, 'symbol': sym,
            'note': note,
            'spread_3': s3, 'spread_10': s10,
        })
    return rows


def section_wall(data):
    """C3a: was α* biased by the 600s timeout?"""
    rows = []
    if data is None:
        return rows
    for ds, d in (data.get('per_dataset') or {}).items():
        mean_delta = d.get('mean_delta', 0)
        n_to       = d.get('n_timed_out', 0)
        if n_to == 0:
            status, colour, sym = 'PASS', '#2ecc71', '✓'
            note = 'no timed-out runs'
        elif abs(mean_delta) < 0.002:
            status, colour, sym = 'PASS', '#2ecc71', '✓'
            note = f'{n_to} timed out; Δerr={mean_delta:+.4f} (negligible)'
        elif abs(mean_delta) < 0.005:
            status, colour, sym = 'WARN', '#f39c12', '~'
            note = f'{n_to} timed out; Δerr={mean_delta:+.4f} (marginal)'
        else:
            status, colour, sym = 'FAIL', '#e74c3c', '✗'
            note = f'{n_to} timed out; Δerr={mean_delta:+.4f} — α* biased'
        rows.append({
            'experiment': 'exp_wall_time',
            'condition': f'600→3600s  {ds}',
            'dataset': ds,
            'beta': None, 'beta_std': None, 'r2': None,
            'deviation_sigma': float('nan'),
            'status': status, 'colour': colour, 'symbol': sym,
            'note': note,
            'mean_delta': mean_delta, 'n_timed_out': n_to,
        })
    return rows


def section_split(data):
    """C3b: does β hold across split choices?"""
    rows = []
    if data is None:
        return rows
    sanity = data.get('sanity_check', {})
    san_status = sanity.get('status', 'NO_DATA')
    san_colour = '#2ecc71' if san_status == 'PASS' else ('#e74c3c' if san_status == 'FAIL' else '#aaaaaa')
    san_sym    = '✓' if san_status == 'PASS' else ('✗' if san_status == 'FAIL' else '?')
    rows.append({
        'experiment': 'exp_binary_split',
        'condition': 'sanity: odd_even == baseline',
        'dataset': PAPER_DATASET,
        'beta': None, 'beta_std': None, 'r2': None,
        'deviation_sigma': float('nan'),
        'status': san_status, 'colour': san_colour, 'symbol': san_sym,
        'note': f"max_diff={sanity.get('max_diff', '?'):.2e}  n={sanity.get('n_checked', '?')}",
    })
    for r in (data.get('exponent_fits') or []):
        ds     = r.get('dataset', '?')
        split  = r.get('split', '?')
        beta, beta_std = r.get('beta'), r.get('beta_std')
        status, colour, sym = _beta_status(beta, beta_std)
        rows.append({
            'experiment': 'exp_binary_split',
            'condition': f'{split}  {ds}',
            'dataset': ds,
            'beta': beta, 'beta_std': beta_std, 'r2': r.get('r2'),
            'deviation_sigma': _dev(beta, beta_std),
            'status': status, 'colour': colour, 'symbol': sym,
            'note': 'SANITY' if r.get('is_sanity') else '',
        })
    return rows


def section_aug(data):
    """C3c: does β hold across augmentations?"""
    rows = []
    if data is None:
        return rows
    sanity = data.get('sanity_check', {})
    san_status = sanity.get('status', 'NO_DATA')
    san_colour = '#2ecc71' if san_status == 'PASS' else ('#e74c3c' if san_status == 'FAIL' else '#aaaaaa')
    san_sym    = '✓' if san_status == 'PASS' else ('✗' if san_status == 'FAIL' else '?')
    rows.append({
        'experiment': 'exp_augmentations',
        'condition': 'sanity: identity×odd_even == baseline',
        'dataset': PAPER_DATASET,
        'beta': None, 'beta_std': None, 'r2': None,
        'deviation_sigma': float('nan'),
        'status': san_status, 'colour': san_colour, 'symbol': san_sym,
        'note': f"n_checked={sanity.get('n_checked', '?')}  max_diff={sanity.get('max_diff', '?'):.2e}",
    })
    for aug, r in (data.get('beta_per_aug_fashion_odd_even') or {}).items():
        beta, beta_std = r.get('beta'), r.get('beta_std')
        status, colour, sym = _beta_status(beta, beta_std)
        rows.append({
            'experiment': 'exp_augmentations',
            'condition': f'aug={aug}',
            'dataset': PAPER_DATASET,
            'beta': beta, 'beta_std': beta_std, 'r2': r.get('r2'),
            'deviation_sigma': _dev(beta, beta_std),
            'status': status, 'colour': colour, 'symbol': sym,
            'note': 'SANITY' if aug == 'identity' else '',
        })
    return rows


def section_depth(depth_dir):
    """C3d: does β hold across depths? Read from depth runs/ directly."""
    rows = []
    runs_dir = Path(depth_dir)
    if not runs_dir.exists():
        return rows

    from collections import defaultdict

    # Group by (L, dataset)
    by_L_ds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in runs_dir.glob('*.json'):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        L  = r.get('L')
        ds = r.get('dataset', '?')
        ap = r.get('alpha_paper')
        h  = r.get('h')
        err = r.get('final_test_err')
        if None in (L, ap, h, err) or err > 0.5:
            continue
        by_L_ds[(L, ds)][h][ap].append(err)

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
                la = math.log(alphas[i]+1e-30) + t*(math.log(alphas[i+1]+1e-30)-math.log(alphas[i]+1e-30))
                return math.exp(la)
        return None

    def fit_beta(h_vals, a_stars):
        lh = np.log(np.array(h_vals, dtype=float))
        la = np.log(np.array(a_stars, dtype=float))
        A = np.vstack([lh, np.ones_like(lh)]).T
        (beta, _), *_ = np.linalg.lstsq(A, la, rcond=None)
        pred = A @ np.array([beta, _])
        ss_res = np.sum((la - pred)**2)
        ss_tot = np.sum((la - la.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float('nan')
        se = np.sqrt(np.sum((la-pred)**2) / max(len(lh)-2, 1))
        beta_std = se / (np.sqrt(np.sum((lh-lh.mean())**2)) + 1e-30)
        return float(beta), float(beta_std), float(r2)

    for (L, ds), by_h in sorted(by_L_ds.items()):
        alpha_stars = {}
        for h, by_ap in by_h.items():
            aps = sorted(by_ap)
            avgs = [np.mean(by_ap[a]) for a in aps]
            astar = find_boundary(aps, avgs)
            if astar:
                alpha_stars[h] = astar
        if len(alpha_stars) < 3:
            continue
        h_arr = sorted(alpha_stars)
        a_arr = [alpha_stars[h] for h in h_arr]
        beta, beta_std, r2 = fit_beta(h_arr, a_arr)
        status, colour, sym = _beta_status(beta, beta_std)
        rows.append({
            'experiment': 'exp_depth',
            'condition': f'L={L}  {ds}',
            'dataset': ds,
            'beta': beta, 'beta_std': beta_std, 'r2': r2,
            'deviation_sigma': _dev(beta, beta_std),
            'status': status, 'colour': colour, 'symbol': sym,
            'note': 'BASELINE' if L == PAPER_L else '',
        })
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

EXP_COLOURS = {
    'exp_h_range':      '#3498db',
    'exp_seeds':        '#9b59b6',
    'exp_wall_time':    '#e67e22',
    'exp_binary_split': '#27ae60',
    'exp_depth':        '#c0392b',
    'exp_augmentations':'#16a085',
}

EXP_LABELS = {
    'exp_h_range':      'H range',
    'exp_seeds':        'Seeds',
    'exp_wall_time':    'Wall time',
    'exp_binary_split': 'Binary split',
    'exp_depth':        'Depth',
    'exp_augmentations':'Augmentations',
}


def plot_summary_table(all_rows, out_dir):
    """Render a table image: condition | β | deviation | status."""
    beta_rows = [r for r in all_rows if r['beta'] is not None]
    if not beta_rows:
        print("  No β rows to plot.")
        return

    n = len(beta_rows)
    fig_h = max(4, 0.35 * n + 2)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis('off')

    col_labels = ['Experiment', 'Condition', 'β ± σ', '|dev|/σ', 'R²', 'Status']
    col_widths = [0.13, 0.35, 0.16, 0.10, 0.08, 0.08]
    col_x = [sum(col_widths[:i]) for i in range(len(col_widths))]

    # Header
    for j, (label, x) in enumerate(zip(col_labels, col_x)):
        ax.text(x, 1.0, label, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')

    ax.axhline(0.97, color='black', lw=1, transform=ax.transAxes)

    row_h = 0.93 / n
    for i, r in enumerate(beta_rows):
        y = 0.95 - i * row_h
        bg = '#f9f9f9' if i % 2 == 0 else 'white'
        ax.axhspan(y - row_h, y, xmin=0, xmax=1, facecolor=bg,
                   transform=ax.transAxes, zorder=0)

        vals = [
            EXP_LABELS.get(r['experiment'], r['experiment']),
            r['condition'],
            _fmt_beta(r['beta'], r['beta_std']),
            f"{r['deviation_sigma']:.1f}σ" if not math.isnan(r['deviation_sigma']) else '—',
            f"{r['r2']:.3f}" if r.get('r2') else '—',
            f"{r['symbol']} {r['status']}",
        ]
        for j, (val, x) in enumerate(zip(vals, col_x)):
            colour = r['colour'] if j == 5 else 'black'
            weight = 'bold' if j == 5 else 'normal'
            ax.text(x + 0.005, y - row_h/2, val,
                    transform=ax.transAxes, fontsize=7.5,
                    va='center', color=colour, fontweight=weight)

    ax.axhline(0.0, color='black', lw=0.5, transform=ax.transAxes)
    fig.suptitle(
        f'All Experiments vs Paper Claim: α* ∝ h^β  (paper: β = {PAPER_BETA})\n'
        f'PASS = |β − (−0.5)| < {CONFIRM_THRESHOLD}σ',
        fontsize=11, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ summary_table.png')


def plot_beta_overview(all_rows, out_dir):
    """Multi-panel bar chart: β for every condition, grouped by experiment."""
    beta_rows = [r for r in all_rows if r['beta'] is not None]
    if not beta_rows:
        return

    exps = list(dict.fromkeys(r['experiment'] for r in beta_rows))
    n_exp = len(exps)
    fig, axes = plt.subplots(1, n_exp, figsize=(4 * n_exp, 5),
                              sharey=True, gridspec_kw={'wspace': 0.05})
    if n_exp == 1:
        axes = [axes]

    for ax, exp in zip(axes, exps):
        rows = [r for r in beta_rows if r['experiment'] == exp]
        labels = [r['condition'].replace(f'  {r["dataset"]}', '').strip() for r in rows]
        betas  = [r['beta'] for r in rows]
        errs   = [r['beta_std'] for r in rows]
        colours= [r['colour'] for r in rows]
        x = np.arange(len(rows))

        ax.bar(x, betas, yerr=errs, capsize=3, color=colours, alpha=0.85, zorder=3)
        ax.axhline(PAPER_BETA, color='red', ls='--', lw=2, zorder=4,
                   label='Paper: β=−0.5')
        ax.axhspan(PAPER_BETA - 0.1, PAPER_BETA + 0.1, alpha=0.08, color='red', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7)
        ax.set_title(EXP_LABELS.get(exp, exp), fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, zorder=1)
        ax.set_ylim(-1.1, 0.2)
        if ax == axes[0]:
            ax.set_ylabel('Fitted β')
        ax.legend(fontsize=7)

    legend_elements = [
        Patch(facecolor='#2ecc71', label='PASS (|dev|<2σ)'),
        Patch(facecolor='#f39c12', label='WARN (2–4σ)'),
        Patch(facecolor='#e74c3c', label='FAIL (>4σ)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Fitted Scaling Exponent β Across All Conditions\n'
                 'Red dashed = paper claim (β=−0.5); shaded band = ±0.1',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'beta_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ beta_overview.png')


def plot_collapse_quality(seeds_rows, out_dir):
    """Collapse spread at n=3 vs n=10 per dataset."""
    rows = [r for r in seeds_rows if 'spread_3' in r and 'spread_10' in r]
    if not rows:
        print('  (no seeds data for collapse_quality.png)')
        return

    datasets = [r['dataset'] for r in rows]
    s3  = [r['spread_3']  for r in rows]
    s10 = [r['spread_10'] for r in rows]
    x = np.arange(len(datasets))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.2), 4))
    ax.bar(x - w/2, s3,  w, label='n=3 seeds (baseline)', color='#3498db', alpha=0.8)
    ax.bar(x + w/2, s10, w, label='n=10 seeds',           color='#2ecc71', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha='right')
    ax.set_ylabel('Collapse spread\n(lower = better collapse)')
    ax.set_title('Collapse Spread: n=3 vs n=10 Seeds\n'
                 'Large change → 3 seeds was insufficient', fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'collapse_quality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ collapse_quality.png')


def plot_timeout_bias(wall_rows, out_dir):
    """Mean Δ test error and timeout count per dataset."""
    rows = [r for r in wall_rows if 'mean_delta' in r]
    if not rows:
        print('  (no wall_time data for timeout_bias.png)')
        return

    datasets  = [r['dataset'] for r in rows]
    deltas    = [r['mean_delta'] for r in rows]
    n_timeouts= [r['n_timed_out'] for r in rows]
    colours   = [r['colour'] for r in rows]
    x = np.arange(len(datasets))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(datasets)*1.2), 6),
                                    sharex=True)

    ax1.bar(x, deltas, color=colours, alpha=0.85)
    ax1.axhline(0, color='black', lw=0.8)
    ax1.axhline( 0.005, color='red', ls=':', lw=1, label='±0.005 threshold')
    ax1.axhline(-0.005, color='red', ls=':', lw=1)
    ax1.set_ylabel('Mean Δ test error\n(baseline − extended)')
    ax1.set_title('Timeout Bias: 600s → 3600s Extension\n'
                  'Positive = baseline overestimated error (α* biased high)', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, n_timeouts, color='#e67e22', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=20, ha='right')
    ax2.set_ylabel('N runs timed out at 600s')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'timeout_bias.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ timeout_bias.png')


# ── Verdict writer ────────────────────────────────────────────────────────────

def write_verdict(all_rows, data, out_dir):
    """Write VERDICT.md mapping every result to the paper's key finding."""

    beta_rows   = [r for r in all_rows if r['beta'] is not None]
    n_pass = sum(1 for r in beta_rows if r['status'] == 'PASS')
    n_warn = sum(1 for r in beta_rows if r['status'] == 'WARN')
    n_fail = sum(1 for r in beta_rows if r['status'] == 'FAIL')
    n_total= len(beta_rows)

    # Overall verdict
    if n_total == 0:
        overall = 'INSUFFICIENT DATA — run all sub-experiments first'
        overall_colour = '🟡'
    elif n_fail > 0:
        overall = f'CONTRADICTED — {n_fail}/{n_total} conditions fail the β=−0.5 claim'
        overall_colour = '🔴'
    elif n_warn > 0:
        overall = f'WEAKLY SUPPORTED — {n_pass} pass, {n_warn} borderline, 0 fail'
        overall_colour = '🟡'
    else:
        overall = f'STRONGLY SUPPORTED — all {n_pass} conditions consistent with β=−0.5'
        overall_colour = '🟢'

    lines = []
    lines += [
        '# Verdict: Paper Key Finding vs Experimental Evidence',
        '',
        '## Paper Claim',
        '',
        '> "The boundary between lazy and feature training regimes scales as',
        '> α\\* = O(h^{−1/2}), confirmed by collapse of test error curves when',
        '> plotted against α√h."',
        '',
        f'## Overall: {overall_colour} {overall}',
        '',
        f'- β rows tested : {n_total}',
        f'- PASS (|dev|<2σ): {n_pass}',
        f'- WARN (2–4σ)   : {n_warn}',
        f'- FAIL (>4σ)    : {n_fail}',
        '',
        '---',
        '',
    ]

    # ── C1: Exponent precision ────────────────────────────────────────────────
    lines += ['## C1 — Exponent Precision (exp_h_range)', '']
    h_rows = [r for r in beta_rows if r['experiment'] == 'exp_h_range']
    if not h_rows:
        lines += ['*No data — run exp_h_range first.*', '']
    else:
        lines += ['With h spanning 3 orders of magnitude, the exponent can be fitted precisely.', '']
        lines += ['| Dataset | β ± σ | |dev|/σ | R² | Status |']
        lines += ['|---------|-------|--------|-----|--------|']
        for r in h_rows:
            lines.append(
                f"| {r['dataset']} | {_fmt_beta(r['beta'],r['beta_std'])} "
                f"| {r['deviation_sigma']:.1f}σ | {r.get('r2',0):.3f} "
                f"| {r['symbol']} {r['status']} |"
            )
        lines += ['']
        worst = max(h_rows, key=lambda r: r['deviation_sigma'])
        if worst['status'] == 'PASS':
            lines += [f"✓ All h-range conditions consistent with β=−0.5. "
                      f"Largest deviation: {worst['deviation_sigma']:.1f}σ ({worst['dataset']}).", '']
        else:
            lines += [f"✗ {worst['dataset']}: β={worst['beta']:.3f}, "
                      f"{worst['deviation_sigma']:.1f}σ from −0.5. Paper claim may be imprecise.", '']

    # ── C2: Collapse quality ──────────────────────────────────────────────────
    lines += ['## C2 — Collapse Quality (exp_seeds)', '']
    s_rows = [r for r in all_rows if r['experiment'] == 'exp_seeds' and 'spread_3' in r]
    if not s_rows:
        lines += ['*No data — run exp_seeds first.*', '']
    else:
        large_change = [r for r in s_rows
                        if abs(r['spread_10'] - r['spread_3']) / (r['spread_3'] + 1e-9) > 0.15]
        if not large_change:
            lines += ['✓ Collapse spread changed < 15% from n=3 to n=10 seeds. '
                      'Baseline error bands were adequate.', '']
        else:
            ds_list = [r['dataset'] for r in large_change]
            lines += [f'⚠ Spread changed >15% for: {ds_list}. '
                      f'Baseline error bands may be unreliable for these datasets.', '']

    # ── C3a: Wall-time bias ───────────────────────────────────────────────────
    lines += ['## C3a — Wall-Time Bias (exp_wall_time)', '']
    w_rows = [r for r in all_rows if r['experiment'] == 'exp_wall_time' and 'mean_delta' in r]
    if not w_rows:
        lines += ['*No data — run exp_wall_time first.*', '']
    else:
        biased = [r for r in w_rows if abs(r['mean_delta']) > 0.005]
        if not biased:
            lines += ['✓ Extending 600→3600s changed test errors by < 0.005 for all datasets. '
                      'The 600s limit did not materially bias α*.', '']
        else:
            for r in biased:
                lines += [f"✗ {r['dataset']}: {r['n_timed_out']} runs timed out; "
                          f"Δerr = {r['mean_delta']:+.4f}. α* estimate was biased upward.", '']

    # ── C3b: Split independence ───────────────────────────────────────────────
    lines += ['## C3b — Split Independence (exp_binary_split)', '']
    sp_rows = [r for r in beta_rows if r['experiment'] == 'exp_binary_split']
    sanity_row = next((r for r in all_rows
                       if r['experiment'] == 'exp_binary_split'
                       and 'sanity' in r['condition'].lower()), None)
    if sanity_row:
        lines += [f"Sanity check (odd_even == baseline): {sanity_row['symbol']} {sanity_row['status']} — {sanity_row['note']}", '']
    if not sp_rows:
        lines += ['*No β data — run exp_binary_split first.*', '']
    else:
        fail_splits = [r for r in sp_rows if r['status'] == 'FAIL']
        if not fail_splits:
            lines += [f'✓ β ≈ −0.5 across all {len(sp_rows)} split conditions. '
                      f'Result is split-independent.', '']
        else:
            for r in fail_splits:
                lines += [f"✗ {r['condition']}: β={r['beta']:.3f} ({r['deviation_sigma']:.1f}σ). "
                          f"Exponent may depend on split difficulty.", '']
        lines += ['']

    # ── C3c: Augmentation independence ───────────────────────────────────────
    lines += ['## C3c — Augmentation Independence (exp_augmentations)', '']
    aug_rows = [r for r in beta_rows if r['experiment'] == 'exp_augmentations']
    aug_sanity = next((r for r in all_rows
                       if r['experiment'] == 'exp_augmentations'
                       and 'sanity' in r['condition'].lower()), None)
    if aug_sanity:
        lines += [f"Sanity check (identity×odd_even == baseline): "
                  f"{aug_sanity['symbol']} {aug_sanity['status']} — {aug_sanity['note']}", '']
    if not aug_rows:
        lines += ['*No β data — run exp_augmentations first.*', '']
    else:
        fail_augs = [r for r in aug_rows if r['status'] == 'FAIL']
        if not fail_augs:
            lines += [f'✓ β ≈ −0.5 across all {len(aug_rows)} augmentation conditions. '
                      f'Result is robust to input distribution perturbations.', '']
        else:
            for r in fail_augs:
                lines += [f"✗ {r['condition']}: β={r['beta']:.3f} ({r['deviation_sigma']:.1f}σ). "
                          f"Exponent changes under this augmentation.", '']
        lines += ['']

    # ── C3d: Depth independence ───────────────────────────────────────────────
    lines += ['## C3d — Depth Independence (exp_depth)', '']
    d_rows = [r for r in beta_rows if r['experiment'] == 'exp_depth']
    if not d_rows:
        lines += ['*No data — run exp_depth first.*', '']
    else:
        fail_depth = [r for r in d_rows if r['status'] == 'FAIL']
        if not fail_depth:
            lines += [f'✓ β ≈ −0.5 for all depths L tested. '
                      f'NTK theory\'s depth-independence prediction holds.', '']
        else:
            for r in fail_depth:
                lines += [f"✗ {r['condition']}: β={r['beta']:.3f} ({r['deviation_sigma']:.1f}σ). "
                          f"Depth affects the exponent.", '']
        lines += ['']

    # ── Final summary ─────────────────────────────────────────────────────────
    lines += [
        '---',
        '',
        '## Summary Table',
        '',
        '| Condition | β ± σ | |dev|/σ | Status |',
        '|-----------|-------|--------|--------|',
    ]
    for r in beta_rows:
        dev_str = f"{r['deviation_sigma']:.1f}σ" if not math.isnan(r['deviation_sigma']) else '—'
        lines.append(
            f"| {r['condition']} "
            f"| {_fmt_beta(r['beta'], r['beta_std'])} "
            f"| {dev_str} "
            f"| {r['symbol']} {r['status']} |"
        )
    lines += ['']

    out_file = out_dir / 'VERDICT.md'
    with open(out_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f'✓ VERDICT.md')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    root = Path(__file__).parent

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--h-range-results',  type=Path,
                    default=root / 'results' / 'exp_h_range')
    ap.add_argument('--seeds-results',    type=Path,
                    default=root / 'results' / 'exp_seeds')
    ap.add_argument('--wall-results',     type=Path,
                    default=root / 'results' / 'exp_wall_time')
    ap.add_argument('--split-results',    type=Path,
                    default=root / 'results' / 'exp_binary_split')
    ap.add_argument('--depth-results',    type=Path,
                    default=root / 'runs' / 'exp_depth')
    ap.add_argument('--aug-results',      type=Path,
                    default=root / 'results' / 'exp_augmentations')
    ap.add_argument('--out',              type=Path,
                    default=root / 'report')
    args = ap.parse_args()

    args.out.mkdir(exist_ok=True)

    print('Loading results...')
    data = load_all(args)

    print('Building row sets...')
    h_rows    = section_h_range(data['h_range'])
    s_rows    = section_seeds(data['seeds'])
    w_rows    = section_wall(data['wall'])
    sp_rows   = section_split(data['split'])
    aug_rows  = section_aug(data['aug'])
    d_rows    = section_depth(data['depth_dir'])

    all_rows = h_rows + s_rows + w_rows + sp_rows + aug_rows + d_rows

    n_with_data = sum(1 for r in all_rows
                      if r['status'] not in ('NO DATA',) and r['beta'] is not None)
    print(f'Total conditions with β data: {n_with_data}')
    print()

    print('Generating plots...')
    plot_summary_table(all_rows, args.out)
    plot_beta_overview(all_rows, args.out)
    plot_collapse_quality(s_rows, args.out)
    plot_timeout_bias(w_rows, args.out)

    print('Writing verdict...')
    write_verdict(all_rows, data, args.out)

    print()
    print('=' * 60)
    print(f'Output in: {args.out}/')
    print('  summary_table.png  — full β table with pass/fail')
    print('  beta_overview.png  — bar charts per experiment')
    print('  collapse_quality.png — seed-count effect on collapse')
    print('  timeout_bias.png   — wall-time extension impact')
    print('  VERDICT.md         — plain-English verdict per claim')
    print('=' * 60)

    # Print quick summary to terminal
    beta_rows = [r for r in all_rows if r['beta'] is not None]
    if beta_rows:
        print()
        print(f"{'Condition':<40} {'β':>9}  {'dev':>6}  Status")
        print('-' * 68)
        for r in beta_rows:
            dev_str = f"{r['deviation_sigma']:.1f}σ" if not math.isnan(r['deviation_sigma']) else '  —'
            print(f"{r['condition']:<40} {_fmt_beta(r['beta'], r['beta_std']):>9}  "
                  f"{dev_str:>6}  {r['symbol']} {r['status']}")


if __name__ == '__main__':
    main()
