#!/usr/bin/env python3
"""
smoke_test.py — Fast end-to-end validation of all experiment scripts.

Runs a minimal subset of each experiment (2-3 runs per script) to verify:
  - No import errors or crashes
  - Output JSON is written and well-formed
  - Wall times are consistent with the 9h budget targets

Does NOT write to the real runs/ directory — uses runs/smoke_test/ instead,
so it is safe to run at any time without corrupting real results.

Usage
-----
    python experiments/smoke_test.py                  # all experiments
    python experiments/smoke_test.py --only exp_h_range,exp_depth
    python experiments/smoke_test.py --workers 4
    python experiments/smoke_test.py --device cpu

Expected runtime: ~2-5 minutes at workers=4 (one tiny run per experiment).

Wall-time targets (for full run at workers=10, cpu, fashion):
    exp_h_range        < 1.5h
    exp_seeds          < 1.5h
    exp_binary_split   < 2.0h
    exp_depth          < 1.5h
    TOTAL              < 9.0h
"""

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _common import DEFAULT_DEVICE

# ── Wall-time targets (seconds, sequential) used to project full run ──────────
# Based on: 525 runs * ~45s avg = 23625s seq / 10 workers = ~0.66h
WALL_TARGETS = {
    'exp_h_range':      525 * 45 / 10,    # ~0.66h
    'exp_seeds':        540 * 45 / 10,    # ~0.68h
    'exp_binary_split': 1071 * 45 / 10,   # ~1.34h
    'exp_depth':        765 * 45 / 10,    # ~0.96h
    'exp_wall_time':    60 * 200 / 10,    # ~0.33h (excluded by default)
    'exp_augmentations':2295 * 45 / 10,   # ~2.87h (excluded by default)
}
TOTAL_TARGET = 9 * 3600  # 9 hours in seconds

# ── Minimal smoke configs per experiment ─────────────────────────────────────
# Each produces 2-4 runs: one fast (large alpha = lazy = converges in seconds),
# one slow-ish (small alpha = feature = may approach wall time).

SMOKE_CONFIGS = {
    'exp_h_range': {
        'h_values':     [100, 1000],
        'alpha_values': [1000.0, 0.01],   # one lazy (fast), one feature (slow-ish)
        'seeds':        [0],
    },
    'exp_seeds': {
        'phase_a_seeds':   [0, 1],         # just 2 of the 30
        'phase_b_h':       [300],
        'phase_b_alphas':  [100.0, 0.1],
        'phase_b_seeds':   [0],
    },
    'exp_wall_time': {
        'pairs':    [(300, 0.1)],
        'seeds':    [0],
        'max_wall': 60,                    # smoke only: 60s not 3600s
    },
    'exp_binary_split': {
        'splits':       ['odd_even'],      # sanity check split only
        'h_values':     [300],
        'alpha_values': [100.0, 0.1],
        'seeds':        [0],
    },
    'exp_depth': {
        'l_values':     [1, 3],
        'h_values':     [300],
        'alpha_values': [100.0, 0.1],
        'seeds':        [0],
    },
    'exp_augmentations': {
        'augs':         ['identity', 'hflip'],
        'splits':       ['odd_even'],
        'h_values':     [300],
        'alpha_values': [100.0],
        'seeds':        [0],
    },
}

ALL_EXPERIMENTS = [
    'exp_h_range', 'exp_seeds', 'exp_binary_split', 'exp_depth',
    'exp_wall_time', 'exp_augmentations',
]
DEFAULT_EXPERIMENTS = ['exp_h_range', 'exp_seeds', 'exp_binary_split', 'exp_depth']


# ── Smoke runners — import run_one directly, bypass argparse ─────────────────

def _load_run_one(script_name):
    """
    Import run_one from a run script without executing main().

    The module is registered in sys.modules under its proper name so that
    multiprocessing spawn workers can pickle run_one by qualified name.
    Without registration, spawn can't locate the function: PicklingError.
    """
    mod_name = script_name.replace('.py', '')
    if mod_name not in sys.modules:
        script_path = Path(__file__).parent / script_name
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod   # register BEFORE exec so pickle can find it
        spec.loader.exec_module(mod)
    return sys.modules[mod_name].run_one


def smoke_exp_h_range(cfg, device, workers, out_dir):
    from _common import run_items, extract_last_metrics
    run_one = _load_run_one('exp_h_range_run.py')

    items = [
        {'dataset': 'fashion', 'h': h, 'alpha': a, 'seed': s, 'device': device}
        for h in cfg['h_values']
        for a in cfg['alpha_values']
        for s in cfg['seeds']
    ]
    return _run_smoke(run_one, items, out_dir, 'exp_h_range', workers)


def smoke_exp_seeds(cfg, device, workers, out_dir):
    run_one = _load_run_one('exp_seeds_run.py')
    items = []
    for s in cfg['phase_a_seeds']:
        items.append({'phase': 'A', 'dataset': 'fashion',
                      'h': 300, 'alpha': 1.0, 'seed': s, 'device': device})
    for h in cfg['phase_b_h']:
        for a in cfg['phase_b_alphas']:
            for s in cfg['phase_b_seeds']:
                items.append({'phase': 'B', 'dataset': 'fashion',
                               'h': h, 'alpha': a, 'seed': s, 'device': device})
    return _run_smoke(run_one, items, out_dir, 'exp_seeds', workers)


def smoke_exp_wall_time(cfg, device, workers, out_dir):
    """Patches EXTENDED_WALL before running, restores after."""
    _load_run_one('exp_wall_time_run.py')   # ensure registered in sys.modules
    import sys as _sys
    m = _sys.modules['exp_wall_time_run']
    original_wall = m.EXTENDED_WALL
    m.EXTENDED_WALL = cfg['max_wall']
    items = [
        {'dataset': 'fashion', 'h': h, 'alpha': a, 'seed': s, 'device': device}
        for h, a in cfg['pairs']
        for s in cfg['seeds']
    ]
    try:
        result = _run_smoke(m.run_one, items, out_dir, 'exp_wall_time', workers)
    finally:
        m.EXTENDED_WALL = original_wall
    return result


def smoke_exp_binary_split(cfg, device, workers, out_dir):
    from splits import SPLITS_BY_DATASET, SANITY_CHECK_SPLIT
    run_one = _load_run_one('exp_binary_split_run.py')
    items = []
    for split_name in cfg['splits']:
        split_map = SPLITS_BY_DATASET.get('fashion', {}).get(split_name, {})
        int_map = {int(k): int(v) for k, v in split_map.items()}
        is_sanity = (split_name == SANITY_CHECK_SPLIT)
        for h in cfg['h_values']:
            for a in cfg['alpha_values']:
                for s in cfg['seeds']:
                    items.append({
                        'dataset': 'fashion', 'split': split_name,
                        'split_map': int_map, 'is_sanity': is_sanity,
                        'h': h, 'alpha': a, 'seed': s, 'device': device,
                    })
    return _run_smoke(run_one, items, out_dir, 'exp_binary_split', workers)


def smoke_exp_depth(cfg, device, workers, out_dir):
    run_one = _load_run_one('exp_depth_run.py')
    items = [
        {'L': L, 'dataset': 'fashion', 'h': h, 'alpha': a, 'seed': s, 'device': device}
        for L in cfg['l_values']
        for h in cfg['h_values']
        for a in cfg['alpha_values']
        for s in cfg['seeds']
    ]
    return _run_smoke(run_one, items, out_dir, 'exp_depth', workers)


def smoke_exp_augmentations(cfg, device, workers, out_dir):
    from splits import SPLITS_BY_DATASET, SANITY_CHECK_SPLIT
    from augmentations import augmentations_for
    run_one = _load_run_one('exp_augmentations_run.py')
    items = []
    for aug in cfg['augs']:
        for split_name in cfg['splits']:
            split_map = SPLITS_BY_DATASET.get('fashion', {}).get(split_name, {})
            int_map = {int(k): int(v) for k, v in split_map.items()}
            for h in cfg['h_values']:
                for a in cfg['alpha_values']:
                    for s in cfg['seeds']:
                        items.append({
                            'aug': aug, 'split': split_name,
                            'dataset': 'fashion', 'split_map': int_map,
                            'h': h, 'alpha': a, 'seed': s, 'device': device,
                        })
    return _run_smoke(run_one, items, out_dir, 'exp_augmentations', workers)


SMOKE_FNS = {
    'exp_h_range':       smoke_exp_h_range,
    'exp_seeds':         smoke_exp_seeds,
    'exp_wall_time':     smoke_exp_wall_time,
    'exp_binary_split':  smoke_exp_binary_split,
    'exp_depth':         smoke_exp_depth,
    'exp_augmentations': smoke_exp_augmentations,
}


# ── Core runner ───────────────────────────────────────────────────────────────

def _run_smoke(run_one, items, out_dir, exp_name, workers):
    """
    Run items through run_one, save results, return timing/status dict.
    """
    from _common import run_items

    exp_out = out_dir / exp_name
    exp_out.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    run_times = []
    errors = []

    start_wall = time.time()
    for item, result in run_items(run_one, items, workers=workers,
                                   desc=exp_name):
        t = time.time() - start_wall   # cumulative, not per-run
        if result is not None:
            key = _make_key(item, exp_name)
            json.dump(result, open(exp_out / key, 'w'), indent=2)
            run_times.append(result.get('wall_time', 0))
            n_ok += 1
        else:
            n_fail += 1
            errors.append(str(item))

    total_wall = time.time() - start_wall

    return {
        'exp':          exp_name,
        'n_runs':       len(items),
        'n_ok':         n_ok,
        'n_fail':       n_fail,
        'total_wall_s': total_wall,
        'avg_sim_s':    (sum(run_times) / len(run_times)) if run_times else None,
        'max_sim_s':    max(run_times) if run_times else None,
        'errors':       errors,
        'passed':       n_fail == 0,
    }


def _make_key(item, exp_name):
    parts = []
    if 'L'    in item: parts.append(f"L{item['L']}")
    if 'aug'  in item: parts.append(item['aug'])
    if 'split'in item: parts.append(item['split'])
    if 'phase'in item: parts.append(item['phase'])
    ds    = item.get('dataset', 'fashion')
    h     = item.get('h', 0)
    alpha = item.get('alpha', 0)
    seed  = item.get('seed', 0)
    parts += [ds, f"h{h}", f"a{alpha:.0e}", f"s{seed}"]
    return '_'.join(parts) + '.json'


# ── Projection ────────────────────────────────────────────────────────────────

def project_full_runtime(results, workers):
    """
    Use measured avg_sim_s to project full run wall time.
    Falls back to target-based estimate if no timing data.
    """
    rows = []
    total_projected = 0
    for r in results:
        exp = r['exp']
        target_full_runs = {
            'exp_h_range':       525,
            'exp_seeds':         540,
            'exp_binary_split': 1071,
            'exp_depth':         765,
            'exp_wall_time':      60,
            'exp_augmentations': 2295,
        }.get(exp, 0)

        if r['avg_sim_s'] and target_full_runs:
            # Use measured avg sim time — conservative: assume 30% will approach max_wall
            projected_seq = target_full_runs * (0.30 * 600 + 0.70 * r['avg_sim_s'])
            projected_wall = projected_seq / workers
            source = 'measured'
        else:
            projected_wall = WALL_TARGETS.get(exp, 0) / workers * (3600 / 3600)
            source = 'estimate'

        target_wall = WALL_TARGETS.get(exp, 0)
        over = projected_wall > target_wall
        rows.append({
            'exp': exp,
            'avg_sim_s': r['avg_sim_s'],
            'projected_wall_h': projected_wall / 3600,
            'target_wall_h': target_wall / 3600,
            'over_budget': over,
            'source': source,
        })
        total_projected += projected_wall

    return rows, total_projected


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results, projections, total_projected, workers, log_path):
    lines = []

    lines += [
        '=' * 65,
        'SMOKE TEST REPORT',
        f'workers={workers}  device={results[0].get("device", "cpu")}',
        '=' * 65,
        '',
        'CORRECTNESS',
        '-' * 65,
        f'  {"Experiment":<25} {"Runs":>5} {"OK":>5} {"FAIL":>5}  Status',
    ]

    all_passed = True
    for r in results:
        status = 'PASS' if r['passed'] else 'FAIL'
        if not r['passed']:
            all_passed = False
        lines.append(
            f"  {r['exp']:<25} {r['n_runs']:>5} {r['n_ok']:>5} {r['n_fail']:>5}"
            f"  {status}"
        )
        for e in r['errors']:
            lines.append(f"    ERROR: {e}")

    lines += [
        '',
        'TIMING — smoke run',
        '-' * 65,
        f'  {"Experiment":<25} {"Wall(s)":>9} {"avg_sim(s)":>11} {"max_sim(s)":>11}',
    ]
    for r in results:
        lines.append(
            f"  {r['exp']:<25} {r['total_wall_s']:>9.1f}"
            f" {(r['avg_sim_s'] or 0):>11.1f}"
            f" {(r['max_sim_s'] or 0):>11.1f}"
        )

    lines += [
        '',
        f'PROJECTED FULL RUN (workers={workers})',
        '-' * 65,
        f'  {"Experiment":<25} {"Projected":>10} {"Target":>10}  {"Status":>8}  Source',
    ]
    for p in projections:
        status = 'OVER   ' if p['over_budget'] else 'OK     '
        lines.append(
            f"  {p['exp']:<25} {p['projected_wall_h']:>9.2f}h"
            f" {p['target_wall_h']:>9.2f}h  {status}  {p['source']}"
        )
    total_h = total_projected / 3600
    budget_h = 9.0
    lines += [
        f"",
        f"  {'TOTAL (excl. opt-in)':<25} {total_h:>9.2f}h {budget_h:>9.1f}h"
        f"  {'OVER   ' if total_h > budget_h else 'OK     '}",
        '',
        '=' * 65,
        f'OVERALL: {"ALL PASS" if all_passed else "FAILURES DETECTED"}',
        f'Full run estimate: {total_h:.2f}h at workers={workers} '
        f'({"within" if total_h <= budget_h else "EXCEEDS"} 9h budget)',
        '=' * 65,
    ]

    report = '\n'.join(lines)
    print(report)
    log_path.write_text(report)
    print(f"\nReport saved: {log_path}")

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only',    default='',
                    help=f'Comma-separated experiments to test. '
                         f'Default: {",".join(DEFAULT_EXPERIMENTS)}')
    ap.add_argument('--all',     action='store_true',
                    help='Include opt-in experiments (wall_time, augmentations)')
    ap.add_argument('--workers', type=int, default=4,
                    help='Workers for smoke runs (default: 4)')
    ap.add_argument('--device',  default=DEFAULT_DEVICE)
    args = ap.parse_args()

    if args.only:
        to_test = [e.strip() for e in args.only.split(',')]
    elif args.all:
        to_test = ALL_EXPERIMENTS
    else:
        to_test = DEFAULT_EXPERIMENTS

    unknown = [e for e in to_test if e not in ALL_EXPERIMENTS]
    if unknown:
        print(f"Unknown experiments: {unknown}")
        print(f"Valid: {ALL_EXPERIMENTS}")
        sys.exit(1)

    out_dir = Path(__file__).parent / 'runs' / 'smoke_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'smoke_report.txt'

    print(f"Smoke testing: {to_test}")
    print(f"device={args.device}  workers={args.workers}")
    print(f"Output: {out_dir}\n")

    results = []
    for exp_name in to_test:
        print(f"\n{'─'*50}")
        print(f"  {exp_name}")
        print(f"{'─'*50}")
        cfg = SMOKE_CONFIGS[exp_name]
        fn = SMOKE_FNS[exp_name]
        try:
            r = fn(cfg, args.device, args.workers, out_dir)
            r['device'] = args.device
        except Exception:
            traceback.print_exc()
            r = {
                'exp': exp_name, 'n_runs': 0, 'n_ok': 0, 'n_fail': 1,
                'total_wall_s': 0, 'avg_sim_s': None, 'max_sim_s': None,
                'errors': [traceback.format_exc()], 'passed': False,
                'device': args.device,
            }
        results.append(r)
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f"  -> {status}  {r['n_ok']}/{r['n_runs']} ok  "
              f"{r['total_wall_s']:.1f}s wall  avg_sim={r['avg_sim_s'] or 0:.1f}s")

    projections, total_projected = project_full_runtime(results, workers=10)
    all_passed = print_report(results, projections, total_projected,
                               workers=10, log_path=log_path)

    json.dump({'results': results, 'projections': projections},
              open(out_dir / 'smoke_data.json', 'w'), indent=2)

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
