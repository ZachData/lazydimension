# Experiments: Systematic Fidelity Improvements

All files live flat in this `experiments/` folder — no subfolders for source code.
Generated outputs go into `runs/` and `results/` subdirectories created at runtime.

## File Layout

```
experiments/
├── datasets.py                 Shared dataset config (ptr/pte for all 9 datasets)
├── splits.py                   Named binary splits; odd_even = sanity check
├── augmentations.py            Named image augmentations; identity = sanity check
│
├── run_all.py                  Master runner — runs all sub-experiments via Modal
├── plot_all.py                 Master analysis — reads all results, writes report/
│
├── exp_h_range_run.py          Run: expanded width range h∈[10…10000]
├── exp_h_range_plot.py         Plot: exponent fit, collapse figure
├── exp_seeds_run.py            Run: 30-seed variance study
├── exp_seeds_plot.py           Plot: collapse spread vs n_seeds
├── exp_wall_time_run.py        Run: 600s → 3600s timeout extension
├── exp_wall_time_plot.py       Plot: timeout bias, Δ test error
├── exp_binary_split_run.py     Run: all named binary splits
├── exp_binary_split_plot.py    Plot: β per split, sanity check
├── exp_depth_run.py            Run: depths L∈{1,2,3,4,6}
├── exp_augmentations_run.py    Run: 16 augmentations × all splits
└── exp_augmentations_plot.py   Plot: β per augmentation, heatmap
```

Generated at runtime (not committed):
```
experiments/
├── runs/
│   ├── baseline/               Symlink or copy of project-level runs/ (for wall_time comparison)
│   ├── exp_h_range/            JSON results from exp_h_range_run.py
│   ├── exp_seeds/
│   ├── exp_wall_time/
│   ├── exp_binary_split/
│   ├── exp_depth/
│   └── exp_augmentations/
├── results/
│   ├── exp_h_range/            Plots + JSONs from exp_h_range_plot.py
│   ├── exp_seeds/
│   ├── exp_wall_time/
│   ├── exp_binary_split/
│   └── exp_augmentations/
└── report/
    ├── summary_table.png       } From plot_all.py
    ├── beta_overview.png       }
    ├── collapse_quality.png    }
    ├── timeout_bias.png        }
    └── VERDICT.md              }
```

## Quick Start

```bash
# Run all experiments on Fashion-MNIST (default)
modal run experiments/run_all.py

# Run on a different dataset
modal run experiments/run_all.py --dataset mnist

# Run on all 9 datasets
modal run experiments/run_all.py --dataset all

# Skip heavy experiments
modal run experiments/run_all.py --skip exp_augmentations

# Run only one experiment
modal run experiments/run_all.py --only exp_h_range

# Dry run (print commands, don't launch)
modal run experiments/run_all.py --dry-run
```

## Analysing Results

Run each experiment's plot script after its run completes:

```bash
python experiments/exp_h_range_plot.py
python experiments/exp_seeds_plot.py
python experiments/exp_wall_time_plot.py   # needs runs/baseline/ — see note below
python experiments/exp_binary_split_plot.py
python experiments/exp_augmentations_plot.py
```

Then synthesise everything:

```bash
python experiments/plot_all.py
# → experiments/report/VERDICT.md
# → experiments/report/summary_table.png
# → experiments/report/beta_overview.png
```

### Baseline runs for wall_time comparison

`exp_wall_time_plot.py` compares extended-timeout runs against the original
baseline runs. Copy or symlink the project-level `runs/` directory:

```bash
ln -s $(pwd)/runs experiments/runs/baseline
```

## Paper Claim Being Tested

> α* = O(h^{-1/2}): the boundary between lazy and feature training scales
> as the inverse square root of network width, confirmed by curve collapse
> under α√h.

| Experiment | What it tests |
|------------|--------------|
| exp_h_range | Exponent β precisely (3 OOM of h vs baseline's 1 OOM) |
| exp_seeds | Were baseline error bands reliable? |
| exp_wall_time | Was α* biased upward by the 600s timeout? |
| exp_binary_split | Is β independent of the binary class assignment? |
| exp_depth | Is β independent of network depth? |
| exp_augmentations | Is β independent of input augmentation? |
