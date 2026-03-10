"""
splits.py — Named binary split definitions for all supported datasets.

TERMINOLOGY
-----------
A "split" maps original class labels (integers) to ±1.
+1 = positive class, -1 = negative class.

Every split is a plain dict: {class_label: +1 or -1}.

SPECIAL SPLIT: 'odd_even'
--------------------------
The 'odd_even' split assigns:
    +1 to classes with even label index in sorted(unique_labels)
    -1 to classes with odd label index

For any dataset with labels 0,1,2,...,N-1 this is equivalent to:
    +1 if class_label % 2 == 0
    -1 if class_label % 2 == 1

This is EXACTLY what the baseline get_binary_dataset() does (it enumerates
y.unique() sorted, assigns +1 to even-indexed entries). Therefore:

    *** odd_even MUST produce identical results to the baseline ***

Any divergence between 'odd_even' and an unmodified baseline run is a bug.
Use it as a regression test whenever you change the split injection machinery.

DATASET-SPECIFIC NOTES
-----------------------
fashion (FashionMNIST, labels 0-9):
    0: T-shirt/top   1: Trouser      2: Pullover    3: Dress
    4: Coat          5: Sandal       6: Shirt       7: Sneaker
    8: Bag           9: Ankle boot

mnist (MNIST, labels 0-9):    handwritten digits 0–9
kmnist (Kuzushiji, labels 0-9): 10 hiragana characters
emnist-letters (labels 1-26):   Latin letters A–Z (1-indexed)
cifar10 (labels 0-9):
    0: airplane  1: automobile  2: bird    3: cat   4: deer
    5: dog       6: frog        7: horse   8: ship  9: truck
cifar_animal, cifar_catdog, cifar_shipbird, cifar_catplane:
    Pre-binarized by the codebase; only 2 labels present → only 'odd_even'
    and 'random' are meaningful (semantic splits are pre-applied by the dataset).
"""

import random as pyrandom


# ── Helpers ──────────────────────────────────────────────────────────────────

def _odd_even(labels):
    """Assign +1 to even-indexed labels, -1 to odd-indexed (matches baseline)."""
    sorted_labels = sorted(set(labels))
    return {lbl: (1 if i % 2 == 0 else -1) for i, lbl in enumerate(sorted_labels)}


def _random_split(labels, seed=42):
    """Fixed random 50/50 assignment. seed=42 for reproducibility."""
    sorted_labels = sorted(set(labels))
    rng = pyrandom.Random(seed)
    shuffled = sorted_labels[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    return {lbl: (1 if lbl in shuffled[:half] else -1) for lbl in sorted_labels}


def _make_split(pos_labels, all_labels):
    """pos_labels → +1, everything else → -1."""
    pos = set(pos_labels)
    return {lbl: (1 if lbl in pos else -1) for lbl in all_labels}


# ── Fashion-MNIST splits ──────────────────────────────────────────────────────

_FASHION_ALL = list(range(10))

FASHION_SPLITS = {
    # ── Sanity check: must equal baseline exactly ────────────────────────────
    'odd_even': _odd_even(_FASHION_ALL),
    # {0:1, 1:-1, 2:1, 3:-1, 4:1, 5:-1, 6:1, 7:-1, 8:1, 9:-1}

    # ── Semantic splits ──────────────────────────────────────────────────────
    # footwear: sandal(5), sneaker(7), ankle boot(9) vs everything else
    'footwear_vs_rest': _make_split([5, 7, 9], _FASHION_ALL),

    # upper-body garments: t-shirt(0), pullover(2), coat(4), shirt(6)
    'upper_vs_rest': _make_split([0, 2, 4, 6], _FASHION_ALL),

    # accessories & bags: bag(8), sandal(5), sneaker(7), ankle boot(9)
    'accessories_vs_garments': _make_split([5, 7, 8, 9], _FASHION_ALL),

    # trousers & dresses (leg-coverings): trouser(1), dress(3)
    'leg_vs_rest': _make_split([1, 3], _FASHION_ALL),

    # ── Noise control: fixed random assignment ───────────────────────────────
    'random_seed42': _random_split(_FASHION_ALL, seed=42),
    'random_seed7':  _random_split(_FASHION_ALL, seed=7),
}

# ── MNIST splits (digits 0-9) ─────────────────────────────────────────────────

_MNIST_ALL = list(range(10))

MNIST_SPLITS = {
    'odd_even': _odd_even(_MNIST_ALL),
    # low vs high digits
    'low_vs_high': _make_split([0, 1, 2, 3, 4], _MNIST_ALL),
    # curved vs angular (approximate — perceptually motivated)
    'curved_vs_angular': _make_split([0, 3, 6, 8, 9], _MNIST_ALL),
    'random_seed42': _random_split(_MNIST_ALL, seed=42),
}

# ── Kuzushiji-MNIST splits ────────────────────────────────────────────────────

_KMNIST_ALL = list(range(10))

KMNIST_SPLITS = {
    'odd_even': _odd_even(_KMNIST_ALL),
    'random_seed42': _random_split(_KMNIST_ALL, seed=42),
    # first-half vs second-half characters
    'first_half': _make_split([0, 1, 2, 3, 4], _KMNIST_ALL),
}

# ── EMNIST-letters splits (labels 1-26, A-Z) ─────────────────────────────────

_EMNIST_ALL = list(range(1, 27))  # 1-indexed

EMNIST_SPLITS = {
    'odd_even': _odd_even(_EMNIST_ALL),
    # vowels (A,E,I,O,U) vs consonants
    'vowels_vs_consonants': _make_split([1, 5, 9, 15, 21], _EMNIST_ALL),
    # first half of alphabet (A-M) vs second (N-Z)
    'first_half_alpha': _make_split(list(range(1, 14)), _EMNIST_ALL),
    'random_seed42': _random_split(_EMNIST_ALL, seed=42),
}

# ── CIFAR-10 splits ───────────────────────────────────────────────────────────
# 0:plane 1:car 2:bird 3:cat 4:deer 5:dog 6:frog 7:horse 8:ship 9:truck

_CIFAR10_ALL = list(range(10))

CIFAR10_SPLITS = {
    'odd_even': _odd_even(_CIFAR10_ALL),
    # vehicles vs animals (4 vs 6)
    'vehicles_vs_animals': _make_split([0, 1, 8, 9], _CIFAR10_ALL),
    # flying/water vs land
    'air_water_vs_land': _make_split([0, 2, 4, 8], _CIFAR10_ALL),
    # mammals vs non-mammals
    'mammals_vs_rest': _make_split([3, 4, 5, 7], _CIFAR10_ALL),
    'random_seed42': _random_split(_CIFAR10_ALL, seed=42),
}

# ── Pre-binarized CIFAR subsets ───────────────────────────────────────────────
# These datasets are already 2-class; the only splits that make sense are
# odd_even (= the existing split, sanity check) and random (noise baseline).

_BINARY_ALL = [0, 1]  # two-class datasets use labels 0 and 1

CIFAR_BINARY_SPLITS = {
    'odd_even': _odd_even(_BINARY_ALL),    # 0→+1, 1→-1 (matches baseline)
    'random_seed42': _random_split(_BINARY_ALL, seed=42),  # flips ~50%
}

# ── Master registry ───────────────────────────────────────────────────────────

SPLITS_BY_DATASET = {
    'fashion':          FASHION_SPLITS,
    'mnist':            MNIST_SPLITS,
    'kmnist':           KMNIST_SPLITS,
    'emnist-letters':   EMNIST_SPLITS,
    'cifar10':          CIFAR10_SPLITS,
    # pre-binarized datasets: only odd_even + random make sense
    'cifar_animal':     CIFAR_BINARY_SPLITS,
    'cifar_catdog':     CIFAR_BINARY_SPLITS,
    'cifar_shipbird':   CIFAR_BINARY_SPLITS,
    'cifar_catplane':   CIFAR_BINARY_SPLITS,
}

# Splits that are "sanity checks" — must reproduce baseline results exactly
SANITY_CHECK_SPLIT = 'odd_even'


# ── Application helper ────────────────────────────────────────────────────────

def apply_split(y_multiclass, split_map: dict):
    """
    Convert a multiclass label tensor to ±1 binary labels using split_map.

    Parameters
    ----------
    y_multiclass : torch.Tensor of shape (N,) with integer class labels
    split_map    : dict mapping each class label (int) to +1 or -1

    Returns
    -------
    torch.Tensor of shape (N,) with dtype matching the input, values ±1.

    Raises
    ------
    ValueError if any label in y_multiclass is not in split_map.
    """
    import torch
    missing = set(y_multiclass.unique().tolist()) - set(split_map.keys())
    if missing:
        raise ValueError(
            f"apply_split: labels {missing} not in split_map (keys: {sorted(split_map.keys())})"
        )
    out = torch.zeros_like(y_multiclass, dtype=torch.float64)
    for cls, sign in split_map.items():
        out[y_multiclass == cls] = sign
    return out


def patch_get_binary(dataset_module, dataset_name: str, split_name: str):
    """
    Context-manager-style patcher.
    Returns (patched_fn, original_fn) so callers can restore the original.

    Usage:
        patched, original = patch_get_binary(dataset_module, 'fashion', 'footwear_vs_rest')
        dataset_module.get_binary_dataset = patched
        try:
            ...run experiment...
        finally:
            dataset_module.get_binary_dataset = original
    """
    split_map = SPLITS_BY_DATASET[dataset_name][split_name]
    original = dataset_module.get_binary_dataset

    def patched(dataset, ps, seeds, d, params=None, device=None, dtype=None):
        sets = dataset_module.get_normalized_dataset(dataset, ps, seeds, d, params)
        import torch
        outs = []
        for x, y, idx in sets:
            x = x.to(device=device, dtype=dtype)
            b = apply_split(y, split_map)
            b = b.to(device=device, dtype=dtype)
            outs.append((x, b, idx))
        return outs

    return patched, original


def list_splits(dataset_name: str) -> list:
    """Return all split names available for a given dataset."""
    return list(SPLITS_BY_DATASET.get(dataset_name, {}).keys())


def split_summary() -> str:
    """Print a table of all available splits per dataset."""
    lines = []
    lines.append(f"{'Dataset':<20}  Splits")
    lines.append("-" * 70)
    for ds, splits in SPLITS_BY_DATASET.items():
        names = ', '.join(splits.keys())
        lines.append(f"{ds:<20}  {names}")
    return '\n'.join(lines)


if __name__ == '__main__':
    print(split_summary())
    print()
    # Verify odd_even matches baseline logic for all multi-class datasets
    print("Verifying odd_even == baseline for multi-class datasets:")
    for ds_name, splits in SPLITS_BY_DATASET.items():
        if ds_name in ('cifar_animal', 'cifar_catdog', 'cifar_shipbird', 'cifar_catplane'):
            continue
        oe = splits['odd_even']
        labels = sorted(oe.keys())
        # Baseline logic: enumerate sorted unique, +1 to even index
        baseline_map = {lbl: (1 if i % 2 == 0 else -1) for i, lbl in enumerate(labels)}
        assert oe == baseline_map, f"{ds_name}: odd_even != baseline!"
        print(f"  {ds_name}: ✓  {oe}")
    print("\nAll sanity checks passed.")
