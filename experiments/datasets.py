"""
datasets.py — shared dataset configuration for all fidelity experiments.

Every dataset entry specifies:
  name        : the string passed to --dataset / args['dataset']
  ptr         : training set size
  pte         : test set size
  n_classes   : number of original classes (must be even for binary split)
  input_dim   : raw input dimension (for documentation; handled automatically)
  notes       : brief description of the binary split applied

Sizing rationale
----------------
fashion / mnist / kmnist : 70k total → ptr=10000, pte=50000 (60k used, 10k spare)
emnist-letters           : 145k total, 26 even classes → ptr=10000, pte=50000
cifar10                  : 60k total → ptr=10000, pte=50000 (uses all of it)
cifar_animal             : all CIFAR classes, binary (vehicles vs animals).
                           Intertwine truncates to min(24k vehicle, 36k animal) = 48k.
                           → ptr=10000, pte=36000
cifar_catdog             : only cat(3)+dog(5) from CIFAR, ~12k total
                           → ptr=3000, pte=6000
cifar_shipbird           : only ship(8)+bird(2) from CIFAR, ~12k total
                           → ptr=3000, pte=6000
cifar_catplane           : only cat(3)+plane(0) from CIFAR, ~12k total
                           → ptr=3000, pte=6000
"""

DATASETS = [
    # ── 10-class image datasets (balanced, large) ──────────────────────────
    {
        'name': 'fashion',
        'ptr': 10000,
        'pte': 50000,
        'n_classes': 10,
        'input_dim': 784,
        'notes': 'baseline; alternating split: {T-shirt,Pullover,Coat,Shirt,Bag} vs rest',
    },
    {
        'name': 'mnist',
        'ptr': 10000,
        'pte': 50000,
        'n_classes': 10,
        'input_dim': 784,
        'notes': 'alternating split: {0,2,4,6,8} vs {1,3,5,7,9}',
    },
    {
        'name': 'kmnist',
        'ptr': 10000,
        'pte': 50000,
        'n_classes': 10,
        'input_dim': 784,
        'notes': 'Kuzushiji-MNIST; alternating split on 10 hiragana characters',
    },
    {
        'name': 'emnist-letters',
        'ptr': 10000,
        'pte': 50000,
        'n_classes': 26,
        'input_dim': 784,
        'notes': 'EMNIST letters (A-Z); alternating split on 26 classes',
    },
    {
        'name': 'cifar10',
        'ptr': 10000,
        'pte': 50000,
        'n_classes': 10,
        'input_dim': 3072,
        'notes': 'CIFAR-10; alternating split: {plane,bird,deer,frog,ship} vs rest',
    },
    # ── Binary-only CIFAR subsets (small, pre-binarized) ───────────────────
    {
        'name': 'cifar_animal',
        'ptr': 10000,
        'pte': 36000,
        'n_classes': 2,
        'input_dim': 3072,
        'notes': 'vehicles {plane,car,ship,truck} vs animals {bird,cat,deer,dog,frog,horse}',
    },
    {
        'name': 'cifar_catdog',
        'ptr': 3000,
        'pte': 6000,
        'n_classes': 2,
        'input_dim': 3072,
        'notes': 'cat vs dog; only ~12k images total',
    },
    {
        'name': 'cifar_shipbird',
        'ptr': 3000,
        'pte': 6000,
        'n_classes': 2,
        'input_dim': 3072,
        'notes': 'ship vs bird; only ~12k images total',
    },
    {
        'name': 'cifar_catplane',
        'ptr': 3000,
        'pte': 6000,
        'n_classes': 2,
        'input_dim': 3072,
        'notes': 'cat vs airplane; only ~12k images total',
    },
]

# Quick lookup by name
DATASET_BY_NAME = {d['name']: d for d in DATASETS}


def dataset_args(name: str, base_args: dict) -> dict:
    """
    Return a copy of base_args with dataset-specific ptr/pte/dataset fields set.
    """
    cfg = DATASET_BY_NAME[name]
    return {
        **base_args,
        'dataset': name,
        'ptr': cfg['ptr'],
        'pte': cfg['pte'],
        # chunk must be at least as large as the biggest set
        'chunk': max(cfg['ptr'], cfg['pte'], 100000),
    }


def available_splits(name: str) -> list:
    """Return the list of named binary splits available for a dataset."""
    from splits import SPLITS_BY_DATASET
    return list(SPLITS_BY_DATASET.get(name, {}).keys())
