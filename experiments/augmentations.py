"""
augmentations.py — Named image augmentations for all spatial datasets.

DESIGN
------
Augmentations operate on the raw float64 pixel tensor of shape (N, C, H, W)
in [0, 1] — BEFORE center_normalize is applied. This is the correct place:

    dataset_to_tensors  →  intertwine_labels  →  [AUGMENT HERE]  →  center_normalize

Working before normalization means:
  - Geometric transforms (flip, rotate, crop) correctly manipulate the image
  - After center_normalize the output has the same scale as baseline
  - The network sees the same representation format regardless of augmentation

SANITY CHECK: 'identity'
-------------------------
The 'identity' augmentation applies no transform. It MUST produce the same
result as the unaugmented baseline. Any divergence is a bug.

Because get_normalized_dataset is @lru_cache, we clear the cache before
patching so old results don't contaminate augmented runs.

COMBINING WITH SPLITS
---------------------
Augmentations and splits are orthogonal axes — any augmentation can be
combined with any split. The `patch_augmentation` context manager handles
augmentation. `patch_get_binary` in splits.py handles the label assignment.
Both patches can be active simultaneously.

Usage:
    augmented, orig_norm = patch_augmentation(dataset_module, 'fashion', 'hflip')
    patched_bin, orig_bin = patch_get_binary(dataset_module, 'fashion', 'footwear_vs_rest')
    dataset_module.get_normalized_dataset = augmented
    dataset_module.get_binary_dataset = patched_bin
    try:
        ...run experiment...
    finally:
        dataset_module.get_normalized_dataset = orig_norm
        dataset_module.get_binary_dataset = orig_bin
        dataset_module.get_normalized_dataset.cache_clear()

SUPPORTED DATASETS
------------------
Augmentations are defined for: fashion, mnist, kmnist, emnist-letters, cifar10
and all cifar_* binary subsets. For non-spatial datasets (higgs, stripe, sphere,
etc.) augmentation is 'identity' only — they have no image structure to perturb.

AUGMENTATIONS
-------------
identity         — no change (sanity check; must equal baseline)

-- Geometric transforms (deterministic, label-preserving) --
hflip            — horizontal flip (left↔right)
vflip            — vertical flip (up↔down)
rot90_cw         — 90° clockwise rotation
rot90_ccw        — 90° counter-clockwise rotation
rot180           — 180° rotation
transpose        — transpose (reflect across main diagonal)
center_crop_20   — centre-crop to 20px then resize back to 28px
center_crop_24   — centre-crop to 24px then resize back to 28px (mild)

-- Noise (additive Gaussian, clipped to [0,1]) --
noise_005        — σ = 0.05
noise_010        — σ = 0.10
noise_020        — σ = 0.20

-- Photometric transforms --
invert           — negate pixels: 1 - x
brightness_up    — add 0.15, clip to [0,1]
brightness_down  — subtract 0.15, clip to [0,1]
blur_3x3         — 3×3 box blur
"""

import math
import torch
import torch.nn.functional as F


# ── Core augmentation functions ───────────────────────────────────────────────
# Each fn: (N, C, H, W) float64 → (N, C, H, W) float64

def _identity(x):
    return x


def _hflip(x):
    return x.flip(-1)


def _vflip(x):
    return x.flip(-2)


def _rot90_cw(x):
    # torch.rot90 rotates counter-clockwise by default; k=3 → clockwise
    return torch.rot90(x, k=3, dims=(-2, -1))


def _rot90_ccw(x):
    return torch.rot90(x, k=1, dims=(-2, -1))


def _rot180(x):
    return torch.rot90(x, k=2, dims=(-2, -1))


def _transpose(x):
    # Reflect across main diagonal: swap H and W dimensions
    return x.transpose(-2, -1)


def _center_crop_then_resize(x, crop_size):
    """Centre-crop to crop_size × crop_size, then bilinear-resize to original."""
    N, C, H, W = x.shape
    assert H == W, "centre-crop only implemented for square images"
    pad = (H - crop_size) // 2
    cropped = x[:, :, pad:pad+crop_size, pad:pad+crop_size]
    # F.interpolate expects float; we're float64 which it supports
    resized = F.interpolate(cropped.float(), size=(H, W), mode='bilinear',
                            align_corners=False).double()
    return resized


def _center_crop_20(x):
    return _center_crop_then_resize(x, 20)


def _center_crop_24(x):
    return _center_crop_then_resize(x, 24)


def _make_noise(sigma):
    def _noise(x):
        return (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)
    return _noise


def _invert(x):
    return 1.0 - x


def _brightness_up(x):
    return (x + 0.15).clamp(0.0, 1.0)


def _brightness_down(x):
    return (x - 0.15).clamp(0.0, 1.0)


def _blur_3x3(x):
    N, C, H, W = x.shape
    kernel = torch.ones(1, 1, 3, 3, dtype=x.dtype, device=x.device) / 9.0
    # Apply per channel
    x_f = x.reshape(N * C, 1, H, W)
    blurred = F.conv2d(x_f, kernel, padding=1).reshape(N, C, H, W)
    return blurred


# ── Registry ──────────────────────────────────────────────────────────────────

_AUGMENTATION_FNS = {
    'identity':        _identity,
    'hflip':           _hflip,
    'vflip':           _vflip,
    'rot90_cw':        _rot90_cw,
    'rot90_ccw':       _rot90_ccw,
    'rot180':          _rot180,
    'transpose':       _transpose,
    'center_crop_20':  _center_crop_20,
    'center_crop_24':  _center_crop_24,
    'noise_005':       _make_noise(0.05),
    'noise_010':       _make_noise(0.10),
    'noise_020':       _make_noise(0.20),
    'invert':          _invert,
    'brightness_up':   _brightness_up,
    'brightness_down': _brightness_down,
    'blur_3x3':        _blur_3x3,
}

# Datasets that have image structure and support non-identity augmentations
SPATIAL_DATASETS = frozenset({
    'fashion', 'pca_fashion',
    'mnist', 'pca_mnist',
    'kmnist',
    'emnist-letters',
    'cifar10', 'pca_cifar10',
    'cifar_animal', 'cifar_catdog', 'cifar_shipbird', 'cifar_catplane',
})

# Augmentations available per dataset type
# All spatial datasets support the full list.
# Non-spatial datasets only support 'identity'.
AUGMENTATIONS_FOR_SPATIAL = list(_AUGMENTATION_FNS.keys())
AUGMENTATIONS_FOR_NONSPATIAL = ['identity']


def augmentations_for(dataset_name: str) -> list:
    if dataset_name in SPATIAL_DATASETS:
        return AUGMENTATIONS_FOR_SPATIAL
    return AUGMENTATIONS_FOR_NONSPATIAL


def get_augmentation_fn(name: str):
    if name not in _AUGMENTATION_FNS:
        raise ValueError(f"Unknown augmentation '{name}'. Available: {list(_AUGMENTATION_FNS)}")
    return _AUGMENTATION_FNS[name]


# ── Patcher ───────────────────────────────────────────────────────────────────

def patch_augmentation(dataset_module, dataset_name: str, aug_name: str):
    """
    Patch dataset_module.get_normalized_dataset to apply `aug_name` to the
    raw pixel data for `dataset_name` before center_normalize is called.

    Returns (patched_fn, original_fn). Callers MUST restore the original:

        patched, original = patch_augmentation(module, 'fashion', 'hflip')
        module.get_normalized_dataset = patched
        try:
            ...
        finally:
            module.get_normalized_dataset = original
            module.get_normalized_dataset.cache_clear()

    Clears the lru_cache before returning so prior cached results won't
    contaminate augmented runs.

    For 'identity' on any dataset the patched function is equivalent to
    calling the original — use this as a regression test.
    """
    if aug_name not in _AUGMENTATION_FNS:
        raise ValueError(f"Unknown augmentation '{aug_name}'")
    if aug_name != 'identity' and dataset_name not in SPATIAL_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' is non-spatial; "
            f"only 'identity' augmentation is valid."
        )

    aug_fn = _AUGMENTATION_FNS[aug_name]
    original = dataset_module.get_normalized_dataset
    # Clear cache so we don't accidentally get a cached baseline run
    original.cache_clear()

    # We need to intercept the fashion/mnist/cifar loading but leave
    # everything else unchanged. The cleanest way: wrap the original
    # and inject the augmentation into the specific dataset branch.
    #
    # get_normalized_dataset loads data, runs intertwine_labels to get
    # (x, y, i), then calls center_normalize(x). We reproduce that for
    # the target dataset, inserting aug_fn between intertwine_labels and
    # center_normalize.

    def patched(dataset, ps, seeds, d=0, params=None):
        import torchvision
        from datasets import DATASET_BY_NAME  # for ptr/pte metadata — not used here
        from __init__ import (dataset_to_tensors, intertwine_labels,
                               center_normalize, intertwine_split)

        transform = torchvision.transforms.ToTensor()
        torch.manual_seed(seeds[0])

        def _load_and_augment(torchvision_tr, torchvision_te=None):
            items = list(torchvision_tr)
            if torchvision_te is not None:
                items += list(torchvision_te)
            x, y, idx = intertwine_labels(*dataset_to_tensors(items))
            x = aug_fn(x)          # ← augment here, before normalisation
            x = center_normalize(x)
            return intertwine_split(x, y, idx, ps, seeds, y.unique())

        # Only intercept the target dataset; delegate all others to original
        if dataset != dataset_name:
            return original(dataset, ps, seeds, d, params)

        # ── Spatial dataset loading ───────────────────────────────────────
        if dataset in ('fashion', 'pca_fashion'):
            tr = torchvision.datasets.FashionMNIST(
                '~/.torchvision/datasets/FashionMNIST',
                train=True, download=True, transform=transform)
            te = torchvision.datasets.FashionMNIST(
                '~/.torchvision/datasets/FashionMNIST',
                train=False, transform=transform)
            return _load_and_augment(tr, te)

        if dataset in ('mnist', 'pca_mnist'):
            tr = torchvision.datasets.MNIST(
                '~/.torchvision/datasets/MNIST',
                train=True, download=True, transform=transform)
            te = torchvision.datasets.MNIST(
                '~/.torchvision/datasets/MNIST',
                train=False, transform=transform)
            return _load_and_augment(tr, te)

        if dataset == 'kmnist':
            tr = torchvision.datasets.KMNIST(
                '~/.torchvision/datasets/KMNIST',
                train=True, download=True, transform=transform)
            te = torchvision.datasets.KMNIST(
                '~/.torchvision/datasets/KMNIST',
                train=False, transform=transform)
            return _load_and_augment(tr, te)

        if dataset == 'emnist-letters':
            tr = torchvision.datasets.EMNIST(
                '~/.torchvision/datasets/EMNIST',
                train=True, download=True, transform=transform, split='letters')
            te = torchvision.datasets.EMNIST(
                '~/.torchvision/datasets/EMNIST',
                train=False, transform=transform, split='letters')
            return _load_and_augment(tr, te)

        if dataset in ('cifar10', 'pca_cifar10'):
            tr = torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=True, download=True, transform=transform)
            te = torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=False, transform=transform)
            return _load_and_augment(tr, te)

        if dataset == 'cifar_catdog':
            tr = [(x, y) for x, y in torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=True, download=True, transform=transform) if y in [3, 5]]
            te = [(x, y) for x, y in torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=False, transform=transform) if y in [3, 5]]
            x, y, idx = intertwine_labels(*dataset_to_tensors(tr + te))
            x = aug_fn(x)
            x = center_normalize(x)
            return intertwine_split(x, y, idx, ps, seeds, y.unique())

        if dataset == 'cifar_shipbird':
            tr = [(x, y) for x, y in torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=True, download=True, transform=transform) if y in [8, 2]]
            te = [(x, y) for x, y in torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=False, transform=transform) if y in [8, 2]]
            x, y, idx = intertwine_labels(*dataset_to_tensors(tr + te))
            x = aug_fn(x)
            x = center_normalize(x)
            return intertwine_split(x, y, idx, ps, seeds, y.unique())

        if dataset == 'cifar_catplane':
            tr = [(x, y) for x, y in torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=True, download=True, transform=transform) if y in [3, 0]]
            te = [(x, y) for x, y in torchvision.datasets.CIFAR10(
                '~/.torchvision/datasets/CIFAR10',
                train=False, transform=transform) if y in [3, 0]]
            x, y, idx = intertwine_labels(*dataset_to_tensors(tr + te))
            x = aug_fn(x)
            x = center_normalize(x)
            return intertwine_split(x, y, idx, ps, seeds, y.unique())

        if dataset == 'cifar_animal':
            tr = [(x, 0 if y in [0, 1, 8, 9] else 1)
                  for x, y in torchvision.datasets.CIFAR10(
                      '~/.torchvision/datasets/CIFAR10',
                      train=True, download=True, transform=transform)]
            te = [(x, 0 if y in [0, 1, 8, 9] else 1)
                  for x, y in torchvision.datasets.CIFAR10(
                      '~/.torchvision/datasets/CIFAR10',
                      train=False, transform=transform)]
            x, y, idx = intertwine_labels(*dataset_to_tensors(tr + te))
            x = aug_fn(x)
            x = center_normalize(x)
            return intertwine_split(x, y, idx, ps, seeds, y.unique())

        # Fallback (should not reach here for valid spatial datasets)
        return original(dataset, ps, seeds, d, params)

    # Copy lru_cache interface so callers can call cache_clear()
    import functools
    patched = functools.lru_cache(maxsize=2)(patched)

    return patched, original


def augmentation_summary() -> str:
    lines = []
    lines.append(f"{'Augmentation':<20}  Description")
    lines.append("-" * 70)
    descs = {
        'identity':        'No change — sanity check (must equal baseline)',
        'hflip':           'Horizontal flip (left ↔ right)',
        'vflip':           'Vertical flip (up ↔ down)',
        'rot90_cw':        '90° clockwise rotation',
        'rot90_ccw':       '90° counter-clockwise rotation',
        'rot180':          '180° rotation',
        'transpose':       'Transpose (reflect across main diagonal)',
        'center_crop_20':  'Centre-crop to 20px, resize to 28px (strong zoom)',
        'center_crop_24':  'Centre-crop to 24px, resize to 28px (mild zoom)',
        'noise_005':       'Additive Gaussian noise σ=0.05, clipped to [0,1]',
        'noise_010':       'Additive Gaussian noise σ=0.10, clipped to [0,1]',
        'noise_020':       'Additive Gaussian noise σ=0.20, clipped to [0,1]',
        'invert':          'Negate pixels: 1 − x',
        'brightness_up':   'Add 0.15 to all pixels, clip to [0,1]',
        'brightness_down': 'Subtract 0.15 from all pixels, clip to [0,1]',
        'blur_3x3':        '3×3 box blur (average filter)',
    }
    for name in _AUGMENTATION_FNS:
        lines.append(f"{name:<20}  {descs.get(name, '')}")
    return '\n'.join(lines)


if __name__ == '__main__':
    print(augmentation_summary())
    print()
    # Quick smoke test: all augmentations run on a random (4, 1, 28, 28) batch
    x = torch.rand(4, 1, 28, 28, dtype=torch.float64)
    print("Smoke test: augmentations on (4,1,28,28) batch")
    for name, fn in _AUGMENTATION_FNS.items():
        out = fn(x)
        assert out.shape == x.shape, f"{name}: shape mismatch {out.shape}"
        assert out.dtype == x.dtype, f"{name}: dtype changed {out.dtype}"
        print(f"  {name:<20}  ✓  range=[{out.min():.3f}, {out.max():.3f}]")
    print("\nSanity: identity(x) == x?", torch.allclose(_identity(x), x))
