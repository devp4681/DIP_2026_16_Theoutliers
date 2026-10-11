"""
week2_spatial_features.py

Adds 2x2 SPATIAL features on top of the Week 1 cache:
    shsv : HSV histogram computed separately in each of the 4 quadrants
    slbp : uniform-LBP histogram computed separately in each of the 4 quadrants

Each vector is stored as [region1 | region2 | region3 | region4], so it can be
reshaped to (n_images, 4, bins) later and compared region-by-region.

Run once (after cache_features.py):
    python week2_spatial_features.py
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import sys
from pathlib import Path

# This finds the Week1 folder relative to this file and adds it to Python's search path
week1_path = Path(__file__).resolve().parent.parent / "Week1"
sys.path.append(str(week1_path))
import cache_features as cf
from cache_features import (
    load_image, list_images, cached_features,
    TRAIN_DIR, TEST_DIR, LBP_RADIUS, LBP_POINTS,
)

# ---------------- settings (changing them changes the cache names) -------------
GRID = (2, 2)                     # rows, cols  -> 4 regions
N_REGIONS = GRID[0] * GRID[1]
SHSV_BINS = (8, 8)                # coarser than global 16x16: each region has 1/4 of the pixels
SHSV_TAG = f"shsv_{GRID[0]}x{GRID[1]}_{SHSV_BINS[0]}x{SHSV_BINS[1]}"
SLBP_TAG = f"slbp_{GRID[0]}x{GRID[1]}_r{LBP_RADIUS}_p{LBP_POINTS}"


def _cells(h, w):
    """Yield (row_slice, col_slice) for every grid cell, row-major order."""
    ys = np.linspace(0, h, GRID[0] + 1).astype(int)
    xs = np.linspace(0, w, GRID[1] + 1).astype(int)
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            yield slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1])


def spatial_hsv(image_path):
    hsv = cv2.cvtColor(load_image(image_path), cv2.COLOR_RGB2HSV)
    out = []
    for ys, xs in _cells(*hsv.shape[:2]):
        h = cv2.calcHist([hsv[ys, xs]], [0, 1], None, list(SHSV_BINS), [0, 180, 0, 256])
        h = h.flatten().astype(np.float64)
        out.append(h / (h.sum() + 1e-7))
    return np.concatenate(out)


def spatial_lbp(image_path):
    gray = cv2.cvtColor(load_image(image_path), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_POINTS + 2
    out = []
    for ys, xs in _cells(*lbp.shape):
        h, _ = np.histogram(lbp[ys, xs].ravel(), bins=n_bins, range=(0, n_bins))
        h = h.astype(np.float64)
        out.append(h / (h.sum() + 1e-7))
    return np.concatenate(out)


def main():
    train_paths, test_paths = list_images(TRAIN_DIR), list_images(TEST_DIR)
    jobs = [
        (f"train_{SHSV_TAG}", train_paths, spatial_hsv),
        (f"train_{SLBP_TAG}", train_paths, spatial_lbp),
        (f"test_{SHSV_TAG}", test_paths, spatial_hsv),
        (f"test_{SLBP_TAG}", test_paths, spatial_lbp),
    ]
    for name, paths, fn in jobs:
        feats = cached_features(name, paths, fn)
        print(f"{name:<40} shape = {feats.shape}")


if __name__ == "__main__":
    main()
