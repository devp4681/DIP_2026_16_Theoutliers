"""
DIP Spatial Feature Extraction Module
Implements spatial grid partitioning (e.g. 2x2 quadrants) to capture spatial layout:
- Spatial HSV color histograms
- Spatial Uniform LBP texture histograms
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def split_grid_cells(h: int, w: int, grid: tuple = (2, 2)):
    """
    Yield slice ranges (row_slice, col_slice) for each grid cell in row-major order.
    """
    rows, cols = grid
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    for i in range(rows):
        for j in range(cols):
            yield slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1])


def extract_spatial_hsv(
    image: np.ndarray,
    grid: tuple = (2, 2),
    bins: tuple = (8, 8)
) -> np.ndarray:
    """
    Extract concatenated HSV histograms computed independently across spatial grid quadrants.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    out = []
    for ys, xs in split_grid_cells(h, w, grid=grid):
        cell = hsv[ys, xs]
        hist = cv2.calcHist([cell], [0, 1], None, list(bins), [0, 180, 0, 256])
        hist = hist.flatten().astype(np.float64)
        norm = hist.sum()
        if norm > 0:
            hist /= norm
        out.append(hist)
    return np.concatenate(out)


def extract_spatial_lbp(
    image: np.ndarray,
    grid: tuple = (2, 2),
    radius: int = 1,
    points: int = 8
) -> np.ndarray:
    """
    Extract concatenated Uniform LBP histograms computed independently across spatial grid quadrants.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    n_bins = points + 2
    out = []
    h, w = lbp.shape
    for ys, xs in split_grid_cells(h, w, grid=grid):
        cell_lbp = lbp[ys, xs]
        hist, _ = np.histogram(cell_lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64)
        norm = hist.sum()
        if norm > 0:
            hist /= norm
        out.append(hist)
    return np.concatenate(out)
