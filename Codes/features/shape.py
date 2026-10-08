"""
DIP Shape and Boundary Feature Extraction Module
Features from standard DIP curriculum:
1. Edge Direction Histogram (EDH) using Sobel Gradient Operators:
   Computes directional gradients G_x and G_y, calculates edge orientations,
   and bins gradient magnitudes into an orientation histogram.
2. Histogram of Oriented Gradients (HOG) descriptor.
"""

import cv2
import numpy as np
from skimage.feature import hog


def extract_edge_direction_histogram(
    image: np.ndarray,
    num_bins: int = 18,
    magnitude_threshold: float = 20.0
) -> np.ndarray:
    """
    Extract Edge Direction Histogram (EDH) using Sobel operators.
    Calculates gradient magnitude and orientation, accumulating magnitudes into angular bins [0, 180).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angles = np.arctan2(gy, gx) * (180.0 / np.pi)
    angles = np.mod(angles, 180.0)

    mask = magnitude >= magnitude_threshold
    valid_angles = angles[mask]
    valid_weights = magnitude[mask]

    if len(valid_angles) == 0:
        return np.zeros(num_bins, dtype=np.float64)

    hist, _ = np.histogram(valid_angles, bins=num_bins, range=(0.0, 180.0), weights=valid_weights)
    hist = hist.astype(np.float64)
    norm = hist.sum()
    if norm > 0:
        hist /= norm
    return hist


def extract_hog_descriptor(
    image: np.ndarray,
    target_size: tuple = (128, 128),
    orientations: int = 9,
    pixels_per_cell: tuple = (16, 16),
    cells_per_block: tuple = (2, 2)
) -> np.ndarray:
    """
    Extract Histogram of Oriented Gradients (HOG) feature descriptor.
    Resizes image to target_size (e.g. 128x128) and computes block-normalized gradient histograms.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    feat = hog(
        resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys'
    )
    feat = feat.astype(np.float64)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm
    return feat
