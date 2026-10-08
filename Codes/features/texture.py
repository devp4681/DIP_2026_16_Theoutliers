"""
DIP Texture Feature Extraction Module
Features from standard DIP curriculum:
1. Local Binary Patterns (LBP) - micro-texture descriptor
2. Gray-Level Co-occurrence Matrix (GLCM) - statistical 2nd-order texture descriptor
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def extract_lbp_histogram(
    image: np.ndarray,
    radius: int = 1,
    points: int = 8,
    method: str = "uniform"
) -> np.ndarray:
    """
    Extract normalized Local Binary Pattern (LBP) histogram.
    Default: radius=1, points=8, uniform -> 10-D feature vector.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lbp = local_binary_pattern(gray, points, radius, method=method)
    if method == "uniform":
        n_bins = points + 2
    else:
        n_bins = 2 ** points

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    norm = hist.sum()
    if norm > 0:
        hist /= norm
    return hist


def extract_glcm_features(
    image: np.ndarray,
    distances: list = [1],
    angles: list = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels: int = 32
) -> np.ndarray:
    """
    Extract Haralick texture features from Gray-Level Co-occurrence Matrix (GLCM).
    Computes mean and std of Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    quantized = (gray // (256 // levels)).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feature_vals = []
    for prop in props:
        vals = graycoprops(glcm, prop).ravel()
        feature_vals.append(float(np.mean(vals)))
        feature_vals.append(float(np.std(vals)))

    feats = np.array(feature_vals, dtype=np.float64)
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats /= norm
    return feats
