"""
DIP Image Similarity and Distance Metrics Module
Classical distance and similarity measures:
- Histogram Correlation
- Chi-Square Distance & Similarity
- Histogram Intersection
- Bhattacharyya / Hellinger Similarity
- Cosine Similarity
- Euclidean Distance
"""

import numpy as np


def correlation_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Compute Pearson correlation between two 1D feature histograms.
    Matches cv2.HISTCMP_CORREL formulation.
    """
    f1_c = f1 - np.mean(f1)
    f2_c = f2 - np.mean(f2)
    denom = np.sqrt(np.sum(f1_c ** 2) * np.sum(f2_c ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(f1_c * f2_c) / denom)


def chi_square_distance(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Chi-Square distance between two histograms.
    """
    denom = f1 + f2 + eps
    return float(0.5 * np.sum(((f1 - f2) ** 2) / denom))


def chi_square_similarity(f1: np.ndarray, f2: np.ndarray, gamma: float = 1.0) -> float:
    """
    Exponential Chi-Square similarity in range (0, 1].
    """
    d = chi_square_distance(f1, f2)
    return float(np.exp(-gamma * d))


def histogram_intersection(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Histogram intersection similarity in range [0, 1].
    """
    return float(np.sum(np.minimum(f1, f2)))


def bhattacharyya_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Bhattacharyya coefficient similarity in range [0, 1].
    """
    p1 = np.maximum(f1, 0.0)
    p2 = np.maximum(f2, 0.0)
    return float(np.sum(np.sqrt(p1 * p2)))


def cosine_similarity(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-12) -> float:
    """
    Cosine similarity between two arbitrary feature vectors.
    """
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    if norm1 < eps or norm2 < eps:
        return 0.0
    return float(np.dot(f1, f2) / (norm1 * norm2))


def euclidean_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    L2 Euclidean distance.
    """
    return float(np.linalg.norm(f1 - f2))
