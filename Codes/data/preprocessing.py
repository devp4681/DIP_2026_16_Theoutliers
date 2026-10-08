"""
DIP Preprocessing Module
Focuses on classical Digital Image Processing techniques for aerial images:
- Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination balancing
- Global Histogram Equalization & Linear Contrast Stretching
- Gaussian and Bilateral filtering for noise smoothing while preserving structural boundaries
"""

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on Luminance (L) channel in LAB color space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    enhanced_lab = cv2.merge((l_clahe, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def apply_gaussian_blur(image: np.ndarray, kernel_size: tuple = (3, 3), sigma: float = 0.5) -> np.ndarray:
    """
    Smooth high-frequency aerial noise using Gaussian smoothing.
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)


def apply_bilateral_filter(image: np.ndarray, d: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Apply edge-preserving bilateral filter.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def linear_contrast_stretching(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """
    Normalize dynamic range via linear contrast stretching between specified percentiles.
    """
    img_float = image.astype(np.float32)
    p_low, p_high = np.percentile(img_float, (low_percentile, high_percentile))
    if p_high > p_low:
        stretched = np.clip((img_float - p_low) / (p_high - p_low) * 255.0, 0, 255)
        return stretched.astype(np.uint8)
    return image


def preprocess_image(
    image: np.ndarray,
    use_clahe: bool = True,
    use_gaussian: bool = False,
    clip_limit: float = 2.0,
    kernel_size: tuple = (3, 3),
    sigma: float = 0.5
) -> np.ndarray:
    """
    Standard preprocessing pipeline combining optional CLAHE and optional Gaussian blur.
    """
    processed = image.copy()
    if use_clahe:
        processed = apply_clahe(processed, clip_limit=clip_limit)
    if use_gaussian:
        processed = apply_gaussian_blur(processed, kernel_size=kernel_size, sigma=sigma)
    return processed
