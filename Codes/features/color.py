"""
DIP Color Feature Extraction Module
Features from standard DIP curriculum:
1. 2D HSV Color Histogram (Hue-Saturation) - captures chromatic distribution independent of illumination intensity.
2. Color Moments (1st: Mean, 2nd: Std Dev, 3rd: Skewness) per channel in HSV and RGB color spaces.
"""

import cv2
import numpy as np


def extract_hsv_histogram(image: np.ndarray, bins: tuple = (16, 16)) -> np.ndarray:
    """
    Extract 2D Hue-Saturation normalized histogram from an RGB image.
    Bins: (H_bins, S_bins), default (16, 16) -> 256-D feature vector.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float64)
    norm = hist.sum()
    if norm > 0:
        hist /= norm
    return hist


def extract_color_moments(image: np.ndarray) -> np.ndarray:
    """
    Extract first 3 color moments (Mean, Standard Deviation, Skewness)
    across each channel in both RGB and HSV spaces (2 x 3 x 3 = 18-D feature vector).
    """
    moments = []
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for color_img in (image, hsv):
        for c in range(3):
            channel = color_img[:, :, c].astype(np.float64)
            mu = float(np.mean(channel))
            sigma = float(np.std(channel))
            diff = channel - mu
            skew = float(np.cbrt(np.mean(diff ** 3)))
            moments.extend([mu, sigma, skew])
    moments = np.array(moments, dtype=np.float64)
    norm = np.linalg.norm(moments)
    if norm > 0:
        moments /= norm
    return moments
