import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
import sys

week1_path = Path(__file__).resolve().parent.parent / "Week1"
sys.path.append(str(week1_path))
from cache_features import list_images, cached_features, TRAIN_DIR

def load_image(image_path):
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))

def texture_lbp_r2_p16(image_path):
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 16, 2, method="uniform")
    n_bins = 16 + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-7
    return hist

def shape_hog_c8(image_path):
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

def main():
    train_paths = list_images(TRAIN_DIR)
    
    # 1. Multi-scale LBP (r=2, p=16)
    name_lbp2 = "train_lbp_r2_p16"
    cached_features(name_lbp2, train_paths, texture_lbp_r2_p16)
    
    # 2. Alternative HOG (c8, b2)
    name_hog_fine = "train_hog_128_o9_c8_b2"
    cached_features(name_hog_fine, train_paths, shape_hog_c8)

if __name__ == "__main__":
    main()
