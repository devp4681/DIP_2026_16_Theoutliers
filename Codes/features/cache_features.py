"""
cache_features.py

Extracts HSV, LBP and HOG features for every training and test image
and saves them to disk as .npy files, so later experiments (fusion
weights, evaluation, etc.) never have to recompute them.

Run once:
    python cache_features.py

Produces 6 feature files + 6 path files in ./cache/
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

# ============================================================
# 0. SETTINGS  (if you change any of these, the cache file names
#    change too, so old caches are never reused by mistake)
# ============================================================

TRAIN_DIR = Path("images_tr")
TEST_DIR = Path("images_test")
CACHE_DIR = Path("cache")

# HSV histogram
HSV_BINS = (16, 16)            # H bins, S bins

# LBP
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# HOG
HOG_SIZE = (128, 128)          # images are resized to this before HOG
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# Tags used in file names
HSV_TAG = f"hsv_{HSV_BINS[0]}x{HSV_BINS[1]}"
LBP_TAG = f"lbp_r{LBP_RADIUS}_p{LBP_POINTS}"
HOG_TAG = (
    f"hog_{HOG_SIZE[0]}_o{HOG_ORIENTATIONS}"
    f"_c{HOG_PIXELS_PER_CELL[0]}_b{HOG_CELLS_PER_BLOCK[0]}"
)


# ============================================================
# 1. IMAGE LOADING (Pillow, avoids the JPEG warnings from cv2)
# ============================================================

def load_image(image_path):
    """Return the image as an RGB uint8 numpy array."""
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"))


def list_images(folder):
    """Sorted list of all .jpg files under a folder (sorted = stable order)."""
    return sorted(Path(folder).rglob("*.jpg"))


# ============================================================
# 2. COLOR FEATURE: HSV histogram
# ============================================================

def color_hsv(image_path):
    image = load_image(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1], None,
        list(HSV_BINS),
        [0, 180, 0, 256],
    )
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


# ============================================================
# 3. TEXTURE FEATURE: uniform LBP histogram
# ============================================================

def texture_lbp(image_path):
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(
        gray, LBP_POINTS, LBP_RADIUS, method="uniform"
    )

    n_bins = LBP_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-7
    return hist


# ============================================================
# 4. SHAPE FEATURE: HOG
# ============================================================

def shape_hog(image_path):
    image = load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Fixed size so every image gives a vector of the same length
    gray = cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_AREA)

    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )


# ============================================================
# 5. CACHING
# ============================================================

def cached_features(name, paths, extractor):
    """
    Compute features once and save them; reload from disk afterwards.
    The cache is reused only if it was built from the same images
    in the same order.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    paths = [str(p) for p in paths]
    feat_file = CACHE_DIR / f"{name}.npy"
    path_file = CACHE_DIR / f"{name}_paths.json"

    if feat_file.exists() and path_file.exists():
        if json.loads(path_file.read_text()) == paths:
            print(f"[cache hit]  {name}")
            return np.load(feat_file)
        print(f"[cache stale] {name}: image list changed, recomputing")

    feats = []
    for i, p in enumerate(paths, 1):
        feats.append(extractor(p))
        if i % 200 == 0 or i == len(paths):
            print(f"[compute]    {name}: {i}/{len(paths)}")

    feats = np.array(feats, dtype=np.float32)
    np.save(feat_file, feats)
    path_file.write_text(json.dumps(paths))
    return feats


def load_cached(name):
    """Load a cached feature matrix and its image paths (for evaluation)."""
    feats = np.load(CACHE_DIR / f"{name}.npy")
    paths = json.loads((CACHE_DIR / f"{name}_paths.json").read_text())
    return feats, paths


# ============================================================
# 6. GENERATE ALL 6 CACHE FILES
# ============================================================

def main():
    train_paths = list_images(TRAIN_DIR)
    test_paths = list_images(TEST_DIR)
    print(f"Training images: {len(train_paths)}")
    print(f"Test images:     {len(test_paths)}\n")

    jobs = [
        (f"train_{HSV_TAG}", train_paths, color_hsv),
        (f"train_{LBP_TAG}", train_paths, texture_lbp),
        (f"train_{HOG_TAG}", train_paths, shape_hog),
        (f"test_{HSV_TAG}", test_paths, color_hsv),
        (f"test_{LBP_TAG}", test_paths, texture_lbp),
        (f"test_{HOG_TAG}", test_paths, shape_hog),
    ]

    print("Summary")
    print("-" * 50)
    for name, paths, extractor in jobs:
        feats = cached_features(name, paths, extractor)
        print(f"{name:<35} shape = {feats.shape}")

    print("\nDone. Files saved in:", CACHE_DIR.resolve())


if __name__ == "__main__":
    main()