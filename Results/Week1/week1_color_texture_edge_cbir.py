from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import local_binary_pattern, hog


# ============================================================
# 1. FIND TRAINING AND TEST IMAGES
# ============================================================

train_dir = Path("images_tr")
test_dir = Path("images_test")

train_images = list(train_dir.rglob("*.jpg"))
test_images = list(test_dir.rglob("*.jpg"))

print("Training images:", len(train_images))
print("Test images:", len(test_images))


# ============================================================
# 2. SAFE IMAGE LOADING
# ============================================================

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    return np.array(image)


# ============================================================
# 3. COLOR FEATURE — HSV HISTOGRAM
# ============================================================

def color_histogram(image_path):

    image = load_image(image_path)

    # RGB → HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256]
    )

    hist = cv2.normalize(
        hist,
        hist
    )

    return hist.flatten()


# ============================================================
# 4. TEXTURE FEATURE — LBP
# ============================================================

def texture_lbp(image_path):

    image = load_image(image_path)

    # RGB → grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    n_bins = points + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    hist = hist.astype(float)

    hist /= (
        hist.sum() + 1e-7
    )

    return hist


# ============================================================
# 5. SHAPE FEATURE — HOG
# ============================================================

def shape_hog(image_path):

    image = load_image(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Resize all images to the same size
    gray = cv2.resize(
        gray,
        (128, 128)
    )

    # HOG feature extraction
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.astype(np.float32)


# ============================================================
# 6. HISTOGRAM SIMILARITY
# ============================================================

def histogram_similarity(
    feature1,
    feature2
):

    return cv2.compareHist(
        feature1.astype(np.float32),
        feature2.astype(np.float32),
        cv2.HISTCMP_CORREL
    )


# ============================================================
# 7. HOG SIMILARITY — COSINE SIMILARITY
# ============================================================

def hog_similarity(
    feature1,
    feature2
):

    denominator = (
        np.linalg.norm(feature1)
        * np.linalg.norm(feature2)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(feature1, feature2)
        / denominator
    )


# ============================================================
# 8. SELECT QUERY IMAGE
# ============================================================

query_path = test_images[0]

print("\nQuery image:")
print(query_path)


# ============================================================
# 9. EXTRACT QUERY FEATURES
# ============================================================

query_color = color_histogram(
    query_path
)

query_texture = texture_lbp(
    query_path
)

query_hog = shape_hog(
    query_path
)


# ============================================================
# 10. SEARCH TRAINING DATABASE
# ============================================================

results = []

for i, image_path in enumerate(train_images):

    # ----------------------------------------
    # Extract features
    # ----------------------------------------

    color_feature = color_histogram(
        image_path
    )

    texture_feature = texture_lbp(
        image_path
    )

    hog_feature = shape_hog(
        image_path
    )

    # ----------------------------------------
    # Calculate individual similarities
    # ----------------------------------------

    color_score = histogram_similarity(
        query_color,
        color_feature
    )

    texture_score = histogram_similarity(
        query_texture,
        texture_feature
    )

    hog_score = hog_similarity(
        query_hog,
        hog_feature
    )

    # ----------------------------------------
    # FEATURE FUSION
    # ----------------------------------------
    #
    # 50% Color
    # 25% Texture
    # 25% Shape
    #

    combined_score = (
        0.50 * color_score
        + 0.25 * texture_score
        + 0.25 * hog_score
    )

    results.append(
        (
            image_path,
            combined_score,
            color_score,
            texture_score,
            hog_score
        )
    )

    # Progress
    if (i + 1) % 200 == 0:

        print(
            f"Processed "
            f"{i + 1}/{len(train_images)}"
        )


# ============================================================
# 11. SORT RESULTS
# ============================================================

results.sort(
    key=lambda x: x[1],
    reverse=True
)

top_k = results[:5]


# ============================================================
# 12. PRINT TOP 5
# ============================================================

print("\n==========================================")
print("TOP 5 RETRIEVED IMAGES")
print("HSV + LBP + HOG")
print("==========================================")

for rank, (
    image_path,
    combined_score,
    color_score,
    texture_score,
    hog_score
) in enumerate(top_k, 1):

    print(f"\nRank {rank}")

    print(
        "Image:",
        image_path
    )

    print(
        f"Combined Similarity: "
        f"{combined_score:.4f}"
    )

    print(
        f"Color Similarity:    "
        f"{color_score:.4f}"
    )

    print(
        f"Texture Similarity:  "
        f"{texture_score:.4f}"
    )

    print(
        f"HOG Similarity:      "
        f"{hog_score:.4f}"
    )


# ============================================================
# 13. DISPLAY QUERY + TOP 5
# ============================================================

fig, axes = plt.subplots(
    1,
    6,
    figsize=(18, 4)
)


# ----------------------------------------
# Query
# ----------------------------------------

query_image = load_image(
    query_path
)

axes[0].imshow(
    query_image
)

axes[0].set_title(
    "QUERY"
)

axes[0].axis("off")


# ----------------------------------------
# Retrieved images
# ----------------------------------------

for ax, (
    image_path,
    combined_score,
    color_score,
    texture_score,
    hog_score
) in zip(
    axes[1:],
    top_k
):

    image = load_image(
        image_path
    )

    ax.imshow(
        image
    )

    ax.set_title(
        f"Combined\n"
        f"{combined_score:.3f}"
    )

    ax.axis("off")


plt.tight_layout()

plt.show()