from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import local_binary_pattern


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
    """
    Load JPEG using Pillow instead of OpenCV's JPEG decoder.
    Returns an RGB NumPy array.
    """
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


# ============================================================
# 3. COLOR FEATURE — HSV HISTOGRAM
# ============================================================

def color_histogram(image_path):

    image = load_image(image_path)

    # RGB → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256]
    )

    # Normalize
    hist = cv2.normalize(hist, hist)

    return hist.flatten()


# ============================================================
# 4. TEXTURE FEATURE — LBP
# ============================================================

def texture_lbp(image_path):

    image = load_image(image_path)

    # RGB → grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # LBP parameters
    radius = 1
    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    # Histogram of LBP patterns
    n_bins = points + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )

    # Normalize
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)

    return hist


# ============================================================
# 5. SIMILARITY FUNCTION
# ============================================================

def calculate_similarity(feature1, feature2):

    return cv2.compareHist(
        feature1.astype(np.float32),
        feature2.astype(np.float32),
        cv2.HISTCMP_CORREL
    )


# ============================================================
# 6. SELECT QUERY IMAGE
# ============================================================

query_path = test_images[0]

print("\nQuery image:")
print(query_path)


# ============================================================
# 7. EXTRACT QUERY FEATURES
# ============================================================

query_color = color_histogram(query_path)
query_texture = texture_lbp(query_path)


# ============================================================
# 8. SEARCH TRAINING DATABASE
# ============================================================

results = []

for i, image_path in enumerate(train_images):

    # Extract features
    color_feature = color_histogram(image_path)
    texture_feature = texture_lbp(image_path)

    # Color similarity
    color_score = calculate_similarity(
        query_color,
        color_feature
    )

    # Texture similarity
    texture_score = calculate_similarity(
        query_texture,
        texture_feature
    )

    # ========================================================
    # FEATURE FUSION
    # ========================================================

    # 70% Color + 30% Texture
    combined_score = (
        0.7 * color_score +
        0.3 * texture_score
    )

    results.append(
        (
            image_path,
            combined_score,
            color_score,
            texture_score
        )
    )

    # Progress
    if (i + 1) % 200 == 0:
        print(
            f"Processed {i + 1}/{len(train_images)}"
        )


# ============================================================
# 9. SORT RESULTS
# ============================================================

results.sort(
    key=lambda x: x[1],
    reverse=True
)

top_k = results[:5]


# ============================================================
# 10. PRINT TOP 5
# ============================================================

print("\n==========================================")
print("TOP 5 RETRIEVED IMAGES")
print("Color + Texture (70% + 30%)")
print("==========================================")

for rank, (
    image_path,
    combined_score,
    color_score,
    texture_score
) in enumerate(top_k, 1):

    print(f"\nRank {rank}")
    print("Image:", image_path)
    print(f"Combined Similarity: {combined_score:.4f}")
    print(f"Color Similarity:    {color_score:.4f}")
    print(f"Texture Similarity:  {texture_score:.4f}")


# ============================================================
# 11. DISPLAY QUERY + TOP 5
# ============================================================

fig, axes = plt.subplots(
    1,
    6,
    figsize=(18, 4)
)


# ---------------------------
# Query image
# ---------------------------

query_image = load_image(query_path)

axes[0].imshow(query_image)
axes[0].set_title("QUERY")
axes[0].axis("off")


# ---------------------------
# Retrieved images
# ---------------------------

for ax, (
    image_path,
    combined_score,
    color_score,
    texture_score
) in zip(axes[1:], top_k):

    image = load_image(image_path)

    ax.imshow(image)

    ax.set_title(
        f"Combined\n{combined_score:.3f}"
    )

    ax.axis("off")


plt.tight_layout()
plt.show()