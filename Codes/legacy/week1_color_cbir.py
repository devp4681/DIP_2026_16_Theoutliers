from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Find training and test images
# ==========================================

train_dir = Path("images_tr")
test_dir = Path("images_test")

train_images = list(train_dir.rglob("*.jpg"))
test_images = list(test_dir.rglob("*.jpg"))

print("Training images:", len(train_images))
print("Test images:", len(test_images))


# ==========================================
# 2. Extract color histogram
# ==========================================

def color_histogram(image_path):

    image = cv2.imread(str(image_path))

    # BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2D Hue-Saturation histogram
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


# ==========================================
# 3. Similarity function
# ==========================================

def calculate_similarity(feature1, feature2):

    return cv2.compareHist(
        feature1.astype(np.float32),
        feature2.astype(np.float32),
        cv2.HISTCMP_CORREL
    )


# ==========================================
# 4. Select a query image
# ==========================================

query_path = test_images[0]

print("\nQuery image:")
print(query_path)


# ==========================================
# 5. Extract query feature
# ==========================================

query_feature = color_histogram(query_path)


# ==========================================
# 6. Search training database
# ==========================================

results = []

for image_path in train_images:

    feature = color_histogram(image_path)

    score = calculate_similarity(
        query_feature,
        feature
    )

    results.append(
        (image_path, score)
    )


# ==========================================
# 7. Rank results
# ==========================================

results.sort(
    key=lambda x: x[1],
    reverse=True
)

top_k = results[:5]


# ==========================================
# 8. Print results
# ==========================================

print("\nTop 5 retrieved images:")

for rank, (image_path, score) in enumerate(top_k, 1):

    print(
        f"{rank}. "
        f"{image_path} "
        f"Similarity = {score:.4f}"
    )


# ==========================================
# 9. Display results
# ==========================================

fig, axes = plt.subplots(
    1,
    6,
    figsize=(18, 4)
)


# Query
query_image = cv2.imread(str(query_path))
query_image = cv2.cvtColor(
    query_image,
    cv2.COLOR_BGR2RGB
)

axes[0].imshow(query_image)
axes[0].set_title("QUERY")
axes[0].axis("off")


# Retrieved images
for ax, (image_path, score) in zip(
    axes[1:],
    top_k
):

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(image)

    ax.set_title(
        f"Similarity\n{score:.3f}"
    )

    ax.axis("off")


plt.tight_layout()
plt.show()