from pathlib import Path
from PIL import Image

train_dir = Path("images_tr")
test_dir = Path("images_test")

all_images = (
    list(train_dir.rglob("*.jpg")) +
    list(test_dir.rglob("*.jpg"))
)

bad_images = []

print("Checking images...")

for i, image_path in enumerate(all_images, 1):

    try:
        with Image.open(image_path) as img:
            img.verify()

    except Exception as e:
        bad_images.append((image_path, str(e)))

    if i % 500 == 0:
        print(f"Checked {i}/{len(all_images)}")

print("\n--------------------------------")
print("CHECK COMPLETE")
print("--------------------------------")

print("Total images:", len(all_images))
print("Problematic images:", len(bad_images))

for image_path, error in bad_images:
    print("\nProblem image:")
    print(image_path)
    print("Error:", error)