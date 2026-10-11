import json
from pathlib import Path
import sys

week3_path = Path(__file__).resolve().parent
sys.path.append(str(week3_path.parent / "Week1"))

from cache_features import list_images, cached_features, TEST_DIR
from week3_variants import texture_lbp_r2_p16

def main():
    test_paths = list_images(TEST_DIR)
    name_lbp2 = "test_lbp_r2_p16"
    cached_features(name_lbp2, test_paths, texture_lbp_r2_p16)

if __name__ == "__main__":
    main()
