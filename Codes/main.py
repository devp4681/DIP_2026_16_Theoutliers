"""
Main Entry-Point for Explainable CBIR System
Digital Image Processing (DIP 2026) - Project 16 | Theoutliers

Usage:
    python Codes/main.py --query images_test/airport_1.jpg --top-k 5
    python Codes/main.py --gallery images_tr --top-k 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Codes.data.dataset import safe_load_image, parse_image_name
from Codes.data.preprocessing import preprocess_image
from Codes.features.color import extract_hsv_histogram
from Codes.features.texture import extract_lbp_histogram
from Codes.features.shape import extract_hog_descriptor
from Codes.retrieval.ranker import CBIRRanker


def extract_all_features(image: np.ndarray) -> dict:
    """
    Extract HSV color, Uniform LBP texture, and HOG shape features.
    """
    return {
        'color': extract_hsv_histogram(image),
        'texture': extract_lbp_histogram(image),
        'shape': extract_hog_descriptor(image)
    }


def run_query(
    query_path: Path,
    gallery_dir: Path,
    top_k: int = 5,
    weights: dict = None,
    znorm: bool = False,
    use_clahe: bool = False
):
    """
    Execute retrieval for a given query image path against gallery directory.
    """
    print(f"Loading query image: {query_path}")
    q_img = safe_load_image(query_path)
    if use_clahe:
        q_img = preprocess_image(q_img, use_clahe=True)
    query_feats = extract_all_features(q_img)

    gallery_files = sorted(list(gallery_dir.rglob("*.jpg")) + list(gallery_dir.rglob("*.png")))
    if not gallery_files:
        print(f"No gallery images found in {gallery_dir}")
        return

    print(f"Scanning {len(gallery_files)} gallery images...")
    gallery_names = []
    gallery_feats = {'color': [], 'texture': [], 'shape': []}

    for gf in gallery_files:
        gallery_names.append(gf.name)
        img = safe_load_image(gf)
        if use_clahe:
            img = preprocess_image(img, use_clahe=True)
        f = extract_all_features(img)
        gallery_feats['color'].append(f['color'])
        gallery_feats['texture'].append(f['texture'])
        gallery_feats['shape'].append(f['shape'])

    ranker = CBIRRanker(weights=weights, znorm=znorm)
    results = ranker.rank(query_feats, gallery_feats, gallery_names, top_k=top_k)

    query_stem, query_class = parse_image_name(query_path)
    print(f"\nTop-{top_k} Retrieval Results for '{query_stem}' (Scene: {query_class}):")
    print(f"{'Rank':<5} | {'Image Name':<25} | {'Score':<8} | {'Color':<8} | {'Texture':<8} | {'Shape':<8}")
    print("------------------------------------------------------------------------")
    for res in results:
        sub = res['subscores']
        c_score = sub.get('color', 0.0)
        t_score = sub.get('texture', 0.0)
        s_score = sub.get('shape', 0.0)
        print(f"{res['rank']:<5} | {res['name']:<25} | {res['score']:<8.4f} | {c_score:<8.4f} | {t_score:<8.4f} | {s_score:<8.4f}")


def main():
    parser = argparse.ArgumentParser(description='Explainable CBIR System - DIP 2026')
    parser.add_argument('--query', type=str, default=None, help='Path to query image')
    parser.add_argument('--gallery', type=str, default='images_tr', help='Gallery images directory')
    parser.add_argument('--top-k', type=int, default=5, help='Number of retrieved items')
    parser.add_argument('--clahe', action='store_true', help='Enable CLAHE illumination preprocessing')
    parser.add_argument('--znorm', action='store_true', help='Enable per-query z-score normalization')
    parser.add_argument('--w_color', type=float, default=0.1, help='Weight for color HSV')
    parser.add_argument('--w_texture', type=float, default=0.7, help='Weight for texture LBP')
    parser.add_argument('--w_shape', type=float, default=0.2, help='Weight for shape HOG')

    args = parser.parse_args()
    weights = {'color': args.w_color, 'texture': args.w_texture, 'shape': args.w_shape}

    if args.query:
        query_path = Path(args.query)
        if not query_path.exists():
            print(f"Error: Query file '{query_path}' does not exist.")
            sys.exit(1)
        run_query(query_path, Path(args.gallery), args.top_k, weights, args.znorm, args.clahe)
    else:
        print("Explainable CBIR Engine v1 ready.")
        print("Please specify --query <path_to_image> to retrieve images.")


if __name__ == '__main__':
    main()
