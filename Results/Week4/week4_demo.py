import sys
import json
from pathlib import Path

week4_path = Path(__file__).resolve().parent
sys.path.append(str(week4_path))

from retrieval_engine_v2 import RetrievalEngineV2

def main():
    if len(sys.argv) < 2:
        print("Usage: python week4_demo.py <path_to_query_image.jpg>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    print("Loading V2 Retrieval Engine...")
    eng = RetrievalEngineV2(gallery_split="train")
    
    print(f"\nQUERY:\n    {Path(image_path).name}\n")
    
    res = eng.search_image(image_path, k=5)
    
    print("TOP 5 RESULTS:\n")
    for r in res["retrieved"]:
        print(f"{r['rank']}. {r['name']} (Class: {r['scene_class']})")
        print(f"   final similarity: {r['final_similarity']:.4f}")
        print(f"   HSV percentile: {r['hsv_percentile']:.1f}%")
        print(f"   LBP-r1 percentile: {r['lbp_r1_percentile']:.1f}%")
        print(f"   LBP-r2 percentile: {r['lbp_r2_percentile']:.1f}%")
        print(f"   HOG percentile: {r['hog_percentile']:.1f}%\n")
        
    print("RELIABILITY:")
    print(f"    score: {res['reliability_score']:.4f}")
    print(f"    label: {res['reliability_label']}")

if __name__ == "__main__":
    main()
