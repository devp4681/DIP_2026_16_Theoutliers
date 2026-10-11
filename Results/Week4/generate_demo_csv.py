import sys
import pandas as pd
from pathlib import Path

week4_path = Path(__file__).resolve().parent
sys.path.append(str(week4_path))

from retrieval_engine_v2 import RetrievalEngineV2

def main():
    eng = RetrievalEngineV2(gallery_split="train")
    
    # Test queries covering HIGH, MEDIUM, and LOW reliability
    queries = [
        ("images_test/Parking/parking_84.jpg", "parking"),
        ("images_test/Center/center_90.jpg", "center"),
        ("images_test/River/river_85.jpg", "river")
    ]
    
    demo_results = []
    
    for q_path, q_cls in queries:
        try:
            res = eng.search_image(q_path, k=5, exclude_self=True)
            for r in res["retrieved"]:
                demo_results.append({
                    "query_image": Path(q_path).name,
                    "query_scene_class": q_cls,
                    "reliability_label": res["reliability_label"],
                    "reliability_score": res["reliability_score"],
                    "rank": r["rank"],
                    "retrieved_image": r["name"],
                    "retrieved_scene_class": r["scene_class"],
                    "final_similarity": r["final_similarity"],
                    "hsv_percentile": r["hsv_percentile"],
                    "lbp_r1_percentile": r["lbp_r1_percentile"],
                    "lbp_r2_percentile": r["lbp_r2_percentile"],
                    "hog_percentile": r["hog_percentile"]
                })
        except Exception as e:
            print(f"Error processing {q_path}: {e}")
            
    df = pd.DataFrame(demo_results)
    out_csv = week4_path / "final_test" / "final_demo_examples.csv"
    df.to_csv(out_csv, index=False)
    print(f"final_demo_examples.csv created successfully at {out_csv}.")

if __name__ == "__main__":
    main()
