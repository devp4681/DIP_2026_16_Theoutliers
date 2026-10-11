import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import scipy.stats as stats

week4_dir = Path(__file__).resolve().parent
project_root = week4_dir.parent
sys.path.append(str(project_root / "Week1"))
sys.path.append(str(project_root / "Week2"))

import retrieval_engine_v1 as eng
from week1_evaluate import load_labels, corr_sim, cos_sim
from cache_features import load_cached, TRAIN_DIR

def main():
    config = json.loads((week4_dir / "v2_config.json").read_text())
    
    tr_f, tr_names, tr_cls, tr_paths = eng.load_split_features("train")
    Lg_all = load_labels(tr_names)
    
    f_lbp2, _ = load_cached("train_lbp_r2_p16")
    tr_f["lbp2"] = f_lbp2.astype(np.float64)
    
    folds = json.loads((project_root / "Week2" / "folds_5.json").read_text())
    
    data = []
    
    for fold_i, fold in enumerate(folds):
        q_idx = np.array(fold["query_idx"])
        g_idx = np.array(fold["gallery_idx"])
        
        q_cls, g_cls = tr_cls[q_idx], tr_cls[g_idx]
        
        q_f = eng.subset(tr_f, q_idx)
        g_f = eng.subset(tr_f, g_idx)
        
        sims = {
            "hsv": corr_sim(q_f["hsv"], g_f["hsv"]),
            "lbp_r1": corr_sim(q_f["lbp"], g_f["lbp"]),
            "lbp_r2": corr_sim(q_f["lbp2"], g_f["lbp2"]),
            "hog": cos_sim(q_f["hog"], g_f["hog"])
        }
        
        w = {
            "hsv": config["hsv"],
            "lbp_r1": config["lbp_r1_p8"],
            "lbp_r2": config["lbp_r2_p16"],
            "hog": config["hog"]
        }
        
        S = w["hsv"]*sims["hsv"] + w["lbp_r1"]*sims["lbp_r1"] + w["lbp_r2"]*sims["lbp_r2"] + w["hog"]*sims["hog"]
        order = np.argsort(-S, axis=1, kind="stable")
        
        n = S.shape[0]
        rows = np.arange(n)[:, None]
        rc = (q_cls[:, None] == g_cls[None, :])[rows, order]
        p10 = rc[:, :10].mean(axis=1)
        
        for i in range(n):
            top_idx = order[i, :10]
            top_S = S[i, top_idx]
            
            s1 = top_S[0]
            s2 = top_S[1]
            gap = s1 - s2
            
            # Feature agreement: standard deviation of similarities for the top result
            sim_vals = [sims["hsv"][i, top_idx[0]], sims["lbp_r1"][i, top_idx[0]], sims["lbp_r2"][i, top_idx[0]], sims["hog"][i, top_idx[0]]]
            agreement_std = np.std(sim_vals)
            
            data.append({
                "top1_similarity": s1,
                "gap": gap,
                "feature_agreement_std": agreement_std,
                "class_p10": p10[i]
            })
            
    df = pd.DataFrame(data)
    
    print("--- CORRELATIONS WITH P@10 ---")
    c1, _ = stats.spearmanr(df["top1_similarity"], df["class_p10"])
    c2, _ = stats.spearmanr(df["gap"], df["class_p10"])
    c3, _ = stats.spearmanr(df["feature_agreement_std"], df["class_p10"])
    
    print(f"Top-1 Similarity : {c1:.3f}")
    print(f"Top-1/Top-2 Gap  : {c2:.3f}")
    print(f"Feature Agrm Std : {c3:.3f} (Lower std = higher agreement)")
    
    # We will compute a simple formula if top1_sim is the best
    # Then we will pick absolute thresholds
    # Let's see percentiles of top1_similarity
    print("\n--- Top-1 Similarity Percentiles ---")
    print(df["top1_similarity"].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    print("\n--- Gap Percentiles ---")
    print(df["gap"].describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    
if __name__ == "__main__":
    main()
