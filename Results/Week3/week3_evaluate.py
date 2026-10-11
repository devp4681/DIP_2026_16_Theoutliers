import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

week1_path = Path(__file__).resolve().parent.parent / "Week1"
week2_path = Path(__file__).resolve().parent.parent / "Week2"
sys.path.append(str(week1_path))
sys.path.append(str(week2_path))

import retrieval_engine_v1 as eng
from week1_evaluate import evaluate, load_labels, corr_sim, cos_sim
# week2_stability might not be in path or might execute script. Let's redefine jaccard_matrix.
def jaccard_matrix(Lq, Lg):
    inter = Lq @ Lg.T
    union = Lq.sum(1)[:, None] + Lg.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def main():
    tr_f, tr_names, tr_cls, _ = eng.load_split_features("train")
    Lg_all = load_labels(tr_names)
    
    # Load new features
    from cache_features import load_cached
    f_lbp2, _ = load_cached("train_lbp_r2_p16")
    f_hog_fine, _ = load_cached("train_hog_128_o9_c8_b2")
    
    tr_f["lbp2"] = f_lbp2.astype(np.float64)
    tr_f["hog_fine"] = f_hog_fine.astype(np.float64)
    
    folds = json.loads((week2_path / "folds_5.json").read_text())
    
    variants = [
        {"name": "V1", "w": {"hsv": 0.1, "lbp": 0.7, "hog": 0.2}},
        {"name": "V3_spatial", "w": {"hsv": 0.1, "lbp": 0.6, "hog": 0.2, "shsv": 0.05, "slbp": 0.05}},
        {"name": "V3_multiscale_lbp", "w": {"hsv": 0.1, "lbp": 0.35, "lbp2": 0.35, "hog": 0.2}}, 
        {"name": "V3_hog_fine", "w": {"hsv": 0.1, "lbp": 0.7, "hog_fine": 0.2}}
    ]
    
    results = []
    
    weak_classes = ["bridge", "square", "stadium", "railwaystation", "airport", "park"]
    weak_class_res = {v["name"]: defaultdict(list) for v in variants}
    
    for cand in variants:
        v_name = cand["name"]
        w = cand["w"]
        
        for i, fold in enumerate(folds):
            q_idx = np.array(fold["query_idx"])
            g_idx = np.array(fold["gallery_idx"])
            
            q_cls, g_cls = tr_cls[q_idx], tr_cls[g_idx]
            Lq, Lg = Lg_all[q_idx], Lg_all[g_idx]
            J = jaccard_matrix(Lq, Lg)
            
            q_f = eng.subset(tr_f, q_idx)
            g_f = eng.subset(tr_f, g_idx)
            
            comps = [c for c in w.keys() if c in eng.COMPONENTS]
            sims = eng.component_sims(q_f, g_f, comps=comps)
            
            if "lbp2" in w:
                sims["lbp2"] = corr_sim(q_f["lbp2"], g_f["lbp2"])
            if "hog_fine" in w:
                sims["hog_fine"] = cos_sim(q_f["hog_fine"], g_f["hog_fine"])
                
            S = sum(w[c] * sims[c] for c in w)
            
            out, _ = evaluate(S, q_cls, g_cls, J)
            row = {"variant": v_name, "fold": i + 1}
            row.update(out)
            sel_metrics = ["Class P@10", "ML P@10", "nDCG@10", "Class mAP", "ML mAP"]
            row["selection_score"] = np.mean([out[m] for m in sel_metrics])
            results.append(row)
            
            order = np.argsort(-S, axis=1, kind="stable")
            n = S.shape[0]
            rows = np.arange(n)[:, None]
            rc = (q_cls[:, None] == g_cls[None, :])[rows, order]
            p10_c = rc[:, :10].mean(axis=1)
            
            for j in range(n):
                c = q_cls[j]
                if c in weak_classes:
                    weak_class_res[v_name][c].append(p10_c[j])
                    
    df_res = pd.DataFrame(results)
    df_res["rank_in_fold"] = df_res.groupby("fold")["selection_score"].rank(ascending=False, method="min")
    df_res.columns = [c.replace(" ", "_").lower().replace("@", "_") for c in df_res.columns]
    Path("Week3").mkdir(exist_ok=True)
    df_res.to_csv("Week3/week3_results.csv", index=False)
    
    summary = []
    for v_name in [v["name"] for v in variants]:
        df_v = df_res[df_res["variant"] == v_name]
        row = {
            "variant": v_name,
            "mean_class_p5": df_v["class_p_5"].mean(),
            "std_class_p5": df_v["class_p_5"].std(),
            "mean_class_p10": df_v["class_p_10"].mean(),
            "std_class_p10": df_v["class_p_10"].std(),
            "mean_ml_p5": df_v["ml_p_5"].mean(),
            "std_ml_p5": df_v["ml_p_5"].std(),
            "mean_ml_p10": df_v["ml_p_10"].mean(),
            "std_ml_p10": df_v["ml_p_10"].std(),
            "mean_jaccard_p5": df_v["jaccard_5"].mean(),
            "mean_jaccard_p10": df_v["jaccard_10"].mean(),
            "mean_ndcg_p5": df_v["ndcg_5"].mean(),
            "mean_ndcg_p10": df_v["ndcg_10"].mean(),
            "mean_ml_map": df_v["ml_map"].mean(),
            "mean_selection_score": df_v["selection_score"].mean(),
            "std_selection_score": df_v["selection_score"].std(),
            "number_of_folds_ranked_first": (df_v["rank_in_fold"] == 1).sum()
        }
        summary.append(row)
        
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv("Week3/week3_summary.csv", index=False)

    print("\n--- 5-FOLD EVALUATION RESULTS ---")
    cols = ["variant", "mean_selection_score", "std_selection_score", "number_of_folds_ranked_first"]
    print(df_sum[cols].to_string(index=False))
    
    print("\n--- WEAK CLASS P@10 IMPROVEMENTS ---")
    for v_name in [v["name"] for v in variants]:
        print(f"\n{v_name}:")
        for c in weak_classes:
            m = np.mean(weak_class_res[v_name][c])
            print(f"  {c}: {m:.3f}")

if __name__ == "__main__":
    main()
