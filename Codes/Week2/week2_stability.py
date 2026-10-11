import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Setup imports
WEEK2_DIR = Path(__file__).resolve().parent
week1_path = WEEK2_DIR.parent / "Week1"
sys.path.append(str(week1_path))

import retrieval_engine_v1 as eng
from week1_evaluate import evaluate, load_labels

def jaccard_matrix(Lq, Lg):
    inter = Lq @ Lg.T
    union = Lq.sum(1)[:, None] + Lg.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def main():
    # 1. 5-Fold Split
    tr_f, tr_names, tr_cls, _ = eng.load_split_features("train")
    Lg_all = load_labels(tr_names)
    
    folds_file = WEEK2_DIR / "folds_5.json"
    if folds_file.exists():
        folds = json.loads(folds_file.read_text())
    else:
        rng = np.random.default_rng(42)
        fold_lists = [[] for _ in range(5)]
        for c in np.unique(tr_cls):
            idx = np.where(tr_cls == c)[0]
            rng.shuffle(idx)
            for i, val in enumerate(idx):
                fold_lists[i % 5].append(val)
        
        folds = []
        n_total = len(tr_names)
        for test_idx in fold_lists:
            test_idx = np.array(test_idx)
            train_idx = np.setdiff1d(np.arange(n_total), test_idx)
            folds.append({
                "query_idx": test_idx.tolist(),
                "gallery_idx": train_idx.tolist()
            })
        folds_file.write_text(json.dumps(folds))

    # 2. Candidate Configurations
    cands_file = WEEK2_DIR / "stability_candidates.json"
    if cands_file.exists():
        cands = json.loads(cands_file.read_text())
    else:
        cands = [
            {"name": "Week 1 best", "weights": {"hsv": 0.5, "lbp": 0.25, "hog": 0.25}},
            {"name": "Equal weights", "weights": {"hsv": 1/3, "lbp": 1/3, "hog": 1/3}},
            {"name": "Engine V1", "weights": {"hsv": 0.1, "lbp": 0.7, "hog": 0.2}}
        ]
        
        df_val = pd.read_csv(WEEK2_DIR / "validation_results.csv")
        df_val = df_val[df_val["family"] == "grid-global"].sort_values("selection_score", ascending=False).head(10)
        for _, row in df_val.iterrows():
            cands.append({
                "name": row["config"],
                "weights": {"hsv": row["w_hsv"], "lbp": row["w_lbp"], "hog": row["w_hog"]}
            })
        cands_file.write_text(json.dumps(cands, indent=2))

    # 3. Evaluation
    results = []
    query_metrics = {cand["name"]: {"p10_class": [], "p10_ml": []} for cand in cands}
    
    for i, fold in enumerate(folds):
        q_idx = np.array(fold["query_idx"])
        g_idx = np.array(fold["gallery_idx"])
        
        q_cls = tr_cls[q_idx]
        g_cls = tr_cls[g_idx]
        
        Lq = Lg_all[q_idx]
        Lg = Lg_all[g_idx]
        J = jaccard_matrix(Lq, Lg)
        
        q_f = eng.subset(tr_f, q_idx)
        g_f = eng.subset(tr_f, g_idx)
        sims = eng.component_sims(q_f, g_f, comps=["hsv", "lbp", "hog"])
        
        for cand in cands:
            w = cand["weights"]
            S = w["hsv"] * sims["hsv"] + w["lbp"] * sims["lbp"] + w["hog"] * sims["hog"]
            out, _ = evaluate(S, q_cls, g_cls, J)
            
            row = {
                "config": cand["name"],
                "fold": i + 1,
                "w_hsv": w["hsv"],
                "w_lbp": w["lbp"],
                "w_hog": w["hog"],
            }
            row.update(out)
            results.append(row)
            
            order = np.argsort(-S, axis=1, kind="stable")
            n = S.shape[0]
            rows = np.arange(n)[:, None]
            rc = (q_cls[:, None] == g_cls[None, :])[rows, order]
            rm = (J >= 0.8)[rows, order]
            
            p10_class = rc[:, :10].mean(axis=1)
            p10_ml = rm[:, :10].mean(axis=1)
            
            query_metrics[cand["name"]]["p10_class"].extend(p10_class.tolist())
            query_metrics[cand["name"]]["p10_ml"].extend(p10_ml.tolist())

    df_res = pd.DataFrame(results)
    
    sel_metrics = ["Class P@10", "ML P@10", "nDCG@10", "Class mAP", "ML mAP"]
    df_res["selection_score"] = df_res[sel_metrics].mean(axis=1)
    df_res["rank_in_fold"] = df_res.groupby("fold")["selection_score"].rank(ascending=False, method="min")
    
    df_res.to_csv(WEEK2_DIR / "stability_results.csv", index=False)
    
    # 4. Bootstrap and Summary
    summary = []
    rng = np.random.default_rng(42)
    
    for cand in cands:
        name = cand["name"]
        df_cand = df_res[df_res["config"] == name]
        
        mean_score = df_cand["selection_score"].mean()
        std_score = df_cand["selection_score"].std()
        num_first = (df_cand["rank_in_fold"] == 1).sum()
        
        p10_c = np.array(query_metrics[name]["p10_class"])
        p10_m = np.array(query_metrics[name]["p10_ml"])
        n_queries = len(p10_c)
        
        bs_c, bs_m = [], []
        for _ in range(1000):
            idx = rng.choice(n_queries, n_queries, replace=True)
            bs_c.append(p10_c[idx].mean())
            bs_m.append(p10_m[idx].mean())
            
        ci_c = np.percentile(bs_c, [2.5, 97.5])
        ci_m = np.percentile(bs_m, [2.5, 97.5])
        
        summary.append({
            "config": name,
            "mean_selection_score": mean_score,
            "std_selection_score": std_score,
            "folds_ranked_1st": num_first,
            "ci_95_class_p10": f"[{ci_c[0]:.4f}, {ci_c[1]:.4f}]",
            "ci_95_ml_p10": f"[{ci_m[0]:.4f}, {ci_m[1]:.4f}]"
        })
        
    df_sum = pd.DataFrame(summary).sort_values("mean_selection_score", ascending=False)
    df_sum.to_csv(WEEK2_DIR / "stability_summary.csv", index=False)
    
    print("Stability check complete.")

if __name__ == "__main__":
    main()
