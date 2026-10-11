import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

week4_path = Path(__file__).resolve().parent
project_root = week4_path.parent
sys.path.append(str(project_root / "Week1"))
sys.path.append(str(project_root / "Week2"))

from retrieval_engine_v1 import load_split_features
from cache_features import load_cached
from week1_evaluate import load_labels, corr_sim, cos_sim
def jaccard_matrix(Lq, Lg):
    inter = Lq @ Lg.T
    union = Lq.sum(1)[:, None] + Lg.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def main():
    tr_f, tr_names, tr_cls, _ = load_split_features("train")
    Lg_all = load_labels(tr_names)
    
    f_lbp2, _ = load_cached("train_lbp_r2_p16")
    tr_f["lbp2"] = f_lbp2.astype(np.float64)
    
    folds = json.loads((project_root / "Week2" / "folds_5.json").read_text())
    
    v1_p10_list = []
    v2_p10_list = []
    classes_list = []
    
    for fold_i, fold in enumerate(folds):
        q_idx = np.array(fold["query_idx"])
        g_idx = np.array(fold["gallery_idx"])
        
        q_cls, g_cls = tr_cls[q_idx], tr_cls[g_idx]
        Lq, Lg = Lg_all[q_idx], Lg_all[g_idx]
        J = jaccard_matrix(Lq, Lg)
        
        q_f = {c: tr_f[c][q_idx] for c in tr_f}
        g_f = {c: tr_f[c][g_idx] for c in tr_f}
        
        sims = {
            "hsv": corr_sim(q_f["hsv"], g_f["hsv"]),
            "lbp": corr_sim(q_f["lbp"], g_f["lbp"]),
            "hog": cos_sim(q_f["hog"], g_f["hog"]),
            "lbp2": corr_sim(q_f["lbp2"], g_f["lbp2"])
        }
        
        S_v1 = 0.1 * sims["hsv"] + 0.7 * sims["lbp"] + 0.2 * sims["hog"]
        order_v1 = np.argsort(-S_v1, axis=1, kind="stable")
        
        S_v2 = 0.1 * sims["hsv"] + 0.35 * sims["lbp"] + 0.35 * sims["lbp2"] + 0.2 * sims["hog"]
        order_v2 = np.argsort(-S_v2, axis=1, kind="stable")
        
        n = S_v1.shape[0]
        rows = np.arange(n)[:, None]
        
        rc_v1 = (q_cls[:, None] == g_cls[None, :])[rows, order_v1]
        p10_v1 = rc_v1[:, :10].mean(axis=1)
        
        rc_v2 = (q_cls[:, None] == g_cls[None, :])[rows, order_v2]
        p10_v2 = rc_v2[:, :10].mean(axis=1)
        
        v1_p10_list.extend(p10_v1)
        v2_p10_list.extend(p10_v2)
        classes_list.extend(q_cls)
        
    v1_p10 = np.array(v1_p10_list)
    v2_p10 = np.array(v2_p10_list)
    classes = np.array(classes_list)
    
    diff = v2_p10 - v1_p10
    observed_mean_diff = diff.mean()
    
    # Paired Bootstrap
    rng = np.random.default_rng(42)
    n_queries = len(diff)
    n_boot = 10000
    boot_diffs = np.zeros(n_boot)
    
    for i in range(n_boot):
        idx = rng.choice(n_queries, n_queries, replace=True)
        boot_diffs[i] = diff[idx].mean()
        
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    
    res = {
        "V1_mean_p10": v1_p10.mean(),
        "V2_mean_p10": v2_p10.mean(),
        "observed_mean_diff": observed_mean_diff,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "significant": ci_lower > 0 or ci_upper < 0
    }
    
    pd.DataFrame([res]).to_csv(week4_path / "v1_v2_bootstrap_results.csv", index=False)
    
    print("\n--- BOOTSTRAP RESULTS ---")
    print(f"Observed Mean Diff: {observed_mean_diff:.5f}")
    print(f"95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]")
    print(f"Significant: {res['significant']}")
    
    # Per-class deltas
    class_deltas = []
    for c in np.unique(classes):
        idx = (classes == c)
        v1_m = v1_p10[idx].mean()
        v2_m = v2_p10[idx].mean()
        class_deltas.append({
            "class": c,
            "v1_p10": v1_m,
            "v2_p10": v2_m,
            "delta": v2_m - v1_m
        })
        
    df_cls = pd.DataFrame(class_deltas).sort_values("delta", ascending=False)
    df_cls.to_csv(week4_path / "v1_v2_per_class_delta.csv", index=False)
    
    print("\n--- PER-CLASS DELTAS (Top 5 & Bottom 5) ---")
    print(df_cls.head(5))
    print("...")
    print(df_cls.tail(5))
    
if __name__ == "__main__":
    main()
