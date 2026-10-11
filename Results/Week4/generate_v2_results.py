import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
from PIL import Image

week4_dir = Path(__file__).resolve().parent
project_root = week4_dir.parent
sys.path.append(str(project_root / "Week1"))
sys.path.append(str(project_root / "Week2"))

from retrieval_engine_v2 import RetrievalEngineV2, load_split_features_v2, COMPONENTS
from week1_evaluate import load_labels

def jaccard_matrix(Lq, Lg):
    inter = Lq @ Lg.T
    union = Lq.sum(1)[:, None] + Lg.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def main():
    folds = json.loads((project_root / "Week2" / "folds_5.json").read_text())
    
    feats, names, classes, paths = load_split_features_v2("train")
    Lg_all = load_labels(names)
    
    eng = RetrievalEngineV2(gallery_split="train")
    
    explanations = []
    rel_validation = []
    
    fig_dir = week4_dir / "figures_v2"
    fig_dir.mkdir(exist_ok=True)
    
    # Track examples for plotting
    high_eg = None
    med_eg = None
    low_eg = None
    
    for fold_i, fold in enumerate(folds):
        q_idx = np.array(fold["query_idx"])
        g_idx = np.array(fold["gallery_idx"])
        
        q_cls, g_cls = classes[q_idx], classes[g_idx]
        q_names = [names[i] for i in q_idx]
        q_paths = [paths[i] for i in q_idx]
        
        Lq = Lg_all[q_idx]
        Lg = Lg_all[g_idx]
        J = jaccard_matrix(Lq, Lg)
        
        q_f = {c: feats[c][q_idx] for c in COMPONENTS}
        
        # Engine needs to search against g_idx only
        eng_fold = RetrievalEngineV2(gallery_split="train", gallery_idx=g_idx)
        results = eng_fold.search(q_f, k=10)
        
        for i, q_res in enumerate(results):
            retrieved = q_res["retrieved"]
            
            # P@10 and ML P@10
            rc = np.array([r["scene_class"] == q_cls[i] for r in retrieved])
            rm = np.array([J[i, r["index"]] >= 0.8 for r in retrieved])
            
            p10 = rc[:10].mean()
            ml_p10 = rm[:10].mean()
            
            gap = retrieved[0]["final_similarity"] - retrieved[1]["final_similarity"]
            sim_vals = [retrieved[0]["hsv_similarity"], retrieved[0]["lbp_r1_similarity"], retrieved[0]["lbp_r2_similarity"], retrieved[0]["hog_similarity"]]
            agreement = 1.0 - np.std(sim_vals)
            
            rel_score = q_res["reliability_score"]
            rel_label = q_res["reliability_label"]
            
            row_dict = {
                "query": q_names[i],
                "query_class": q_cls[i],
                "query_path": q_paths[i],
                "reliability_score": rel_score,
                "reliability_label": rel_label,
                "gap": gap,
                "feature_agreement": agreement,
                "class_p10": p10,
                "ml_p10": ml_p10,
                "retrieved": retrieved[:5]
            }
            rel_validation.append(row_dict)
            
            for k in range(5):
                r = retrieved[k]
                explanations.append({
                    "query": q_names[i],
                    "rank": r["rank"],
                    "retrieved_image": r["name"],
                    "final_similarity": r["final_similarity"],
                    "hsv_similarity": r["hsv_similarity"],
                    "lbp_r1_similarity": r["lbp_r1_similarity"],
                    "lbp_r2_similarity": r["lbp_r2_similarity"],
                    "hog_similarity": r["hog_similarity"],
                    "hsv_percentile": r["hsv_percentile"],
                    "lbp_r1_percentile": r["lbp_r1_percentile"],
                    "lbp_r2_percentile": r["lbp_r2_percentile"],
                    "hog_percentile": r["hog_percentile"],
                    "hsv_contribution": r["hsv_contribution"],
                    "lbp_r1_contribution": r["lbp_r1_contribution"],
                    "lbp_r2_contribution": r["lbp_r2_contribution"],
                    "hog_contribution": r["hog_contribution"]
                })
                
            if high_eg is None and rel_label == "HIGH" and p10 == 1.0:
                high_eg = row_dict
            elif med_eg is None and rel_label == "MEDIUM":
                med_eg = row_dict
            elif low_eg is None and rel_label == "LOW" and p10 <= 0.3:
                low_eg = row_dict

    df_exp = pd.DataFrame(explanations)
    df_exp.to_csv(week4_dir / "explanation_results_v2.csv", index=False)
    
    df_rel = pd.DataFrame(rel_validation)
    
    # Save validation stats
    summary = []
    for label in ["HIGH", "MEDIUM", "LOW"]:
        d = df_rel[df_rel["reliability_label"] == label]
        summary.append({
            "Label": label,
            "Count": len(d),
            "Class P@10": d["class_p10"].mean(),
            "ML P@10": d["ml_p10"].mean()
        })
    df_sum = pd.DataFrame(summary)
    
    print("\n--- RELIABILITY VALIDATION ---")
    print(df_sum.to_string(index=False))
    
    print(f"\nMean Top-1 Similarity: {df_rel['reliability_score'].mean():.4f}")
    print(f"Mean Top-1/Top-2 Gap: {df_rel['gap'].mean():.4f}")
    print(f"Mean Feature Agreement: {df_rel['feature_agreement'].mean():.4f}")
    print(f"Mean Reliability Score: {df_rel['reliability_score'].mean():.4f}")
    
    c1, _ = stats.spearmanr(df_rel["reliability_score"], df_rel["class_p10"])
    print(f"\nSpearman Correlation (Reliability vs Class P@10): {c1:.4f}")
    
    df_rel.drop(columns=["retrieved", "query_path"]).to_csv(week4_dir / "reliability_validation_v2.csv", index=False)
    
    # Plotting logic with clear spacing and explicit Validation labeling
    def plot_query(row, filename):
        fig, axes = plt.subplots(2, 3, figsize=(18, 14))
        fig.suptitle(f"Validation Example \u2014 V2 ({row['reliability_label']} Reliability)", fontsize=16, y=0.98)
        
        q_img = Image.open(project_root / row["query_path"])
        axes[0,0].imshow(q_img)
        axes[0,0].set_title(f"Query Image: {row['query']}\nClass: {row['query_class']}\nReliability: {row['reliability_label']} ({row['reliability_score']:.3f})", fontsize=12, pad=12)
        axes[0,0].axis("off")
        
        results = row["retrieved"]
        for k in range(5):
            ax = axes[(k+1)//3, (k+1)%3]
            res = results[k]
            img = Image.open(project_root / res["path"])
            ax.imshow(img)
            color = "green" if res["scene_class"] == row["query_class"] else "red"
            ax.set_title(f"Rank {k+1}: {res['name']}\nClass: {res['scene_class']} | Sim: {res['final_similarity']:.3f}", color=color, pad=12, fontsize=11)
            ax.axis("off")
            
            # Sub-plot for percentiles: component similarity evidence
            ins_ax = ax.inset_axes([0, -0.42, 1, 0.28])
            x_pos = np.arange(4)
            ins_ax.bar(x_pos, [res["hsv_percentile"], res["lbp_r1_percentile"], res["lbp_r2_percentile"], res["hog_percentile"]], color='skyblue')
            ins_ax.set_xticks(x_pos)
            ins_ax.set_xticklabels(["HSV", "LBP1", "LBP2", "HOG"])
            ins_ax.set_ylim(0, 100)
            ins_ax.set_title("Component Similarity Evidence (Percentiles)", fontsize=9)
            ins_ax.tick_params(axis='x', labelsize=9)
            ins_ax.tick_params(axis='y', labelsize=9)
                
        plt.subplots_adjust(top=0.92, hspace=0.85, wspace=0.35)
        plt.savefig(fig_dir / filename, bbox_inches="tight")
        plt.close(fig)
        
    plot_query(high_eg, "example_high_reliability_v2.png")
    plot_query(med_eg, "example_medium_reliability_v2.png")
    plot_query(low_eg, "example_low_reliability_v2.png")
    
if __name__ == "__main__":
    main()
