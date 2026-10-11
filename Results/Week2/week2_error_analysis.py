import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys

week1_path = Path(__file__).resolve().parent.parent / "Week1"
sys.path.append(str(week1_path))

import retrieval_engine_v1 as eng
from week1_evaluate import load_labels

def main():
    tr_f, tr_names, tr_cls, _ = eng.load_split_features("train")
    Lg_all = load_labels(tr_names)
    
    label_csv_p = Path(week1_path).parent / "multilabel_.csv"
    if not label_csv_p.exists():
        label_csv_p = Path(week1_path).parent / "multilabel .csv"
    df_labels = pd.read_csv(label_csv_p)
    label_names = df_labels.columns[1:].tolist()
    
    folds = json.loads(Path("Week2/folds_5.json").read_text() if Path("Week2/folds_5.json").exists() else Path("folds_5.json").read_text())
    
    w = {"hsv": 0.1, "lbp": 0.7, "hog": 0.2}
    
    all_q_cls = []
    all_p5_class = []
    all_p10_class = []
    
    label_p10_sum = {lbl: 0.0 for lbl in label_names}
    label_q_count = {lbl: 0 for lbl in label_names}
    
    confusions = defaultdict(list)
    
    for fold in folds:
        q_idx = np.array(fold["query_idx"])
        g_idx = np.array(fold["gallery_idx"])
        
        q_cls = tr_cls[q_idx]
        g_cls = tr_cls[g_idx]
        
        Lq = Lg_all[q_idx]
        Lg = Lg_all[g_idx]
        
        q_f = eng.subset(tr_f, q_idx)
        g_f = eng.subset(tr_f, g_idx)
        sims = eng.component_sims(q_f, g_f, comps=["hsv", "lbp", "hog"])
        
        S = w["hsv"] * sims["hsv"] + w["lbp"] * sims["lbp"] + w["hog"] * sims["hog"]
        
        order = np.argsort(-S, axis=1, kind="stable")
        n = S.shape[0]
        rows = np.arange(n)[:, None]
        
        rc = (q_cls[:, None] == g_cls[None, :])[rows, order]
        
        p5_c = rc[:, :5].mean(axis=1)
        p10_c = rc[:, :10].mean(axis=1)
        
        all_q_cls.extend(q_cls.tolist())
        all_p5_class.extend(p5_c.tolist())
        all_p10_class.extend(p10_c.tolist())
        
        for i in range(n):
            true_cls = q_cls[i]
            top5_idx = order[i, :5]
            top5_cls = g_cls[top5_idx]
            
            wrong_cls = top5_cls[top5_cls != true_cls]
            confusions[true_cls].extend(wrong_cls.tolist())
            
            for l_idx, lbl in enumerate(label_names):
                if Lq[i, l_idx] == 1:
                    top10_idx = order[i, :10]
                    g_lbls = Lg[top10_idx, l_idx]
                    rel_count = g_lbls.sum()
                    label_p10_sum[lbl] += rel_count / 10.0
                    label_q_count[lbl] += 1
                    
    df_class = pd.DataFrame({
        "scene_class": all_q_cls,
        "Class P@5": all_p5_class,
        "Class P@10": all_p10_class
    }).groupby("scene_class").mean().reset_index()
    df_class.to_csv("Week2/v1_per_class.csv", index=False)
    
    label_res = []
    for lbl in label_names:
        if label_q_count[lbl] > 0:
            label_res.append({
                "label": lbl,
                "precision@10": label_p10_sum[lbl] / label_q_count[lbl],
                "query_count": label_q_count[lbl]
            })
    df_label = pd.DataFrame(label_res)
    df_label.to_csv("Week2/v1_per_label.csv", index=False)
    
    conf_res = []
    for cls in confusions:
        if len(confusions[cls]) == 0:
            continue
        counts = pd.Series(confusions[cls]).value_counts()
        top3 = counts.head(3)
        row = {"scene_class": cls}
        for i, (wrong_c, count) in enumerate(top3.items()):
            row[f"wrong_class_{i+1}"] = wrong_c
            row[f"count_{i+1}"] = count
        conf_res.append(row)
    df_conf = pd.DataFrame(conf_res)
    df_conf.to_csv("Week2/v1_confusions.csv", index=False)
    
    weak = df_class.sort_values("Class P@10").head(6)
    print("Six weakest scene classes (by Class P@10):")
    print(weak[["scene_class", "Class P@10"]].to_string(index=False))

if __name__ == "__main__":
    main()
