"""
reliability_recalibration.py - Week 7: recalibrate the HIGH / MEDIUM / LOW reliability label for V4 (LOFO/train data only).

Week 4 procedure, reused: compare simple per-query confidence scores by Spearman correlation with per-query Class P@10,
take the best one, and set LOW / HIGH thresholds at the 20th / 70th percentile of the score distribution.
Candidate scores (all label-free, computed from the ranking only):
  top1_sim_v4        = 1 - D_V4(q, top-1)                    (V4 analogue of the Week 4 "top-1 similarity")
  gap_v4             = D_V4(q, 2nd) - D_V4(q, 1st)
  top1_sim_v3_of_v4  = 1 - D_V3(q, V4's top-1 image)
Leakage-safe check: for each held-out fold, thresholds come from the OTHER four folds only.
The frozen thresholds (all 2,400 LOFO queries) are written to reliability_recalibration/reliability_config_v4.json.
The test set is NOT used here; evaluating the frozen categories on the test set belongs to the final test.

Run:  python "Week 7/reliability_recalibration.py"
"""
import json

import numpy as np
import pandas as pd
import scipy.stats as st

import week7_common as c

OUT = c.HERE / "reliability_recalibration"
OUT.mkdir(exist_ok=True)
CAND = ["top1_sim_v4", "gap_v4", "top1_sim_v3_of_v4"]
LOW_PCT, HIGH_PCT = 20, 70                    # same percentiles as Week 4
OLD_V2_THRESHOLDS = {"LOW": 0.930, "HIGH": 0.960}


def label(score, lo, hi):
    return np.where(score < lo, "LOW", np.where(score >= hi, "HIGH", "MEDIUM"))


def main():
    pipe = c.load_pipeline()
    rows = []
    for fi, f in enumerate(pipe.fold_data):
        fd = f["fd"]
        D4 = c.v4_distance(f["D_qg"], f["D_gg"], k1=30, k2=3, lam=0.5)
        p10, ap, order = c.per_query_metrics(D4, fd)
        rel_ml = (fd["J_mat"] >= c.cv.REL_JACCARD)
        r = np.arange(D4.shape[0])[:, None]
        ml10 = rel_ml[r, order][:, :10].mean(axis=1)
        d_sorted = np.take_along_axis(D4, order[:, :2], axis=1)
        top1 = order[:, 0]
        for qi in range(D4.shape[0]):
            rows.append({"fold": fi + 1, "class_p10": p10[qi], "ml_p10": ml10[qi], "class_ap": ap[qi],
                         "top1_sim_v4": 1.0 - d_sorted[qi, 0], "gap_v4": d_sorted[qi, 1] - d_sorted[qi, 0],
                         "top1_sim_v3_of_v4": 1.0 - f["D_qg"][qi, top1[qi]]})
    df = pd.DataFrame(rows)

    # 1) which score predicts quality best (pooled LOFO queries)
    corr = pd.DataFrame([{"score": s, "spearman_vs_class_p10": st.spearmanr(df[s], df.class_p10)[0],
                          "spearman_vs_ml_p10": st.spearmanr(df[s], df.ml_p10)[0],
                          "spearman_vs_class_ap": st.spearmanr(df[s], df.class_ap)[0]} for s in CAND])
    corr.to_csv(OUT / "reliability_correlations.csv", index=False)
    best = corr.sort_values("spearman_vs_class_p10", ascending=False).iloc[0]["score"]

    # 2) leave-fold-out thresholds -> held-out categories
    held = []
    for test in range(1, 6):
        tr, te = df[df.fold != test], df[df.fold == test].copy()
        lo, hi = np.percentile(tr[best], [LOW_PCT, HIGH_PCT])
        te["label"] = label(te[best].values, lo, hi)
        held.append(te)
    held = pd.concat(held)
    def summ(g):
        return pd.Series({"n_queries": len(g), "pct_queries": 100 * len(g) / len(held), "class_p10": g.class_p10.mean(),
                          "ml_p10": g.ml_p10.mean(), "class_ap": g.class_ap.mean()})
    cat = held.groupby("label").apply(summ, include_groups=False).reindex(["HIGH", "MEDIUM", "LOW"])
    cat.to_csv(OUT / "reliability_categories_lofo_heldout.csv")
    per_fold = held.groupby(["fold", "label"]).class_p10.mean().unstack().reindex(columns=["HIGH", "MEDIUM", "LOW"])
    per_fold.to_csv(OUT / "reliability_categories_per_fold_class_p10.csv")
    monotone_folds = int(((per_fold.HIGH > per_fold.MEDIUM) & (per_fold.MEDIUM > per_fold.LOW)).sum())

    # 3) frozen thresholds from all 2,400 LOFO queries
    lo, hi = np.percentile(df[best], [LOW_PCT, HIGH_PCT])
    cfg = {"version": "V4 (k1=30, k2=3, lambda=0.5)", "formula": f"reliability_score = {best}",
           "thresholds": {"LOW": float(lo), "HIGH": float(hi)},
           "threshold_method": f"{LOW_PCT}th / {HIGH_PCT}th percentile of the score over all 2,400 LOFO held-out queries (train data only)",
           "spearman_vs_class_p10": float(corr.set_index('score').loc[best, 'spearman_vs_class_p10']),
           "test_set_used": False, "replaces": "Week4/reliability_config.json (V2 scale; not valid for V4 scores)"}
    (OUT / "reliability_config_v4.json").write_text(json.dumps(cfg, indent=2))

    old = label(df["top1_sim_v3_of_v4"].values, OLD_V2_THRESHOLDS["LOW"], OLD_V2_THRESHOLDS["HIGH"])
    print("Spearman correlations (pooled 2,400 LOFO queries):"); print(corr.round(3).to_string(index=False))
    print(f"\nchosen score: {best}; thresholds LOW={lo:.4f} HIGH={hi:.4f}")
    print("\nHeld-out categories (thresholds from the other 4 folds):"); print(cat.round(4).to_string())
    print("\nper-fold Class P@10 by category:"); print(per_fold.round(3).to_string())
    print(f"folds with HIGH > MEDIUM > LOW: {monotone_folds}/5")
    print("\nOld Week 4 thresholds (V2 scale) applied to V4's V3-similarity score put this share in LOW/MED/HIGH:",
          {k: round(float((old == k).mean()), 3) for k in ["LOW", "MEDIUM", "HIGH"]})


if __name__ == "__main__":
    main()
