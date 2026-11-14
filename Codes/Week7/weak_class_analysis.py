"""
weak_class_analysis.py - Week 7: where does V4 help or fail compared with V3? (LOFO data only, no test set)

V3 = fused distance (D_V3); V4 = frozen V4 (k1=30, k2=3, lambda=0.5). Pooled over the 5 held-out folds
(2,400 queries, 80 per class). Focus classes: square, bridge, airport (plus the full 30-class table).
Labels are used only to score and to count confusions; they never influence a ranking.

Run:  python "Week 7/weak_class_analysis.py"
"""
import numpy as np
import pandas as pd

import week7_common as c

FOCUS = ["square", "bridge", "airport"]
OUT = c.HERE / "weak_class_analysis"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def main():
    pipe = c.load_pipeline()
    recs, conf = [], {"V3": {}, "V4": {}}
    for fi, f in enumerate(pipe.fold_data):
        fd = f["fd"]
        D4 = c.v4_distance(f["D_qg"], f["D_gg"], k1=30, k2=3, lam=0.5)
        for name, D in [("V3", f["D_qg"]), ("V4", D4)]:
            p10, ap, order = c.per_query_metrics(D, fd)
            top10_cls = np.asarray(fd["g_cls"])[order[:, :10]]
            for qi in range(len(p10)):
                q_c = fd["q_cls"][qi]
                recs.append({"system": name, "fold": fi + 1, "qi": f"{fi}_{qi}", "class": q_c,
                             "p10": p10[qi], "ap": ap[qi]})
                d = conf[name].setdefault(q_c, {})
                for t in top10_cls[qi]:
                    d[t] = d.get(t, 0) + 1
    df = pd.DataFrame(recs)
    v3 = df[df.system == "V3"].set_index("qi")
    v4 = df[df.system == "V4"].set_index("qi")
    q = v3[["fold", "class", "p10", "ap"]].join(v4[["p10", "ap"]], lsuffix="_v3", rsuffix="_v4")
    q["d_p10"], q["d_ap"] = q.p10_v4 - q.p10_v3, q.ap_v4 - q.ap_v3

    # ---- all 30 classes -------------------------------------------------------
    rows = []
    for cls, g in q.groupby("class"):
        row = {"class": cls, "n_queries": len(g),
               "p10_v3": g.p10_v3.mean(), "p10_v4": g.p10_v4.mean(), "d_p10": g.d_p10.mean(),
               "ap_v3": g.ap_v3.mean(), "ap_v4": g.ap_v4.mean(), "d_ap": g.d_ap.mean(),
               "zero_p10_v3": int((g.p10_v3 == 0).sum()), "zero_p10_v4": int((g.p10_v4 == 0).sum()),
               "queries_p10_improved": int((g.d_p10 > 1e-12).sum()), "queries_p10_worsened": int((g.d_p10 < -1e-12).sum())}
        for col in ["d_p10", "d_ap"]:
            m = np.array([g[col].values[rng.integers(0, len(g), len(g))].mean() for _ in range(2000)])
            row[f"{col}_ci_low"], row[f"{col}_ci_high"] = np.percentile(m, 2.5), np.percentile(m, 97.5)
        rows.append(row)
    pc = pd.DataFrame(rows).sort_values("p10_v4")
    pc.to_csv(OUT / "per_class_v3_vs_v4.csv", index=False)

    # ---- confusion (share of top-10 slots by returned class) -------------------
    classes = sorted(q["class"].unique())
    for name in ["V3", "V4"]:
        M = pd.DataFrame([[conf[name][a].get(b, 0) / (conf[name][a] and sum(conf[name][a].values())) for b in classes]
                          for a in classes], index=classes, columns=classes)
        M.to_csv(OUT / f"confusion_top10_share_{name}.csv")
        conf[name] = M
    crow = []
    for cls in FOCUS:
        for name in ["V3", "V4"]:
            s = conf[name].loc[cls].drop(cls).sort_values(ascending=False).head(3)
            crow.append({"query_class": cls, "system": name, "own_class_share": conf[name].loc[cls, cls],
                         **{f"confusion_{i + 1}": f"{k} ({v:.3f})" for i, (k, v) in enumerate(s.items())}})
    pd.DataFrame(crow).to_csv(OUT / "focus_class_top_confusions.csv", index=False)

    # ---- overall V3 vs V4 summary ---------------------------------------------
    allq = {"queries": len(q), "p10_improved": int((q.d_p10 > 1e-12).sum()), "p10_worsened": int((q.d_p10 < -1e-12).sum()),
            "p10_unchanged": int((q.d_p10.abs() <= 1e-12).sum()), "ap_improved": int((q.d_ap > 1e-12).sum()),
            "ap_worsened": int((q.d_ap < -1e-12).sum()), "classes_p10_improved": int((pc.d_p10 > 0).sum()),
            "classes_p10_worsened": int((pc.d_p10 < 0).sum()), "classes_ap_improved": int((pc.d_ap > 0).sum()),
            "classes_ap_worsened": int((pc.d_ap < 0).sum())}
    pd.Series(allq).to_csv(OUT / "overall_v3_vs_v4_query_counts.csv", header=["value"])

    pd.set_option("display.width", 220)
    print("overall:", allq)
    print("\nweakest 8 classes under V4 (by P@10):")
    print(pc.head(8)[["class", "p10_v3", "p10_v4", "d_p10", "ap_v3", "ap_v4", "d_ap", "zero_p10_v3", "zero_p10_v4"]].round(3).to_string(index=False))
    print("\nfocus classes:")
    print(pc[pc["class"].isin(FOCUS)][["class", "p10_v3", "p10_v4", "d_p10", "d_p10_ci_low", "d_p10_ci_high", "ap_v3", "ap_v4", "d_ap", "d_ap_ci_low", "d_ap_ci_high", "queries_p10_improved", "queries_p10_worsened"]].round(3).to_string(index=False))
    print("\nfocus-class confusions:")
    print(pd.DataFrame(crow).round(3).to_string(index=False))
    print("\nbest 5 / worst 5 class changes in AP:")
    s = pc.sort_values("d_ap")
    print(s.head(5)[["class", "d_p10", "d_ap"]].round(3).to_string(index=False)); print(s.tail(5)[["class", "d_p10", "d_ap"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
