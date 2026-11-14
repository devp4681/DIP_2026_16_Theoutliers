"""
k2_check.py - Week 7 model check: k2 in {1, 2, 3} for V4's gallery-only k-reciprocal re-ranking.

Only k2 varies. k1=30 and lambda=0.5 stay frozen. Train data / 5-fold LOFO only; the test set is NOT touched.
Pre-specified decision (Week 6 rule, unchanged): V4 changes to a different k2 only if that k2 beats the current
k2=3 by > 0.005 selection score, is positive on >= 4/5 folds, and does not lower Class P@10. Otherwise V4 stays.

Run:  python "Week 7/k2_check.py"
"""
import json
import time

import numpy as np
import pandas as pd

import week7_common as c

K2_VALUES = [1, 2, 3]
CURRENT_K2 = 3
OUT = c.HERE / "k2_check"
OUT.mkdir(exist_ok=True)


def main():
    pipe = c.load_pipeline()
    folds = pipe.fold_data

    v3_rows = [c.evaluate(f["D_qg"], f["fd"]) for f in folds]
    res, times = {}, {}
    for k2 in K2_VALUES:
        t0 = time.time()
        Ds = [c.v4_distance(f["D_qg"], f["D_gg"], k1=30, k2=k2, lam=0.5) for f in folds]
        times[k2] = (time.time() - t0) / len(folds)
        res[k2] = [c.evaluate(D, f["fd"]) for D, f in zip(Ds, folds)]

    # reproduction check (V3, and V4 = k2=3) against Week 6
    repro = c.check_reproduction("V3", v3_rows) + c.check_reproduction("V4", res[CURRENT_K2])
    pd.DataFrame(repro).to_csv(OUT / "baseline_reproduction.csv", index=False)
    assert all(r["match"] for r in repro), "baseline not reproduced"
    print("baseline reproduced: V3 and V4(k2=3) match Week 6 on all six metrics")

    # per-fold table
    rows = []
    for name, rr in [("V3", v3_rows)] + [(f"V4 k2={k}", res[k]) for k in K2_VALUES]:
        for fi, r in enumerate(rr):
            rows.append({"system": name, "fold": fi + 1, **{m: r[m] for m in c.METRICS}})
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(OUT / "k2_per_fold.csv", index=False)

    summ = []
    for name, rr in [("V3", v3_rows)] + [(f"V4 k2={k}", res[k]) for k in K2_VALUES]:
        row = {"system": name}
        for m in c.METRICS:
            v = np.array([r[m] for r in rr])
            row[f"{m}_mean"], row[f"{m}_std"] = v.mean(), v.std(ddof=1)
        row["sec_per_fold_KR"] = times.get(int(name[-1]), np.nan) if name.startswith("V4") else np.nan
        summ.append(row)
    pd.DataFrame(summ).to_csv(OUT / "k2_summary.csv", index=False)

    # paired comparison vs current V4 (k2=3) + Week 6 rule
    base_ss = np.array([r["selection_score"] for r in res[CURRENT_K2]])
    base_p10 = np.array([r["Class P@10"] for r in res[CURRENT_K2]])
    comp = []
    for k2 in K2_VALUES:
        if k2 == CURRENT_K2:
            continue
        ss = np.array([r["selection_score"] for r in res[k2]])
        p10 = np.array([r["Class P@10"] for r in res[k2]])
        d = ss - base_ss
        crit = {
            "gain_gt_0.005": bool(d.mean() > c.RULE_MIN_DELTA),
            "positive_folds_ge_4": bool((d > 0).sum() >= c.RULE_MIN_POSITIVE_FOLDS),
            "class_p10_not_lower": bool((p10 - base_p10).mean() >= 0),
        }
        comp.append({"candidate": f"k2={k2}", "mean_delta_selection": d.mean(), "std_delta": d.std(ddof=1),
                     "per_fold_delta": d.round(5).tolist(), "positive_folds": int((d > 0).sum()),
                     "mean_delta_class_p10": float((p10 - base_p10).mean()), **crit,
                     "replaces_k2_3": all(crit.values())})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(OUT / "k2_vs_current_v4.csv", index=False)

    # LOFO selection of k2 (choose on 4 folds by selection score, report held-out fold)
    lofo = []
    for test in range(5):
        train = [i for i in range(5) if i != test]
        scores = {k: float(np.mean([res[k][i]["selection_score"] for i in train])) for k in K2_VALUES}
        best = max(K2_VALUES, key=lambda k: (scores[k], k == CURRENT_K2))     # ties -> current k2
        lofo.append({"held_out_fold": test + 1, "selected_k2": best, "heldout_selection_score": res[best][test]["selection_score"]})
    pd.DataFrame(lofo).to_csv(OUT / "k2_lofo_selection.csv", index=False)

    final_k2 = CURRENT_K2 if not any(x["replaces_k2_3"] for x in comp) else \
        max([x for x in comp if x["replaces_k2_3"]], key=lambda x: x["mean_delta_selection"])["candidate"]
    decision = {"k2_values_tested": K2_VALUES, "k1": 30, "lambda": 0.5, "frozen_k2": final_k2,
                "v4_changed": final_k2 != CURRENT_K2, "lofo_selected_k2_per_fold": [x["selected_k2"] for x in lofo],
                "rule": "gain>0.005 selection score vs k2=3, >=4/5 folds positive, Class P@10 not lower",
                "test_set_used": False}
    (OUT / "k2_decision.json").write_text(json.dumps(decision, indent=2, default=str))

    pd.set_option("display.width", 200)
    print("\n=== mean over 5 folds ===")
    print(pd.DataFrame(summ)[["system"] + [f"{m}_mean" for m in c.METRICS] + ["sec_per_fold_KR"]].round(4).to_string(index=False))
    print("\n=== selection score per fold ===")
    print(per_fold.pivot(index="system", columns="fold", values="selection_score").round(4))
    print("\n=== vs current V4 (k2=3) ===")
    print(comp_df.round(5).to_string(index=False))
    print("\nLOFO-selected k2 per fold:", [x["selected_k2"] for x in lofo])
    print("DECISION:", json.dumps(decision))


if __name__ == "__main__":
    main()
