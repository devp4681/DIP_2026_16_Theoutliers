"""
week7_context.py

Week 7 ABLATION (negative result): V4-stage neighbourhood expansion ("V4 + NE") on top of the frozen V4 system.

V4 = V3 + gallery-only k-reciprocal re-ranking (k1=30, k2=3, lambda=0.5)   [unchanged, from Week 6]
Candidate = V4 distance blended with the mean gallery distance to the query's top-m V4 results:

    D_v4(q, g)   = lambda * Jaccard_KR(q, g) + (1 - lambda) * D_V3(q, g)          (Week 6, unchanged)
    N(q, g)      = mean_{r in top-m of D_v4(q, .)}  D_V3(r, g)                    (new, label-free)
    D_cand(q, g) = (1 - beta) * D_v4(q, g) + beta * N(q, g)

beta = 0 returns V4 exactly. Only gallery-side information is used (D_V3 between gallery images).
No labels are used for retrieval or parameter selection; labels only enter the metrics.

Run (from the project root, i.e. the folder containing Week1 ... Week 7 and cache/):
    python "Week 7/ne_ablation/ne_ablation.py" --sanity-check
    python "Week 7/ne_ablation/ne_ablation.py"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent          # project root (ne_ablation/ sits inside 'Week 7/')
for p in ["Week1", "Week2", "Week 5", "Week 6"]:
    sys.path.insert(0, str(ROOT / p))

import cv_experiment as cv                      # noqa: E402  (Week 5 evaluator, unchanged)
import week1_evaluate as w1                     # noqa: E402
import neighborhood_context as nc               # noqa: E402  (Week 6: V3 matrices + gallery-only KR)

FOLDS_FILE = ROOT / "Week2" / "folds_5.json"
LABEL_CSV_CANDIDATES = [ROOT / "multilabel_.csv", ROOT / "multilabel .csv", Path("multilabel_.csv")]

# ---- frozen V4 ----------------------------------------------------------------
V4_PARAMS = {"k1": 30, "k2": 3, "lambda": 0.5}

# ---- Week 6 reference numbers (from Week 6/selected_parameters.json) -----------
W6_REF = {
    "V3": {"Class P@10": 0.5069, "Class mAP": 0.2547},
    "V4": {"Class P@10": 0.5361, "Class mAP": 0.3206},
}
W6_REF_ML = {   # only checked when multilabel_.csv is present
    "V3": {"ML P@10": 0.4556, "ML mAP": 0.2738, "nDCG@10": 0.7286, "selection_score": 0.4439},
    "V4": {"ML P@10": 0.4626, "ML mAP": 0.2938, "nDCG@10": 0.7298, "selection_score": 0.4686},
}

# ---- small candidate grid (decided before looking at any result) --------------
M_GRID = [3, 5, 10]
BETA_GRID = [0.2, 0.35, 0.5]

CLASS_METRICS = ["Class P@10", "Class mAP"]
ALL_ML_METRICS = ["ML P@10", "ML mAP", "nDCG@10", "selection_score"]

# Decision rule from Week 6 (unchanged): delta selection score > 0.005, >=4/5 folds positive,
# Class P@10 does not decrease overall.
RULE_MIN_DELTA = 0.005
RULE_MIN_POSITIVE_FOLDS = 4


# ==============================================================================
# 1. THE NEW METHOD
# ==============================================================================

def neighbourhood_expansion(D_base, D_gg, m=5, beta=0.35):
    """Blend a base distance matrix with the mean gallery distance to its top-m results.

    D_base : (n_q, n_g) distances produced by V4 (smaller = more similar)
    D_gg   : (n_g, n_g) V3 gallery-gallery distances (zero diagonal)
    m      : number of top-ranked gallery neighbours used as context
    beta   : weight of the neighbourhood term, 0 = V4 unchanged
    """
    assert D_base.ndim == 2 and D_gg.ndim == 2, "distance matrices must be 2D"
    assert D_gg.shape[0] == D_gg.shape[1], "D_gg must be square"
    assert D_base.shape[1] == D_gg.shape[0], (
        f"gallery dimension mismatch: D_base {D_base.shape[1]} vs D_gg {D_gg.shape[0]}")
    assert 1 <= m <= D_gg.shape[0], f"m={m} out of range"
    assert 0.0 <= beta <= 1.0, f"beta={beta} must be in [0, 1]"
    if beta == 0.0:
        return D_base.copy()
    top_m = np.argsort(D_base, axis=1, kind="stable")[:, :m]        # (n_q, m)
    N = D_gg[top_m].mean(axis=1)                                      # (n_q, n_g)
    return (1.0 - beta) * D_base + beta * N


# ==============================================================================
# 2. DATA LOADING (real labels if available, class-only fallback otherwise)
# ==============================================================================

def find_label_csv():
    for p in LABEL_CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def load_pipeline():
    """Builds the Week 6 pipeline object (V3 matrices for all 5 folds) with the unchanged dataset code."""
    csv = find_label_csv()
    class_only = csv is None
    if class_only:
        print("!" * 78)
        print("multilabel_.csv NOT FOUND -> CLASS-ONLY MODE.")
        print("Only Class P@10 and Class mAP are valid. ML P@10, ML mAP, nDCG@10 and the selection")
        print("score are NOT computed (reported as NaN). Put multilabel_.csv in the project root and rerun.")
        print("!" * 78)

        def class_only_labels(names):
            paths = cv.load_cached("train_hsv_16x16")[1]
            cls = [w1.parse_path(p)[1] for p in paths]
            u = sorted(set(cls))
            L = np.zeros((len(cls), len(u)))
            for i, c in enumerate(cls):
                L[i, u.index(c)] = 1.0
            return L
        cv.load_labels = class_only_labels

    ds = cv.CVDataset(folds_file=FOLDS_FILE)
    pipe = nc.Week6Pipeline.__new__(nc.Week6Pipeline)
    pipe.dataset = ds
    pipe.folds_file = FOLDS_FILE
    pipe.cache_dir = ROOT / "cache"
    pipe.precompute_v3_matrices()
    return pipe, class_only


def evaluate(D, fd, class_only):
    """Uses the unchanged Week 5 evaluator. In class-only mode the multi-label entries are masked."""
    m = cv.evaluate_ranking(-D, fd["q_cls"], fd["g_cls"], fd["J_mat"], fd["exclude_mask"])
    if class_only:
        for k in ALL_ML_METRICS:
            m[k] = float("nan")
        m["selection_score"] = float("nan")
    # development proxy for class-only mode: mean of the two valid class metrics
    m["class_proxy"] = float(np.mean([m["Class P@10"], m["Class mAP"]]))
    return m


def per_query_metrics(D, fd):
    """Per-query Class P@10 and Class AP (same definitions as the evaluator), post-hoc analysis only."""
    S = -D.copy()
    if fd["exclude_mask"] is not None:
        S[fd["exclude_mask"]] = -np.inf
    order = np.argsort(-S, axis=1, kind="stable")
    rows = np.arange(S.shape[0])[:, None]
    rc = (fd["q_cls"][:, None] == fd["g_cls"][None, :])[rows, order]
    p10 = rc[:, :10].mean(axis=1)
    cum = np.cumsum(rc, axis=1)
    prec = cum / np.arange(1, rc.shape[1] + 1)
    ap = (prec * rc).sum(axis=1) / np.maximum(rc.sum(axis=1), 1)
    return p10, ap, order


# ==============================================================================
# 3. CHECKS
# ==============================================================================

def sanity_checks(pipe):
    print("\n--- sanity checks ---")
    f = pipe.fold_data[0]
    D_qg, D_gg = f["D_qg"][:20], f["D_gg"]
    J = nc.compute_gallery_only_k_reciprocal_jaccard(D_qg, D_gg, **{"k1": 30, "k2": 3})
    D4 = V4_PARAMS["lambda"] * J + (1 - V4_PARAMS["lambda"]) * D_qg
    # 1. beta = 0 returns V4 exactly
    assert np.array_equal(neighbourhood_expansion(D4, D_gg, m=5, beta=0.0), D4)
    # 2. output shape preserved and finite
    out = neighbourhood_expansion(D4, D_gg, m=5, beta=0.35)
    assert out.shape == D4.shape and np.isfinite(out).all()
    # 3. query independence: single query == batch row
    for q in range(3):
        one = neighbourhood_expansion(D4[q:q + 1], D_gg, m=5, beta=0.35)[0]
        assert np.max(np.abs(one - out[q])) < 1e-12
    # 4. invalid configs raise
    for bad in [dict(m=0, beta=0.3), dict(m=5, beta=1.5), dict(m=10 ** 6, beta=0.3)]:
        try:
            neighbourhood_expansion(D4, D_gg, **bad)
            raise RuntimeError(f"invalid config not rejected: {bad}")
        except AssertionError:
            pass
    try:
        neighbourhood_expansion(D4, D_gg[:100, :100], m=5, beta=0.3)
        raise RuntimeError("dimension mismatch not rejected")
    except AssertionError:
        pass
    # 5. labels are not an argument anywhere in the method
    import inspect
    assert list(inspect.signature(neighbourhood_expansion).parameters) == ["D_base", "D_gg", "m", "beta"]
    print("all sanity checks passed")


# ==============================================================================
# 4. MAIN EXPERIMENT
# ==============================================================================

def fold_table(rows, name, keys):
    df = pd.DataFrame(rows)
    out = {"Method": name}
    for i in range(5):
        out[f"Fold {i + 1}"] = df.loc[i, keys]
    out["Mean"] = df[keys].mean()
    out["Std"] = df[keys].std()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-check", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    pipe, class_only = load_pipeline()
    sanity_checks(pipe)
    if args.sanity_check:
        return

    # ---------------- Stage 2: reproduce V3 and V4 --------------------------------
    v3_rows = [evaluate(f["D_qg"], f["fd"], class_only) for f in pipe.fold_data]
    t0 = time.time()
    D4 = []
    for f in pipe.fold_data:
        J = nc.compute_gallery_only_k_reciprocal_jaccard(f["D_qg"], f["D_gg"], k1=V4_PARAMS["k1"], k2=V4_PARAMS["k2"])
        D4.append(V4_PARAMS["lambda"] * J + (1 - V4_PARAMS["lambda"]) * f["D_qg"])
    v4_time = time.time() - t0
    v4_rows = [evaluate(D4[i], f["fd"], class_only) for i, f in enumerate(pipe.fold_data)]

    print("\n=== BASELINE REPRODUCTION (mean over 5 folds) ===")
    repro = []
    for name, rows in [("V3", v3_rows), ("V4", v4_rows)]:
        ref_all = dict(W6_REF[name])
        if not class_only:
            ref_all.update(W6_REF_ML[name])
        for k, ref in ref_all.items():
            obs = float(np.mean([r[k] for r in rows]))
            ok = abs(obs - ref) <= 1e-4
            repro.append({"system": name, "metric": k, "week6_recorded": ref, "reproduced": round(obs, 4), "match_1e-4": ok})
            print(f"{name} {k:11s} recorded={ref:.4f} reproduced={obs:.4f} {'MATCH' if ok else 'MISMATCH'}")
    pd.DataFrame(repro).to_csv(OUT / "baseline_reproduction.csv", index=False)
    assert all(r["match_1e-4"] for r in repro), "baseline not reproduced; stop and investigate"

    # ---------------- Stage 5: candidate grid on all 5 folds ----------------------
    configs = [{"m": m, "beta": b} for m in M_GRID for b in BETA_GRID]
    grid = {i: [] for i in range(len(configs))}
    D_cand_store = {}
    t0 = time.time()
    for fi, f in enumerate(pipe.fold_data):
        for ci, c in enumerate(configs):
            Dc = neighbourhood_expansion(D4[fi], f["D_gg"], m=c["m"], beta=c["beta"])
            grid[ci].append(evaluate(Dc, f["fd"], class_only))
    ne_time_grid = time.time() - t0

    gkeys = CLASS_METRICS + ["class_proxy"] + ([] if class_only else ALL_ML_METRICS)
    grid_df = pd.DataFrame([
        {**c, "fold": fi + 1, **{k: grid[ci][fi][k] for k in gkeys}}
        for ci, c in enumerate(configs) for fi in range(5)])
    grid_df.to_csv(OUT / "grid_all_configs_per_fold.csv", index=False)

    # ---------------- LOFO selection (same scheme as Week 6) -----------------------
    # selection criterion on the 4 training folds only; class_proxy in class-only mode
    sel_key = "selection_score" if not class_only else "class_proxy"
    lofo = []
    for test in range(5):
        train = [i for i in range(5) if i != test]
        best, best_s = None, -1
        for ci in range(len(configs)):
            s = float(np.mean([grid[ci][i][sel_key] for i in train]))
            if s > best_s:
                best, best_s = ci, s
        lofo.append({"outer_fold": test + 1, "cfg_idx": best, "cfg": configs[best], "train_score": best_s})
    print("\nLOFO-selected configs:", [(r["outer_fold"], r["cfg"]) for r in lofo])
    cand_rows = [grid[r["cfg_idx"]][r["outer_fold"] - 1] for r in lofo]

    # ---------------- Stage 6: comparison ----------------------------------------
    metric_keys = CLASS_METRICS + ["class_proxy"] + ([] if class_only else ALL_ML_METRICS)
    tables = {}
    for k in metric_keys:
        v4 = np.array([r[k] for r in v4_rows])
        cd = np.array([r[k] for r in cand_rows])
        d = cd - v4
        tables[k] = {
            "V4": v4, "candidate": cd, "delta": d,
            "mean_delta": float(d.mean()), "std_delta": float(d.std(ddof=1)),
            "rel_pct": float(100 * d.mean() / v4.mean()),
            "paired_ratio": float(d.mean() / (d.std(ddof=1) + 1e-12)),
            "positive_folds": int((d > 0).sum()),
        }
    comp_rows = []
    for k, t in tables.items():
        for name, arr in [("V4 baseline", t["V4"]), ("V4 + NE (LOFO-selected)", t["candidate"])]:
            comp_rows.append({"metric": k, "method": name, **{f"Fold {i + 1}": arr[i] for i in range(5)},
                              "Mean": arr.mean(), "Std": arr.std(ddof=1),
                              "Change vs V4": 0.0 if name.startswith("V4 b") else t["mean_delta"]})
        comp_rows.append({"metric": k, "method": "paired delta (cand - V4)", **{f"Fold {i + 1}": t["delta"][i] for i in range(5)},
                          "Mean": t["mean_delta"], "Std": t["std_delta"], "Change vs V4": t["mean_delta"]})
    pd.DataFrame(comp_rows).to_csv(OUT / "week7_comparison_table.csv", index=False)

    # fixed-config view of the whole grid (descriptive only, NOT used for selection on held-out)
    summ = []
    for ci, c in enumerate(configs):
        row = {**c}
        for k in gkeys:
            v = np.array([grid[ci][i][k] for i in range(5)])
            base = np.array([r[k] for r in v4_rows])
            row[f"{k}_mean"] = v.mean()
            row[f"{k}_delta_vs_V4"] = (v - base).mean()
            row[f"{k}_positive_folds"] = int(((v - base) > 0).sum())
        summ.append(row)
    pd.DataFrame(summ).to_csv(OUT / "grid_summary_vs_v4.csv", index=False)

    # ---------------- query-level analysis + paired bootstrap -----------------------
    qrows, ex_lists = [], []
    sel_cfg_by_fold = {r["outer_fold"]: r["cfg"] for r in lofo}
    rng = np.random.default_rng(0)
    for fi, f in enumerate(pipe.fold_data):
        c = sel_cfg_by_fold[fi + 1]
        Dc = neighbourhood_expansion(D4[fi], f["D_gg"], m=c["m"], beta=c["beta"])
        p4, ap4, o4 = per_query_metrics(D4[fi], f["fd"])
        pc, apc, oc = per_query_metrics(Dc, f["fd"])
        pv3, apv3, _ = per_query_metrics(f["D_qg"], f["fd"])
        for qi in range(len(p4)):
            gi = int(f["fd"]["q_idx"][qi])
            qrows.append({"fold": fi + 1, "query_path": pipe.dataset.paths[gi], "class": f["fd"]["q_cls"][qi],
                          "p10_v3": pv3[qi], "p10_v4": p4[qi], "p10_cand": pc[qi],
                          "ap_v3": apv3[qi], "ap_v4": ap4[qi], "ap_cand": apc[qi]})
            g_idx = f["fd"]["g_idx"]
            ex_lists.append({
                "query_path": pipe.dataset.paths[gi],
                "v4_top10": [pipe.dataset.paths[int(g_idx[j])] for j in o4[qi, :10]],
                "cand_top10": [pipe.dataset.paths[int(g_idx[j])] for j in oc[qi, :10]],
                "v4_top10_dist": [float(D4[fi][qi, j]) for j in o4[qi, :10]],
                "cand_top10_dist": [float(Dc[qi, j]) for j in oc[qi, :10]],
            })
    q = pd.DataFrame(qrows)
    q["d_p10"] = q.p10_cand - q.p10_v4
    q["d_ap"] = q.ap_cand - q.ap_v4
    q["outcome_p10"] = np.where(q.d_p10 > 1e-12, "improved", np.where(q.d_p10 < -1e-12, "worsened", "unchanged"))
    q["outcome_ap"] = np.where(q.d_ap > 1e-12, "improved", np.where(q.d_ap < -1e-12, "worsened", "unchanged"))
    q["changed_top10"] = [a["v4_top10"] != a["cand_top10"] for a in ex_lists]
    q.to_csv(OUT / "per_query_results.csv", index=False)

    boot = {}
    for col in ["d_p10", "d_ap"]:
        x = q[col].values
        means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)])
        boot[col] = {"mean": float(x.mean()), "ci95_low": float(np.percentile(means, 2.5)),
                     "ci95_high": float(np.percentile(means, 97.5))}

    pc = q.groupby("class").agg(p10_v4=("p10_v4", "mean"), p10_cand=("p10_cand", "mean"),
                                ap_v4=("ap_v4", "mean"), ap_cand=("ap_cand", "mean")).reset_index()
    pc["d_p10"] = pc.p10_cand - pc.p10_v4
    pc["d_ap"] = pc.ap_cand - pc.ap_v4
    pc.sort_values("d_p10").to_csv(OUT / "per_class_delta.csv", index=False)

    # representative examples (selected AFTER evaluation, post-hoc analysis only)
    def pick(df, n, asc):
        return df.sort_values("d_ap", ascending=asc).head(n)
    examples = {
        "most_improved": pick(q, 3, False),
        "most_worsened": pick(q, 3, True),
        "hard_queries_v4": q.sort_values("ap_v4").head(3),
        "easy_successes_v4": q.sort_values(["p10_v4", "ap_v4"], ascending=False).head(3),
    }
    ex_idx = {p: i for i, p in enumerate(q.query_path)}
    ex_out = []
    for cat, df in examples.items():
        for _, r in df.iterrows():
            e = ex_lists[ex_idx[r.query_path]]
            cls = lambda p: w1.parse_path(p)[1]
            ex_out.append({
                "category": cat, "query": r.query_path, "class": r["class"], "fold": int(r.fold),
                "ap_v4": round(r.ap_v4, 4), "ap_cand": round(r.ap_cand, 4),
                "p10_v4": r.p10_v4, "p10_cand": r.p10_cand,
                "v4_top10_classes": ";".join(cls(p) for p in e["v4_top10"]),
                "cand_top10_classes": ";".join(cls(p) for p in e["cand_top10"]),
                "v4_top10_files": ";".join(Path(p).name for p in e["v4_top10"]),
                "cand_top10_files": ";".join(Path(p).name for p in e["cand_top10"]),
            })
    pd.DataFrame(ex_out).to_csv(OUT / "error_analysis_examples.csv", index=False)

    # ---------------- runtime ------------------------------------------------------
    t0 = time.time()
    f = pipe.fold_data[0]
    for _ in range(3):
        neighbourhood_expansion(D4[0], f["D_gg"], m=5, beta=0.35)
    ne_per_fold = (time.time() - t0) / 3

    # ---------------- Stage 8: decision (Week 6 rule, unchanged) -------------------
    dp = tables[sel_key]
    d_p10 = tables["Class P@10"]
    rule = {
        "mode": "CLASS_ONLY_PROVISIONAL" if class_only else "FULL",
        "criterion_1_delta_gt_0.005": {"value": dp["mean_delta"], "metric": sel_key, "pass": dp["mean_delta"] > RULE_MIN_DELTA},
        "criterion_2_>=4_of_5_folds_positive": {"positive_folds": dp["positive_folds"], "pass": dp["positive_folds"] >= RULE_MIN_POSITIVE_FOLDS},
        "criterion_3_class_p10_not_decreased": {"delta": d_p10["mean_delta"], "pass": d_p10["mean_delta"] >= 0},
    }
    rule["all_pass"] = all(v["pass"] for k, v in rule.items() if isinstance(v, dict))
    decision = ("V5 accepted" if rule["all_pass"] else "V4 retained") if not class_only else \
               ("PROVISIONAL: V5 would pass on class metrics only" if rule["all_pass"] else "PROVISIONAL: V4 retained")

    summary = {
        "decision": decision, "rule": rule, "class_only_mode": class_only,
        "lofo_selected": [{"outer_fold": r["outer_fold"], **r["cfg"], "train_score": r["train_score"]} for r in lofo],
        "v4_params": V4_PARAMS, "grid": {"m": M_GRID, "beta": BETA_GRID},
        "tables": {k: {"V4": t["V4"].tolist(), "candidate": t["candidate"].tolist(), "delta": t["delta"].tolist(),
                       "mean_delta": t["mean_delta"], "std_delta": t["std_delta"], "rel_pct": t["rel_pct"],
                       "paired_ratio": t["paired_ratio"], "positive_folds": t["positive_folds"]} for k, t in tables.items()},
        "query_level": {
            "p10": q.outcome_p10.value_counts().to_dict(), "ap": q.outcome_ap.value_counts().to_dict(),
            "top10_changed": int(q.changed_top10.sum()), "n_queries": int(len(q))},
        "bootstrap_query_level": boot,
        "runtime_sec": {"v4_stage_all_folds": v4_time, "ne_grid_all_configs_all_folds": ne_time_grid,
                        "ne_single_fold_480_queries": ne_per_fold, "total_script": time.time() - t_all},
    }
    (OUT / "week7_results.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== RESULTS (class metrics) ===")
    for k in metric_keys:
        t = tables[k]
        print(f"{k:15s} V4 {t['V4'].mean():.4f}±{t['V4'].std(ddof=1):.4f} | cand {t['candidate'].mean():.4f}±{t['candidate'].std(ddof=1):.4f} "
              f"| delta {t['mean_delta']:+.4f} ({t['rel_pct']:+.2f}%) | pos folds {t['positive_folds']}/5 | ratio {t['paired_ratio']:.2f}")
        print("   per-fold delta:", np.round(t["delta"], 4))
    print("query-level P@10:", q.outcome_p10.value_counts().to_dict(), "| AP:", q.outcome_ap.value_counts().to_dict())
    print("bootstrap:", boot)
    print("rule:", json.dumps(rule, default=float))
    print("DECISION:", decision)


if __name__ == "__main__":
    main()
