"""
build_final_results.py - Week 8: assemble the FINAL RESULTS folder from the Week 1-7 evidence.

Nothing here is tuned. It (1) recomputes V2 / V3 / V4 on the same 5-fold LOFO protocol and checks the numbers against the
values recorded in Weeks 5-6, (2) adds paired fold-level and query-level bootstrap comparisons, (3) collects the weak-class,
reliability, k2 and failed-experiment (AQE, NE, ...) evidence from the Week 6/7 files, (4) draws the figures, (5) writes
the frozen configuration, and (6) copies the final-test results ONLY if Week 7's final_test.py has actually completed.

Run from anywhere:  python "Week 8/build_final_results.py"
Then:               python "Week 8/verify_final_results.py"
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "Week 7"))
import week7_common as c          # noqa: E402
import final_test as ft           # noqa: E402  (reused: V2/V3/V4 scoring, per-query metrics, paired bootstrap)

FR = Path(os.environ.get("W8_OUT", HERE / "final_results"))          # env overrides are for testing only
W7 = ROOT / "Week 7"
FT_DIR = Path(os.environ.get("W8_FINAL_TEST_DIR", W7 / "final_test"))
CMP = ft.CMP
KEY6 = ["Class P@10", "Class mAP", "ML P@10", "ML mAP", "nDCG@10", "selection_score"]
WEAK = ["airport", "bridge", "square"]
COL = {"V2": "#9e9e9e", "V3": "#1f77b4", "V4": "#ff7f0e"}

# recorded reference values (Week 5 v3_validation_summary.md, V2 baseline fold 1 + means; Week 6 for V3/V4)
V2_FOLD1 = {"Class P@10": 0.4312, "ML P@10": 0.4002, "ML mAP": 0.2387, "nDCG@10": 0.6814, "Class mAP": 0.2125, "selection_score": 0.3928}
V2_MEAN = {"Class P@10": 0.4336, "selection_score": 0.3970}


def sub(name):
    d = FR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def md(df, floatfmt=4):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(x) for x in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join((f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


def write_readme(summ, pf, test_done, d_t):
    pm = pd.read_csv(FR / "lofo_comparison/paired_fold_deltas.csv")
    bs = pd.read_csv(FR / "lofo_comparison/paired_bootstrap_lofo_queries.csv")
    fx = pd.read_csv(FR / "failed_experiments/failed_experiments.csv")
    wk = pd.read_csv(FR / "weak_classes/weak_classes_v2_v3_v4_lofo.csv")
    rc = pd.read_csv(FR / "reliability/reliability_categories_lofo_heldout.csv")
    rows = []
    for k in ["V2", "V3", "V4"]:
        g = summ[summ.system == k].iloc[0]
        rows.append({"System": k, **{m: f"{g[f'{m}_mean']:.4f} +/- {g[f'{m}_std']:.4f}" for m in CMP}})
    main_t = pd.DataFrame(rows)
    pd_t = pm[pm.metric.isin(KEY6) & pm.comparison.isin(["V4 - V3", "V4 - V2"])][["comparison", "metric", "mean_delta", "relative_pct", "folds_improved"]]
    bs_t = bs[bs.comparison.isin(["V4 - V3", "V4 - V2"])][["comparison", "metric", "mean_diff", "ci95_low", "ci95_high", "queries_better", "queries_worse"]]
    fx_t = fx[["experiment", "week", "reference", "selection_score", "delta_vs_reference", "outcome", "reason"]]
    cfgj = json.loads((FR / "config/final_config.json").read_text())
    t = []
    t.append("# Final results folder (Week 8, task 1)\n")
    t.append(f"**Final-test status: {cfgj['final_test_status']}.**\n")
    if not test_done:
        t.append("Every number below comes from the 5-fold LOFO evaluation on the 2,400 training images (480 held-out queries vs 1,920 gallery per fold). "
                 "**No test-set number for V3 or V4 exists yet**, so none is reported. The test-set comparison slots into `final_test/` once "
                 "`Week 7/final_test.py` has been run once; rerun `build_final_results.py` and `verify_final_results.py` afterwards.\n")
    t.append("## Frozen system\nV4 = V3 + gallery-only k-reciprocal re-ranking. V3 weights HSV 0.20 / LBP-r1 0.40 / LBP-r2 0.20 / HOG 0.20 (Hellinger, Hellinger, Hellinger, cosine; raw fusion). "
             "k1 = 30, k2 = 3, lambda = 0.5. Reliability score = 1 - D_V4(top-1), LOW < %.4f, HIGH >= %.4f. Full details: `config/final_config.json`.\n"
             % (cfgj["reliability"]["thresholds"]["LOW"], cfgj["reliability"]["thresholds"]["HIGH"]))
    t.append("## 1. V2 vs V3 vs V4 (LOFO, mean +/- std over 5 folds)\n" + md(main_t) + "\n")
    t.append("Each recorded Week 5/6 value was reproduced to 4 decimals (`lofo_comparison/reproduction_checks.csv`, 20 of 20).\n")
    t.append("## 2. Paired comparison\nFold-level deltas (5 folds):\n\n" + md(pd_t) + "\n")
    t.append("Query-level paired bootstrap (10,000 resamples over the 2,400 pooled held-out queries; queries are treated as independent, so the intervals are approximate):\n\n" + md(bs_t) + "\n")
    d5 = pm[(pm.comparison == "V4 - V3") & pm.metric.isin(["Jaccard@5", "nDCG@5"])].set_index("metric")
    top5 = (f"Caution on top-5 multi-label ranking: V4 is slightly LOWER than V3 on Jaccard@5 ({d5.loc['Jaccard@5', 'mean_delta']:+.4f}, {int(d5.loc['Jaccard@5', 'folds_improved'])}/5 folds up) "
            f"and nDCG@5 ({d5.loc['nDCG@5', 'mean_delta']:+.4f}, {int(d5.loc['nDCG@5', 'folds_improved'])}/5 folds up); see `lofo_comparison/paired_fold_deltas.csv`. ")
    t.append(top5 + "")
    t.append("How to read this: V4 improves on V3 clearly in Class mAP, Class P@10 and ML mAP (all 5 folds, intervals far from zero). "
             "ML P@10 is a small gain (about +0.007, 4 of 5 folds, interval barely above zero) and nDCG@10 is **unchanged within noise** (+0.0012, 2 of 5 folds). "
             "A bootstrap interval above zero is not the same as a practically meaningful gain; five folds are limited evidence.\n")
    t.append("## 3. Weak classes (LOFO, 80 queries per class)\n" + md(wk, 3) + "\n")
    t.append("Airport gains clearly from V3 to V4. Bridge and square gain slightly; their V3-to-V4 P@10 intervals include zero (`weak_classes/per_class_v3_vs_v4.csv`) and both remain among the weakest classes. "
             "Top confusions: `weak_classes/focus_class_top_confusions.csv`. V4 also made some classes worse (e.g. park, commercial, denseresidential).\n")
    t.append("## 4. Reliability (LOFO held-out; thresholds from the other four folds)\n" + md(rc[["label", "n_queries", "pct_queries", "class_p10", "ml_p10", "class_ap"]], 3) + "\n")
    t.append("The thresholds were fixed on LOFO data with 1,920-image galleries; shares on the 2,400-image test gallery may differ and will be reported, not re-tuned.\n")
    t.append("## 5. Unsuccessful / not-adopted experiments\n" + md(fx_t) + "\n")
    if test_done:
        ov = pd.read_csv(d_t / "final_overall_results_v2_v3_v4.csv")
        tb = pd.read_csv(d_t / "paired_bootstrap.csv")[["comparison", "metric", "mean_diff", "ci95_low", "ci95_high", "queries_better", "queries_worse"]]
        rt = pd.read_csv(d_t / "reliability_categories_test.csv")
        t.append("## 6. FINAL TEST (600 queries vs 2,400 gallery, run once)\n" + md(ov) + "\n\nPaired bootstrap:\n\n" + md(tb) + "\n\nReliability on test (frozen thresholds):\n\n" + md(rt.drop(columns=[c_ for c_ in rt.columns if c_ == "note"]), 3) + "\n")
        o = ov.set_index("Model"); dd = o.loc["V4"] - o.loc["V3"]
        up = [m for m in CMP if dd[m] > 1e-4]; down = [m for m in CMP if dd[m] < -1e-4]
        t.append("Reading of the test result: V4 vs V3 is higher on " + ", ".join(f"{m} ({dd[m]:+.4f})" for m in up) +
                 " and lower on " + (", ".join(f"{m} ({dd[m]:+.4f})" for m in down) if down else "none") +
                 ". V4 vs V2 is higher on " + ", ".join(m for m in CMP if (o.loc['V4'] - o.loc['V2'])[m] > 0) + ". "
                 "Check `final_test/paired_bootstrap.csv` before calling any difference real: e.g. the V4 - V3 ML P@10 interval includes zero. "
                 "The weak-class test numbers (20 queries per class) are noisy and differ from LOFO (bridge does not improve on test).\n")
        t.append("Note: the test set was also used in earlier weeks (Week 1 fusion configurations, V1 and V2), so it is not a fully blind holdout for the project as a whole; V3 and V4 were selected on train/LOFO data only.\n")
    else:
        t.append("## 6. Final test\nNot run - see `final_test/PENDING.md`.\n")
    if test_done:
        t.append("Status note: this README was regenerated after the final test; sections 1-5 remain LOFO (training data) results.\n")
    t.append("## Folder map\n`config/` frozen configurations | `lofo_comparison/` per-fold, summary, paired deltas, bootstrap, reproduction checks | `weak_classes/` | `reliability/` | `k2_check/` | "
             "`failed_experiments/` (AQE, NE and others, with NE grid files) | `figures/` | `final_test/` | `MANIFEST.sha256` (file hashes).\n")
    t.append("## Reproduce\n```\npython \"Week 7/k2_check.py\"\npython \"Week 7/weak_class_analysis.py\"\npython \"Week 7/reliability_recalibration.py\"\npython \"Week 8/build_final_results.py\"\npython \"Week 8/verify_final_results.py\"\n```\n")
    (FR / "README.md").write_text("\n".join(t))


def main():
    FR.mkdir(exist_ok=True)
    # ------------------------------------------------------------------ 1. LOFO V2 / V3 / V4
    pipe = c.load_pipeline()
    per_fold, pq_all = [], {k: [] for k in ["V2", "V3", "V4"]}
    for fi, f in enumerate(pipe.fold_data):
        fd = f["fd"]
        _, mets, pq = ft.run_systems(fd["q_feats"], fd["g_feats"], fd["q_cls"], fd["g_cls"], fd["J_mat"], fd["exclude_mask"])
        for k in mets:
            per_fold.append({"system": k, "fold": fi + 1, **{m: mets[k][m] for m in CMP}})
            t = pq[k].copy(); t["fold"] = fi + 1; t["class"] = fd["q_cls"]
            pq_all[k].append(t)
    pf = pd.DataFrame(per_fold)
    pq_all = {k: pd.concat(v, ignore_index=True) for k, v in pq_all.items()}

    # reproduction checks against recorded numbers (stop if anything differs)
    checks = []
    for sysname in ["V3", "V4"]:
        rows = [{m: r[m] for m in KEY6} for _, r in pf[pf.system == sysname].iterrows()]
        checks += c.check_reproduction(sysname, rows)
    v2f1 = pf[(pf.system == "V2") & (pf.fold == 1)].iloc[0]
    for m, ref in V2_FOLD1.items():
        checks.append({"system": "V2 (fold 1)", "metric": m, "week6_recorded": ref, "reproduced": round(float(v2f1[m]), 4), "match": abs(float(v2f1[m]) - ref) <= 1e-4})
    for m, ref in V2_MEAN.items():
        obs = float(pf[pf.system == "V2"][m].mean())
        checks.append({"system": "V2 (mean)", "metric": m, "week6_recorded": ref, "reproduced": round(obs, 4), "match": abs(obs - ref) <= 1e-4})
    chk = pd.DataFrame(checks).rename(columns={"week6_recorded": "recorded_in_earlier_week"})
    d_lofo = sub("lofo_comparison")
    chk.to_csv(d_lofo / "reproduction_checks.csv", index=False)
    assert chk["match"].all(), "a recorded value was not reproduced:\n" + chk[~chk["match"]].to_string()

    pf.to_csv(d_lofo / "v2_v3_v4_per_fold.csv", index=False)
    summ = []
    for k in ["V2", "V3", "V4"]:
        g = pf[pf.system == k]
        summ.append({"system": k, **{f"{m}_mean": g[m].mean() for m in CMP}, **{f"{m}_std": g[m].std(ddof=1) for m in CMP}})
    summ = pd.DataFrame(summ)
    summ.to_csv(d_lofo / "v2_v3_v4_summary.csv", index=False)

    # paired fold-level deltas
    rows = []
    for a, b in [("V4", "V2"), ("V4", "V3"), ("V3", "V2")]:
        A = pf[pf.system == a].sort_values("fold"); B = pf[pf.system == b].sort_values("fold")
        for m in CMP:
            d = A[m].values - B[m].values
            rows.append({"comparison": f"{a} - {b}", "metric": m, "mean_delta": d.mean(), "std_delta": d.std(ddof=1),
                         "relative_pct": 100 * d.mean() / B[m].mean(), "folds_improved": int((d > 0).sum()),
                         **{f"fold{i + 1}": d[i] for i in range(5)}})
    pd.DataFrame(rows).to_csv(d_lofo / "paired_fold_deltas.csv", index=False)

    # paired query-level bootstrap on the 2,400 pooled held-out queries (approximate: queries treated as independent)
    rng = np.random.default_rng(0)
    bs = []
    for a, b in [("V4", "V2"), ("V4", "V3"), ("V3", "V2")]:
        for col in ["class_p10", "ml_p10", "class_ap", "ml_ap"]:
            bs.append(ft.paired(pq_all[a], pq_all[b], a, b, col, rng))
    pd.DataFrame(bs).to_csv(d_lofo / "paired_bootstrap_lofo_queries.csv", index=False)

    # ------------------------------------------------------------------ 2. weak classes
    d_w = sub("weak_classes")
    for fn in ["per_class_v3_vs_v4.csv", "focus_class_top_confusions.csv", "overall_v3_vs_v4_query_counts.csv",
               "confusion_top10_share_V3.csv", "confusion_top10_share_V4.csv"]:
        shutil.copy(W7 / "weak_class_analysis" / fn, d_w / fn)
    wk = []
    for cls in WEAK:
        r = {"class": cls}
        for k in ["V2", "V3", "V4"]:
            g = pq_all[k][pq_all[k]["class"] == cls]
            r[f"{k}_class_p10"], r[f"{k}_class_ap"] = g.class_p10.mean(), g.class_ap.mean()
            r["n_queries"] = len(g)
        wk.append(r)
    pd.DataFrame(wk).to_csv(d_w / "weak_classes_v2_v3_v4_lofo.csv", index=False)

    # ------------------------------------------------------------------ 3. reliability, k2, NE
    d_r = sub("reliability")
    for fn in ["reliability_config_v4.json", "reliability_correlations.csv", "reliability_categories_lofo_heldout.csv",
               "reliability_categories_per_fold_class_p10.csv"]:
        shutil.copy(W7 / "reliability_recalibration" / fn, d_r / fn)
    d_k = sub("k2_check")
    for fn in ["k2_summary.csv", "k2_per_fold.csv", "k2_vs_current_v4.csv", "k2_lofo_selection.csv", "k2_decision.json"]:
        shutil.copy(W7 / "k2_check" / fn, d_k / fn)

    # ------------------------------------------------------------------ 4. failed / rejected experiments (read from source files)
    d_f = sub("failed_experiments")
    for fn in ["week7_results.json", "week7_comparison_table.csv", "grid_summary_vs_v4.csv"]:
        shutil.copy(W7 / "ne_ablation" / fn, d_f / f"ne_ablation_{fn}")
    w6 = pd.read_csv(ROOT / "Week 6" / "neighborhood_results.csv").set_index("experiment")
    ne = json.loads((W7 / "ne_ablation" / "week7_results.json").read_text())
    k2s = pd.read_csv(W7 / "k2_check" / "k2_summary.csv").set_index("system")
    v3ss, v4ss = w6.loc["Frozen V3 Baseline", "selection_score_mean"], w6.loc["Gallery-Only KR (V4 Proposed)", "selection_score_mean"]
    fx = []
    def add(**kw): fx.append(kw)
    add(experiment="Average Query Expansion (AQE) on V3", week=6, reference="V3", selection_score=w6.loc["AQE (Ablation)", "selection_score_mean"],
        delta_vs_reference=w6.loc["AQE (Ablation)", "selection_score_mean"] - v3ss, class_p10=w6.loc["AQE (Ablation)", "Class P@10_mean"],
        class_map=w6.loc["AQE (Ablation)", "Class mAP_mean"], outcome="rejected (ablation)",
        reason="gain +0.0023 is below the +0.005 threshold; Class P@10 -0.0042, negative on 4 of 5 folds", source="Week 6/neighborhood_summary.md, neighborhood_results.csv")
    add(experiment="AQE then k-reciprocal (Order 1)", week=6, reference="V4", selection_score=w6.loc["Combined (Order 1: AQE -> KR)", "selection_score_mean"],
        delta_vs_reference=w6.loc["Combined (Order 1: AQE -> KR)", "selection_score_mean"] - v4ss, class_p10=w6.loc["Combined (Order 1: AQE -> KR)", "Class P@10_mean"],
        class_map=w6.loc["Combined (Order 1: AQE -> KR)", "Class mAP_mean"], outcome="rejected", reason="lower than gallery-only k-reciprocal alone",
        source="Week 6/neighborhood_results.csv")
    add(experiment="Transductive k-reciprocal (earlier version)", week=6, reference="V4", selection_score=w6.loc["Transductive KR (Ablation)", "selection_score_mean"],
        delta_vs_reference=w6.loc["Transductive KR (Ablation)", "selection_score_mean"] - v4ss, class_p10=w6.loc["Transductive KR (Ablation)", "Class P@10_mean"],
        class_map=w6.loc["Transductive KR (Ablation)", "Class mAP_mean"], outcome="replaced by gallery-only version",
        reason="Week 6 kept it as an ablation; V4 uses the gallery-only single-query protocol instead", source="Week 6/neighborhood_results.csv")
    add(experiment="Per-query Z-score normalisation (on V2)", week=5, reference="V2 raw fusion", selection_score=0.3813, delta_vs_reference=0.3813 - 0.3970,
        class_p10=0.4030, class_map=np.nan, outcome="rejected", reason="lower than unnormalised V2 even after retuning (best 0.3954); recorded value, not rerun here",
        source="Week 5/v3_validation_summary.md")
    t = ne["tables"]["selection_score"]
    add(experiment="Neighbourhood expansion (NE) on V4", week=7, reference="V4", selection_score=float(np.mean(t["candidate"])), delta_vs_reference=t["mean_delta"],
        class_p10=float(np.mean(ne["tables"]["Class P@10"]["candidate"])), class_map=float(np.mean(ne["tables"]["Class mAP"]["candidate"])),
        outcome="rejected (ablation)", reason=f"selection score {t['mean_delta']:+.4f}, {t['positive_folds']}/5 folds up; failed all three Week 6 criteria; every one of 9 settings below V4",
        source="Week 7/ne_ablation/")
    for k in [1, 2]:
        r = k2s.loc[f"V4 k2={k}"]
        add(experiment=f"V4 with k2={k}", week=7, reference="V4 (k2=3)", selection_score=r["selection_score_mean"], delta_vs_reference=r["selection_score_mean"] - k2s.loc["V4 k2=3", "selection_score_mean"],
            class_p10=r["Class P@10_mean"], class_map=r["Class mAP_mean"], outcome="not adopted",
            reason="gain < 0.005 needed to replace k2=3 (selection score lower); LOFO selected k2=3 on all folds", source="Week 7/k2_check/")
    fdf = pd.DataFrame(fx)
    fdf.to_csv(d_f / "failed_experiments.csv", index=False)

    # ------------------------------------------------------------------ 5. figures
    fg = sub("figures")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (ms, title) in enumerate([(["Class P@10", "ML P@10", "nDCG@10"], "Top-10 metrics"), (["Class mAP", "ML mAP", "selection_score"], "mAP and selection score")]):
        x = np.arange(len(ms))
        for j, k in enumerate(["V2", "V3", "V4"]):
            g = summ[summ.system == k].iloc[0]
            ax[i].bar(x + (j - 1) * 0.26, [g[f"{m}_mean"] for m in ms], 0.26, yerr=[g[f"{m}_std"] for m in ms], capsize=3, label=k, color=COL[k])
        ax[i].set_xticks(x); ax[i].set_xticklabels(ms); ax[i].set_title(title); ax[i].legend()
    fig.suptitle("V2 vs V3 vs V4 - 5-fold LOFO (train data), mean +/- std"); fig.tight_layout(); fig.savefig(fg / "fig_lofo_v2_v3_v4.png", dpi=150); plt.close(fig)

    pc = pd.read_csv(W7 / "weak_class_analysis" / "per_class_v3_vs_v4.csv").sort_values("d_p10")
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(pc["class"], pc.d_p10, color=np.where(pc.d_p10 >= 0, COL["V4"], "#d62728")); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Change in Class P@10 (V4 - V3), LOFO"); ax.set_title("Class-wise change, V3 to V4"); fig.tight_layout(); fig.savefig(fg / "fig_classwise_p10_change.png", dpi=150); plt.close(fig)

    wdf = pd.DataFrame(wk); x = np.arange(3)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (m, t_) in enumerate([("class_p10", "Class P@10"), ("class_ap", "Class AP")]):
        for j, k in enumerate(["V2", "V3", "V4"]):
            ax[i].bar(x + (j - 1) * 0.26, wdf[f"{k}_{m}"], 0.26, label=k, color=COL[k])
        ax[i].set_xticks(x); ax[i].set_xticklabels(WEAK); ax[i].set_title(t_); ax[i].legend()
    fig.suptitle("Weak classes, LOFO (80 queries per class)"); fig.tight_layout(); fig.savefig(fg / "fig_weak_classes.png", dpi=150); plt.close(fig)

    rc = pd.read_csv(d_r / "reliability_categories_lofo_heldout.csv").set_index("label").loc[["HIGH", "MEDIUM", "LOW"]]
    fig, ax = plt.subplots(figsize=(5.5, 4)); x = np.arange(3)
    ax.bar(x - 0.2, rc.class_p10, 0.4, label="Class P@10"); ax.bar(x + 0.2, rc.ml_p10, 0.4, label="ML P@10")
    ax.set_xticks(x); ax.set_xticklabels(rc.index); ax.legend(); ax.set_title("Reliability categories (LOFO held-out)")
    fig.tight_layout(); fig.savefig(fg / "fig_reliability.png", dpi=150); plt.close(fig)

    ks = pd.read_csv(d_k / "k2_summary.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(ks.system, ks.selection_score_mean, yerr=ks.selection_score_std, capsize=3, color=[COL["V3"]] + [COL["V4"]] * 3)
    ax.set_ylim(0.43, 0.48); ax.set_ylabel("LOFO selection score"); ax.set_title("k2 check (k1=30, lambda=0.5)"); fig.tight_layout(); fig.savefig(fg / "fig_k2_check.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    f2 = fdf.sort_values("delta_vs_reference")
    ax.barh(f2.experiment, f2.delta_vs_reference, color="#d62728"); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Selection-score change vs the stated reference"); ax.set_title("Unsuccessful / not-adopted experiments")
    fig.tight_layout(); fig.savefig(fg / "fig_failed_experiments.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ 6. frozen configuration
    d_c = sub("config")
    for src, dst in [(ROOT / "Week4" / "v2_config.json", "v2_config.json"), (ROOT / "Week 5" / "v3_config.json", "v3_config.json"),
                     (ROOT / "Week 6" / "selected_parameters.json", "week6_selected_parameters.json"),
                     (W7 / "reliability_recalibration" / "reliability_config_v4.json", "reliability_config_v4.json")]:
        shutil.copy(src, d_c / dst)
    rel = json.loads((d_c / "reliability_config_v4.json").read_text())
    done = FT_DIR / "FINAL_TEST_DONE.json"
    test_done = done.exists() and json.loads(done.read_text()).get("status") == "completed"
    cfg = {
        "frozen_final_system": "V4 = V3 + gallery-only k-reciprocal re-ranking",
        "V2": json.loads((d_c / "v2_config.json").read_text()),
        "V3_weights": {"hsv": 0.20, "lbp_r1_p8": 0.40, "lbp_r2_p16": 0.20, "hog": 0.20},
        "V3_similarities": {"hsv": "hellinger", "lbp_r1_p8": "hellinger", "lbp_r2_p16": "hellinger", "hog": "cosine"},
        "V3_normalisation": "raw fusion (no z-score); HSV/LBP histograms L1-normalised; HOG L2-Hys",
        "V4_k_reciprocal": {"k1": 30, "k2": 3, "lambda": 0.5, "gallery_only": True},
        "reliability": {"score": rel["formula"], "thresholds": rel["thresholds"], "calibrated_on": "LOFO held-out queries (train data) only"},
        "evaluation_protocol": {"folds": "Week2/folds_5.json (5-fold LOFO, 480 queries / 1,920 gallery per fold)",
                                "final_test": "600 test queries vs 2,400 gallery", "metrics": CMP},
        "not_adopted": ["AQE", "AQE then k-reciprocal", "neighbourhood expansion (NE)", "k2 = 1, 2", "z-score normalisation"],
        "not_tested": ["spatial pyramid / Gabor / LAB features"],
        "final_test_status": "completed" if test_done else "NOT RUN (test-split feature caches not available when this folder was built)",
    }
    (d_c / "final_config.json").write_text(json.dumps(cfg, indent=2))

    # ------------------------------------------------------------------ 7. final test results (only if it really ran)
    d_t = sub("final_test")
    for p in d_t.glob("*"):
        p.unlink()
    if test_done:
        for p in FT_DIR.glob("*.csv"):
            shutil.copy(p, d_t / p.name)
        shutil.copy(done, d_t / "FINAL_TEST_DONE.json")
        ov = pd.read_csv(d_t / "final_overall_results_v2_v3_v4.csv").set_index("Model")
        fig, ax = plt.subplots(figsize=(10, 4)); ms = ["Class P@10", "ML P@10", "nDCG@10", "Class mAP", "ML mAP", "selection_score"]; x = np.arange(len(ms))
        for j, k in enumerate(["V2", "V3", "V4"]):
            ax.bar(x + (j - 1) * 0.26, [ov.loc[k, m] for m in ms], 0.26, label=k, color=COL[k])
        ax.set_xticks(x); ax.set_xticklabels(ms); ax.legend(); ax.set_title("FINAL TEST: 600 queries vs 2,400 gallery")
        fig.tight_layout(); fig.savefig(fg / "fig_final_test_v2_v3_v4.png", dpi=150); plt.close(fig)
    else:
        (d_t / "PENDING.md").write_text(
            "# Final test (V2 vs V3 vs V4) - NOT RUN\n\nNo test numbers exist yet, so none are reported anywhere in this folder.\n"
            "Needs `cache/test_hsv_16x16`, `test_lbp_r1_p8`, `test_lbp_r2_p16`, `test_hog_128_o9_c16_b2` (.npy + _paths.json).\n"
            "Then: `python \"Week 7/final_test.py\"` (once), and rerun `python \"Week 8/build_final_results.py\"` to import the results.\n")

    write_readme(summ, pf, test_done, d_t)
    pd.set_option("display.width", 220)
    print(summ[["system"] + [f"{m}_mean" for m in KEY6]].round(4).to_string(index=False))
    print("\nreproduction checks:", int(chk["match"].sum()), "of", len(chk), "match")
    print("final test status:", cfg["final_test_status"])


if __name__ == "__main__":
    main()
