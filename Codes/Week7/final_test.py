"""
final_test.py - Week 7 FINAL TEST (run ONCE): V2 vs V3 vs V4 on the held-out test split.

Queries : 600 test images          Gallery : the 2,400 train images (same as Week 4)
Systems : V2 (Week 4 frozen, corr/cos similarities), V3 (Week 5 frozen), V4 (V3 + gallery-only KR, k1=30, k2=3, lambda=0.5)
Metrics : unchanged Week 5 evaluator (Class/ML P@5, P@10, Jaccard@K, nDCG@K, Class mAP, ML mAP, selection score)

Order of work (nothing is tuned here):
  1. check that the test-split feature caches exist (cache/test_<tag>.npy, built with Week 1 cache_features.py / Week 3 cache_test_r2.py)
  2. reproduce V2 against the old Week 4 result (overall table + per-query Class P@10); STOP if it does not match
  3. evaluate V3 and V4, paired bootstrap V4-vs-V2 and V4-vs-V3 (query level) + per-class sign tests
  4. weak classes (square, bridge, airport): per-class rows, a pooled 60-query bootstrap and top-10 confusions (reporting only)
  5. evaluate the frozen V4 reliability categories (Week 7/reliability_recalibration/reliability_config_v4.json)
The lock file final_test/FINAL_TEST_DONE.json is written as soon as the V2 check passes, so a crash after that point cannot
silently lead to a second look at the test set. No parameter is read from test results.

Run:
  python "Week 7/final_test.py" --dry-run     # plumbing check on LOFO fold 1 (train data only, writes nothing)
  python "Week 7/final_test.py"               # the real, single final test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

import week7_common as c

os.chdir(c.ROOT)                              # cache_features uses a cwd-relative cache folder
OUT = c.HERE / "final_test"
LOCK = OUT / "FINAL_TEST_DONE.json"
V4 = {"k1": 30, "k2": 3, "lambda": 0.5}       # frozen in Week 7 (k2 check: k2=3 kept)
N_BOOT = 10000
WEAK = ["square", "bridge", "airport"]
CMP = ["Class P@5", "ML P@5", "Jaccard@5", "nDCG@5", "Class P@10", "ML P@10", "Jaccard@10", "nDCG@10",
       "Class mAP", "ML mAP", "selection_score"]


def v2_scores(q_feats, g_feats):
    cfg = json.loads((c.ROOT / "Week4" / "v2_config.json").read_text())
    fns = {"hsv": c.w1.corr_sim, "lbp_r1_p8": c.w1.corr_sim, "lbp_r2_p16": c.w1.corr_sim, "hog": c.w1.cos_sim}
    return sum(cfg[k] * fns[k](q_feats[k], g_feats[k]) for k in fns)


def v3_matrices(q_feats, g_feats):
    W, F = c.nc.FROZEN_V3_WEIGHTS, c.nc.FROZEN_V3_SIM_FNS
    S_qg = sum(W[k] * F[k](q_feats[k], g_feats[k]) for k in W)
    D_qg = np.clip(1.0 - S_qg, 0.0, None)
    S_gg = sum(W[k] * F[k](g_feats[k], g_feats[k]) for k in W)
    np.fill_diagonal(S_gg, 1.0)
    D_gg = np.clip(1.0 - S_gg, 0.0, None)
    np.fill_diagonal(D_gg, 0.0)
    return D_qg, D_gg


def per_query(D, q_cls, g_cls, J_mat, mask):
    """Per-query Class P@10, ML P@10, Class AP, ML AP (same definitions as the Week 5 evaluator)."""
    S = -D.copy()
    if mask is not None:
        S[mask] = -np.inf
    order = np.argsort(-S, axis=1, kind="stable")
    r = np.arange(S.shape[0])[:, None]
    out = {}
    for tag, rel in [("class", (q_cls[:, None] == g_cls[None, :])), ("ml", J_mat >= c.cv.REL_JACCARD)]:
        rr = rel[r, order]
        prec = np.cumsum(rr, axis=1) / np.arange(1, rr.shape[1] + 1)
        out[f"{tag}_p10"] = rr[:, :10].mean(axis=1)
        out[f"{tag}_ap"] = (prec * rr).sum(axis=1) / np.maximum(rr.sum(axis=1), 1)
    return pd.DataFrame(out)


def run_systems(q_feats, g_feats, q_cls, g_cls, J_mat, mask):
    D_qg, D_gg = v3_matrices(q_feats, g_feats)
    Ds = {"V2": 1.0 - v2_scores(q_feats, g_feats), "V3": D_qg,
          "V4": c.v4_distance(D_qg, D_gg, V4["k1"], V4["k2"], V4["lambda"])}
    mets = {k: c.cv.evaluate_ranking(-D, q_cls, g_cls, J_mat, mask) for k, D in Ds.items()}
    pq = {k: per_query(D, q_cls, g_cls, J_mat, mask) for k, D in Ds.items()}
    return Ds, mets, pq


def top10_classes(D, g_cls, mask):
    S = -D.copy()
    if mask is not None:
        S[mask] = -np.inf
    order = np.argsort(-S, axis=1, kind="stable")[:, :10]
    return np.asarray(g_cls)[order]


def analyse(out, q_names, q_cls, g_cls, Ds, mets, pq, mask, seed=0):
    """All reporting (no tuning). Used by the real run and by --dry-run (with a temporary output folder)."""
    out.mkdir(parents=True, exist_ok=True)
    assert set(WEAK) <= set(q_cls) and set(WEAK) <= set(g_cls), f"weak classes {WEAK} not all present in query/gallery labels"
    overall = pd.DataFrame([{"Model": k, **{m: mets[k][m] for m in CMP}} for k in ["V2", "V3", "V4"]])
    overall.to_csv(out / "final_overall_results_v2_v3_v4.csv", index=False)
    rng = np.random.default_rng(seed)
    stat = []
    for a, b in [("V4", "V2"), ("V4", "V3")]:
        for col in ["class_p10", "ml_p10", "class_ap", "ml_ap"]:
            stat.append(paired(pq[a], pq[b], a, b, col, rng))
    stat = pd.DataFrame(stat)
    stat.to_csv(out / "paired_bootstrap.csv", index=False)

    # per-class table + sign tests (30 classes, ~20 test queries each)
    cls_rows = []
    for k in ["V2", "V3", "V4"]:
        t = pq[k].copy(); t["class"] = q_cls
        g = t.groupby("class")[["class_p10", "class_ap"]].mean()
        g.columns = [f"{k}_{x}" for x in g.columns]; cls_rows.append(g)
    pc = pd.concat(cls_rows, axis=1)
    pc.to_csv(out / "per_class_v2_v3_v4.csv")
    sign = []
    for ref in ["V2", "V3"]:
        for col in ["class_p10", "class_ap"]:
            d = pc[f"V4_{col}"] - pc[f"{ref}_{col}"]
            n_pos, n_neg = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            p = st.binomtest(n_pos, n_pos + n_neg, 0.5).pvalue if n_pos + n_neg else float("nan")
            sign.append({"comparison": f"V4 vs {ref}", "metric": col, "classes_improved": n_pos, "classes_worsened": n_neg, "sign_test_p": p})
    sign = pd.DataFrame(sign)
    sign.to_csv(out / "per_class_sign_tests.csv", index=False)

    # weak classes (reporting only): per-class rows, pooled bootstrap, top-10 confusions
    wk = np.isin(q_cls, WEAK)
    pc.loc[[w for w in WEAK if w in pc.index]].to_csv(out / "weak_classes_per_class.csv")
    wb = []
    for a, b in [("V4", "V2"), ("V4", "V3")]:
        for col in ["class_p10", "class_ap", "ml_p10"]:
            r = paired(pq[a][wk].reset_index(drop=True), pq[b][wk].reset_index(drop=True), a, b, col, rng)
            r["n_queries_pooled"] = int(wk.sum()); wb.append(r)
    wb = pd.DataFrame(wb)
    wb.to_csv(out / "weak_classes_pooled_bootstrap.csv", index=False)
    conf = []
    for name in ["V2", "V3", "V4"]:
        t10 = top10_classes(Ds[name], g_cls, mask)
        for cls in WEAK:
            idx = np.where(q_cls == cls)[0]
            if len(idx) == 0:
                continue
            vals, cnt = np.unique(t10[idx].ravel(), return_counts=True)
            share = pd.Series(cnt / cnt.sum(), index=vals).sort_values(ascending=False)
            top = share.drop(cls, errors="ignore").head(3)
            conf.append({"system": name, "query_class": cls, "n_queries": len(idx), "own_class_share": float(share.get(cls, 0.0)),
                         **{f"confusion_{i + 1}": f"{k} ({v:.3f})" for i, (k, v) in enumerate(top.items())}})
    conf = pd.DataFrame(conf)
    conf.to_csv(out / "weak_classes_top10_confusions.csv", index=False)

    # frozen V4 reliability categories (thresholds from LOFO; NOT re-tuned)
    rel = json.loads((c.HERE / "reliability_recalibration" / "reliability_config_v4.json").read_text())
    score = 1.0 - np.sort(Ds["V4"], axis=1)[:, 0]
    lab = np.where(score < rel["thresholds"]["LOW"], "LOW", np.where(score >= rel["thresholds"]["HIGH"], "HIGH", "MEDIUM"))
    r4 = pq["V4"].copy(); r4["label"] = lab
    cat = r4.groupby("label").agg(n_queries=("class_p10", "size"), class_p10=("class_p10", "mean"),
                                  ml_p10=("ml_p10", "mean"), class_ap=("class_ap", "mean")).reindex(["HIGH", "MEDIUM", "LOW"])
    cat["pct_queries"] = 100 * cat.n_queries / len(r4)
    cat["note"] = ("thresholds were set on 1,920-image LOFO galleries; the Jaccard part of D_V4 depends on gallery density, "
                   "so the HIGH/MEDIUM/LOW shares here may differ from LOFO (about 30/50/20). Reported, not re-tuned.")
    cat.to_csv(out / "reliability_categories_test.csv")

    pqo = pd.concat({k: v for k, v in pq.items()}, axis=1); pqo.index = q_names; pqo["class"] = q_cls
    pqo["v4_reliability_score"] = score; pqo["v4_reliability_label"] = lab
    pqo.to_csv(out / "per_query_v2_v3_v4.csv")
    return overall, stat, sign, wb, conf, cat


def dry_run():
    print("DRY RUN on LOFO fold 1 (train data only; writes only to a temporary folder)")
    import tempfile
    pipe = c.load_pipeline()
    f = pipe.fold_data[0]
    fd = f["fd"]
    Ds, mets, pq = run_systems(fd["q_feats"], fd["g_feats"], fd["q_cls"], fd["g_cls"], fd["J_mat"], fd["exclude_mask"])
    for name in ["V3", "V4"]:
        ref = pipe.v3_baseline_metrics[0] if name == "V3" else c.evaluate(c.v4_distance(f["D_qg"], f["D_gg"]), fd)
        for m in ["Class P@10", "ML P@10", "nDCG@10", "Class mAP", "ML mAP", "selection_score"]:
            assert abs(mets[name][m] - ref[m]) < 1e-9, (name, m, mets[name][m], ref[m])
    print("V3 and V4 from final_test code == LOFO fold-1 values (all six metrics, 1e-9)")
    tmp = Path(tempfile.mkdtemp())
    names = [f"q{i}" for i in range(len(fd["q_cls"]))]
    overall, stat, sign, wb, conf, cat = analyse(tmp, names, fd["q_cls"], fd["g_cls"], Ds, mets, pq, fd["exclude_mask"])
    print("reporting code (bootstrap, sign tests, weak-class pooled bootstrap, confusions, reliability) ran on fold-1 data:")
    print(sorted(p.name for p in tmp.iterdir()))
    print("V2 is only a smoke test here (its reproduction check needs the real test set).")
    print("dry run passed")


def paired(a, b, name_a, name_b, col, rng):
    d = (a[col] - b[col]).values
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return {"comparison": f"{name_a} - {name_b}", "metric": col, "mean_diff": d.mean(),
            "ci95_low": np.percentile(boots, 2.5), "ci95_high": np.percentile(boots, 97.5),
            "queries_better": int((d > 1e-12).sum()), "queries_worse": int((d < -1e-12).sum()),
            "queries_same": int((np.abs(d) <= 1e-12).sum()), "ci_excludes_zero": bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        return dry_run()

    if LOCK.exists():
        sys.exit(f"REFUSING TO RUN: the final test was already started/run ({LOCK}). If it crashed after the lock, redo the analysis from final_test/test_distance_matrices_v2_v3_v4.npz instead of recomputing. The test set must not be re-used for tuning.")

    # ---- 1. test features ----------------------------------------------------
    tags = {k: v.replace("train_", "test_") for k, v in c.cv.CACHE_TAGS.items()}
    missing = [t for t in tags.values() if not ((c.ROOT / "cache" / f"{t}.npy").exists() and (c.ROOT / "cache" / f"{t}_paths.json").exists())]
    if missing:
        sys.exit("CANNOT RUN: missing test-split feature caches in cache/: " + ", ".join(missing) +
                 "\nBuild them with Week1/cache_features.py (and Week3/cache_test_r2.py for lbp_r2_p16), then rerun.")
    q_feats, q_paths = {}, None
    for k, t in tags.items():
        f_, p_ = c.cv.load_cached(t)
        q_paths = q_paths or p_
        assert p_ == q_paths, f"image order differs between test caches ({t})"
        q_feats[k] = f_.astype(np.float64)
    ds = c.cv.CVDataset(folds_file=c.FOLDS_FILE)           # train features/labels (gallery)
    g_feats = ds.feats
    q_names, q_cls = zip(*[c.w1.parse_path(p) for p in q_paths])
    q_names, q_cls = list(q_names), np.array(q_cls)
    assert len(q_names) == 600 and len(ds.names) == 2400, (len(q_names), len(ds.names))
    Lall = c.cv.load_labels(q_names + ds.names)
    J_mat = c.cv.compute_jaccard_matrix(Lall[:600], Lall[600:])
    g_cls = ds.classes

    t0 = time.time()
    Ds, mets, pq = run_systems(q_feats, g_feats, q_cls, g_cls, J_mat, None)

    # ---- 2. V2 reproduction (must pass before anything else is reported) ------
    OUT.mkdir(exist_ok=True)
    old = pd.read_csv(c.ROOT / "Week4" / "final_test" / "final_overall_results.csv")
    old_v2 = old[old.iloc[:, 0].astype(str).str.startswith("V2")].iloc[0]
    rep = [{"metric": m, "week4_recorded": float(old_v2[m]), "reproduced": mets["V2"][m],
            "match_1e-4": abs(float(old_v2[m]) - mets["V2"][m]) <= 1e-4} for m in old.columns[1:] if m in mets["V2"]]
    old_pq = pd.read_csv(c.ROOT / "Week4" / "final_test" / "v2_per_query_results.csv").set_index("query_image")
    new_pq = pq["V2"].copy(); new_pq.index = q_names
    common = old_pq.index.intersection(new_pq.index)
    if len(common) != len(q_names):
        sys.exit(f"STOP: only {len(common)} of {len(q_names)} test query names match Week4/final_test/v2_per_query_results.csv "
                 "(query naming differs). This is a naming problem, not a V2 mismatch. No lock written, nothing reported.")
    mad = float(np.abs(old_pq.loc[common, "Class_P10"] - new_pq.loc[common, "class_p10"]).max())
    rep.append({"metric": "per-query Class P@10 (max abs diff)", "week4_recorded": 0.0, "reproduced": mad, "match_1e-4": bool(mad <= 1e-4)})
    pd.DataFrame(rep).to_csv(OUT / "v2_reproduction_check.csv", index=False)
    if not all(r["match_1e-4"] for r in rep):
        sys.exit("V2 DOES NOT REPRODUCE the Week 4 result - stop and investigate (see final_test/v2_reproduction_check.csv). "
                 "No lock written and nothing else reported.")
    print("V2 reproduces the Week 4 test result")

    # lock NOW: from here on the V3/V4 test numbers are being looked at
    # (distance matrices are saved first-thing so the analysis can be redone from disk without touching the test set again)
    LOCK.write_text(json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "started", "v4": V4,
                                "note": "final test run once; no tuning allowed after this point"}, indent=2))
    np.savez_compressed(OUT / "test_distance_matrices_v2_v3_v4.npz", D_V2=Ds["V2"], D_V3=Ds["V3"], D_V4=Ds["V4"],
                        q_names=np.array(q_names), q_cls=q_cls, g_cls=np.asarray(g_cls))

    # ---- 3-5. results, statistics, weak classes, reliability ------------------
    overall, stat, sign, wb, conf, cat = analyse(OUT, q_names, q_cls, g_cls, Ds, mets, pq, None)
    LOCK.write_text(json.dumps({"started_at": json.loads(LOCK.read_text())["started_at"], "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "status": "completed", "v4": V4, "seconds": time.time() - t0,
                                "note": "final test run once; no tuning allowed after this point"}, indent=2))
    pd.set_option("display.width", 220)
    for t in (overall, stat, sign, wb, conf, cat):
        print("\n", t.round(4).to_string(index=not isinstance(t.index, pd.RangeIndex)))


if __name__ == "__main__":
    main()
