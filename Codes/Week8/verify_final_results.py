"""
verify_final_results.py - Week 8: Comprehensive consistency, integrity, and manifest verifier.

Checks:
  1. Required files exist and are non-empty.
  2. Reproduction checks against recorded values from earlier weeks.
  3. Frozen configuration consistency across config files.
  4. Cross-checks between summary, per-fold, weak-class, and reliability tables.
  5. Failed experiments table validation (removes ineffective pass loop).
  6. ML mAP aggregation verification (reconciles headline mAP on 588 queries vs 600 per-query rows).
  7. README markdown table vs CSV consistency checks.
  8. Final-test execution record validation & run-once lock verification.
  9. SHA-256 manifest verification (POSIX path normalization; does not overwrite by default).

Usage:
  python "Week 8/verify_final_results.py"                  # runs all checks and verifies manifest
  python "Week 8/verify_final_results.py" --check-only    # alias for default verification
  python "Week 8/verify_final_results.py" --update-manifest # updates MANIFEST.sha256 after intentional changes

Exit code 1 if any check fails.
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FR = Path(os.environ.get("W8_OUT", HERE / "final_results"))
W7 = ROOT / "Week 7"
fails, n_ok = [], 0


def check(cond, msg):
    global n_ok
    if cond:
        n_ok += 1
    else:
        fails.append(msg)
        print("FAIL:", msg)


def close(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def parse_markdown_tables(md_text):
    """Extracts markdown tables from text as a dict of title/preceding-header -> DataFrame."""
    lines = md_text.splitlines()
    tables = {}
    current_header = "Intro"
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#"):
            current_header = line.lstrip("#").strip()
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|(\s*:?-+:?\s*\|)+$", lines[i + 1].strip()):
            headers = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip("|").split("|")]
                if len(row) == len(headers):
                    rows.append(row)
                i += 1
            if rows:
                df = pd.DataFrame(rows, columns=headers)
                t_key = current_header
                suffix = 1
                while t_key in tables:
                    suffix += 1
                    t_key = f"{current_header}_{suffix}"
                tables[t_key] = df
            continue
        i += 1
    return tables


def main():
    update_manifest = "--update-manifest" in sys.argv or "--write-manifest" in sys.argv
    check_only = "--check-only" in sys.argv or not update_manifest

    # 1. Required files exist and are non-empty
    req = [
        "config/final_config.json", "config/v2_config.json", "config/v3_config.json",
        "config/week6_selected_parameters.json", "config/reliability_config_v4.json",
        "lofo_comparison/v2_v3_v4_per_fold.csv", "lofo_comparison/v2_v3_v4_summary.csv",
        "lofo_comparison/paired_fold_deltas.csv", "lofo_comparison/paired_bootstrap_lofo_queries.csv",
        "lofo_comparison/reproduction_checks.csv", "weak_classes/weak_classes_v2_v3_v4_lofo.csv",
        "weak_classes/per_class_v3_vs_v4.csv", "reliability/reliability_categories_lofo_heldout.csv",
        "k2_check/k2_summary.csv", "failed_experiments/failed_experiments.csv",
        "figures/fig_lofo_v2_v3_v4.png", "figures/fig_weak_classes.png",
        "figures/fig_classwise_p10_change.png", "figures/fig_reliability.png",
        "figures/fig_k2_check.png", "figures/fig_failed_experiments.png",
        "MANIFEST.sha256", "README.md"
    ]
    for r in req:
        p = FR / r
        check(p.exists() and p.stat().st_size > 0, f"missing or empty: {r}")
    if fails:
        print(f"Aborting: {len(fails)} required files missing.")
        sys.exit(1)

    cfg = json.loads((FR / "config/final_config.json").read_text())
    summ = pd.read_csv(FR / "lofo_comparison/v2_v3_v4_summary.csv").set_index("system")
    pf = pd.read_csv(FR / "lofo_comparison/v2_v3_v4_per_fold.csv")

    # 2. Recorded values from earlier weeks
    w6_path = ROOT / "Week 6/selected_parameters.json"
    if not w6_path.exists():
        w6_path = FR / "config/week6_selected_parameters.json"
    check(w6_path.exists(), "Week 6 selected_parameters.json missing (also checked bundled config copy)")
    w6 = json.loads(w6_path.read_text())
    ref = {
        "V3": {"Class P@10": 0.5069, "Class mAP": 0.2547, "ML P@10": 0.4556, "ML mAP": 0.2738, "nDCG@10": 0.7286, "selection_score": 0.4439},
        "V4": {"Class P@10": 0.5361, "Class mAP": 0.3206, "ML P@10": 0.4626, "ML mAP": 0.2938, "nDCG@10": 0.7298, "selection_score": 0.4686},
        "V2": {"Class P@10": 0.4336, "selection_score": 0.3970}
    }
    for s, d in ref.items():
        for m, v in d.items():
            check(close(summ.loc[s, f"{m}_mean"], v), f"{s} {m} mean {summ.loc[s, f'{m}_mean']:.4f} != recorded {v}")
    check(len(pf) == 15 and pf.groupby("system").fold.nunique().eq(5).all(), "per-fold table must have 5 folds x 3 systems")
    check(pd.read_csv(FR / "lofo_comparison/reproduction_checks.csv")["match"].all(), "a reproduction check is False")

    # 3. Frozen configuration agrees with the sources
    k = cfg["V4_k_reciprocal"]
    sel = w6["selected_configuration"]
    check((k["k1"], k["k2"], k["lambda"]) == (sel["k1"], sel["k2"], sel["lambda"]), "final_config V4 parameters differ from Week 6 selection")
    check(abs(sum(cfg["V3_weights"].values()) - 1.0) < 1e-9, "V3 weights do not sum to 1")
    v3c = json.loads((FR / "config/v3_config.json").read_text())["weights"]
    check(all(abs(v3c[x] - cfg["V3_weights"][x]) < 1e-12 for x in v3c), "V3 weights differ from Week 5 v3_config.json")
    rel = json.loads((FR / "config/reliability_config_v4.json").read_text())
    check(cfg["reliability"]["thresholds"] == rel["thresholds"] and rel["test_set_used"] is False, "reliability config mismatch or test set used")
    d = json.loads((FR / "k2_check/k2_decision.json").read_text())
    check(d["frozen_k2"] == 3 and d["test_set_used"] is False and k["k2"] == 3, "k2 decision inconsistent with frozen k2=3")

    # 4. Cross-checks between tables
    k2s = pd.read_csv(FR / "k2_check/k2_summary.csv").set_index("system")
    check(close(k2s.loc["V4 k2=3", "selection_score_mean"], summ.loc["V4", "selection_score_mean"], 1e-9), "k2=3 row differs from V4 row")
    wk = pd.read_csv(FR / "weak_classes/weak_classes_v2_v3_v4_lofo.csv").set_index("class")
    pc = pd.read_csv(FR / "weak_classes/per_class_v3_vs_v4.csv").set_index("class")
    for cls in ["airport", "bridge", "square"]:
        check(close(wk.loc[cls, "V3_class_p10"], pc.loc[cls, "p10_v3"], 1e-9) and close(wk.loc[cls, "V4_class_p10"], pc.loc[cls, "p10_v4"], 1e-9),
              f"weak-class table disagrees with Week 7 analysis for {cls}")
    bs = pd.read_csv(FR / "lofo_comparison/paired_bootstrap_lofo_queries.csv")
    check((bs.ci95_low <= bs.mean_diff).all() and (bs.mean_diff <= bs.ci95_high).all(), "bootstrap mean outside its own CI")
    pr = pd.read_csv(FR / "reliability/reliability_categories_lofo_heldout.csv").set_index("label")
    check(pr.loc["HIGH", "class_p10"] > pr.loc["MEDIUM", "class_p10"] > pr.loc["LOW", "class_p10"], "reliability categories are not ordered HIGH>MEDIUM>LOW")

    # 5. Failed experiments table validation (removes ineffective pass loop)
    fx = pd.read_csv(FR / "failed_experiments/failed_experiments.csv")
    check(len(fx) >= 7, f"failed_experiments.csv should contain at least 7 experiments, got {len(fx)}")
    req_cols = ["experiment", "week", "reference", "selection_score", "delta_vs_reference", "outcome", "reason"]
    check(all(c in fx.columns for c in req_cols), "failed_experiments.csv missing required columns")
    check(not fx["delta_vs_reference"].isna().any(), "NaN found in failed_experiments delta_vs_reference")
    check(not fx["reason"].isna().any() and (fx["reason"].str.len() > 10).all(), "empty or trivial reason in failed_experiments")
    check({"Average Query Expansion (AQE) on V3", "Neighbourhood expansion (NE) on V4"} <= set(fx.experiment), "AQE and NE must be documented")
    ne = fx[fx.experiment.str.startswith("Neighbourhood")].iloc[0]
    check(ne.delta_vs_reference < 0, "NE should show a negative selection-score change (it was rejected)")

    # 6. ML mAP aggregation verification
    t = FR / "final_test"
    done = (t / "FINAL_TEST_DONE.json").exists()
    check(done == cfg["final_test_status"].startswith("completed"), "final_test_status in config does not match the final_test folder")
    check((t / "PENDING.md").exists() != done, "final_test folder must hold either PENDING.md or real results, not both")
    
    if done:
        pq = pd.read_csv(t / "per_query_v2_v3_v4.csv", header=[0, 1], index_col=0)
        ov = pd.read_csv(t / "final_overall_results_v2_v3_v4.csv").set_index("Model")
        
        # Verify the 12 queries with zero ML AP (Jaccard < 0.8 with all gallery items)
        for m in ["V2", "V3", "V4"]:
            ml_ap = pq[(m, "ml_ap")].astype(float)
            zero_count = int((ml_ap == 0.0).sum())
            check(zero_count == 12, f"{m} per-query ml_ap zero count is {zero_count}, expected 12")
            
            # Verify headline ML mAP matches sum / 588
            headline_mAP = ov.loc[m, "ML mAP"]
            calc_headline = ml_ap.sum() / 588.0
            check(abs(calc_headline - headline_mAP) < 1e-6, f"{m} headline ML mAP ({headline_mAP:.6f}) does not match sum/588 ({calc_headline:.6f})")
            
            # Verify per-query average (all 600) matches headline * (588/600)
            mean_600 = ml_ap.mean()
            check(abs(mean_600 - headline_mAP * (588.0 / 600.0)) < 1e-6, f"{m} mean_600 does not equal headline * 588/600")

    # 7. Final-test execution record validation & run-once protection
    if done:
        tj = json.loads((t / "FINAL_TEST_DONE.json").read_text())
        check(tj.get("status") == "completed", "FINAL_TEST_DONE.json status is not completed")
        check(bool(tj.get("started_at")) and bool(tj.get("finished_at")), "FINAL_TEST_DONE.json missing timestamps")
        check(tj.get("seconds", 0) > 0, "FINAL_TEST_DONE.json recorded runtime seconds <= 0")
        check(tj.get("v4") == {"k1": 30, "k2": 3, "lambda": 0.5}, "FINAL_TEST_DONE.json recorded V4 parameters mismatch")
        check("run once" in tj.get("note", "").lower(), "FINAL_TEST_DONE.json note must confirm run-once lock")
        check((t / "v2_reproduction_check.csv").exists() and pd.read_csv(t / "v2_reproduction_check.csv")["match_1e-4"].all(), "V2 reproduction check missing or failed")

    # 8. README table vs CSV consistency checks
    readme_text = (FR / "README.md").read_text()
    md_tables = parse_markdown_tables(readme_text)
    
    # Check Section 1 table against summary
    t1_key = [k for k in md_tables if "V2 vs V3 vs V4 (LOFO" in k]
    if t1_key:
        t1_df = md_tables[t1_key[0]].set_index("System")
        for sysname in ["V2", "V3", "V4"]:
            for m in ["Class P@10", "Class mAP", "ML P@10", "ML mAP", "nDCG@10", "selection_score"]:
                expected_str = f"{summ.loc[sysname, f'{m}_mean']:.4f} +/- {summ.loc[sysname, f'{m}_std']:.4f}"
                obs_str = t1_df.loc[sysname, m]
                check(obs_str == expected_str, f"README table 1 mismatch for {sysname} {m}: '{obs_str}' != '{expected_str}'")
                
    # Check Section 6 Final Test table against overall CSV
    if done:
        t6_key = [k for k in md_tables if "FINAL TEST" in k]
        if t6_key:
            t6_df = md_tables[t6_key[0]].set_index("Model")
            for sysname in ["V2", "V3", "V4"]:
                for m in ["Class P@10", "Class mAP", "ML P@10", "ML mAP", "selection_score"]:
                    expected_val = ov.loc[sysname, m]
                    obs_val = float(t6_df.loc[sysname, m])
                    check(close(obs_val, expected_val, 1e-4), f"README final test table mismatch for {sysname} {m}: {obs_val} != {expected_val}")

    print(f"\n{n_ok} checks passed, {len(fails)} failed")
    if fails:
        sys.exit(1)

    # 9. Manifest verification (POSIX path normalization; does not overwrite by default)
    man = FR / "MANIFEST.sha256"
    files = sorted(p for p in FR.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    hashes = {p.relative_to(FR).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    
    if check_only:
        old = dict(line.split("  ", 1)[::-1] for line in man.read_text().splitlines() if line.strip())
        bad = [f for f in hashes if old.get(f) != hashes[f]]
        missing = [f for f in old if f not in hashes]
        extra = [f for f in hashes if f not in old]
        if bad or missing or extra:
            print(f"FAIL manifest: MISMATCH in {bad}, missing {missing}, extra {extra}")
            sys.exit(1)
        print(f"manifest: all {len(hashes)} files match {man.name} exactly.")
        sys.exit(0)
    else:
        man.write_text("\n".join(f"{h}  {f}" for f, h in sorted(hashes.items(), key=lambda x: x[0])) + "\n")
        print(f"wrote {man.name} ({len(hashes)} files with POSIX paths)")


if __name__ == "__main__":
    main()
