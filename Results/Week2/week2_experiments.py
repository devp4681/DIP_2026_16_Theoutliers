"""
week2_experiments.py

Week 2: pick the final retrieval configuration using a VALIDATION set only.

    python week2_experiments.py                # tuning on validation (never touches test)
    python week2_experiments.py --final-test   # run ONCE at the end, after best_weights.json exists

Protocol
    2400 train  ->  1920 reference + 480 validation   (stratified by scene class, fixed seed)
    validation queries are searched against the 1920 reference images only.
    final test: 600 test queries vs all 2400 train images (same setting as Week 1).

Outputs
    split_indices.json        the split (reused on every run so results are reproducible)
    validation_results.csv    every configuration tried, all metrics
    feature_comparison.csv    compact table for the report
    best_weights.json         the selected configuration  (-> RetrievalEngineV1)
    test_results.csv          only with --final-test
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import retrieval_engine_v1 as eng
from week1_evaluate import evaluate, load_labels

SEED, VAL_FRAC = 42, 0.20
SPLIT_FILE = "split_indices.json"
SELECTION_METRICS = ["Class P@10", "ML P@10", "nDCG@10", "Class mAP", "ML mAP"]
TIE_TOL = 0.002          # configs within this of the best score count as "tied"; the simplest wins
WEEK1_BEST = ({"hsv": 0.5, "lbp": 0.25, "hog": 0.25}, False)


# ------------------------------------------------------------------ helpers
def make_split(classes, names):
    """Stratified 80/20 split of the training images (by scene class)."""
    if Path(SPLIT_FILE).exists():
        d = json.loads(Path(SPLIT_FILE).read_text())
        if d["names"] == names:
            return np.array(d["ref_idx"]), np.array(d["val_idx"])
        print("[split] image list changed -> creating a new split")
    rng = np.random.default_rng(SEED)
    ref, val = [], []
    for c in np.unique(classes):
        idx = rng.permutation(np.where(classes == c)[0])
        n_val = int(round(VAL_FRAC * len(idx)))
        val += idx[:n_val].tolist()
        ref += idx[n_val:].tolist()
    ref, val = np.sort(ref), np.sort(val)
    Path(SPLIT_FILE).write_text(json.dumps(
        {"seed": SEED, "names": names, "ref_idx": ref.tolist(), "val_idx": val.tolist()}))
    return ref, val


def jaccard_matrix(Lq, Lg):
    inter = Lq @ Lg.T
    union = Lq.sum(1)[:, None] + Lg.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def simplex(n, step):
    k = int(round(1 / step))
    for c in itertools.product(range(k + 1), repeat=n):
        if sum(c) == k:
            yield tuple(x / k for x in c)


def score_of(m):
    return float(np.mean([m[k] for k in SELECTION_METRICS]))


class Runner:
    """Evaluates fused configurations against one fixed set of queries/gallery."""

    def __init__(self, sims, q_cls, g_cls, J):
        self.sims = sims
        self.sims_z = {c: eng.zscore_rows(s) for c, s in sims.items()}
        self.q_cls, self.g_cls, self.J = q_cls, g_cls, J
        self.rows = []

    def run(self, name, family, weights, znorm):
        weights = {c: w for c, w in weights.items() if w > 0}
        src = self.sims_z if znorm else self.sims
        S = sum(w * src[c] for c, w in weights.items())
        m, _ = evaluate(S, self.q_cls, self.g_cls, self.J)
        row = {"config": name, "family": family, "znorm": znorm,
               **{f"w_{c}": weights.get(c, 0.0) for c in eng.COMPONENTS},
               **m, "selection_score": score_of(m)}
        self.rows.append(row)
        return row


def label(weights):
    return "+".join(f"{c.upper()}{w:.1f}" for c, w in weights.items() if w > 0)


# --------------------------------------------------------------- validation
def tune(step):
    tr_f, tr_names, tr_cls, _ = eng.load_split_features("train")
    ref, val = make_split(tr_cls, tr_names)
    print(f"Reference: {len(ref)}   Validation: {len(val)}   (test is NOT loaded)")

    Lg_all = load_labels(tr_names)
    J = jaccard_matrix(Lg_all[val], Lg_all[ref])
    sims = eng.component_sims(eng.subset(tr_f, val), eng.subset(tr_f, ref))
    R = Runner(sims, tr_cls[val], tr_cls[ref], J)

    # 1. Week 1 baseline, reproduced on validation ---------------------------
    base = {
        "HSV": {"hsv": 1}, "LBP": {"lbp": 1}, "HOG": {"hog": 1},
        "HSV+LBP (0.7/0.3)": {"hsv": .7, "lbp": .3},
        "HSV+HOG (0.5/0.5)": {"hsv": .5, "hog": .5},
        "LBP+HOG (0.5/0.5)": {"lbp": .5, "hog": .5},
        "HSV+LBP+HOG (0.5/0.25/0.25)": {"hsv": .5, "lbp": .25, "hog": .25},
        "HSV+LBP+HOG (1/3 each)": {"hsv": 1/3, "lbp": 1/3, "hog": 1/3},
    }
    for n, w in base.items():
        R.run(n, "baseline", w, False)
        if len(w) > 1:
            R.run(n + " z-norm", "baseline", w, True)

    # 2. Spatial components on their own ------------------------------------
    for n, w in {"Spatial HSV": {"shsv": 1}, "Spatial LBP": {"slbp": 1},
                 "Spatial HSV+LBP (0.7/0.3)": {"shsv": .7, "slbp": .3}}.items():
        R.run(n, "spatial-single", w, False)

    # 3. Global-only weight grid (HSV, LBP, HOG) -----------------------------
    for z in (False, True):
        for w in simplex(3, step):
            R.run("grid-global " + label(dict(zip(("hsv", "lbp", "hog"), w))),
                  "grid-global", dict(zip(("hsv", "lbp", "hog"), w)), z)

    # 4. Full grid: global + spatial -----------------------------------------
    for z in (False, True):
        for w in simplex(5, step):
            wd = dict(zip(eng.COMPONENTS, w))
            R.run("grid-full " + label(wd), "grid-full", wd, z)

    df = pd.DataFrame(R.rows)
    df.round(4).to_csv("validation_results.csv", index=False)

    # 5. Pick the final configuration ----------------------------------------
    best = df["selection_score"].max()
    tied = df[df["selection_score"] >= best - TIE_TOL].copy()
    tied["n_comp"] = (tied[[f"w_{c}" for c in eng.COMPONENTS]] > 0).sum(axis=1)
    final = tied.sort_values(["n_comp", "znorm", "selection_score"],
                             ascending=[True, True, False]).iloc[0]
    weights = {c: float(final[f"w_{c}"]) for c in eng.COMPONENTS if final[f"w_{c}"] > 0}

    cfg = {
        "name": "Final Retrieval Engine v1", "weights": weights, "znorm": bool(final["znorm"]),
        "selection_metrics": SELECTION_METRICS, "tie_tolerance": TIE_TOL,
        "validation_selection_score": round(float(final["selection_score"]), 4),
        "validation_metrics": {k: round(float(final[k]), 4) for k in
                               final.index if k.startswith(("Class", "ML", "Jaccard", "nDCG"))},
        "feature_tags": {c: eng.TAGS[c] for c in weights},
        "split": {"seed": SEED, "val_fraction": VAL_FRAC},
    }
    Path(eng.CONFIG_FILE).write_text(json.dumps(cfg, indent=2))

    # 6. Compact comparison table -------------------------------------------
    pick = lambda fam: df[df.family == fam].sort_values("selection_score", ascending=False).iloc[0]
    best_global = pick("grid-global").copy(); best_global["config"] = "Best weighted fusion (global only)"
    best_full = pick("grid-full").copy();     best_full["config"] = "Best fusion incl. spatial"
    final = final.copy();                      final["config"] = "SELECTED: Engine v1 (" + label(weights) + (", z-norm)" if cfg["znorm"] else ")")
    keep = df[df.family.isin(["baseline", "spatial-single"])]
    comp = pd.concat([keep, best_global.to_frame().T, best_full.to_frame().T, final.to_frame().T])
    cols = ["config", "znorm"] + [f"w_{c}" for c in eng.COMPONENTS] + \
           ["Class P@5", "Class P@10", "Class mAP", "ML P@5", "ML P@10", "ML mAP",
            "Jaccard@5", "Jaccard@10", "nDCG@5", "nDCG@10", "selection_score"]
    comp = comp[cols]
    comp[cols[2:]] = comp[cols[2:]].astype(float)
    comp.round(4).to_csv("feature_comparison.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    show = ["config", "Class P@5", "Class P@10", "Class mAP", "ML P@10", "Jaccard@5", "nDCG@10", "selection_score"]
    print("\n=== VALIDATION: feature comparison ===")
    print(comp[show].round(3).to_string(index=False))
    print("\nSelected configuration:", json.dumps({"weights": weights, "znorm": cfg["znorm"]}))
    print("Saved: validation_results.csv, feature_comparison.csv, best_weights.json")

    # global vs spatial answer in one line
    g, f = best_global["selection_score"], best_full["selection_score"]
    print(f"\nGlobal-only best score: {g:.4f}   with spatial: {f:.4f}   gain: {f - g:+.4f}")


# --------------------------------------------------------------- final test
def final_test():
    cfg = json.loads(Path(eng.CONFIG_FILE).read_text())
    tr_f, tr_names, tr_cls, _ = eng.load_split_features("train")
    te_f, te_names, te_cls, _ = eng.load_split_features("test")
    J = jaccard_matrix(load_labels(te_names), load_labels(tr_names))
    R = Runner(eng.component_sims(te_f, tr_f), te_cls, tr_cls, J)
    R.run("Week 1 best (HSV+LBP+HOG 0.5/0.25/0.25)", "test", *WEEK1_BEST)
    R.run("Engine v1", "test", cfg["weights"], cfg["znorm"])
    df = pd.DataFrame(R.rows)
    df.round(4).to_csv("test_results.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n=== FINAL TEST (600 queries vs 2400 train images) ===")
    print(df.drop(columns=["family"]).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="evaluate selected config on the 600 test images (run once)")
    ap.add_argument("--step", type=float, default=0.1, help="weight-grid step (0.1 fine, 0.2 faster)")
    a = ap.parse_args()
    final_test() if a.final_test else tune(a.step)
