"""
evaluate_cbir.py

Evaluates several feature combinations on ALL test queries using the
cached features from cache_features.py and the multi-label annotations.

Run (after cache_features.py has been run once):
    python evaluate_cbir.py

Outputs:
    evaluation_results.csv        overall table (one row per method)
    per_class_precision.csv       class Precision@5 per scene class
"""

import numpy as np
import pandas as pd

from cache_features import load_cached, HSV_TAG, LBP_TAG, HOG_TAG

# ============================================================
# SETTINGS
# ============================================================

LABEL_CSV = "multilabel .csv"
KS = (5, 10)                 # cut-offs for Precision@K, nDCG@K
REL_JACCARD = 0.8            # multi-label "relevant" if Jaccard >= this
                             # (chance level: 0.5 -> 50%, 0.7 -> 25%, 0.8 -> 14%)
OUT_CSV = "evaluation_results.csv"
PER_CLASS_CSV = "per_class_precision.csv"

# (weights, use z-score normalisation per query before fusing)
METHODS = {
    "HSV":                              ({"hsv": 1.0}, False),
    "LBP":                              ({"lbp": 1.0}, False),
    "HOG":                              ({"hog": 1.0}, False),
    "HSV+LBP (0.7/0.3)":                ({"hsv": 0.7, "lbp": 0.3}, False),
    "HSV+LBP+HOG (0.5/0.25/0.25)":      ({"hsv": 0.5, "lbp": 0.25, "hog": 0.25}, False),
    "HSV+LBP (0.7/0.3) z-norm":         ({"hsv": 0.7, "lbp": 0.3}, True),
    "HSV+LBP+HOG equal z-norm":         ({"hsv": 1/3, "lbp": 1/3, "hog": 1/3}, True),
    "HSV+LBP+HOG (0.5/0.25/0.25) z-norm": ({"hsv": 0.5, "lbp": 0.25, "hog": 0.25}, True),
}


# ============================================================
# 1. LOAD CACHED FEATURES + LABELS
# ============================================================

def parse_path(p):
    """'images_tr\\Airport\\airport_1.jpg' -> ('airport_1', 'airport')"""
    parts = p.replace("\\", "/").split("/")
    return parts[-1].rsplit(".", 1)[0], parts[-2].lower()


def load_split(split):
    feats, paths = {}, None
    for key, tag in (("hsv", HSV_TAG), ("lbp", LBP_TAG), ("hog", HOG_TAG)):
        f, p = load_cached(f"{split}_{tag}")
        if paths is None:
            paths = p
        assert p == paths, f"{split}: image order differs between caches"
        feats[key] = f.astype(np.float64)
    names, classes = zip(*[parse_path(p) for p in paths])
    return feats, list(names), np.array(classes)


def load_labels(names):
    df = pd.read_csv(LABEL_CSV, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "name"}).set_index("name")
    df.index = df.index.str.lower()
    missing = [n for n in names if n.lower() not in df.index]
    assert not missing, f"No labels for: {missing[:5]}"
    return df.loc[[n.lower() for n in names]].values.astype(np.float64)


# ============================================================
# 2. SIMILARITY MATRICES (query x gallery)
# ============================================================

def corr_sim(Q, G):
    """Pearson correlation = cv2.HISTCMP_CORREL, for all pairs at once."""
    Q = Q - Q.mean(1, keepdims=True)
    G = G - G.mean(1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    return Q @ G.T


def cos_sim(Q, G):
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    return Q @ G.T


def zscore_rows(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)


# ============================================================
# 3. METRICS
# ============================================================

def average_precision(rel_sorted):
    """rel_sorted: (n_queries, n_gallery) bool, in ranked order."""
    r = rel_sorted.astype(np.float64)
    cum = np.cumsum(r, axis=1)
    prec = cum / np.arange(1, r.shape[1] + 1)
    n_rel = r.sum(1)
    ok = n_rel > 0
    ap = np.full(r.shape[0], np.nan)
    ap[ok] = (prec[ok] * r[ok]).sum(1) / n_rel[ok]
    return np.nanmean(ap)


def evaluate(S, q_cls, g_cls, J):
    n = S.shape[0]
    order = np.argsort(-S, axis=1, kind="stable")
    rows = np.arange(n)[:, None]

    same_class = (q_cls[:, None] == g_cls[None, :])
    rel_ml = J >= REL_JACCARD

    rc = same_class[rows, order]
    rm = rel_ml[rows, order]
    jg = J[rows, order]

    out = {}
    for K in KS:
        out[f"Class P@{K}"] = rc[:, :K].mean()
        out[f"ML P@{K}"] = rm[:, :K].mean()
        out[f"Jaccard@{K}"] = jg[:, :K].mean()

        disc = 1.0 / np.log2(np.arange(2, K + 2))
        dcg = (jg[:, :K] * disc).sum(1)
        ideal = -np.sort(-J, axis=1)[:, :K]
        idcg = (ideal * disc).sum(1)
        ok = idcg > 0
        out[f"nDCG@{K}"] = (dcg[ok] / idcg[ok]).mean()

    out["Class mAP"] = average_precision(rc)
    out["ML mAP"] = average_precision(rm)

    # per-class P@5 (for the CSV)
    p5 = rc[:, :5].mean(1)
    per_class = pd.Series(p5).groupby(q_cls).mean()
    return out, per_class


# ============================================================
# 4. MAIN
# ============================================================

def main():
    tr_feats, tr_names, tr_cls = load_split("train")
    te_feats, te_names, te_cls = load_split("test")
    print(f"Train: {len(tr_names)}   Test: {len(te_names)}")

    Lq = load_labels(te_names)
    Lg = load_labels(tr_names)

    # Jaccard overlap between every query and every gallery label set
    inter = Lq @ Lg.T
    union = Lq.sum(1)[:, None] + Lg.sum(1)[None, :] - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    print(f"\nPairs sharing >= 1 label:      {(inter > 0).mean():.1%}")
    print(f"Pairs with Jaccard >= {REL_JACCARD}:      {(J >= REL_JACCARD).mean():.1%}")
    print(f"Pairs from the same scene class: {(te_cls[:, None] == tr_cls[None, :]).mean():.1%}")
    print("(these are the chance levels for 'relevant' under each definition)\n")

    # component similarity matrices
    sims = {
        "hsv": corr_sim(te_feats["hsv"], tr_feats["hsv"]),
        "lbp": corr_sim(te_feats["lbp"], tr_feats["lbp"]),
        "hog": cos_sim(te_feats["hog"], tr_feats["hog"]),
    }
    sims_z = {k: zscore_rows(v) for k, v in sims.items()}

    results, per_class = {}, {}

    # random reference
    rng = np.random.default_rng(0)
    results["Random (reference)"], per_class["Random (reference)"] = evaluate(
        rng.random(sims["hsv"].shape), te_cls, tr_cls, J
    )

    for name, (weights, znorm) in METHODS.items():
        src = sims_z if znorm else sims
        S = sum(w * src[k] for k, w in weights.items())
        results[name], per_class[name] = evaluate(S, te_cls, tr_cls, J)

    table = pd.DataFrame(results).T
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.round(3).to_string())

    table.round(4).to_csv(OUT_CSV)
    pd.DataFrame(per_class).round(3).to_csv(PER_CLASS_CSV)
    print(f"\nSaved: {OUT_CSV}, {PER_CLASS_CSV}")


if __name__ == "__main__":
    main()