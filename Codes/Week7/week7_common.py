"""
week7_common.py - shared helpers for the Week 7 scripts (no new method here).

Everything is imported from the existing Week 1-6 code. Nothing is retrained and no fold changes.
Run all scripts from anywhere; paths are resolved from this file. Needs multilabel_.csv in the project root.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in ["Week1", "Week2", "Week 5", "Week 6"]:
    sys.path.insert(0, str(ROOT / p))

import cv_experiment as cv                    # noqa: E402  Week 5 evaluator (unchanged)
import week1_evaluate as w1                   # noqa: E402
try:                                          # the Week 6 file is spelled 'neighborhood_context.py' here
    import neighborhood_context as nc         # noqa: E402  Week 6: V3 matrices + gallery-only KR
except ImportError:                           # ... and 'neighbourhood_context.py' in some working copies
    import neighbourhood_context as nc        # noqa: E402

FOLDS_FILE = ROOT / "Week2" / "folds_5.json"
LABEL_CSV = ROOT / "multilabel_.csv"

# metrics and Week 6 reference values (selected_parameters.json / neighborhood_summary.md)
METRICS = ["Class P@10", "ML P@10", "nDCG@10", "Class mAP", "ML mAP", "selection_score"]
W6_REF = {
    "V3": {"Class P@10": 0.5069, "ML P@10": 0.4556, "nDCG@10": 0.7286, "Class mAP": 0.2547, "ML mAP": 0.2738, "selection_score": 0.4439},
    "V4": {"Class P@10": 0.5361, "ML P@10": 0.4626, "nDCG@10": 0.7298, "Class mAP": 0.3206, "ML mAP": 0.2938, "selection_score": 0.4686},
}

# Week 6 adoption rule (unchanged): gain > 0.005 selection score, >= 4/5 folds positive, Class P@10 not decreased
RULE_MIN_DELTA = 0.005
RULE_MIN_POSITIVE_FOLDS = 4


def load_pipeline():
    """Week 6 pipeline object holding V3 distance matrices (D_qg, D_gg) and eval data for the 5 LOFO folds."""
    if not LABEL_CSV.exists():
        raise FileNotFoundError(f"{LABEL_CSV} not found: the multilabel metrics need it.")
    ds = cv.CVDataset(folds_file=FOLDS_FILE)
    pipe = nc.Week6Pipeline.__new__(nc.Week6Pipeline)
    pipe.dataset = ds
    pipe.folds_file = FOLDS_FILE
    pipe.cache_dir = ROOT / "cache"
    pipe.precompute_v3_matrices()
    return pipe


def evaluate(D, fd):
    """Unchanged Week 5 evaluator on a distance matrix D (smaller = more similar)."""
    return cv.evaluate_ranking(-D, fd["q_cls"], fd["g_cls"], fd["J_mat"], fd["exclude_mask"])


def v4_distance(D_qg, D_gg, k1=30, k2=3, lam=0.5):
    J = nc.compute_gallery_only_k_reciprocal_jaccard(D_qg, D_gg, k1=k1, k2=k2)
    return lam * J + (1 - lam) * D_qg


def per_query_metrics(D, fd):
    """Per-query Class P@10 / Class AP and the ranking order (post-hoc analysis; same definitions as the evaluator)."""
    S = -D.copy()
    if fd["exclude_mask"] is not None:
        S[fd["exclude_mask"]] = -np.inf
    order = np.argsort(-S, axis=1, kind="stable")
    rows = np.arange(S.shape[0])[:, None]
    rc = (fd["q_cls"][:, None] == fd["g_cls"][None, :])[rows, order]
    p10 = rc[:, :10].mean(axis=1)
    prec = np.cumsum(rc, axis=1) / np.arange(1, rc.shape[1] + 1)
    ap = (prec * rc).sum(axis=1) / np.maximum(rc.sum(axis=1), 1)
    return p10, ap, order


def check_reproduction(name, rows, tol=1e-4):
    out = []
    for k, ref in W6_REF[name].items():
        obs = float(np.mean([r[k] for r in rows]))
        out.append({"system": name, "metric": k, "week6_recorded": ref, "reproduced": round(obs, 4),
                    "match": abs(obs - ref) <= tol})
    return out
