"""
cv_experiment.py

Central, reusable 5-Fold Cross-Validation evaluation pipeline for CBIR experiments on AID Multi-Label.

Protocol:
- Loads folds_5.json (5 stratified folds of 2,400 training images: 480 val queries vs 1,920 gallery).
- Strictly partitions query and gallery sets for every fold.
- Excludes query self-matches if present in gallery.
- Evaluates:
    * Class P@10 (and Class P@5)
    * ML P@10 (and ML P@5, Jaccard >= 0.8)
    * ML mAP (full-gallery average precision over multi-label relevant items)
    * nDCG@10 (graded multi-label Jaccard DCG vs ideal DCG)
- Computes per-fold results, then mean ± standard deviation across all 5 folds.
- Can precompute and cache fold similarity matrices for instantaneous grid weight searches.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

# Setup imports relative to project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

sys.path.append(str(project_root / "Week1"))
sys.path.append(str(project_root / "Week2"))
sys.path.append(str(project_root / "Week4"))
sys.path.append(str(current_dir))

from cache_features import load_cached
from week1_evaluate import load_labels, parse_path
from distances import (
    corr_sim, cos_sim, chi2_sim, bhattacharyya_sim, hellinger_sim,
    hist_intersection_sim, zscore_per_query, AVAILABLE_SIM_FNS
)

COMPONENTS = ("hsv", "lbp_r1_p8", "lbp_r2_p16", "hog")
CACHE_TAGS = {
    "hsv": "train_hsv_16x16",
    "lbp_r1_p8": "train_lbp_r1_p8",
    "lbp_r2_p16": "train_lbp_r2_p16",
    "hog": "train_hog_128_o9_c16_b2"
}

DEFAULT_SIM_FNS = {
    "hsv": corr_sim,
    "lbp_r1_p8": corr_sim,
    "lbp_r2_p16": corr_sim,
    "hog": cos_sim
}

DEFAULT_WEIGHTS = {
    "hsv": 0.10,
    "lbp_r1_p8": 0.35,
    "lbp_r2_p16": 0.35,
    "hog": 0.20
}

REL_JACCARD = 0.8


def compute_jaccard_matrix(Lq, Lg):
    """
    Computes pairwise Jaccard similarity between multi-label query vectors Lq and gallery vectors Lg.
    Matrix shape: (n_queries, n_gallery).
    """
    inter = Lq @ Lg.T
    union = Lq.sum(axis=1)[:, None] + Lg.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)


def average_precision(rel_sorted):
    """
    Computes Mean Average Precision over full gallery for multi-label binary relevance.
    rel_sorted: (n_queries, n_gallery) boolean array in ranked retrieval order.
    """
    r = rel_sorted.astype(np.float64)
    cum = np.cumsum(r, axis=1)
    prec = cum / np.arange(1, r.shape[1] + 1)
    n_rel = r.sum(axis=1)
    ok = n_rel > 0
    ap = np.full(r.shape[0], np.nan)
    ap[ok] = (prec[ok] * r[ok]).sum(axis=1) / n_rel[ok]
    return float(np.nanmean(ap))


def evaluate_ranking(S, q_cls, g_cls, J_mat, exclude_mask=None, ks=(5, 10)):
    """
    Evaluates retrieval ranking given similarity matrix S of shape (n_queries, n_gallery).
    If exclude_mask is provided, masked entries (True) are set to -infinity before sorting.
    """
    n_q = S.shape[0]
    S_eval = S.copy()
    if exclude_mask is not None:
        S_eval[exclude_mask] = -np.inf

    order = np.argsort(-S_eval, axis=1, kind="stable")
    rows = np.arange(n_q)[:, None]

    same_class = (q_cls[:, None] == g_cls[None, :])
    rel_ml = (J_mat >= REL_JACCARD)

    rc = same_class[rows, order]
    rm = rel_ml[rows, order]
    jg = J_mat[rows, order]

    metrics = {}
    for K in ks:
        metrics[f"Class P@{K}"] = float(rc[:, :K].mean())
        metrics[f"ML P@{K}"] = float(rm[:, :K].mean())
        metrics[f"Jaccard@{K}"] = float(jg[:, :K].mean())

        disc = 1.0 / np.log2(np.arange(2, K + 2))
        dcg = (jg[:, :K] * disc).sum(axis=1)
        ideal = -np.sort(-J_mat, axis=1)[:, :K]
        idcg = (ideal * disc).sum(axis=1)
        ok = idcg > 0
        metrics[f"nDCG@{K}"] = float((dcg[ok] / idcg[ok]).mean())

    metrics["Class mAP"] = average_precision(rc)
    metrics["ML mAP"] = average_precision(rm)

    # Standard selection score combining all key metrics
    sel_metrics = ["Class P@10", "ML P@10", "nDCG@10", "Class mAP", "ML mAP"]
    metrics["selection_score"] = float(np.mean([metrics[m] for m in sel_metrics]))

    return metrics


class CVDataset:
    """
    Loads all training features, labels, paths, and 5-fold CV splits once.
    Provides fast access to fold subsets.
    """
    def __init__(self, folds_file=None):
        if folds_file is None:
            folds_file = project_root / "folds_5.json"
        self.folds = json.loads(Path(folds_file).read_text())
        assert len(self.folds) == 5, f"Expected 5 folds, found {len(self.folds)}"

        # Load all 4 training feature components
        self.feats = {}
        paths = None
        for comp in COMPONENTS:
            f, p = load_cached(CACHE_TAGS[comp])
            if paths is None:
                paths = p
            assert p == paths, f"Component {comp} image ordering does not match reference paths"
            self.feats[comp] = f.astype(np.float64)

        self.paths = paths
        names, classes = zip(*[parse_path(p) for p in paths])
        self.names = list(names)
        self.classes = np.array(classes)
        self.labels = load_labels(self.names)
        assert len(self.names) == 2400, f"Expected 2400 train images, got {len(self.names)}"

    def get_fold_data(self, fold_idx):
        fold = self.folds[fold_idx]
        q_idx = np.array(fold["query_idx"])
        g_idx = np.array(fold["gallery_idx"])

        q_feats = {c: self.feats[c][q_idx] for c in COMPONENTS}
        g_feats = {c: self.feats[c][g_idx] for c in COMPONENTS}

        q_cls = self.classes[q_idx]
        g_cls = self.classes[g_idx]

        Lq = self.labels[q_idx]
        Lg = self.labels[g_idx]
        J_mat = compute_jaccard_matrix(Lq, Lg)

        # Build self-exclusion mask (True if query image is also in gallery)
        q_paths = [self.paths[i] for i in q_idx]
        g_paths = [self.paths[i] for i in g_idx]
        g_path_set = {p: idx for idx, p in enumerate(g_paths)}

        exclude_mask = np.zeros((len(q_idx), len(g_idx)), dtype=bool)
        for i, qp in enumerate(q_paths):
            if qp in g_path_set:
                exclude_mask[i, g_path_set[qp]] = True

        return {
            "q_idx": q_idx,
            "g_idx": g_idx,
            "q_feats": q_feats,
            "g_feats": g_feats,
            "q_cls": q_cls,
            "g_cls": g_cls,
            "J_mat": J_mat,
            "exclude_mask": exclude_mask
        }


def compute_fold_sims(fold_data, sim_fns):
    """
    Computes raw similarity matrices for all 4 components on a single fold.
    Returns dict: comp -> (480, 1920) array.
    """
    sims = {}
    for c in COMPONENTS:
        fn = sim_fns.get(c, DEFAULT_SIM_FNS[c])
        sims[c] = fn(fold_data["q_feats"][c], fold_data["g_feats"][c])
    return sims


def evaluate_fused_fold(sims, weights, fold_data, znorm=False):
    """
    Fuses similarities and evaluates the resulting ranking on a single fold.
    """
    active_comps = [c for c, w in weights.items() if w > 0]
    processed_sims = {}
    for c in active_comps:
        if znorm:
            processed_sims[c] = zscore_per_query(sims[c])
        else:
            processed_sims[c] = sims[c]

    S = sum(weights[c] * processed_sims[c] for c in active_comps)
    return evaluate_ranking(S, fold_data["q_cls"], fold_data["g_cls"], fold_data["J_mat"], fold_data["exclude_mask"])


def run_cv_experiment(experiment_name, weights=None, sim_fns=None, znorm=False, dataset=None, precomputed_sims=None):
    """
    Runs full 5-fold cross-validation experiment.

    Returns:
    - summary: dict with mean and std for all metrics
    - fold_df: DataFrame of per-fold metrics
    - precomputed_sims: list of per-fold raw component similarity dicts (reusable for fast weight search)
    """
    if dataset is None:
        dataset = CVDataset()

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    if sim_fns is None:
        sim_fns = DEFAULT_SIM_FNS.copy()

    start_time = time.time()
    per_fold_results = []
    saved_sims = [] if precomputed_sims is None else precomputed_sims

    for fold_i in range(5):
        fold_data = dataset.get_fold_data(fold_i)

        if precomputed_sims is None:
            sims = compute_fold_sims(fold_data, sim_fns)
            saved_sims.append(sims)
        else:
            sims = precomputed_sims[fold_i]

        fold_metrics = evaluate_fused_fold(sims, weights, fold_data, znorm=znorm)
        fold_metrics["fold"] = fold_i + 1
        fold_metrics["experiment"] = experiment_name
        per_fold_results.append(fold_metrics)

    runtime = time.time() - start_time
    df_folds = pd.DataFrame(per_fold_results)

    # Core required metrics
    key_metrics = ["Class P@10", "ML P@10", "ML mAP", "nDCG@10", "Class P@5", "ML P@5", "selection_score"]
    summary = {
        "experiment": experiment_name,
        "znorm": znorm,
        "runtime_sec": round(runtime, 2),
        "weights": weights,
        "sim_fns": {c: getattr(fn, "__name__", str(fn)) for c, fn in sim_fns.items()}
    }

    for m in key_metrics:
        summary[f"{m}_mean"] = float(df_folds[m].mean())
        summary[f"{m}_std"] = float(df_folds[m].std())

    return summary, df_folds, saved_sims


def format_summary_row(summary):
    """Formats one clean line for display tables."""
    return (
        f"{summary['experiment']:<32} "
        f"{summary['Class P@10_mean']:.4f} +/- {summary['Class P@10_std']:.4f}   "
        f"{summary['ML P@10_mean']:.4f} +/- {summary['ML P@10_std']:.4f}   "
        f"{summary['ML mAP_mean']:.4f} +/- {summary['ML mAP_std']:.4f}   "
        f"{summary['nDCG@10_mean']:.4f} +/- {summary['nDCG@10_std']:.4f}"
    )


if __name__ == "__main__":
    print("=" * 90)
    print("RUNNING REUSABLE 5-FOLD CV PIPELINE: V2 BASELINE VERIFICATION")
    print("=" * 90)

    dataset = CVDataset()
    summary, df_folds, _ = run_cv_experiment("V2 baseline", znorm=False, dataset=dataset)

    print("\nPer-Fold Results:")
    cols_show = ["fold", "Class P@10", "ML P@10", "ML mAP", "nDCG@10", "selection_score"]
    print(df_folds[cols_show].round(4).to_string(index=False))

    print("\n" + "=" * 90)
    print(f"{'Experiment':<32} {'Class P@10':<19} {'ML P@10':<19} {'ML mAP':<19} {'nDCG@10':<19}")
    print("-" * 90)
    print(format_summary_row(summary))
    print("=" * 90)
