"""
retrieval_engine_v1.py   --   "Final Retrieval Engine v1"

Everything needed to turn cached features into a ranked list:
    * loading all cached components (global + spatial) for a split
    * per-component similarity matrices
    * (optional) per-query z-score normalisation and weighted fusion
    * a small RetrievalEngineV1 class that reads best_weights.json

week2_experiments.py uses the same functions, so what was tuned on the
validation set is exactly what the engine runs.
"""

import json
from pathlib import Path

import numpy as np
import sys
from pathlib import Path

# This finds the Week1 folder relative to this file and adds it to Python's search path
week1_path = Path(__file__).resolve().parent.parent / "Week1"
sys.path.append(str(week1_path))

from cache_features import (
    load_cached, HSV_TAG, LBP_TAG, HOG_TAG, color_hsv, texture_lbp, shape_hog,
)
from week1_evaluate import corr_sim, cos_sim, zscore_rows, parse_path, load_labels
from week2_spatial_features import (
    SHSV_TAG, SLBP_TAG, N_REGIONS, spatial_hsv, spatial_lbp,
)

COMPONENTS = ("hsv", "lbp", "hog", "shsv", "slbp")
TAGS = {"hsv": HSV_TAG, "lbp": LBP_TAG, "hog": HOG_TAG, "shsv": SHSV_TAG, "slbp": SLBP_TAG}
EXTRACTORS = {"hsv": color_hsv, "lbp": texture_lbp, "hog": shape_hog,
              "shsv": spatial_hsv, "slbp": spatial_lbp}
CONFIG_FILE = "best_weights.json"


# ------------------------------------------------------------------ loading
def load_split_features(split):
    """split = 'train' or 'test'. Returns (feats dict, names list, scene-class array)."""
    feats, paths = {}, None
    for comp in COMPONENTS:
        f, p = load_cached(f"{split}_{TAGS[comp]}")
        if paths is None:
            paths = p
        assert p == paths, f"{split}/{comp}: image order differs between caches"
        feats[comp] = f.astype(np.float64)
    names, classes = zip(*[parse_path(p) for p in paths])
    return feats, list(names), np.array(classes), paths


def subset(feats, idx):
    return {k: v[idx] for k, v in feats.items()}


# --------------------------------------------------------------- similarity
def region_corr_sim(Q, G, n_regions=N_REGIONS):
    """Correlation computed per region, then averaged over the regions."""
    Qr = Q.reshape(len(Q), n_regions, -1)
    Gr = G.reshape(len(G), n_regions, -1)
    return sum(corr_sim(Qr[:, r], Gr[:, r]) for r in range(n_regions)) / n_regions


def component_sims(Qf, Gf, comps=COMPONENTS):
    """Dict component -> (n_query x n_gallery) similarity matrix."""
    fn = {"hsv": corr_sim, "lbp": corr_sim, "hog": cos_sim,
          "shsv": region_corr_sim, "slbp": region_corr_sim}
    return {c: fn[c](Qf[c], Gf[c]) for c in comps}


def fuse(sims, weights, znorm=False, return_parts=False):
    """Weighted sum of component similarities (z-scored per query if znorm)."""
    parts = {}
    for c, w in weights.items():
        if w == 0:
            continue
        s = zscore_rows(sims[c]) if znorm else sims[c]
        parts[c] = w * s
    S = sum(parts.values())
    return (S, parts) if return_parts else S


# ------------------------------------------------------------------- engine
class RetrievalEngineV1:
    """
    engine = RetrievalEngineV1()                     # reads best_weights.json
    engine = RetrievalEngineV1(gallery_idx=ref_idx)  # restrict gallery (e.g. 1920 reference images)

    hits = engine.search(query_feats, k=10)          # query_feats: dict comp -> (n, D)
    hits = engine.search_image("some.jpg", k=10)     # compute features on the fly
    """

    def __init__(self, config_path=CONFIG_FILE, gallery_split="train", gallery_idx=None):
        cfg = json.loads(Path(config_path).read_text())
        self.weights, self.znorm = cfg["weights"], cfg["znorm"]
        feats, names, classes, paths = load_split_features(gallery_split)
        idx = np.arange(len(names)) if gallery_idx is None else np.asarray(gallery_idx)
        self.feats = subset(feats, idx)
        self.names = [names[i] for i in idx]
        self.classes = classes[idx]
        self.paths = [paths[i] for i in idx]

    def search(self, qfeats, k=10):
        """Returns, for each query, a list of k dicts with rank, index, name, class,
        fused score and the weighted contribution of every component."""
        used = [c for c, w in self.weights.items() if w > 0]
        sims = component_sims(qfeats, self.feats, used)
        S, parts = fuse(sims, self.weights, self.znorm, return_parts=True)
        order = np.argsort(-S, axis=1, kind="stable")[:, :k]
        results = []
        for q in range(S.shape[0]):
            results.append([{
                "rank": r + 1, "index": int(j), "name": self.names[j],
                "scene_class": str(self.classes[j]), "score": float(S[q, j]),
                "contributions": {c: float(parts[c][q, j]) for c in parts},
            } for r, j in enumerate(order[q])])
        return results

    def search_image(self, image_path, k=10):
        used = [c for c, w in self.weights.items() if w > 0]
        q = {c: EXTRACTORS[c](image_path)[None, :].astype(np.float64) for c in used}
        return self.search(q, k)[0]
