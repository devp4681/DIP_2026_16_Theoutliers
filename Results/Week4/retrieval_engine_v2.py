import json
from pathlib import Path
import numpy as np
import sys
import scipy.stats as stats

week4_path = Path(__file__).resolve().parent
week2_path = week4_path.parent / "Week2"
week1_path = week4_path.parent / "Week1"
week3_path = week4_path.parent / "Week3"

sys.path.append(str(week1_path))
sys.path.append(str(week2_path))
sys.path.append(str(week3_path))

from cache_features import load_cached, color_hsv, texture_lbp, shape_hog, load_image
from week1_evaluate import corr_sim, cos_sim, parse_path
from week3_variants import texture_lbp_r2_p16

COMPONENTS = ("hsv", "lbp_r1_p8", "lbp_r2_p16", "hog")
EXTRACTORS = {
    "hsv": color_hsv,
    "lbp_r1_p8": texture_lbp,
    "lbp_r2_p16": texture_lbp_r2_p16,
    "hog": shape_hog
}
SIM_FNS = {
    "hsv": corr_sim,
    "lbp_r1_p8": corr_sim,
    "lbp_r2_p16": corr_sim,
    "hog": cos_sim
}
TAGS = {
    "hsv": "hsv_16x16",
    "lbp_r1_p8": "lbp_r1_p8",
    "lbp_r2_p16": "lbp_r2_p16",
    "hog": "hog_128_o9_c16_b2"
}

def load_split_features_v2(split):
    feats, paths = {}, None
    for comp in COMPONENTS:
        f, p = load_cached(f"{split}_{TAGS[comp]}")
        if paths is None:
            paths = p
        assert p == paths, f"{split}/{comp}: image order differs between caches"
        feats[comp] = f.astype(np.float64)
    names, classes = zip(*[parse_path(p) for p in paths])
    return feats, list(names), np.array(classes), paths

class RetrievalEngineV2:
    def __init__(self, gallery_split="train", gallery_idx=None):
        # Authoritative active configuration for the final V2 retrieval engine
        cfg = json.loads((week4_path / "v2_config.json").read_text())
        self.weights = {
            "hsv": cfg["hsv"],
            "lbp_r1_p8": cfg["lbp_r1_p8"],
            "lbp_r2_p16": cfg["lbp_r2_p16"],
            "hog": cfg["hog"]
        }
        
        rel_cfg = json.loads((week4_path / "reliability_config.json").read_text())
        self.rel_thresholds = rel_cfg["thresholds"]
        
        feats, names, classes, paths = load_split_features_v2(gallery_split)
        idx = np.arange(len(names)) if gallery_idx is None else np.asarray(gallery_idx)
        
        self.feats = {k: v[idx] for k, v in feats.items()}
        self.names = [names[i] for i in idx]
        self.classes = classes[idx]
        self.paths = [paths[i] for i in idx]
        
    def get_reliability_label(self, score):
        if score < self.rel_thresholds["LOW"]:
            return "LOW"
        elif score >= self.rel_thresholds["HIGH"]:
            return "HIGH"
        return "MEDIUM"

    def search(self, qfeats, k=5, exclude_gallery_idx=None):
        sims = {c: SIM_FNS[c](qfeats[c], self.feats[c]) for c in COMPONENTS}
        S = sum(self.weights[c] * sims[c] for c in COMPONENTS)
        
        if exclude_gallery_idx is not None:
            # exclude_gallery_idx should be a list of lists/arrays, one per query
            for q in range(S.shape[0]):
                if len(exclude_gallery_idx[q]) > 0:
                    S[q, exclude_gallery_idx[q]] = -np.inf
        
        order = np.argsort(-S, axis=1, kind="stable")[:, :k]
        results = []
        
        for q in range(S.shape[0]):
            q_res = []
            
            top1_sim = S[q, order[q, 0]]
            rel_score = float(top1_sim)
            rel_label = self.get_reliability_label(rel_score)
            
            for r, j in enumerate(order[q]):
                # Component percentiles
                pct = {}
                for c in COMPONENTS:
                    arr = sims[c][q]
                    val = arr[j]
                    pct[c] = float((arr < val).mean() * 100)
                
                q_res.append({
                    "rank": r + 1,
                    "index": int(j),
                    "name": self.names[j],
                    "path": self.paths[j],
                    "scene_class": str(self.classes[j]),
                    "final_similarity": float(S[q, j]),
                    "hsv_similarity": float(sims["hsv"][q, j]),
                    "lbp_r1_similarity": float(sims["lbp_r1_p8"][q, j]),
                    "lbp_r2_similarity": float(sims["lbp_r2_p16"][q, j]),
                    "hog_similarity": float(sims["hog"][q, j]),
                    "hsv_percentile": pct["hsv"],
                    "lbp_r1_percentile": pct["lbp_r1_p8"],
                    "lbp_r2_percentile": pct["lbp_r2_p16"],
                    "hog_percentile": pct["hog"],
                    "hsv_contribution": float(self.weights["hsv"] * sims["hsv"][q, j]),
                    "lbp_r1_contribution": float(self.weights["lbp_r1_p8"] * sims["lbp_r1_p8"][q, j]),
                    "lbp_r2_contribution": float(self.weights["lbp_r2_p16"] * sims["lbp_r2_p16"][q, j]),
                    "hog_contribution": float(self.weights["hog"] * sims["hog"][q, j])
                })
            
            results.append({
                "reliability_score": rel_score,
                "reliability_label": rel_label,
                "retrieved": q_res
            })
            
        return results

    def search_image(self, image_path, k=5, exclude_self=True):
        qfeats = {c: EXTRACTORS[c](image_path)[None, :].astype(np.float64) for c in COMPONENTS}
        ex_idx = [[]]
        if exclude_self:
            try:
                norm_q = Path(image_path).as_posix().replace('\\', '/')
                idx = [i for i, p in enumerate(self.paths) if Path(p).as_posix().replace('\\', '/') == norm_q or Path(p).name == Path(image_path).name]
                ex_idx = [idx]
            except ValueError:
                pass
        return self.search(qfeats, k, exclude_gallery_idx=ex_idx)[0]
