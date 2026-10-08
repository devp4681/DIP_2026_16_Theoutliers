"""
DIP Feature Fusion and Ranking Engine
Handles:
- Score normalization across candidate gallery (per-query z-norm)
- Weighted linear feature fusion
- Top-K ranking
"""

from typing import Dict, List, Tuple
import numpy as np
from .similarity import correlation_similarity, cosine_similarity, chi_square_similarity


class CBIRRanker:
    """
    Multi-feature fusion ranking engine.
    Computes individual component similarities, applies optional z-score normalization,
    and calculates weighted sum for Top-K ranking.
    """

    def __init__(self, weights: Dict[str, float] = None, znorm: bool = False):
        if weights:
            self.weights = weights
        else:
            self.weights = {'color': 0.4, 'texture': 0.4, 'shape': 0.2}
        self.znorm = znorm

    def score_single_feature(
        self,
        query_feat: np.ndarray,
        gallery_feats: np.ndarray,
        metric: str = 'corr'
    ) -> np.ndarray:
        """
        Compute similarity score vector between query feature and gallery features.
        """
        scores = []
        for g_feat in gallery_feats:
            if metric == 'corr':
                s = correlation_similarity(query_feat, g_feat)
            elif metric == 'cos':
                s = cosine_similarity(query_feat, g_feat)
            elif metric == 'chi2':
                s = chi_square_similarity(query_feat, g_feat)
            else:
                s = correlation_similarity(query_feat, g_feat)
            scores.append(s)
        return np.array(scores, dtype=np.float64)

    def rank(
        self,
        query_feats: Dict[str, np.ndarray],
        gallery_feats: Dict[str, np.ndarray],
        gallery_names: List[str],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rank gallery images for a query feature dictionary.
        Returns list of top_k dictionaries with rank, name, index, fused score, and component subscores.
        """
        n_gallery = len(gallery_names)
        combined_scores = np.zeros(n_gallery, dtype=np.float64)
        subscores = {}

        for feat_name, w in self.weights.items():
            if feat_name not in query_feats or feat_name not in gallery_feats or w <= 0:
                continue

            q_f = query_feats[feat_name]
            g_f = gallery_feats[feat_name]

            if 'hog' in feat_name.lower() or 'moments' in feat_name.lower():
                metric = 'cos'
            else:
                metric = 'corr'

            s = self.score_single_feature(q_f, g_f, metric=metric)

            if self.znorm:
                mu = np.mean(s)
                sigma = np.std(s)
                s = (s - mu) / (sigma + 1e-7)

            subscores[feat_name] = s
            combined_scores += w * s

        ranked_indices = np.argsort(combined_scores)[::-1]

        results = []
        for rank_idx, idx in enumerate(ranked_indices[:top_k], 1):
            results.append({
                'rank': rank_idx,
                'name': gallery_names[idx],
                'index': int(idx),
                'score': float(combined_scores[idx]),
                'subscores': {fn: float(subscores[fn][idx]) for fn in subscores}
            })

        return results
