"""
DIP Evaluation Metrics for Multi-Label Aerial Image Retrieval
Includes:
- Precision@K & Recall@K (Scene Class & Multi-Label)
- Multi-Label Jaccard Similarity / IoU @ K
- Average Precision (AP) & Mean Average Precision (mAP)
- Normalized Discounted Cumulative Gain (nDCG@K)
"""

from typing import List
import numpy as np


def compute_jaccard(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Jaccard similarity (Intersection over Union) between two binary multi-label vectors.
    """
    intersection = np.sum(np.logical_and(vec1, vec2))
    union = np.sum(np.logical_or(vec1, vec2))
    if union == 0:
        return 1.0
    return float(intersection / union)


def class_precision_at_k(query_class: str, retrieved_classes: List[str], k: int = 10) -> float:
    """
    Precision@K based on exact scene-level category match.
    """
    top_k = retrieved_classes[:k]
    if len(top_k) == 0:
        return 0.0
    matches = sum(1 for c in top_k if c == query_class)
    return float(matches / len(top_k))


def multilabel_precision_at_k(query_vec: np.ndarray, retrieved_vecs: List[np.ndarray], k: int = 10) -> float:
    """
    Multi-label Precision@K: fraction of top-K results sharing at least one label with query.
    """
    top_k = retrieved_vecs[:k]
    if len(top_k) == 0:
        return 0.0
    relevant = sum(1 for r_vec in top_k if np.sum(np.logical_and(query_vec, r_vec)) > 0)
    return float(relevant / len(top_k))


def jaccard_at_k(query_vec: np.ndarray, retrieved_vecs: List[np.ndarray], k: int = 10) -> float:
    """
    Average Jaccard similarity across the top-K retrieved images.
    """
    top_k = retrieved_vecs[:k]
    if len(top_k) == 0:
        return 0.0
    scores = [compute_jaccard(query_vec, r_vec) for r_vec in top_k]
    return float(np.mean(scores))


def compute_average_precision(relevance: List[int]) -> float:
    """
    Compute Average Precision (AP) for binary relevance list.
    """
    relevance = np.asarray(relevance)
    num_relevant = np.sum(relevance)
    if num_relevant == 0:
        return 0.0

    cumulative_hits = np.cumsum(relevance)
    ranks = np.arange(1, len(relevance) + 1)
    precision_at_k = cumulative_hits / ranks
    ap = np.sum(precision_at_k * relevance) / num_relevant
    return float(ap)


def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank K.
    Supports graded relevance (e.g. multi-label Jaccard similarities).
    """
    scores = np.asarray(relevance_scores[:k], dtype=np.float64)
    if len(scores) == 0 or np.sum(scores) == 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(scores) + 2))
    dcg = np.sum((2.0 ** scores - 1.0) / discounts)

    ideal_scores = np.sort(relevance_scores)[::-1][:k]
    idcg = np.sum((2.0 ** ideal_scores - 1.0) / discounts[:len(ideal_scores)])

    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)
