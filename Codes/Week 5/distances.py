"""
distances.py

Vectorized similarity and distance functions for CBIR feature fusion,
including per-query Z-score normalization and alternative histogram metrics.

All functions return similarity matrices S of shape (n_queries, n_gallery)
where LARGER values indicate GREATER similarity (compatible with np.argsort(-S)).
"""

import numpy as np


def corr_sim(Q, G):
    """
    Pearson correlation similarity (cv2.HISTCMP_CORREL).
    Centered and L2-normalized dot product.
    Output range: [-1.0, 1.0]. Larger = more similar.
    """
    Q_c = Q - Q.mean(axis=1, keepdims=True)
    G_c = G - G.mean(axis=1, keepdims=True)
    Q_norm = np.linalg.norm(Q_c, axis=1, keepdims=True) + 1e-12
    G_norm = np.linalg.norm(G_c, axis=1, keepdims=True) + 1e-12
    return (Q_c / Q_norm) @ (G_c / G_norm).T


def cos_sim(Q, G):
    """
    Cosine similarity.
    L2-normalized dot product.
    Output range: [-1.0, 1.0] (non-negative descriptors: [0.0, 1.0]). Larger = more similar.
    """
    Q_norm = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    G_norm = np.linalg.norm(G, axis=1, keepdims=True) + 1e-12
    return (Q / Q_norm) @ (G / G_norm).T


def chi2_distance(Q, G, eps=1e-12, chunk_size=120):
    """
    Symmetric Chi-Square distance:
        D(p, q) = sum_k (p_k - q_k)^2 / (p_k + q_k + eps)
    Non-negative distance: 0 = identical. Smaller = more similar.
    Computed in memory-efficient query chunks.
    """
    n_q = len(Q)
    n_g = len(G)
    D = np.zeros((n_q, n_g), dtype=np.float64)
    for s in range(0, n_q, chunk_size):
        e = min(s + chunk_size, n_q)
        q_chunk = Q[s:e, None, :]
        g_chunk = G[None, :, :]
        diff = q_chunk - g_chunk
        denom = q_chunk + g_chunk + eps
        D[s:e] = np.sum((diff * diff) / denom, axis=-1)
    return D


def chi2_sim(Q, G, eps=1e-12, chunk_size=120):
    """
    Chi-Square converted to similarity score:
        S = -D_chi2
    Larger = more similar (rank direction preserved).
    """
    return -chi2_distance(Q, G, eps=eps, chunk_size=chunk_size)


def bhattacharyya_sim(Q, G):
    """
    Bhattacharyya coefficient:
        BC(p, q) = sum_k sqrt(p_k * q_k)
    For normalized histograms, BC in [0.0, 1.0], where 1.0 = identical.
    Larger = more similar.
    """
    sq_Q = np.sqrt(np.maximum(Q, 0.0))
    sq_G = np.sqrt(np.maximum(G, 0.0))
    norm_Q = np.linalg.norm(sq_Q, axis=1, keepdims=True) + 1e-12
    norm_G = np.linalg.norm(sq_G, axis=1, keepdims=True) + 1e-12
    return (sq_Q / norm_Q) @ (sq_G / norm_G).T


def hellinger_sim(Q, G):
    """
    Hellinger distance converted to similarity:
        H(p, q) = sqrt(1 - BC(p, q)) in [0, 1]
        S = -H(p, q)  (or 1 - H)
    Here returning 1.0 - H(p, q) so similarity is in [0.0, 1.0], larger = more similar.
    """
    bc = np.clip(bhattacharyya_sim(Q, G), 0.0, 1.0)
    H = np.sqrt(np.maximum(1.0 - bc, 0.0))
    return 1.0 - H


def hist_intersection_sim(Q, G, chunk_size=120):
    """
    Histogram Intersection similarity:
        K(p, q) = sum_k min(p_k, q_k) / max(sum(p), sum(q))
    Normalized so K in [0.0, 1.0], larger = more similar.
    Calculated efficiently using the L1 identity:
        sum_k min(p_k, q_k) = 0.5 * (sum(p) + sum(q) - sum|p_k - q_k|)
    """
    sum_q = Q.sum(axis=1, keepdims=True)
    sum_g = G.sum(axis=1, keepdims=True).T
    n_q = len(Q)
    n_g = len(G)
    l1 = np.zeros((n_q, n_g), dtype=np.float64)
    for s in range(0, n_q, chunk_size):
        e = min(s + chunk_size, n_q)
        l1[s:e] = np.sum(np.abs(Q[s:e, None, :] - G[None, :, :]), axis=-1)
    inter = 0.5 * (sum_q + sum_g - l1)
    denom = np.maximum(sum_q, sum_g) + 1e-12
    return inter / denom


def zscore_per_query(S, eps=1e-12):
    """
    Per-query Z-score normalization for a similarity matrix S of shape (n_queries, n_gallery).
    Each row (query) is normalized independently across gallery similarity scores:
        Z[q, g] = (S[q, g] - mean_q) / (std_q + eps)
    Zero standard deviation is safely handled by zeroing the entire row.
    """
    mean = S.mean(axis=1, keepdims=True)
    std = S.std(axis=1, keepdims=True)
    return np.where(std < eps, 0.0, (S - mean) / (std + eps))


AVAILABLE_SIM_FNS = {
    "corr": corr_sim,
    "cos": cos_sim,
    "chi2": chi2_sim,
    "bhatt": bhattacharyya_sim,
    "hellinger": hellinger_sim,
    "intersection": hist_intersection_sim
}
