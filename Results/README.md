# Results Directory

This directory stores all visual, graphical, and tabular outputs generated during experiments for the Explainable Multi-Label Aerial CBIR system.

---

## 📂 Subdirectories & Artifacts

### 1. `tables/` — Quantitative Benchmark Logs
- **`feature_comparison.csv`**: Full validation results across individual features (HSV, LBP, HOG), pairwise fusions, equal weights, z-score variations, and spatial grid features across 10 evaluation metrics.
- **`test_results.csv`**: Unseen test evaluation comparing Week 1 baseline against the selected Retrieval Engine v1.
- **`best_weights.json`**: Selected hyperparameter weights determined via validation grid-search (`HSV: 0.1, LBP: 0.7, HOG: 0.2`).

### 2. `images/` — Qualitative Retrieval Demonstrations
- **`top5_easy.png`**: High-confidence retrieval scenario where query characteristics are cleanly separated and retrieved images achieve 100% semantic agreement.
- **`top5_difficult.png`**: Complex query scenario with cluttered multi-label scene semantics demonstrating explainability of feature contributions.
- **`top5_color_dominated.png`**: Comparison showing instances where strong chromatic similarity (e.g., bare soil, water) dominates retrieval vs texture-informed ranking.
- **`top5_spatial.png`**: Retrieval results illustrating the effect of spatial quadrant layout constraints.

### 3. `graphs/` — Benchmark Plots
- **`feature_performance_comparison.png`**: High-resolution multi-metric benchmark bar chart comparing Class Precision@10, Multi-Label Precision@10, nDCG@10, and Class mAP across all core feature combinations up to Engine v1.

---

## 📊 Summary Benchmark Performance

| Method / Configuration | Class P@10 | ML P@10 | nDCG@10 | Class mAP | ML mAP |
|---|---|---|---|---|---|
| **Color (HSV)** | 0.3412 | 0.3510 | 0.6414 | 0.1596 | 0.2292 |
| **Texture (LBP)** | 0.2944 | 0.2983 | 0.6175 | 0.1452 | 0.2114 |
| **Shape (HOG)** | 0.1252 | 0.1498 | 0.3758 | 0.0853 | 0.1705 |
| **HSV + LBP (0.7 / 0.3)** | 0.3823 | 0.3777 | 0.6650 | 0.1827 | 0.2373 |
| **Week 1 Baseline (0.5 / 0.25 / 0.25)** | 0.3879 | 0.3838 | 0.6655 | 0.1882 | 0.2399 |
| **Equal Weights (1/3 each)** | 0.4021 | 0.3929 | 0.6702 | 0.1992 | 0.2427 |
| **Engine v1 (0.1 HSV + 0.7 LBP + 0.2 HOG)** | **0.4396** | **0.4094** | **0.6894** | **0.2131** | **0.2416** |
