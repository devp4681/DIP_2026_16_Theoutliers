# Week 3 Progress Report

**Date Range**: September 21 – September 27, 2026  
**Milestone Focus**: Evaluation Protocol Setup, Multi-Label Metrics Implementation & Baseline Fusion Benchmarking

---

## 1. Objectives for the Week
1. Establish a rigorous, leak-free evaluation protocol by partitioning the 2,400 training images into reference and validation subsets.
2. Develop standard quantitative multi-label CBIR metrics:
   - Scene Class Precision@K (Class P@K)
   - Multi-Label Precision@K (ML P@K)
   - Multi-Label Jaccard Similarity / IoU @ K (Jaccard@K)
   - Normalized Discounted Cumulative Gain (nDCG@K)
   - Mean Average Precision (Class mAP & Multi-Label mAP)
3. Evaluate baseline feature combinations (HSV, LBP, HOG, HSV+LBP, and Week 1 heuristic weights) on the validation set.

---

## 2. Key Implementations & Progress

### A. Stratified Split Protocol (`split_indices.json`)
- To ensure unbiased hyperparameter tuning without touching the 600 test images, the 2,400 training set images were partitioned into:
  - **1,920 Reference Images (Gallery)** (80%)
  - **480 Validation Queries** (20%)
- The split is stratified by scene category with a fixed random seed (`seed = 42`) and persisted as JSON for exact reproducibility across all experiments.

### B. Multi-Label Metric Suite (`Codes/utils/metrics.py`)
- **Class P@K**: Percentage of top-$K$ retrieved images matching the exact query scene category.
- **ML P@K**: Percentage of top-$K$ retrievals that share at least one active ground-truth label with the query.
- **Jaccard@K**: Mean multi-label Intersection-over-Union (IoU) across top-$K$ retrieved images:
  $$\text{Jaccard}(y_q, y_r) = \frac{|y_q \cap y_r|}{|y_q \cup y_r|}$$
- **nDCG@K**: Measures ranking quality with graded multi-label relevance, penalizing relevant retrievals positioned at lower ranks.
- **mAP**: Mean Average Precision computed over the ranked gallery.

### C. Preprocessing Pipeline (`Codes/data/preprocessing.py`)
- Implemented **Contrast Limited Adaptive Histogram Equalization (CLAHE)** on the Luminance channel in LAB color space to mitigate aerial contrast variations and uneven sunlight illumination.
- Implemented Gaussian smoothing and bilateral filtering for noise reduction while maintaining sharp structural edges.

---

## 3. Quantitative Baseline Results (Validation Set)

| Method / Configuration | Class P@5 | Class P@10 | ML P@10 | Jaccard@10 | nDCG@10 | Class mAP | ML mAP |
|---|---|---|---|---|---|---|---|
| **Color (HSV alone)** | 0.4042 | 0.3412 | 0.3510 | 0.6149 | 0.6414 | 0.1596 | 0.2292 |
| **Texture (LBP alone)** | 0.3338 | 0.2944 | 0.2983 | 0.5969 | 0.6175 | 0.1452 | 0.2114 |
| **Shape (HOG alone)** | 0.1429 | 0.1252 | 0.1498 | 0.3615 | 0.3758 | 0.0853 | 0.1705 |
| **HSV + LBP (0.7 / 0.3)** | 0.4542 | 0.3823 | 0.3777 | 0.6382 | 0.6650 | 0.1827 | 0.2373 |
| **Week 1 Baseline (0.5 / 0.25 / 0.25)** | 0.4621 | 0.3879 | 0.3838 | 0.6397 | 0.6655 | 0.1882 | 0.2399 |

---

## 4. Key Observations & Insights
1. **Multi-Feature Complementarity**: Combining HSV and LBP yields significant gains over either modality alone (+4.1% Class P@10 over pure HSV; +8.8% over pure LBP).
2. **HOG Sensitivity**: Unweighted HOG achieves lower precision on aerial images due to rotation variance, but when combined with color and texture, it provides essential geometric boundary filtering.
3. **Multi-Label vs Single-Label**: Jaccard similarity remains consistently higher (~0.64) than single-class precision (~0.39), indicating that retrieved images capture semantic scene contents (e.g. grass, trees, roads) even when the high-level scene classification differs.

---

## 5. Team Task Split & Contributions

| Member | Focus & Contributions This Week |
|---|---|
| **devp4681 (Dev)** | Stratified train/validation splitting protocol, split JSON serialization, dataset reproducibility tests. |
| **dhruvif16 (Dhruvi)** | Preprocessing pipeline implementation (CLAHE in LAB space, bilateral and Gaussian smoothing). |
| **aastha280406 (Aastha)** | Similarity ranking function refinement, score fusion implementation, baseline execution. |
| **Aashilearnstocode (Aashi)** | Multi-label evaluation metrics implementation (Jaccard@K, nDCG@K, mAP) and benchmark logging. |

---

## 6. Next Steps for Week 4
- Perform systematic grid-search over feature weights to discover optimal fusion parameters.
- Explore spatial partitioning (2×2 regional pyramids) for color and texture to capture spatial layout.
- Evaluate the finalized engine against unseen test queries.
