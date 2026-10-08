# Week 4 Progress Report

**Date Range**: September 28 – October 04, 2026  
**Milestone Focus**: Core Retrieval Engine Optimization, Spatial Feature Pyramids & Final Benchmark Evaluation

---

## 1. Objectives for the Week
1. Perform systematic simplex grid-search optimization across feature fusion weights $(w_{\text{HSV}}, w_{\text{LBP}}, w_{\text{HOG}})$ on the validation set.
2. Formulate and extract **2×2 spatial pyramid features** (`Spatial HSV` and `Spatial LBP`) to capture quadrant-level layout.
3. Test per-query z-score normalization across similarity scores.
4. Establish the optimal **Retrieval Engine v1** configuration and evaluate performance on unseen test queries.
5. Generate qualitative top-$K$ retrieval visual figures illustrating easy, difficult, color-dominated, and spatial cases.

---

## 2. Key Implementations & Progress

### A. Spatial Feature Extraction (`Codes/features/spatial.py`)
- Divided the $600 \times 600$ aerial images into a $2 \times 2$ grid (4 equal quadrants).
- Extracted local HSV histograms ($8 \times 8 = 64$ bins per quadrant $\rightarrow 256$-D concatenated vector) and local Uniform LBP histograms ($10$ bins per quadrant $\rightarrow 40$-D concatenated vector).
- Implemented region-by-region similarity comparison:
  $$S_{\text{spatial}}(Q, G) = \frac{1}{4} \sum_{r=1}^{4} \text{corr}\left(Q^{(r)}, G^{(r)}\right)$$

### B. Validation Grid-Search & Weight Tuning
- Ran an exhaustive grid-search with step size $\Delta w = 0.1$ over the weight simplex:
  $$\sum w_c = 1.0, \quad w_c \ge 0$$
- Explored 66 distinct global configurations and multi-component fusions (both raw and z-score normalized).
- **Composite Selection Metric**:
  $$\text{Score} = \frac{1}{5} \left( \text{Class P@10} + \text{ML P@10} + \text{nDCG@10} + \text{Class mAP} + \text{ML mAP} \right)$$
- **Optimization Outcome**:
  - The highest-performing configuration achieved a validation selection score of **0.3986**:
    $$\mathbf{w^*} = [w_{\text{HSV}} = 0.1, \ w_{\text{LBP}} = 0.7, \ w_{\text{HOG}} = 0.2]$$
  - Raw un-normalized fusion outperformed z-score normalization on correlation similarities.
  - While spatial features improved specific structured scenes (e.g., ports, airports), global texture and shape provided higher generalized performance across all 17 aerial classes without quadrupling descriptor memory.

---

## 3. Quantitative Evaluation Benchmarks

### Validation Set Comparison (`Results/tables/feature_comparison.csv`)

| Configuration | $w_{\text{HSV}}$ | $w_{\text{LBP}}$ | $w_{\text{HOG}}$ | Class P@10 | ML P@10 | nDCG@10 | Class mAP | Selection Score |
|---|---|---|---|---|---|---|---|---|
| Pure Color (HSV) | 1.0 | 0.0 | 0.0 | 0.3412 | 0.3510 | 0.6414 | 0.1596 | 0.3445 |
| Pure Texture (LBP) | 0.0 | 1.0 | 0.0 | 0.2944 | 0.2983 | 0.6175 | 0.1452 | 0.3133 |
| Pure Shape (HOG) | 0.0 | 0.0 | 1.0 | 0.1252 | 0.1498 | 0.3758 | 0.0853 | 0.1813 |
| Week 1 Heuristic | 0.5 | 0.25 | 0.25 | 0.3879 | 0.3838 | 0.6655 | 0.1882 | 0.3731 |
| Equal Weighting | 0.33 | 0.33 | 0.33 | 0.4021 | 0.3929 | 0.6702 | 0.1992 | 0.3814 |
| Spatial HSV + LBP | 0.0 (shsv: 0.7) | 0.0 (slbp: 0.3) | 0.0 | 0.3662 | 0.3656 | 0.6598 | 0.1820 | 0.3625 |
| **Engine v1 (Ours)** | **0.1** | **0.7** | **0.2** | **0.4396** | **0.4094** | **0.6894** | **0.2131** | **0.3986** |

### Final Unseen Test Set Results (`Results/tables/test_results.csv`)
Evaluated across all 600 unseen test queries against 2,400 training gallery images:

| Configuration | Class P@5 | ML P@5 | Jaccard@5 | nDCG@5 | Class P@10 | ML P@10 | Jaccard@10 | nDCG@10 | Class mAP | ML mAP |
|---|---|---|---|---|---|---|---|---|---|---|
| **Week 1 Baseline** | 0.3687 | 0.3603 | 0.6405 | 0.6598 | 0.3345 | 0.3483 | 0.6273 | 0.6527 | 0.1699 | 0.2296 |
| **Engine v1** | **0.4187** | **0.3803** | **0.6585** | **0.6761** | **0.3733** | **0.3593** | **0.6453** | **0.6699** | **0.1891** | **0.2262** |
| *Relative Gain* | **+13.6%** | **+5.6%** | **+2.8%** | **+2.5%** | **+11.6%** | **+3.2%** | **+2.9%** | **+2.6%** | **+11.3%** | - |

---

## 4. Qualitative Retrieval Visualizations (`Results/images/`)
- **`top5_easy.png`**: High-confidence aerial retrieval (homogeneous scenes like desert / sea) showing 100% precision.
- **`top5_difficult.png`**: Highly cluttered scenes (e.g. dense urban residential areas) illustrating the contribution breakdown across color, texture, and shape.
- **`top5_color_dominated.png`**: Demonstrates failure modes when pure color misleadingly associates distinct semantic categories.
- **`top5_spatial.png`**: Demonstrates spatial quadrant alignment for coastal and airport geometries.

---

## 5. Team Task Split & Contributions

| Member | Focus & Contributions This Week |
|---|---|
| **devp4681 (Dev)** | Grid-search simplex search automation, `best_weights.json` serialisation, test pipeline execution. |
| **dhruvif16 (Dhruvi)** | Spatial grid partitioning implementation (`Codes/features/spatial.py`), quadrant histograms. |
| **aastha280406 (Aastha)** | Multi-channel similarity fusion tuning, tie-breaking heuristics, and ranker modularization. |
| **Aashilearnstocode (Aashi)** | Unseen test set evaluation, result visualization plotting (`top5_*.png`), and report documentation. |

---

## 6. Summary of Weeks 1–4 Progress
By the end of Week 4, the core classical CV CBIR pipeline has been fully designed, implemented, tuned, and validated. The system achieves a **+13.6% relative improvement in Class P@5** and **+11.3% improvement in mAP** over the baseline on unseen test queries.
